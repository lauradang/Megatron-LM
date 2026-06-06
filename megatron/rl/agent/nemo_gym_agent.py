# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import json
import os
import random
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from .api import (
    AgentBaseModel,
    EvaluationAgent,
    EvaluationRequest,
    GroupedRolloutGenerator,
    GroupedRolloutRequest,
    RewardEvaluationResult,
    TokenRollout,
)
from .reward_only_agent import RewardOnlyEvaluationResponse


_AGENT_REF_KEY = "agent_ref"
_RESPONSES_CREATE_PARAMS_KEY = "responses_create_params"
_TASK_INDEX_KEY = "_ng_task_index"
_ROLLOUT_INDEX_KEY = "_ng_rollout_index"


class NemoGymOutputItem(AgentBaseModel):
    """Subset of a NeMo Gym Responses output item consumed by Megatron RL."""

    prompt_token_ids: list[int] | None = None
    generation_token_ids: list[int] | None = None
    generation_log_probs: list[float] | None = None
    policy_epoch: list[tuple[int, int]] | None = None
    kv_cache_epoch: list[tuple[int, int]] | None = None
    num_evictions: int | None = None

    @property
    def is_trainable(self) -> bool:
        return self.generation_token_ids is not None


class NemoGymResponsePayload(AgentBaseModel):
    output: list[NemoGymOutputItem] = Field(default_factory=list)


class NemoGymInstanceConfig(AgentBaseModel):
    mask_sample: bool = False
    problem_info: dict[str, Any] = Field(default_factory=dict)
    persistent_dir: str | None = None


class NemoGymRunResult(AgentBaseModel):
    response: NemoGymResponsePayload = Field(default_factory=NemoGymResponsePayload)
    reward: float = 0.0
    resolved: bool | None = None
    instance_config: NemoGymInstanceConfig = Field(default_factory=NemoGymInstanceConfig)
    responses_create_params: dict[str, Any] | None = None


class NemoGymAgent(GroupedRolloutGenerator, EvaluationAgent):
    """Megatron RL adapter for NeMo Gym `/run` environments."""

    env_id: str = "nemo_gym/swe_agents"
    input_jsonl_fpath: str | None = None
    validation_input_jsonl_fpath: str | None = None
    rows: list[dict[str, Any]] | None = None
    validation_rows: list[dict[str, Any]] | None = None

    train_agent_name: str = "swe_agents_train"
    validation_agent_name: str | None = "swe_agents_val"
    gym_root: str | None = None

    policy_model_name: str = "megatron-policy"
    policy_api_key: str = "NONE"
    policy_base_url: str | list[str] = Field(default_factory=lambda: ["http://127.0.0.1:8294/v1"])
    nemo_gym: dict[str, Any] = Field(default_factory=dict)

    _rh: Any = PrivateAttr(default=None)
    _rch: Any = PrivateAttr(default=None)
    _head_server_config: Any = PrivateAttr(default=None)
    _train_rows: list[dict[str, Any]] | None = PrivateAttr(default=None)
    _validation_rows: list[dict[str, Any]] | None = PrivateAttr(default=None)

    def _resolve_path(self, path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate

        search_roots = [Path.cwd()]
        if self.gym_root:
            search_roots.append(Path(self.gym_root))
        if os.environ.get("NEMO_GYM_ROOT"):
            search_roots.append(Path(os.environ["NEMO_GYM_ROOT"]))
        search_roots.append(Path("/workspace/Gym"))

        for root in search_roots:
            resolved = root / candidate
            if resolved.exists():
                return resolved
        return search_roots[0] / candidate

    def _load_rows(self, validation: bool = False) -> list[dict[str, Any]]:
        cached_rows = self._validation_rows if validation else self._train_rows
        if cached_rows is not None:
            return cached_rows

        configured_rows = self.validation_rows if validation and self.validation_rows else self.rows
        if configured_rows is not None:
            rows = [copy.deepcopy(row) for row in configured_rows]
        else:
            fpath = (
                self.validation_input_jsonl_fpath
                if validation and self.validation_input_jsonl_fpath
                else self.input_jsonl_fpath
            )
            if not fpath:
                raise ValueError(
                    "NemoGymAgent requires input_jsonl_fpath or rows in agent_args."
                )
            resolved_fpath = self._resolve_path(fpath)
            with resolved_fpath.open() as f:
                rows = [json.loads(line) for line in f if line.strip()]

        if not rows:
            raise ValueError("NemoGymAgent dataset is empty.")

        if validation:
            self._validation_rows = rows
        else:
            self._train_rows = rows
        return rows

    def _initial_global_config_dict(self) -> dict[str, Any]:
        config = copy.deepcopy(self.nemo_gym)
        config.setdefault("policy_model_name", self.policy_model_name)
        config.setdefault("policy_api_key", self.policy_api_key)
        config.setdefault("policy_base_url", self.policy_base_url)
        return config

    def _ensure_started(self) -> None:
        if self._rh is not None:
            return

        try:
            from nemo_gym.cli import GlobalConfigDictParserConfig, RunHelper
            from nemo_gym.rollout_collection import RolloutCollectionHelper
            from omegaconf import DictConfig
        except ImportError as exc:
            raise ImportError(
                "NemoGymAgent requires NeMo Gym to be installed. Use the Megatron "
                "`rl-gym` container target or install the Gym checkout into this environment."
            ) from exc

        self._rh = RunHelper()
        self._rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig(self._initial_global_config_dict()),
                skip_load_from_cli=True,
            )
        )
        self._rch = RolloutCollectionHelper()
        self._head_server_config = self._rh._server_client.head_server_config

    def shutdown(self) -> None:
        if self._rh is not None:
            self._rh.shutdown()
        self._rh = None
        self._rch = None
        self._head_server_config = None

    def _agent_name(self, validation: bool = False) -> str:
        if validation and self.validation_agent_name:
            return self.validation_agent_name
        return self.train_agent_name

    def _apply_generation_args(
        self,
        responses_create_params: dict[str, Any],
        request: GroupedRolloutRequest | EvaluationRequest,
    ) -> None:
        generation_args = request.generation_args
        responses_create_params["model"] = self.policy_model_name
        if generation_args.temperature is not None:
            responses_create_params["temperature"] = generation_args.temperature
        if generation_args.top_p is not None:
            responses_create_params["top_p"] = generation_args.top_p
        if generation_args.max_tokens is not None:
            existing = responses_create_params.get("max_output_tokens")
            responses_create_params["max_output_tokens"] = (
                min(existing, generation_args.max_tokens)
                if existing is not None
                else generation_args.max_tokens
            )

    def _build_example(
        self,
        row: dict[str, Any],
        *,
        task_index: int,
        rollout_index: int,
        request: GroupedRolloutRequest | EvaluationRequest,
    ) -> dict[str, Any]:
        example = copy.deepcopy(row)
        example.setdefault(_AGENT_REF_KEY, {})
        example[_AGENT_REF_KEY]["name"] = self._agent_name(request.validation)

        if _RESPONSES_CREATE_PARAMS_KEY not in example:
            raise ValueError(f"NeMo Gym row is missing `{_RESPONSES_CREATE_PARAMS_KEY}`.")
        self._apply_generation_args(example[_RESPONSES_CREATE_PARAMS_KEY], request)

        example[_TASK_INDEX_KEY] = task_index
        example[_ROLLOUT_INDEX_KEY] = rollout_index
        return example

    async def _run_examples(
        self, examples: list[dict[str, Any]]
    ) -> list[tuple[dict[str, Any], NemoGymRunResult]]:
        self._ensure_started()
        tasks = list(
            self._rch.run_examples(
                examples=examples,
                head_server_config=self._head_server_config,
            )
        )

        results = []
        for task in tasks:
            row, result = await task
            results.append((row, NemoGymRunResult.model_validate(result)))

        return sorted(results, key=lambda pair: pair[0].get(_ROLLOUT_INDEX_KEY, 0))

    @staticmethod
    def _is_contiguous_prefix(seen_token_ids: list[int], prompt_token_ids: list[int]) -> bool:
        return seen_token_ids == prompt_token_ids[: len(seen_token_ids)]

    def rollout_from_gym_result(self, gym_result: NemoGymRunResult) -> TokenRollout:
        trajectories: list[list[int]] = []
        generation_masks: list[list[bool]] = []
        logprobs: list[list[float]] = []
        policy_epoch: list[list[tuple[int, int]]] = []
        kv_cache_epoch: list[list[tuple[int, int]]] = []
        num_evictions: list[int] = []
        seen_token_ids: list[int] = []

        for item in gym_result.response.output:
            prompt_token_ids = item.prompt_token_ids or []
            generation_token_ids = item.generation_token_ids
            if generation_token_ids is None:
                continue

            if not self._is_contiguous_prefix(seen_token_ids, prompt_token_ids):
                raise ValueError(
                    "NeMo Gym returned non-contiguous prompt token IDs. This usually means "
                    "the agent trajectory was truncated or retokenized between turns."
                )

            prompt_delta = prompt_token_ids[len(seen_token_ids) :]
            turn_tokens = prompt_delta + generation_token_ids
            turn_generation_mask = [False] * len(prompt_delta) + [True] * len(generation_token_ids)

            trajectories.append(turn_tokens)
            generation_masks.append(turn_generation_mask)
            logprobs.append(item.generation_log_probs or [])
            policy_epoch.append(item.policy_epoch or [(0, 0)])
            kv_cache_epoch.append(item.kv_cache_epoch or [(0, 0)])
            num_evictions.append(item.num_evictions or 0)

            seen_token_ids.extend(prompt_delta)
            seen_token_ids.extend(generation_token_ids)

        if not trajectories:
            prompt_only_items = [
                item.prompt_token_ids
                for item in gym_result.response.output
                if item.prompt_token_ids
            ]
            if not prompt_only_items:
                raise ValueError(
                    "NeMo Gym returned no trainable token output and no prompt token IDs to mask."
                )
            trajectories.append(prompt_only_items[-1])
            generation_masks.append([False] * len(prompt_only_items[-1]))
            logprobs.append([])
            policy_epoch.append([(0, 0)])
            kv_cache_epoch.append([(0, 0)])
            num_evictions.append(0)

        mask_sample = gym_result.instance_config.mask_sample
        if mask_sample:
            generation_masks = [[False] * len(turn) for turn in trajectories]

        problem_id = gym_result.instance_config.problem_info.get("instance_id")

        return TokenRollout(
            trajectory=trajectories,
            reward=gym_result.reward,
            generation_mask=generation_masks,
            logprobs=logprobs,
            env_id=self.env_id,
            problem_id=problem_id,
            policy_epoch=policy_epoch,
            kv_cache_epoch=kv_cache_epoch,
            num_evictions=num_evictions,
            resolved=gym_result.resolved,
            mask_sample=mask_sample,
            persistent_dir=gym_result.instance_config.persistent_dir,
            full_result=gym_result.model_dump(),
        )

    async def group_rollout(self, request: GroupedRolloutRequest) -> list[TokenRollout]:
        rows = self._load_rows(validation=request.validation)
        task_index = random.randrange(len(rows))
        examples = [
            self._build_example(
                rows[task_index],
                task_index=task_index,
                rollout_index=rollout_index,
                request=request,
            )
            for rollout_index in range(request.rollouts_per_group)
        ]
        results = await self._run_examples(examples)
        return [self.rollout_from_gym_result(result) for _, result in results]

    async def run_evaluation(self, request: EvaluationRequest) -> RewardOnlyEvaluationResponse:
        rows = self._load_rows(validation=request.validation)
        count = min(request.num_prompts, len(rows))
        examples = [
            self._build_example(
                rows[task_index],
                task_index=task_index,
                rollout_index=0,
                request=request,
            )
            for task_index in range(count)
        ]
        results = await self._run_examples(examples)

        evaluation_results = []
        for row, gym_result in results:
            responses_create_params = row[_RESPONSES_CREATE_PARAMS_KEY]
            metadata = responses_create_params.get("metadata", {})
            evaluation_results.append(
                RewardEvaluationResult(
                    prompt=json.dumps(responses_create_params.get("input", [])),
                    response="",
                    reward=gym_result.reward,
                    problem_id=gym_result.instance_config.problem_info.get(
                        "instance_id", metadata.get("instance_id")
                    ),
                    resolved=gym_result.resolved,
                    mask_sample=gym_result.instance_config.mask_sample,
                    full_result=gym_result.model_dump(),
                )
            )

        return RewardOnlyEvaluationResponse(results=evaluation_results, env_id=self.env_id)
