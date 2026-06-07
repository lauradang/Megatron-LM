# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import copy
import json
import logging
import math
import os
import random
import time
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
_OPTIONAL_ROLLOUT_METRIC_KEYS = (
    "patch_exists",
    "agent_timed_out",
    "eval_timed_out",
    "ray_queue_time",
    "openhands_run_time",
    "generation_apptainer_spinup_time",
    "final_eval_apptainer_spinup_time",
)
_AGENT_ERROR_KIND_KEY = "agent_error_kind"
_AGENT_ERROR_KIND_VALUES = ("max_iteration", "context_window", "stuck_in_loop", "other")
_SKIP_RESULT_METRIC_SEARCH_KEYS = {
    "completion",
    "completions",
    "input",
    "messages",
    "output",
    "patch",
    "response",
    "trajectory",
    "trajectories",
}

logger = logging.getLogger(__name__)


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


class NemoGymEvaluationResponse(RewardOnlyEvaluationResponse):
    """Evaluation response with NeMo Gym health metrics."""

    type_name: str = "NemoGymEvaluationResponse"

    def metrics(self):
        return {
            "reward": [el.reward for el in self.results],
            "resolved_rate": [float(getattr(el, "resolved", False) is True) for el in self.results],
            "mask_sample_rate": [float(bool(getattr(el, "mask_sample", False))) for el in self.results],
            "trainable_rate": [float(bool(getattr(el, "trainable", False))) for el in self.results],
            "agent_timed_out_rate": [
                float(bool(getattr(el, "agent_timed_out", False))) for el in self.results
            ],
            "eval_timed_out_rate": [
                float(bool(getattr(el, "eval_timed_out", False))) for el in self.results
            ],
        }


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

    @staticmethod
    def _get_attr_or_key(value: Any, *keys: str) -> Any:
        for key in keys:
            if isinstance(value, dict) and key in value:
                return value[key]
            if hasattr(value, key):
                return getattr(value, key)
        return None

    @classmethod
    def _head_server_address(cls, head_server_config: Any) -> str:
        host = cls._get_attr_or_key(
            head_server_config, "host", "hostname", "server_host", "head_host"
        )
        port = cls._get_attr_or_key(
            head_server_config, "port", "server_port", "head_port"
        )
        if host is not None and port is not None:
            return f"{host}:{port}"
        return str(head_server_config)

    @staticmethod
    def _example_agent_name(example: dict[str, Any]) -> str | None:
        agent_ref = example.get(_AGENT_REF_KEY, {})
        return agent_ref.get("name") if isinstance(agent_ref, dict) else None

    @staticmethod
    def _example_instance_id(example: dict[str, Any]) -> Any:
        responses_create_params = example.get(_RESPONSES_CREATE_PARAMS_KEY, {})
        metadata = responses_create_params.get("metadata", {})
        if isinstance(metadata, dict) and "instance_id" in metadata:
            return metadata["instance_id"]
        if "instance_id" in example:
            return example["instance_id"]
        if "problem_id" in example:
            return example["problem_id"]
        return None

    @staticmethod
    def _result_health_counts(gym_result: NemoGymRunResult) -> dict[str, int]:
        output_items = len(gym_result.response.output)
        trainable_items = 0
        generated_tokens = 0
        for item in gym_result.response.output:
            if item.generation_token_ids is None:
                continue
            trainable_items += 1
            generated_tokens += len(item.generation_token_ids)
        return {
            "output_items": output_items,
            "trainable_items": trainable_items,
            "generated_tokens": generated_tokens,
        }

    @staticmethod
    def _to_finite_float(value: Any) -> float | None:
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, int | float):
            float_value = float(value)
            if math.isfinite(float_value):
                return float_value
        return None

    @classmethod
    def _find_result_value(cls, payload: Any, key: str, depth: int = 0) -> Any:
        if depth > 4:
            return None
        if isinstance(payload, AgentBaseModel):
            return cls._find_result_value(payload.model_dump(exclude={"response"}), key, depth + 1)
        if isinstance(payload, dict):
            if key in payload:
                return payload[key]
            for nested_key, nested_value in payload.items():
                if nested_key in _SKIP_RESULT_METRIC_SEARCH_KEYS:
                    continue
                found = cls._find_result_value(nested_value, key, depth + 1)
                if found is not None:
                    return found
        if isinstance(payload, list):
            for item in payload:
                found = cls._find_result_value(item, key, depth + 1)
                if found is not None:
                    return found
        return None

    @classmethod
    def _rollout_metrics_from_gym_result(
        cls, gym_result: NemoGymRunResult, counts: dict[str, int] | None = None
    ) -> dict[str, float]:
        if counts is None:
            counts = cls._result_health_counts(gym_result)

        trainable_items = counts["trainable_items"]
        generated_tokens = counts["generated_tokens"]
        output_items = counts["output_items"]
        metrics = {
            "nemo_gym/resolved": float(gym_result.resolved is True),
            "nemo_gym/mask_sample": float(gym_result.instance_config.mask_sample),
            "nemo_gym/trainable": float(trainable_items > 0),
            "nemo_gym/trainable_items": float(trainable_items),
            "nemo_gym/generated_tokens": float(generated_tokens),
            "nemo_gym/output_items": float(output_items),
        }

        for key in _OPTIONAL_ROLLOUT_METRIC_KEYS:
            value = cls._to_finite_float(cls._find_result_value(gym_result, key))
            if value is not None:
                metrics[f"nemo_gym/{key}"] = value

        agent_error_kind = cls._find_result_value(gym_result, _AGENT_ERROR_KIND_KEY)
        if agent_error_kind is not None and agent_error_kind not in _AGENT_ERROR_KIND_VALUES:
            agent_error_kind = "other"
        for error_kind in _AGENT_ERROR_KIND_VALUES:
            metrics[f"nemo_gym/agent_error_kind/{error_kind}"] = float(
                agent_error_kind == error_kind
            )

        return metrics

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

        logger.info(
            "Starting NeMo Gym for env_id=%s\n"
            "policy_model_name=%s\n"
            "policy_base_url=%s\n"
            "train_agent_name=%s\n"
            "validation_agent_name=%s\n"
            "config_paths=%s",
            self.env_id,
            self.policy_model_name,
            self.policy_base_url,
            self.train_agent_name,
            self.validation_agent_name,
            self.nemo_gym.get("config_paths", []),
        )
        startup_start = time.monotonic()
        self._rh = RunHelper()
        self._rh.start(
            global_config_dict_parser_config=GlobalConfigDictParserConfig(
                initial_global_config_dict=DictConfig(self._initial_global_config_dict()),
                skip_load_from_cli=True,
            )
        )
        self._rch = RolloutCollectionHelper()
        self._head_server_config = self._rh._server_client.head_server_config
        logger.info(
            "NeMo Gym started\nhead_server=%s\nstartup_elapsed_sec=%.3f",
            self._head_server_address(self._head_server_config),
            time.monotonic() - startup_start,
        )

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
        batch_start = time.monotonic()
        logger.info(
            "Submitting NeMo Gym rollout batch\n"
            "agent_name=%s\n"
            "num_examples=%d\n"
            "instance_ids=%s\n"
            "task_indices=%s",
            self._example_agent_name(examples[0]) if examples else None,
            len(examples),
            [self._example_instance_id(example) for example in examples],
            [example.get(_TASK_INDEX_KEY) for example in examples],
        )
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

        results = sorted(results, key=lambda pair: pair[0].get(_ROLLOUT_INDEX_KEY, 0))
        counts = [self._result_health_counts(result) for _, result in results]
        logger.info(
            "NeMo Gym rollout batch complete\n"
            "elapsed_sec=%.3f\n"
            "num_examples=%d\n"
            "rewards=%s\n"
            "resolved_count=%d\n"
            "mask_sample_count=%d\n"
            "trainable_count=%d\n"
            "generated_tokens=%d",
            time.monotonic() - batch_start,
            len(results),
            [result.reward for _, result in results],
            sum(1 for _, result in results if result.resolved is True),
            sum(1 for _, result in results if result.instance_config.mask_sample),
            sum(1 for result_counts in counts if result_counts["trainable_items"] > 0),
            sum(result_counts["generated_tokens"] for result_counts in counts),
        )
        for (row, result), result_counts in zip(results, counts):
            instance_id = result.instance_config.problem_info.get(
                "instance_id", self._example_instance_id(row)
            )
            logger.info(
                "NeMo Gym rollout result\n"
                "instance_id=%s\n"
                "reward=%s\n"
                "resolved=%s\n"
                "mask_sample=%s\n"
                "trainable_items=%d\n"
                "generated_tokens=%d\n"
                "output_items=%d\n"
                "persistent_dir=%s",
                instance_id,
                result.reward,
                result.resolved,
                result.instance_config.mask_sample,
                result_counts["trainable_items"],
                result_counts["generated_tokens"],
                result_counts["output_items"],
                result.instance_config.persistent_dir,
            )

        return results

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
                logger.warning("NeMo Gym returned non-contiguous prompt token IDs")
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
            if item.policy_epoch is None or item.kv_cache_epoch is None:
                logger.warning("NeMo Gym result missing epoch metadata; using fallback epochs")
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
            logger.warning(
                "NeMo Gym result has no trainable token output; sample will be fully masked"
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
        counts = self._result_health_counts(gym_result)

        return TokenRollout(
            trajectory=trajectories,
            reward=gym_result.reward,
            generation_mask=generation_masks,
            logprobs=logprobs,
            env_id=self.env_id,
            problem_id=problem_id,
            metrics=self._rollout_metrics_from_gym_result(gym_result, counts),
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

    async def run_evaluation(self, request: EvaluationRequest) -> NemoGymEvaluationResponse:
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
            metrics = self._rollout_metrics_from_gym_result(gym_result)
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
                    trainable=bool(metrics["nemo_gym/trainable"]),
                    agent_timed_out=bool(metrics.get("nemo_gym/agent_timed_out", 0.0)),
                    eval_timed_out=bool(metrics.get("nemo_gym/eval_timed_out", 0.0)),
                    full_result=gym_result.model_dump(),
                )
            )

        return NemoGymEvaluationResponse(results=evaluation_results, env_id=self.env_id)
