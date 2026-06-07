# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import pytest

from megatron.rl.agent.api import RewardEvaluationResult
from megatron.rl.agent.nemo_gym_agent import (
    NemoGymAgent,
    NemoGymEvaluationResponse,
    NemoGymRunResult,
)


def _make_result(**overrides):
    data = {
        "response": {
            "output": [
                {
                    "prompt_token_ids": [1, 2],
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.1, -0.2],
                    "policy_epoch": [(0, 7)],
                    "kv_cache_epoch": [(0, 11)],
                    "num_evictions": 2,
                    "ignored_extra_field": "ok",
                }
            ]
        },
        "reward": 1.0,
        "resolved": True,
        "instance_config": {
            "mask_sample": False,
            "problem_info": {"instance_id": "astropy__astropy-12907"},
            "persistent_dir": "/tmp/nemo-gym-result",
        },
    }
    data.update(overrides)
    return NemoGymRunResult.model_validate(data)


def test_nemo_gym_output_parses_extra_fields():
    result = _make_result(extra_top_level="allowed")

    assert result.response.output[0].generation_token_ids == [3, 4]
    assert result.extra_top_level == "allowed"
    assert result.response.output[0].ignored_extra_field == "ok"


def test_converts_gym_result_to_token_rollout():
    rollout = NemoGymAgent().rollout_from_gym_result(_make_result())

    assert rollout.trajectory == [[1, 2, 3, 4]]
    assert rollout.generation_mask == [[False, False, True, True]]
    assert rollout.logprobs == [[-0.1, -0.2]]
    assert rollout.reward == 1.0
    assert rollout.resolved is True
    assert rollout.mask_sample is False
    assert rollout.problem_id == "astropy__astropy-12907"
    assert rollout.policy_epoch == [[(0, 7)]]
    assert rollout.kv_cache_epoch == [[(0, 11)]]
    assert rollout.num_evictions == [2]
    assert rollout.metrics["nemo_gym/resolved"] == 1.0
    assert rollout.metrics["nemo_gym/mask_sample"] == 0.0
    assert rollout.metrics["nemo_gym/trainable"] == 1.0
    assert rollout.metrics["nemo_gym/generated_tokens"] == 2.0


def test_mask_sample_masks_all_tokens_but_keeps_reward():
    result = _make_result(
        instance_config={
            "mask_sample": True,
            "problem_info": {"instance_id": "masked"},
        }
    )

    rollout = NemoGymAgent().rollout_from_gym_result(result)

    assert rollout.reward == 1.0
    assert rollout.generation_mask == [[False, False, False, False]]
    assert rollout.mask_sample is True
    assert rollout.metrics["nemo_gym/mask_sample"] == 1.0


def test_missing_epoch_metadata_falls_back_safely():
    result = _make_result(
        response={
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.4],
                }
            ]
        }
    )

    rollout = NemoGymAgent().rollout_from_gym_result(result)

    assert rollout.policy_epoch == [[(0, 0)]]
    assert rollout.kv_cache_epoch == [[(0, 0)]]
    assert rollout.num_evictions == [0]


def test_prompt_only_result_is_non_trainable_masked_sample():
    result = _make_result(
        response={
            "output": [
                {
                    "prompt_token_ids": [10, 11, 12],
                }
            ]
        },
        reward=0.0,
        resolved=False,
    )

    rollout = NemoGymAgent().rollout_from_gym_result(result)

    assert rollout.trajectory == [[10, 11, 12]]
    assert rollout.generation_mask == [[False, False, False]]
    assert rollout.logprobs == [[]]
    assert rollout.reward == 0.0
    assert rollout.resolved is False
    assert rollout.metrics["nemo_gym/trainable"] == 0.0
    assert rollout.metrics["nemo_gym/generated_tokens"] == 0.0


def test_missing_optional_gym_fields_do_not_crash_metric_extraction():
    rollout = NemoGymAgent().rollout_from_gym_result(_make_result())

    assert "nemo_gym/patch_exists" not in rollout.metrics
    assert "nemo_gym/agent_timed_out" not in rollout.metrics


def test_agent_error_kind_context_window_produces_one_hot_metric():
    result = _make_result(agent_error_kind="context_window")

    rollout = NemoGymAgent().rollout_from_gym_result(result)

    assert rollout.metrics["nemo_gym/agent_error_kind/max_iteration"] == 0.0
    assert rollout.metrics["nemo_gym/agent_error_kind/context_window"] == 1.0
    assert rollout.metrics["nemo_gym/agent_error_kind/stuck_in_loop"] == 0.0
    assert rollout.metrics["nemo_gym/agent_error_kind/other"] == 0.0


def test_nemo_gym_evaluation_response_metrics_include_health_rates():
    response = NemoGymEvaluationResponse(
        env_id="nemo_gym/swe_agents",
        results=[
            RewardEvaluationResult(
                prompt="",
                response="",
                reward=1.0,
                resolved=True,
                mask_sample=False,
                trainable=True,
                agent_timed_out=False,
                eval_timed_out=True,
            ),
            RewardEvaluationResult(
                prompt="",
                response="",
                reward=0.0,
                resolved=False,
                mask_sample=True,
                trainable=False,
                agent_timed_out=True,
                eval_timed_out=False,
            ),
        ],
    )

    metrics = response.metrics()

    assert metrics["reward"] == [1.0, 0.0]
    assert metrics["resolved_rate"] == [1.0, 0.0]
    assert metrics["mask_sample_rate"] == [0.0, 1.0]
    assert metrics["trainable_rate"] == [1.0, 0.0]
    assert metrics["agent_timed_out_rate"] == [0.0, 1.0]
    assert metrics["eval_timed_out_rate"] == [1.0, 0.0]


def test_non_contiguous_turns_raise():
    result = _make_result(
        response={
            "output": [
                {
                    "prompt_token_ids": [1],
                    "generation_token_ids": [2],
                    "generation_log_probs": [-0.2],
                },
                {
                    "prompt_token_ids": [9, 9],
                    "generation_token_ids": [3],
                    "generation_log_probs": [-0.3],
                },
            ]
        }
    )

    with pytest.raises(ValueError, match="non-contiguous"):
        NemoGymAgent().rollout_from_gym_result(result)
