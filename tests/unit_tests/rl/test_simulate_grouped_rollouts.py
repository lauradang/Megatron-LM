# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Smoke test keeping simulate_grouped_rollouts.py in sync with the pipeline API."""

import pytest

from tests.unit_tests.rl.simulate_grouped_rollouts import (
    MODES,
    SimConfig,
    run_mode,
    write_html_report,
)

TINY_CFG = SimConfig(
    num_groups=2,
    num_batches=2,
    rollouts_per_group=2,
    parallel_generation_tasks=2,
    latency_lo=0.001,
    latency_hi=0.002,
    seed=0,
)


class TestSimulateGroupedRollouts:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("submission, consumption", MODES)
    async def test_run_mode_smoke(self, submission, consumption):
        result = await run_mode(submission, consumption, TINY_CFG)

        expected_groups = TINY_CFG.num_batches * TINY_CFG.num_groups
        assert result.num_groups == expected_groups
        assert result.num_rollouts == expected_groups * TINY_CFG.rollouts_per_group
        assert len(result.rollout_walls) == result.num_rollouts
        assert result.total_wall > 0
        assert all(wall > 0 for wall in result.rollout_walls)
        assert len(result.sampled_latencies) >= result.num_rollouts
        assert result.counters["yielded"] >= result.num_groups
        assert (
            result.counters["prepared"]
            >= result.counters["inferred"]
            >= result.counters["assembled"] * TINY_CFG.rollouts_per_group
        )
        assert len(result.batch_done_at) == TINY_CFG.num_batches
        assert [t for _, t in result.batch_done_at] == sorted(t for _, t in result.batch_done_at)
        assert len(result.batch_inference_steps) == TINY_CFG.num_batches
        assert len(result.batch_max_seq_len) == TINY_CFG.num_batches
        lo_tokens = TINY_CFG.latency_lo / TINY_CFG.seconds_per_token
        hi_tokens = TINY_CFG.latency_hi / TINY_CFG.seconds_per_token
        assert all(lo_tokens <= max_len <= hi_tokens for max_len in result.batch_max_seq_len)
        # The first batch's longest rollout ran entirely inside the first
        # engine-active window, so its inference steps bound its max seq len.
        assert result.batch_inference_steps[0] >= result.batch_max_seq_len[0] * 0.99
        for name in ("infer_queue_dwell", "engine_dwell", "assemble_queue_dwell"):
            assert result.dwell[name], f"{name} collected no samples"
        if submission == "R":
            assert result.max_active_requests <= TINY_CFG.parallel_generation_tasks

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "submission, expected_capacity",
        [("B", 1), ("G", 2), ("R", 4)],  # lag 0: 1 batch / num_groups / num_groups*rollouts
    )
    async def test_generation_lag_capacity(self, submission, expected_capacity):
        cfg = SimConfig(
            num_groups=2,
            num_batches=2,
            rollouts_per_group=2,
            latency_lo=0.001,
            latency_hi=0.002,
            generation_lag=0,
        )
        result = await run_mode(submission, "B", cfg)
        assert result.gate["capacity"] == expected_capacity
        assert result.num_groups == cfg.num_batches * cfg.num_groups

    @pytest.mark.asyncio
    async def test_max_readahead_bounds_lag(self):
        cfg = SimConfig(
            num_groups=2,
            num_batches=6,
            rollouts_per_group=2,
            latency_lo=0.001,
            latency_hi=0.002,
            generation_lag=0,
            max_readahead_batches=2,
        )
        result = await run_mode("R", "G", cfg)
        assert result.num_groups == cfg.num_batches * cfg.num_groups
        assert max(result.readahead_batches) <= 2.0 + 1e-9
        assert max(result.batch_mean_lag) <= 2.0 + 1e-9

    @pytest.mark.asyncio
    async def test_write_html_report(self, tmp_path):
        results = [await run_mode("G", "B", TINY_CFG)]
        report_path = tmp_path / "report.html"
        write_html_report(results, TINY_CFG, report_path)
        report = report_path.read_text()
        assert "<svg" in report and "G/B" in report and "data-tip" in report
