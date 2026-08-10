# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Unit + pipeline tests for the durable rollout bank (queued-group path).

Covers the append -> restore round trip, checksum/torn-write handling for both the
JSONL index and the binary sidecars, the consumption-marker filter, manifest +
compaction, and an end-to-end write-through/restore through the real
``_RolloutPipeline`` (reusing the mocks from ``test_grouped_rollouts``).
"""

import asyncio
import json
import os
from collections import deque

import numpy as np
import pytest

from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest
from megatron.rl import rl_utils, rollout_bank
from megatron.rl.agent.api import (
    GroupedRolloutRequest,
    Rollout,
    RolloutGroup,
    TokenRollout,
    _InferredItem,
    _RolloutPipeline,
)
from megatron.rl.agent.reward_only_agent import RewardOnlyAgent
from megatron.rl.agent.weighted_multi_task import AgentConfig, WeightedMultiTask
from megatron.rl.inference.api import InferenceResponse, LLMChatMessage
from megatron.rl.rollout_bank import (
    _CONSUMED,
    _FORMAT_VERSION,
    _LEDGER,
    _MANIFEST,
    _TOKENS_BIN,
    RolloutBank,
    _segment_name,
)
from megatron.rl.types import Rollout as SharedRollout
from megatron.rl.types import RolloutGroup as SharedRolloutGroup
from megatron.rl.types import TokenRollout as SharedTokenRollout
from megatron.training.checkpointing import _register_rollout_bank_compaction

# Reuse the pipeline mocks so the integration test drives the real pipeline.
from tests.unit_tests.rl.test_grouped_rollouts import MockGenerator, MockInferenceInterface


def test_agent_api_reexports_shared_rollout_types():
    assert SharedRollout is Rollout
    assert SharedRolloutGroup is RolloutGroup
    assert SharedTokenRollout is TokenRollout


def test_token_rollout_declares_advantage_override():
    rollout = TokenRollout(
        trajectory=[[1]],
        reward=1.0,
        policy_epoch=[[(0, 0)]],
        kv_cache_epoch=[[(0, 0)]],
        num_evictions=[0],
        advantage_override="-5.0",
    )

    assert "advantage_override" in TokenRollout.model_fields
    assert rollout.advantage_override == -5.0


def test_rollout_reward_accepts_none():
    rollout = Rollout(
        trajectory=["prompt"],
        reward=None,
        policy_epoch=[[(0, 0)]],
        kv_cache_epoch=[[(0, 0)]],
        num_evictions=[0],
    )

    assert rollout.reward is None


def make_token_group(members, *, batch_id=0, index_in_batch=0):
    """Build a RolloutGroup of TokenRollout members.

    ``members`` is a list of (tokens, logprobs, mask) triples, each a per-turn
    jagged list, so the sidecar packing is exercised with multi-turn, ragged data.
    """
    rollouts = []
    for tokens, logprobs, mask in members:
        rollouts.append(
            TokenRollout(
                trajectory=tokens,
                reward=1.0,
                logprobs=logprobs,
                generation_mask=mask,
                env_id="test",
                problem_id="p",
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
        )
    return RolloutGroup(rollouts=rollouts, batch_id=batch_id, index_in_batch=index_in_batch)


def sample_group(batch_id=0):
    return make_token_group(
        [
            ([[1, 2, 3], [4, 5]], [[-0.1, -0.2, -0.3], [-0.4, -0.5]],
             [[False, True, True], [True, True]]),
            ([[7, 8]], [[-1.5, -2.5]], [[True, True]]),
        ],
        batch_id=batch_id,
    )


def text_group():
    return RolloutGroup(
        rollouts=[
            Rollout(trajectory=["hello world"], reward=0.5, env_id="t",
                    policy_epoch=[[(0, 0)]], kv_cache_epoch=[[(0, 0)]], num_evictions=[0]),
        ],
    )


class TestRoundTrip:
    def test_manifest_and_ledger_record_current_format_version(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        ledger_path = tmp_path / _segment_name(3) / _LEDGER
        record = json.loads(ledger_path.read_text().splitlines()[0])

        assert manifest["format_version"] == _FORMAT_VERSION
        assert record["format_version"] == _FORMAT_VERSION

    @pytest.mark.parametrize(
        "invalid_version",
        [
            pytest.param(None, id="missing"),
            pytest.param(_FORMAT_VERSION + 1, id="unsupported"),
        ],
    )
    def test_restore_rejects_incompatible_manifest_version(self, tmp_path, invalid_version):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        manifest_path = tmp_path / _MANIFEST
        manifest = json.loads(manifest_path.read_text())
        if invalid_version is None:
            manifest.pop("format_version")
        else:
            manifest["format_version"] = invalid_version
        manifest_path.write_text(json.dumps(manifest))

        with pytest.raises(ValueError, match="Unsupported RolloutBank format_version"):
            RolloutBank(str(tmp_path)).restore(0)

    @pytest.mark.parametrize(
        "invalid_version",
        [
            pytest.param(None, id="missing"),
            pytest.param(_FORMAT_VERSION + 1, id="unsupported"),
        ],
    )
    def test_restore_rejects_incompatible_ledger_version(self, tmp_path, invalid_version):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        bank.append(sample_group())
        bank.close()

        ledger_path = tmp_path / _segment_name(3) / _LEDGER
        record = json.loads(ledger_path.read_text())
        if invalid_version is None:
            record.pop("format_version")
        else:
            record["format_version"] = invalid_version
        ledger_path.write_text(json.dumps(record) + "\n")

        with pytest.raises(ValueError, match="Unsupported RolloutBank format_version"):
            RolloutBank(str(tmp_path)).restore(0)

    def test_encode_returns_named_payload(self, tmp_path):
        assert hasattr(rollout_bank, "EncodedGroup")

        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        encoded = bank._encode(sample_group(), "gen-000003/0")

        assert isinstance(encoded, rollout_bank.EncodedGroup)
        assert encoded.record["uid"] == "gen-000003/0"
        assert encoded.tok_bytes
        assert encoded.lp_bytes
        assert encoded.mask_bytes

    def test_token_group_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(3)
        original = sample_group(batch_id=2)
        uid = bank.append(original)
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 1
        g = restored[0]
        assert g.uid == uid
        assert g.batch_id == 2
        # token ids are exact (int32)
        assert g.rollouts[0].trajectory == [[1, 2, 3], [4, 5]]
        assert g.rollouts[1].trajectory == [[7, 8]]
        # generation_mask preserved exactly
        assert g.rollouts[0].generation_mask == [[False, True, True], [True, True]]
        # logprobs recovered within fp16 tolerance
        assert np.allclose(g.rollouts[0].logprobs[0], [-0.1, -0.2, -0.3], atol=1e-3)
        assert np.allclose(g.rollouts[1].logprobs[0], [-1.5, -2.5], atol=1e-3)

    def test_text_group_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        bank.append(text_group())
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 1
        assert isinstance(restored[0].rollouts[0], Rollout)
        assert restored[0].rollouts[0].trajectory == ["hello world"]

    def test_fp16_logprobs_lossy_but_close_tokens_exact(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        toks = list(range(50))
        lps = [round(-0.01 * i, 4) for i in range(50)]
        bank.append(make_token_group([([toks], [lps], [[True] * 50])]))
        bank.close()

        g = RolloutBank(str(tmp_path)).restore(0)[0]
        assert g.rollouts[0].trajectory[0] == toks  # int32 exact
        assert np.allclose(g.rollouts[0].logprobs[0], lps, atol=1e-3)

    @pytest.mark.parametrize("field", ["logprobs", "generation_mask"])
    def test_mixed_optional_field_presence_is_rejected(self, tmp_path, field):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        group = sample_group()
        setattr(group.rollouts[1], field, None)

        with pytest.raises(ValueError, match=f"{field} must be present for all or no rollouts"):
            bank.append(group)


class TestDurability:
    def test_manifest_replace_is_followed_by_bank_directory_fsync(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        events = []
        real_replace = os.replace

        def replace(src, dst):
            real_replace(src, dst)
            events.append("replace")

        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )

        bank._write_manifest_atomic(
            {"trained_through": 1, "segments": [], "compacted_at": 0}
        )

        assert events == ["replace", f"dir:{tmp_path}"]

    def test_first_append_fsyncs_new_entries_after_file_contents(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        segment = tmp_path / _segment_name(0)
        events = []
        monkeypatch.setattr(os, "fsync", lambda fd: events.append("file"))
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )

        bank.append(sample_group())

        assert events == ["file", f"dir:{segment}"] * 4

        events.clear()
        bank.append(sample_group())
        assert events == ["file"] * 4

    def test_new_segment_is_durable_before_manifest_publication(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        events = []
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )
        monkeypatch.setattr(
            bank,
            "_write_manifest_atomic",
            lambda manifest: events.append(f"manifest:{manifest['segments'][-1]}"),
        )

        bank.set_collection(7)

        assert events == [f"dir:{tmp_path}", f"manifest:{_segment_name(7)}"]

    def test_compacted_segment_is_durable_before_manifest_publication(
        self, tmp_path, monkeypatch
    ):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        events = []
        real_replace = os.replace

        def replace(src, dst):
            real_replace(src, dst)
            if str(src).endswith(".compact"):
                events.append("segment_replace")

        monkeypatch.setattr(os, "replace", replace)
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )
        monkeypatch.setattr(bank, "restore", lambda iteration: [])
        monkeypatch.setattr(bank, "_rewrite_segment", lambda *args: None)
        monkeypatch.setattr(
            bank,
            "_write_manifest_atomic",
            lambda manifest: events.append(f"manifest:{manifest['trained_through']}"),
        )

        bank.checkpoint(2)

        replace_index = events.index("segment_replace")
        assert events[replace_index : replace_index + 3] == [
            "segment_replace",
            f"dir:{tmp_path}",
            "manifest:2",
        ]

    def test_first_consumed_marker_fsyncs_bank_directory(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        events = []
        monkeypatch.setattr(os, "fsync", lambda fd: events.append("file"))
        monkeypatch.setattr(
            rollout_bank,
            "_fsync_directory",
            lambda path: events.append(f"dir:{path}"),
        )

        bank.mark_consumed("gen-000000/0", 1)
        bank.mark_consumed("gen-000000/1", 1)

        assert events == ["file", f"dir:{tmp_path}", "file"]

    def test_torn_final_ledger_line_dropped_and_append_recovers_after_restart(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.append(sample_group())
        bank.close()

        # Simulate a kill mid-append: a truncated JSON line at the end of the ledger.
        ledger = os.path.join(str(tmp_path), _segment_name(0), _LEDGER)
        with open(ledger, "a") as f:
            f.write('{"uid": "gen-000000/2", "kind": "toke')  # torn, no newline

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 2  # the two intact records survive

        restarted = RolloutBank(str(tmp_path))
        restarted.set_collection(0)
        new_uid = restarted.append(sample_group())
        restarted.close()

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 3
        assert new_uid == f"{_segment_name(0)}/2"
        assert new_uid in {group.uid for group in restored}

    def test_truncated_sidecar_slice_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.append(sample_group())
        bank.close()

        # Chop the tail of tokens.bin so the second record's slice is short.
        tokens_bin = os.path.join(str(tmp_path), _segment_name(0), _TOKENS_BIN)
        size = os.path.getsize(tokens_bin)
        with open(tokens_bin, "r+b") as f:
            f.truncate(size - 4)

        restored = RolloutBank(str(tmp_path)).restore(0)
        assert len(restored) == 1  # only the first record's slice is intact

    def test_checksum_mismatch_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.close()

        ledger = os.path.join(str(tmp_path), _segment_name(0), _LEDGER)
        with open(ledger) as f:
            rec = json.loads(f.readline())
        rec["checksum"] = "0" * 32  # tamper
        with open(ledger, "w") as f:
            f.write(json.dumps(rec) + "\n")

        assert RolloutBank(str(tmp_path)).restore(0) == []


class TestMarkerFilter:
    def test_marker_filter_rules(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(5)
        trained = bank.append(sample_group())   # consumed at 5 <= T=10 -> discard
        rolled_back = bank.append(sample_group())  # consumed at 12 > T=10 -> restore
        _never = bank.append(sample_group())     # no marker -> restore
        bank.mark_consumed(trained, 5)
        bank.mark_consumed(rolled_back, 12)
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=10)
        uids = {g.uid for g in restored}
        assert trained not in uids
        assert rolled_back in uids
        assert _never in uids


class TestCompaction:
    def test_async_compaction_finalize_runs_with_captured_iteration(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        monkeypatch.setattr(rl_utils, "_ROLLOUT_BANK", bank)
        bank.set_collection(1)
        first = bank.append(sample_group())
        bank.mark_consumed(first, 1)

        first_save = AsyncRequest(None, (), [])
        second_save = AsyncRequest(None, (), [])
        _register_rollout_bank_compaction(first_save, 1)
        _register_rollout_bank_compaction(second_save, 2)

        bank.set_collection(2)
        second = bank.append(sample_group())
        bank.mark_consumed(second, 2)

        first_save.finalize_fns[0]()
        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 1
        assert {group.uid for group in bank.restore(1)} == {second}

        second_save.finalize_fns[0]()
        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 2
        assert manifest["segments"] == [_segment_name(2)]
        assert bank.restore(2) == []

    def test_marker_after_compaction_is_not_orphaned(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        old_uid = bank.append(sample_group())

        bank.checkpoint(2)
        bank.mark_consumed(old_uid, 4)

        assert bank.restore(2)[0].uid == old_uid
        assert bank.restore(4) == []

    def test_fresh_append_after_compaction_has_unique_uid(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(2)
        survivor_uid = bank.append(sample_group())

        bank.checkpoint(2)
        fresh_uid = bank.append(sample_group())

        assert fresh_uid != survivor_uid
        assert {group.uid for group in bank.restore(2)} == {survivor_uid, fresh_uid}

    def test_restore_reads_legacy_segment_marker(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        uid = bank.append(sample_group())
        bank.mark_consumed(uid, 1)
        os.replace(tmp_path / _CONSUMED, tmp_path / _segment_name(1) / _CONSUMED)

        assert RolloutBank(str(tmp_path)).restore(1) == []

    def test_compaction_prunes_and_flips_manifest(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(1)
        consumed = bank.append(sample_group())
        bank.mark_consumed(consumed, 1)
        bank.set_collection(2)
        survivor = bank.append(sample_group())

        bank.checkpoint(2)  # trained_through=2: prune consumed(<=2), keep survivor

        manifest = json.loads((tmp_path / _MANIFEST).read_text())
        assert manifest["trained_through"] == 2
        assert manifest["segments"] == [_segment_name(2)]
        assert manifest["compacted_at"] == 2
        # stale segment dir removed
        assert not (tmp_path / _segment_name(1)).exists()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=2)
        assert len(restored) == 1
        # the survivor's payload is intact after being rewritten by compaction
        assert restored[0].rollouts[0].trajectory == [[1, 2, 3], [4, 5]]
        assert restored[0].rollouts[1].trajectory == [[7, 8]]
        assert survivor  # uid was assigned at append time

    def test_compaction_survivor_survives_next_kill(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(2)
        bank.append(sample_group())
        bank.checkpoint(2)
        # A fresh process restores the compacted survivor.
        assert len(RolloutBank(str(tmp_path)).restore(2)) == 1


class TestPipelineIntegration:
    """Write-through + restore through the real _RolloutPipeline."""

    def _collect(self, tmp_path, num_groups=4, stop_after=None):
        async def run():
            gen = MockGenerator(parallel_generation_tasks=8)
            bank = RolloutBank(str(tmp_path))
            bank.set_collection(0)
            gen._rollout_bank = bank
            request_groups = []
            from megatron.rl.agent.api import GroupedRolloutRequest

            req = GroupedRolloutRequest(
                num_groups=num_groups,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
                submission_granularity="B",
                consumption_granularity="B",
            )
            async for group in gen.get_grouped_rollouts(req):
                request_groups.append(group)
                if stop_after is not None and len(request_groups) >= stop_after:
                    break
            bank.close()
            return request_groups

        return asyncio.run(run())

    def test_write_through_then_restore(self, tmp_path):
        groups = self._collect(tmp_path, num_groups=4)
        assert len(groups) == 4
        assert all(getattr(g, "uid", None) for g in groups)

        # Fresh process (restart): no markers, T=0 -> all completed groups restored,
        # never regenerated.
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 4
        assert {g.uid for g in restored} == {g.uid for g in groups}

    def test_early_exit_keeps_assembled_groups(self, tmp_path):
        # Break after the first group; write-through means at least that group is
        # already durable (assembly precedes consumption).
        groups = self._collect(tmp_path, num_groups=4, stop_after=1)
        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) >= len(groups) >= 1


def _env_group(env_id, problem_id="p"):
    """A minimal inline (text) RolloutGroup tagged with ``env_id``."""
    return RolloutGroup(
        rollouts=[
            Rollout(
                trajectory=["x"],
                reward=1.0,
                env_id=env_id,
                problem_id=problem_id,
                policy_epoch=[[(0, 0)]],
                kv_cache_epoch=[[(0, 0)]],
                num_evictions=[0],
            )
        ]
    )


def _weighted_agent(env_weights):
    return WeightedMultiTask(
        [
            AgentConfig(agent_type=MockGenerator, agent_args={"env_id": e}, weight=w)
            for e, w in env_weights
        ]
    )


class TestRestoreBalancing:
    """Cap-and-defer injection + per-env residual balancing for restored groups."""

    def test_env_targets_matches_distribute_counts(self):
        agent = _weighted_agent([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        for n in (3, 6, 7, 10):
            expected: dict = {}
            for eid, c in zip(agent._env_ids(), agent._distribute_counts(n)):
                expected[eid] = expected.get(eid, 0) + c
            assert rl_utils._env_targets(agent, n) == expected
            assert sum(rl_utils._env_targets(agent, n).values()) == n

    def test_plan_restore_injection_caps_and_defers(self):
        target = {"a": 2, "b": 2, "c": 2}
        restored = {"a": deque(range(6))}  # 6 restored groups, all env "a"
        inject, residual = rl_utils._plan_restore_injection(target, restored)
        assert inject == {"a": 2, "b": 0, "c": 0}  # capped at target["a"], 4 deferred
        assert residual == {"a": 0, "b": 2, "c": 2}
        for env in target:
            assert inject[env] + residual[env] == target[env]

    def test_restore_injection_drain_window_stays_balanced(self):
        target = {"a": 2, "b": 2, "c": 2}
        restored = {"a": deque(f"a{i}" for i in range(6))}
        injected_total = []
        steps = 0
        while any(restored.values()):
            inject, residual = rl_utils._plan_restore_injection(target, restored)
            for env in target:
                # No env ever injects more than its weighted target for the batch.
                assert inject[env] <= target[env]
                assert inject[env] + residual[env] == target[env]
            for env, count in inject.items():
                for _ in range(count):
                    injected_total.append(restored[env].popleft())
            steps += 1
            assert steps < 100, "drain did not terminate"
        # Every restored group is eventually injected; none dropped.
        assert len(injected_total) == 6

    def test_bucket_restored_groups_buckets_by_env(self):
        groups = [_env_group("a"), _env_group("b"), _env_group("a")]
        buckets = rl_utils._bucket_restored_groups(groups, {"a", "b", "c"})
        assert set(buckets) == {"a", "b"}
        assert len(buckets["a"]) == 2 and len(buckets["b"]) == 1

    def test_bucket_restored_groups_asserts_env_config_drift(self):
        with pytest.raises(AssertionError, match="not in the current"):
            rl_utils._bucket_restored_groups([_env_group("z")], {"a", "b"})

    def test_bucket_and_plan_from_real_bank(self, tmp_path):
        # End-to-end: bank 6 groups all env "a", restore, bucket, and plan step 1.
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        for i in range(6):
            bank.append(_env_group("a", problem_id=f"p{i}"))
        bank.close()

        restored = RolloutBank(str(tmp_path)).restore(trained_through=0)
        assert len(restored) == 6
        buckets = rl_utils._bucket_restored_groups(restored, {"a", "b", "c"})
        assert len(buckets["a"]) == 6

        agent = _weighted_agent([("a", 1.0), ("b", 1.0), ("c", 1.0)])
        target = rl_utils._env_targets(agent, 6)
        inject, residual = rl_utils._plan_restore_injection(target, buckets)
        assert inject == {"a": 2, "b": 0, "c": 0}
        assert residual == {"a": 0, "b": 2, "c": 2}


# ------------------------------------------------------ Phase B: partial groups


def make_partial_snapshot(
    problem_id="p", env_id="test", n=2, finished=1, token=False, *, uid=None, agent_index=0
):
    """Build a PartialGroupSnapshot dict for the storage-layer tests."""
    if token:
        resp = {
            "response": {"role": "assistant", "content": "t0"},
            "token_ids": [1, 2, 3],
            "prompt_length": 1,
            "logprobs": [-0.1, -0.2, -0.3],
            "finish_reason": "stop",
            "policy_epoch": [[0, 0]],
            "kv_cache_epoch": [[0, 0]],
            "num_evictions": 0,
        }
    else:
        resp = {
            "response": {"role": "assistant", "content": "t0"},
            "raw_text": "t0",
            "finish_reason": "stop",
            "policy_epoch": [[0, 0]],
            "kv_cache_epoch": [[0, 0]],
            "num_evictions": 0,
        }
    return {
        "partial_uid": uid or f"{env_id}:{problem_id}",
        "env_id": env_id,
        "agent_index": agent_index,
        "problem_id": problem_id,
        "inference_request": {"prompt": [{"role": "user", "content": "hi"}], "tools": None},
        "golden": {"problem_id": problem_id, "answer": 1},
        "rollouts_per_group": n,
        "batch_id": 0,
        "index_in_batch": 0,
        "finished_members": [{"rollout_idx": i, "response": resp} for i in range(finished)],
    }


class RepeatingProblemAgent(RewardOnlyAgent):
    """Minimal reward agent that intentionally serves the same dataset row."""

    env_id: str | None = "test"

    async def get_prompt(self, validation):
        return "t0", {"problem_id": "p", "answer": 0}

    async def get_reward(self, response, golden, finish_reason):
        return float(response == "t0")


class CountingMockInferenceInterface(MockInferenceInterface):
    calls: int = 0

    async def base_generate(self, request):
        self.calls += 1
        return await super().base_generate(request)


class TestPartialSnapshot:
    """Storage-layer round trip, durability, compaction, and dedup for Phase B."""

    def test_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.snapshot_partial([make_partial_snapshot("p0"), make_partial_snapshot("p1")], 0)
        restored = RolloutBank(str(tmp_path)).restore_partial(trained_through=0)
        assert {s["problem_id"] for s in restored} == {"p0", "p1"}
        member = restored[0]["finished_members"][0]["response"]
        assert member["raw_text"] == "t0"

    def test_token_member_round_trip(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.snapshot_partial([make_partial_snapshot("p", token=True)], 0)
        restored = RolloutBank(str(tmp_path)).restore_partial(0)
        resp = restored[0]["finished_members"][0]["response"]
        assert resp["token_ids"] == [1, 2, 3]
        assert resp["logprobs"] == [-0.1, -0.2, -0.3]

    def test_empty_snapshot_clears_stale_partials(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.snapshot_partial([make_partial_snapshot()], 0)
        bank.snapshot_partial([], 1)  # group completed -> next window clears it
        assert RolloutBank(str(tmp_path)).restore_partial(0) == []

    def test_missing_snapshot_returns_empty(self, tmp_path):
        assert RolloutBank(str(tmp_path)).restore_partial(0) == []

    def test_torn_snapshot_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.snapshot_partial([make_partial_snapshot()], 0)
        with open(bank._inflight_path, "r+") as f:
            size = f.seek(0, os.SEEK_END)
            f.truncate(size // 2)
        assert RolloutBank(str(tmp_path)).restore_partial(0) == []

    def test_checksum_mismatch_dropped(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.snapshot_partial([make_partial_snapshot()], 0)
        with open(bank._inflight_path) as f:
            payload = json.load(f)
        payload["partials"][0]["problem_id"] = "tampered"
        with open(bank._inflight_path, "w") as f:
            json.dump(payload, f)
        assert RolloutBank(str(tmp_path)).restore_partial(0) == []

    def test_atomic_replace_survives_kill_mid_write(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.snapshot_partial([make_partial_snapshot("old")], 0)

        def boom(src, dst):
            raise KeyboardInterrupt

        monkeypatch.setattr(rollout_bank.os, "replace", boom)
        with pytest.raises(KeyboardInterrupt):
            bank.snapshot_partial([make_partial_snapshot("new")], 1)
        # The previous complete snapshot is intact (atomic rename never happened).
        restored = RolloutBank(str(tmp_path)).restore_partial(0)
        assert [s["problem_id"] for s in restored] == ["old"]

    def test_compaction_leaves_snapshot_intact(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group())
        bank.snapshot_partial([make_partial_snapshot("p")], 0)
        bank.checkpoint(0)  # rewrites/prunes gen-<iter>/ segments; must not touch inflight.json
        restored = RolloutBank(str(tmp_path)).restore_partial(0)
        assert [s["problem_id"] for s in restored] == ["p"]

    def test_completed_partial_not_double_restored(self, tmp_path):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group(), partial_uid="test:p")  # completed-from-resume record
        bank.snapshot_partial([make_partial_snapshot("p")], 0)  # stale snapshot
        assert RolloutBank(str(tmp_path)).restore_partial(0) == []

    def test_partial_uid_survives_compaction(self, tmp_path):
        # A group completed from resume keeps its partial_uid across a compaction, so
        # a stale snapshot restored after a later kill is still deduped out.
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        bank.append(sample_group(), partial_uid="test:p")
        bank.checkpoint(0)
        assert "test:p" in RolloutBank(str(tmp_path))._completed_partial_uids()

    def test_unclaimed_restored_partial_survives_next_snapshot(self, tmp_path, monkeypatch):
        bank = RolloutBank(str(tmp_path))
        bank.set_collection(0)
        snapshot = make_partial_snapshot("p", uid="unclaimed")
        agent = RepeatingProblemAgent()
        agent._resume_partials = {"p": [snapshot]}
        assert getattr(agent, "_active_pipeline", None) is None
        monkeypatch.setattr(rl_utils, "_ROLLOUT_AGENT", agent)

        rl_utils._snapshot_partial_groups(bank, 1)

        assert RolloutBank(str(tmp_path)).restore_partial(0) == [snapshot]

    def test_repeated_rows_and_shared_env_agents_keep_distinct_occurrences(self):
        async def run():
            indexed = rl_utils._index_partials_by_env(
                [
                    make_partial_snapshot("p", uid="occurrence-a", agent_index=0),
                    make_partial_snapshot("p", uid="occurrence-b", agent_index=0),
                    make_partial_snapshot("p", uid="occurrence-c", agent_index=1),
                ]
            )
            request = GroupedRolloutRequest(
                num_groups=1,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
            )
            first_agent = RepeatingProblemAgent()
            second_agent = RepeatingProblemAgent()
            first_agent._resume_partials = indexed[("test", 0)]
            second_agent._resume_partials = indexed[("test", 1)]
            first = await first_agent.prepare_group_rollout(request)
            second = await first_agent.prepare_group_rollout(request)
            other = await second_agent.prepare_group_rollout(request)
            return first, second, other

        first, second, other = asyncio.run(run())
        assert [first.partial_uid, second.partial_uid, other.partial_uid] == [
            "occurrence-a",
            "occurrence-b",
            "occurrence-c",
        ]

    def test_fresh_partial_completion_dedups_stale_snapshot(self, tmp_path, monkeypatch):
        async def run():
            bank = RolloutBank(str(tmp_path))
            bank.set_collection(0)
            agent = RepeatingProblemAgent(parallel_generation_tasks=8)
            request = GroupedRolloutRequest(
                num_groups=1,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
                submission_granularity="B",
                consumption_granularity="B",
            )
            pipeline = _RolloutPipeline(agent, request, 8, bank)
            await pipeline.stage_prepare()
            first = await pipeline.infer_queue.get()
            second = await pipeline.infer_queue.get()
            response = InferenceResponse(
                response=LLMChatMessage(role="assistant", content="t0"),
                raw_text="t0",
                finish_reason="stop",
                policy_epoch=[(0, 0)],
                kv_cache_epoch=[(0, 0)],
                num_evictions=0,
            )
            pipeline._assemble_pending[0] = [_InferredItem(first, response)]
            agent._active_pipeline = pipeline
            monkeypatch.setattr(rl_utils, "_ROLLOUT_AGENT", agent)
            rl_utils._snapshot_partial_groups(bank, 0)
            assert bank.restore_partial(0)[0]["partial_uid"] == first.params.partial_uid

            await pipeline.assemble_queue.put(_InferredItem(second, response))
            pipeline.assemble_queue.shutdown()
            await pipeline.stage_assemble()
            bank.close()
            return first.params.partial_uid

        partial_uid = asyncio.run(run())
        reopened = RolloutBank(str(tmp_path))
        assert partial_uid in reopened._completed_partial_uids()
        assert reopened.restore_partial(0) == []

    def test_full_restored_group_is_discarded_and_regenerated(self):
        async def run():
            inference = CountingMockInferenceInterface()
            agent = RepeatingProblemAgent(parallel_generation_tasks=4)
            agent._resume_partials = {
                "p": [make_partial_snapshot("p", n=2, finished=2, uid="complete")]
            }
            request = GroupedRolloutRequest(
                num_groups=1,
                rollouts_per_group=2,
                inference_interface=inference,
                submission_granularity="B",
                consumption_granularity="B",
            )

            async def collect():
                return [group async for group in agent.get_grouped_rollouts(request)]

            return await asyncio.wait_for(collect(), timeout=2), inference.calls

        groups, inference_calls = asyncio.run(run())
        assert len(groups) == 1 and len(groups[0]) == 2
        assert inference_calls == 2

    @pytest.mark.parametrize(
        "indices,saved_n",
        [([0], 3), ([2], 2), ([0, 0], 2)],
        ids=["group-size", "member-range", "duplicate-member"],
    )
    def test_incompatible_restored_members_are_discarded(self, indices, saved_n):
        async def run():
            snapshot = make_partial_snapshot("p", n=saved_n, finished=1)
            member = snapshot["finished_members"][0]
            snapshot["finished_members"] = [dict(member, rollout_idx=idx) for idx in indices]
            agent = RepeatingProblemAgent()
            agent._resume_partials = {"p": [snapshot]}
            request = GroupedRolloutRequest(
                num_groups=1,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
            )
            return await agent.prepare_group_rollout(request)

        params = asyncio.run(run())
        assert params.resume_members is None
        assert params.partial_uid == "test:p"


class ResumingMockGenerator(MockGenerator):
    """MockGenerator whose first group resumes from one saved member."""

    def __init__(self, saved_response, **kwargs):
        super().__init__(**kwargs)
        self._saved = saved_response
        self.inferences = 0

    async def get_rollout_response(self, request, inference_request):
        self.inferences += 1
        return await super().get_rollout_response(request, inference_request)

    async def prepare_group_rollout(self, request):
        params = await super().prepare_group_rollout(request)
        if self.prepare_group_rollout_calls == 1:  # first group resumes
            return params._replace(resume_members={0: self._saved}, partial_uid="test:0")
        return params


class TestPartialResumeIntegration:
    """Resume through the real _RolloutPipeline: only missing members regenerate."""

    def test_resumes_missing_members_only(self, tmp_path):
        async def run():
            saved = InferenceResponse(
                response=LLMChatMessage(role="assistant", content="t99"),
                raw_text="t99",
                finish_reason="stop",
                policy_epoch=[(0, 0)],
                kv_cache_epoch=[(0, 0)],
                num_evictions=0,
            )
            gen = ResumingMockGenerator(saved_response=saved, parallel_generation_tasks=8)
            bank = RolloutBank(str(tmp_path))
            bank.set_collection(0)
            gen._rollout_bank = bank
            req = GroupedRolloutRequest(
                num_groups=2,
                rollouts_per_group=2,
                inference_interface=MockInferenceInterface(),
                submission_granularity="B",
                consumption_granularity="B",
            )
            groups = [g async for g in gen.get_grouped_rollouts(req)]
            bank.close()
            return gen, groups

        gen, groups = asyncio.run(run())
        assert len(groups) == 2
        # group0: 1 saved + 1 generated; group1: 2 generated => 3 inferences, not 4.
        assert gen.inferences == 3
        resumed = next(g for g in groups if any(r.trajectory == ["t99"] for r in g.rollouts))
        assert len(resumed.rollouts) == 2  # completed to full size N
        # The completed-from-resume group recorded its partial_uid for dedup.
        assert "test:0" in RolloutBank(str(tmp_path))._completed_partial_uids()
