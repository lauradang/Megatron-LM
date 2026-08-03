# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import TypeAlias, TypedDict

from .rollout import EnvId

#: Identifies a sub-agent by ``(env_id, agent_index)``.
AgentKey: TypeAlias = tuple[EnvId, int]

#: Identifies the dataset row used to reattach a partial snapshot.
ProblemId: TypeAlias = str


class PartialMember(TypedDict):
    """One finished-but-ungrouped member of a partial group (raw response).

    Args:
        rollout_idx: The member's index within its group (0 .. rollouts_per_group-1).
        response: ``InferenceResponse.model_dump()`` for the finished member. Stored
            raw (not a built Rollout) so the snapshot performs zero reward evaluation;
            rebuild happens for free when stage_assemble runs build_rollout on the
            re-seeded bucket at completion.
    """

    rollout_idx: int
    response: dict


class PartialGroupSnapshot(TypedDict):
    """A partial group's finished members plus what is needed to complete it.

    Args:
        partial_uid: Unique per-group occurrence identity used for exactly-once
            dedup against completed ledger records.
        env_id: The environment/sub-agent the group belongs to.
        agent_index: Stable position of the sub-agent in the configured agent list.
            This disambiguates agents that intentionally share an env_id.
        problem_id: The dataset-row identity used to re-attach on resume.
        inference_request: ``InferenceRequest.model_dump()`` (the prompt).
        golden: The dataset reward answer (JSON-serializable).
        rollouts_per_group: Target group size N at snapshot time.
        batch_id: Observability; reassigned when the group is re-served on resume.
        index_in_batch: Observability; reassigned on resume.
        finished_members: The finished members captured this window.
    """

    partial_uid: str
    env_id: EnvId
    agent_index: int
    problem_id: ProblemId
    inference_request: dict
    golden: dict
    rollouts_per_group: int
    batch_id: int
    index_in_batch: int
    finished_members: list[PartialMember]


#: Maps each problem ID to its partial snapshots in occurrence order.
PartialsByProblemId: TypeAlias = dict[ProblemId, list[PartialGroupSnapshot]]

#: Maps each agent to its restored partial snapshots, grouped by problem ID.
#: Per-problem lists preserve repeated dataset-row occurrences in snapshot order.
RestoredPartialsByAgent: TypeAlias = dict[AgentKey, PartialsByProblemId]
