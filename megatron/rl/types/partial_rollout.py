# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

from typing import TypeAlias, TypedDict

from .rollout import EnvId

AgentKey: TypeAlias = tuple[EnvId, int]
ProblemId: TypeAlias = str


class PartialMember(TypedDict):
    """One finished-but-ungrouped member of a partial rollout group.

    ``episode`` is an ``EpisodeResult`` serialized as response and conversation
    dictionaries. Keeping the raw episode defers reward evaluation until the
    group is complete and supports both single- and multi-turn agents.
    """

    rollout_idx: int
    episode: dict


class PartialGroupSnapshot(TypedDict):
    """Finished members and metadata needed to resume one group occurrence."""

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


PartialsByProblemId: TypeAlias = dict[ProblemId, list[PartialGroupSnapshot]]
RestoredPartialsByAgent: TypeAlias = dict[AgentKey, PartialsByProblemId]
