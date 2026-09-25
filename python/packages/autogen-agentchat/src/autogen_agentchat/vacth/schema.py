from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


StateSlot = Literal[
    "fact",
    "constraint",
    "decision",
    "hypothesis",
    "tool_state_delta",
    "artifact",
    "unresolved_issue",
]
EpistemicStatus = Literal[
    "observed",
    "verified",
    "hypothesis",
    "invalidated",
    "superseded",
    "active_constraint",
]
Priority = Literal["low", "medium", "high", "critical"]
EdgeRelation = Literal[
    "supports",
    "contradicts",
    "refines",
    "supersedes",
    "depends_on",
    "duplicates",
    "violates",
]
RenderingMode = Literal[
    "brief",
    "compact",
    "capsule",
    "capsule_with_evidence",
    "full_fragment",
]


class EvidencePointer(BaseModel):
    message_id: str | None = None
    source_agent: str | None = None
    turn_id: int | None = None
    quote: str | None = None


class VACTHStateItem(BaseModel):
    item_id: str
    slot: StateSlot
    content: str
    normalized_predicate: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    merge_key: str
    epistemic_status: EpistemicStatus = "observed"
    confidence: float = 0.5
    priority: Priority = "medium"
    source_agent: str
    evidence: list[EvidencePointer] = Field(default_factory=list)
    created_at_turn: int
    expires_at: str | None = None


class VACTHEdge(BaseModel):
    source: str
    relation: EdgeRelation
    target: str
    confidence: float = 0.5


class VACTHCapsule(BaseModel):
    capsule_id: str
    sender: str
    receiver: str
    task_id: str
    turn_id: int
    selected_items: list[VACTHStateItem]
    selected_edges: list[VACTHEdge] = Field(default_factory=list)
    unresolved_issues: list[VACTHStateItem] = Field(default_factory=list)
    budget: dict[str, int] = Field(default_factory=dict)
    rendering_mode: RenderingMode = "capsule_with_evidence"


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split()))
