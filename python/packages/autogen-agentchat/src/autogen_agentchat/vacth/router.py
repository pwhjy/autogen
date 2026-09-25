from __future__ import annotations

from .schema import VACTHCapsule, VACTHEdge, VACTHStateItem, estimate_tokens


PRIORITY_VALUE = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
STATUS_VALUE = {
    "hypothesis": 0.35,
    "observed": 0.55,
    "active_constraint": 0.9,
    "verified": 1.0,
    "invalidated": 0.8,
    "superseded": 0.1,
}
ROLE_SLOT_VALUE = {
    "global": {
        "constraint": 0.85,
        "unresolved_issue": 0.85,
        "tool_state_delta": 0.8,
        "decision": 0.75,
        "artifact": 0.7,
        "fact": 0.7,
        "hypothesis": 0.55,
    },
    "planner": {
        "constraint": 0.9,
        "unresolved_issue": 0.8,
        "hypothesis": 0.6,
        "fact": 0.5,
    },
    "retriever": {
        "constraint": 0.8,
        "unresolved_issue": 0.9,
        "hypothesis": 0.7,
        "fact": 0.6,
    },
    "coder": {
        "constraint": 0.9,
        "decision": 0.8,
        "artifact": 0.8,
        "tool_state_delta": 0.7,
        "unresolved_issue": 0.8,
        "fact": 0.7,
    },
    "tester": {
        "constraint": 0.8,
        "decision": 0.7,
        "artifact": 0.7,
        "tool_state_delta": 0.9,
        "unresolved_issue": 0.8,
        "fact": 0.6,
    },
    "reviewer": {
        "constraint": 1.0,
        "decision": 0.9,
        "artifact": 0.8,
        "tool_state_delta": 1.0,
        "unresolved_issue": 0.9,
        "fact": 0.8,
        "hypothesis": 0.5,
    },
}
TEST_SIGNAL_TERMS = {
    "fail_to_pass",
    "expected",
    "regression",
    "test",
    "tests",
    "edge case",
    "lowercase",
    "uppercase",
    "case-insensitive",
    "case insensitive",
    "masked",
    "empty",
    "zero",
    "boundary",
}


class ContextualValueEstimator:
    def __init__(
        self,
        *,
        token_budget: int = 900,
        enable_cve: bool = True,
        role_specific_routing: bool = True,
        method_name: str = "vacth_full",
    ) -> None:
        self.token_budget = token_budget
        self.enable_cve = enable_cve
        self.role_specific_routing = role_specific_routing
        self.method_name = method_name

    def build_capsule(
        self,
        *,
        items: list[VACTHStateItem],
        edges: list[VACTHEdge],
        receiver: str,
        sender: str,
        task_id: str,
        turn_id: int,
        token_budget: int | None = None,
    ) -> tuple[VACTHCapsule, dict[str, object]]:
        effective_token_budget = self.token_budget if token_budget is None else token_budget
        scoring_receiver = receiver if self.role_specific_routing else "global"
        scored = self._score_items(
            items=items,
            receiver=scoring_receiver,
            turn_id=turn_id,
        )
        selected: list[VACTHStateItem] = []
        used_tokens = 0
        for record in scored:
            cost = int(record["tokens"])
            if selected and used_tokens + cost > effective_token_budget:
                continue
            selected.append(record["item"])  # type: ignore[arg-type]
            used_tokens += cost

        selected_ids = {item.item_id for item in selected}
        selected_edges = [
            edge
            for edge in edges
            if edge.source in selected_ids and edge.target in selected_ids
        ]
        unresolved = [
            item
            for item in selected
            if item.slot == "unresolved_issue"
            and item.epistemic_status not in {"verified", "superseded", "invalidated"}
            and not item.merge_key.startswith("issue_resolved:")
        ]
        capsule = VACTHCapsule(
            capsule_id=f"capsule_{turn_id}_{receiver}",
            sender=sender,
            receiver=receiver,
            task_id=task_id,
            turn_id=turn_id,
            selected_items=selected,
            selected_edges=selected_edges,
            unresolved_issues=unresolved,
            budget={
                "max_tokens": effective_token_budget,
                "used_tokens": used_tokens,
                "candidate_items": len(scored),
            },
        )
        routing_decision = {
            "turn_id": turn_id,
            "receiver": receiver,
            "sender": sender,
            "method": self.method_name,
            "routing_method": self._routing_method(),
            "budget": effective_token_budget,
            "enable_cve": self.enable_cve,
            "role_specific_routing": self.role_specific_routing,
            "scoring_receiver": scoring_receiver,
            "selected_item_ids": [item.item_id for item in selected],
            "selected_edge_count": len(selected_edges),
            "estimated_tokens": used_tokens,
            "scored_items": [
                {
                    "item_id": record["item"].item_id,  # type: ignore[index,union-attr]
                    "value": round(float(record["value"]), 4),
                    "tokens": record["tokens"],
                }
                for record in scored
            ],
        }
        return capsule, routing_decision

    def _score_items(
        self, *, items: list[VACTHStateItem], receiver: str, turn_id: int
    ) -> list[dict[str, object]]:
        candidates = [item for item in items if item.epistemic_status != "superseded"]
        if not self.enable_cve:
            return [
                {
                    "item": item,
                    "value": 1.0,
                    "tokens": estimate_tokens(item.content),
                }
                for item in candidates
            ]

        scored: list[dict[str, object]] = [
            {
                "item": item,
                "value": self.score_item(item, receiver=receiver, turn_id=turn_id),
                "tokens": estimate_tokens(item.content),
            }
            for item in candidates
        ]
        scored.sort(
            key=lambda record: float(record["value"]) / max(1, int(record["tokens"])),
            reverse=True,
        )
        return scored

    def _routing_method(self) -> str:
        if not self.enable_cve:
            return "global_chronological_budget"
        if not self.role_specific_routing:
            return "cve_global_greedy"
        return "cve_greedy"

    def score_item(self, item: VACTHStateItem, *, receiver: str, turn_id: int) -> float:
        role_values = ROLE_SLOT_VALUE.get(receiver, {})
        role_utility = role_values.get(item.slot, 0.45)
        recency = 1.0 / max(1, turn_id - item.created_at_turn + 1)
        priority = PRIORITY_VALUE[item.priority]
        fidelity = (STATUS_VALUE[item.epistemic_status] + item.confidence) / 2
        evidence_strength = min(1.0, 0.2 * len(item.evidence))
        lower_content = item.content.lower()
        test_signal = 0.35 if any(term in lower_content for term in TEST_SIGNAL_TERMS) else 0.0
        cost_penalty = 0.002 * estimate_tokens(item.content)
        return (
            1.0 * role_utility
            + 0.45 * recency
            + 0.6 * priority
            + 0.7 * fidelity
            + 0.25 * evidence_strength
            + test_signal
            - cost_penalty
        )
