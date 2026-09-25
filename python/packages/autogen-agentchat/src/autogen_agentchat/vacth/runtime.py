from __future__ import annotations

from .aggregator import ProvenanceAwareAggregator
from .renderer import render_capsule
from .router import ContextualValueEstimator
from .schema import VACTHCapsule, VACTHEdge, VACTHStateItem


class VACTHRuntime:
    def __init__(
        self,
        *,
        task_id: str,
        token_budget: int = 900,
        method_name: str = "vacth_full",
        enable_cve: bool = True,
        enable_paa: bool = True,
        enable_provenance: bool = True,
        role_specific_routing: bool = True,
    ) -> None:
        self.task_id = task_id
        self.aggregator = ProvenanceAwareAggregator(
            enable_paa=enable_paa,
            enable_provenance=enable_provenance,
        )
        self.router = ContextualValueEstimator(
            token_budget=token_budget,
            enable_cve=enable_cve,
            role_specific_routing=role_specific_routing,
            method_name=method_name,
        )

    def update(
        self, items: list[VACTHStateItem], edges: list[VACTHEdge] | None = None
    ) -> dict[str, list[VACTHStateItem] | list[VACTHEdge]]:
        return self.aggregator.add_delta(items, edges or [])

    def build_capsule(
        self, *, receiver: str, sender: str, turn_id: int, token_budget: int | None = None
    ) -> tuple[VACTHCapsule, dict[str, object]]:
        return self.router.build_capsule(
            items=self.aggregator.active_items(),
            edges=self.aggregator.edges,
            receiver=receiver,
            sender=sender,
            task_id=self.task_id,
            turn_id=turn_id,
            token_budget=token_budget,
        )

    def render_capsule(
        self,
        capsule: VACTHCapsule,
        *,
        compact: bool = False,
        max_item_chars: int = 240,
    ) -> str:
        return render_capsule(capsule, compact=compact, max_item_chars=max_item_chars)
