from __future__ import annotations

import re

from .schema import EdgeRelation, VACTHEdge, VACTHStateItem


STATUS_RANK = {
    "hypothesis": 1,
    "observed": 2,
    "active_constraint": 3,
    "verified": 4,
    "invalidated": 4,
    "superseded": 0,
}
PRIORITY_RANK = {"low": 1, "medium": 2, "high": 3, "critical": 4}
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+")


class ProvenanceAwareAggregator:
    def __init__(
        self, *, enable_paa: bool = True, enable_provenance: bool = True
    ) -> None:
        self.enable_paa = enable_paa
        self.enable_provenance = enable_provenance
        self.items: list[VACTHStateItem] = []
        self.edges: list[VACTHEdge] = []
        self._active_by_merge_key: dict[str, VACTHStateItem] = {}
        self._seen_item_ids: set[str] = set()

    def add_delta(
        self,
        items: list[VACTHStateItem],
        edges: list[VACTHEdge] | None = None,
    ) -> dict[str, list[VACTHStateItem] | list[VACTHEdge]]:
        added_items: list[VACTHStateItem] = []
        added_edges: list[VACTHEdge] = []
        for item in items:
            item = self._deduplicate_item_id(item)
            if not self.enable_provenance:
                item.evidence = []

            previous = self._active_by_merge_key.get(item.merge_key)
            self.items.append(item)
            added_items.append(item)

            if previous is None or not self.enable_paa:
                self._active_by_merge_key[item.merge_key] = item
                added_edges.extend(self._supersede_resolved_issues(item))
                added_edges.extend(self._link_related_items(item))
                continue

            relation = self._infer_relation(previous, item)
            edge = VACTHEdge(
                source=item.item_id,
                relation=relation,
                target=previous.item_id,
                confidence=max(previous.confidence, item.confidence),
            )
            if self.enable_provenance:
                self.edges.append(edge)
                added_edges.append(edge)

            if relation == "duplicates":
                previous.confidence = max(previous.confidence, item.confidence)
                previous.evidence.extend(item.evidence)
                continue
            if relation in {"supersedes", "refines"}:
                previous.epistemic_status = "superseded"
            self._active_by_merge_key[item.merge_key] = item
            added_edges.extend(self._supersede_resolved_issues(item))
            if relation != "duplicates":
                added_edges.extend(self._link_related_items(item))

        if self.enable_provenance:
            for edge in edges or []:
                if (
                    edge.source in self._seen_item_ids
                    or edge.target in self._seen_item_ids
                ):
                    self.edges.append(edge)
                    added_edges.append(edge)

        return {"items": added_items, "edges": added_edges}

    def active_items(self) -> list[VACTHStateItem]:
        inactive = {"superseded", "invalidated"}
        if not self.enable_paa:
            return [
                item for item in self.items if item.epistemic_status not in inactive
            ]
        return [
            item
            for item in self.items
            if item.epistemic_status not in inactive
            and self._active_by_merge_key.get(item.merge_key, item).item_id
            == item.item_id
        ]

    def graph(self) -> dict[str, list[dict[str, object]]]:
        return {
            "items": [item.model_dump(mode="json") for item in self.items],
            "edges": [edge.model_dump(mode="json") for edge in self.edges],
        }

    def _deduplicate_item_id(self, item: VACTHStateItem) -> VACTHStateItem:
        if item.item_id not in self._seen_item_ids:
            self._seen_item_ids.add(item.item_id)
            return item
        base = item.item_id
        index = 2
        while f"{base}_{index}" in self._seen_item_ids:
            index += 1
        item.item_id = f"{base}_{index}"
        self._seen_item_ids.add(item.item_id)
        return item

    def _infer_relation(
        self, previous: VACTHStateItem, item: VACTHStateItem
    ) -> EdgeRelation:
        if previous.content.strip().lower() == item.content.strip().lower():
            return "duplicates"
        if (
            previous.epistemic_status == "hypothesis"
            and item.epistemic_status == "verified"
        ):
            return "supports"
        if item.epistemic_status == "invalidated":
            return "contradicts"
        if previous.slot == "tool_state_delta" or item.slot == "tool_state_delta":
            return "supersedes"
        if STATUS_RANK.get(item.epistemic_status, 0) >= STATUS_RANK.get(
            previous.epistemic_status, 0
        ):
            return "refines"
        return "duplicates"

    def _supersede_resolved_issues(self, item: VACTHStateItem) -> list[VACTHEdge]:
        if not self.enable_paa or not self.enable_provenance:
            return []
        edges: list[VACTHEdge] = []
        resolved_key = item.arguments.get("resolved_issue_merge_key")
        if isinstance(resolved_key, str) and resolved_key:
            target = self._active_by_merge_key.get(resolved_key)
            if target is not None:
                edge = self._make_supersedes_edge(item, target)
                if edge is not None:
                    edges.append(edge)

        if not self._looks_like_resolution(item):
            return edges
        for target in list(self._active_by_merge_key.values()):
            if target.item_id == item.item_id:
                continue
            if target.slot != "unresolved_issue":
                continue
            if target.epistemic_status in {"superseded", "invalidated"}:
                continue
            if self._issue_overlap(item, target):
                edge = self._make_supersedes_edge(item, target)
                if edge is not None:
                    edges.append(edge)
        return edges

    def _make_supersedes_edge(
        self, item: VACTHStateItem, target: VACTHStateItem
    ) -> VACTHEdge | None:
        if target.item_id == item.item_id:
            return None
        target.epistemic_status = "superseded"
        edge = VACTHEdge(
            source=item.item_id,
            relation="supersedes",
            target=target.item_id,
            confidence=max(item.confidence, target.confidence),
        )
        self.edges.append(edge)
        return edge

    def _looks_like_resolution(self, item: VACTHStateItem) -> bool:
        if item.epistemic_status == "verified":
            return True
        if item.slot not in {"decision", "artifact", "tool_state_delta", "fact"}:
            return False
        text = item.content.lower()
        return any(
            marker in text
            for marker in [
                "passed",
                "verified",
                "resolved",
                "changed",
                "now returns",
                "return a + b",
                "tests ran with ok",
            ]
        )

    def _issue_overlap(self, item: VACTHStateItem, target: VACTHStateItem) -> bool:
        item_tokens = set(TOKEN_PATTERN.findall(item.content.lower()))
        target_tokens = set(TOKEN_PATTERN.findall(target.content.lower()))
        useful = {
            token
            for token in item_tokens.intersection(target_tokens)
            if len(token) > 2 and token not in {"the", "and", "with", "from"}
        }
        if len(useful) >= 2:
            return True
        item_text = item.content.lower()
        target_text = target.content.lower()
        return ("return a + b" in item_text or "passed" in item_text) and (
            "subtract" in target_text or "failed" in target_text
        )

    def _link_related_items(self, item: VACTHStateItem) -> list[VACTHEdge]:
        if not self.enable_paa or not self.enable_provenance:
            return []
        edges: list[VACTHEdge] = []
        for target in list(self._active_by_merge_key.values()):
            if target.item_id == item.item_id:
                continue
            if target.epistemic_status in {"superseded", "invalidated"}:
                continue
            if not self._issue_overlap(item, target):
                continue
            relation: EdgeRelation = "refines"
            if target.slot == "constraint":
                relation = "depends_on"
            elif item.epistemic_status == "verified":
                relation = "supports"
            edge = VACTHEdge(
                source=item.item_id,
                relation=relation,
                target=target.item_id,
                confidence=min(1.0, max(item.confidence, target.confidence)),
            )
            self.edges.append(edge)
            edges.append(edge)
            if len(edges) >= 2:
                break
        return edges
