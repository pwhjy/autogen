from __future__ import annotations

from collections import defaultdict

from .schema import VACTHCapsule, VACTHStateItem, estimate_tokens


SLOT_LABELS = {
    "constraint": "Active Constraints",
    "fact": "Facts",
    "decision": "Decisions",
    "hypothesis": "Hypotheses",
    "tool_state_delta": "Tool State",
    "artifact": "Artifacts",
    "unresolved_issue": "Unresolved Issues",
}


def render_capsule(
    capsule: VACTHCapsule,
    *,
    compact: bool = False,
    max_item_chars: int = 240,
) -> str:
    grouped = defaultdict(list)
    for item in capsule.selected_items:
        grouped[item.slot].append(item)

    if compact:
        return _render_compact_capsule(capsule, grouped, max_item_chars=max_item_chars)

    lines = [
        "VACTH role-specific handoff capsule",
        f"capsule_id: {capsule.capsule_id}",
        f"sender: {capsule.sender}",
        f"receiver: {capsule.receiver}",
        f"turn_id: {capsule.turn_id}",
        (
            "budget: "
            f"{capsule.budget.get('used_tokens', 0)}/"
            f"{capsule.budget.get('max_tokens', 0)} estimated tokens"
        ),
        "",
    ]
    if not capsule.selected_items:
        lines.append("(no selected VACTH state yet)")
    for slot, label in SLOT_LABELS.items():
        items = grouped.get(slot, [])
        if not items:
            continue
        lines.append(f"{label}:")
        for item in items:
            evidence = ""
            if item.evidence:
                pointer = item.evidence[0]
                evidence = (
                    f" evidence={pointer.source_agent or 'unknown'}:"
                    f"{pointer.message_id or 'unknown'}"
                )
            lines.append(
                "- "
                f"[{item.item_id}] "
                f"status={item.epistemic_status} "
                f"priority={item.priority} "
                f"confidence={item.confidence:.2f}"
                f"{evidence}\n"
                f"  {item.content}"
            )
        lines.append("")
    if capsule.selected_edges:
        lines.append("Provenance Edges:")
        for edge in capsule.selected_edges:
            lines.append(
                f"- {edge.source} --{edge.relation}--> {edge.target} "
                f"(confidence={edge.confidence:.2f})"
            )
        lines.append("")
    if capsule.unresolved_issues:
        lines.append("Targeted attention:")
        lines.append(
            "Resolve or explicitly account for the unresolved issues above before "
            "claiming success."
        )
        lines.append("")
    lines.append(f"rendered_tokens_estimate: {estimate_tokens(chr(10).join(lines))}")
    return "\n".join(lines)


def _render_compact_capsule(
    capsule: VACTHCapsule,
    grouped: dict[str, list[VACTHStateItem]],
    *,
    max_item_chars: int,
) -> str:
    lines = [
        "VACTH compact capsule",
        f"id={capsule.capsule_id} sender={capsule.sender} receiver={capsule.receiver} turn={capsule.turn_id}",
        (
            "budget="
            f"{capsule.budget.get('used_tokens', 0)}/"
            f"{capsule.budget.get('max_tokens', 0)} "
            f"items={len(capsule.selected_items)}"
        ),
        "",
    ]
    if not capsule.selected_items:
        lines.append("(empty)")
    for slot, label in SLOT_LABELS.items():
        items = grouped.get(slot, [])
        if not items:
            continue
        lines.append(f"{label}:")
        for item in items:
            content = _truncate(item.content, max_item_chars)
            lines.append(
                "- "
                f"{item.item_id} "
                f"src={item.source_agent} "
                f"status={item.epistemic_status} "
                f"priority={item.priority}: "
                f"{content}"
            )
    if capsule.selected_edges:
        edge_text = ", ".join(
            f"{edge.source}-{edge.relation}->{edge.target}"
            for edge in capsule.selected_edges[:8]
        )
        if len(capsule.selected_edges) > 8:
            edge_text += f", +{len(capsule.selected_edges) - 8} more"
        lines.extend(["", f"Edges: {edge_text}"])
    if capsule.unresolved_issues:
        lines.extend(["", "Attention: unresolved issues above must be addressed before success."])
    lines.append(f"rendered_tokens_estimate: {estimate_tokens(chr(10).join(lines))}")
    return "\n".join(lines)


def _truncate(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."
