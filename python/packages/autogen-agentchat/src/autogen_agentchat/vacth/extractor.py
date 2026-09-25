from __future__ import annotations

import json
import re
from typing import Any

from .schema import EvidencePointer, Priority, StateSlot, VACTHEdge, VACTHStateItem

VALID_SLOTS: set[str] = {
    "fact",
    "constraint",
    "decision",
    "hypothesis",
    "tool_state_delta",
    "artifact",
    "unresolved_issue",
}
VALID_STATUSES: set[str] = {
    "observed",
    "verified",
    "hypothesis",
    "invalidated",
    "superseded",
    "active_constraint",
}
VALID_PRIORITIES: set[str] = {"low", "medium", "high", "critical"}
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+")


def build_extraction_prompt(
    *,
    task: str,
    sender: str,
    source_messages: list[dict[str, Any]],
    existing_items: list[dict[str, Any]],
    max_existing_items: int = 12,
    max_items_per_turn: int | None = None,
) -> str:
    latest_turn = "\n\n".join(
        (
            f"message_id={message.get('message_id')} "
            f"turn={message.get('turn_id')} "
            f"source={message.get('source')} "
            f"type={message.get('type')}\n"
            f"{message.get('content', '')}"
        )
        for message in source_messages
    )
    existing = json.dumps(
        [_compact_existing_item(item) for item in existing_items[-max_existing_items:]],
        ensure_ascii=False,
        indent=2,
    )
    item_limit_instruction = ""
    if max_items_per_turn is not None and max_items_per_turn > 0:
        item_limit_instruction = f" Emit at most {max_items_per_turn} high-value items for this turn."
    return (
        "Extract a VACTH typed handoff delta from the latest multi-agent turn. "
        "Return only valid JSON. The JSON object must have keys 'items' and "
        "'edges'. 'items' must be a list. Each item should include: slot, "
        "content, normalized_predicate, arguments, merge_key, epistemic_status, "
        "confidence, priority, expires_at, evidence. Valid slots are fact, "
        "constraint, decision, hypothesis, tool_state_delta, artifact, "
        "unresolved_issue. Valid epistemic statuses are observed, verified, "
        "hypothesis, invalidated, superseded, active_constraint. Evidence should "
        "point to message_id, source_agent, turn_id, and a short quote from the "
        "latest turn. Edges may use supports, contradicts, refines, supersedes, "
        "depends_on, duplicates, or violates. Do not invent facts."
        f"{item_limit_instruction}\n\n"
        "Software-repair extraction rules:\n"
        "- Preserve concrete test-derived requirements, edge cases, inputs, "
        "expected outputs, and pre-applied test-patch semantics as high-priority "
        "facts or constraints.\n"
        "- If a test helper transforms data broadly, capture the broad behavior "
        "explicitly; do not collapse it into a narrower implementation preference.\n"
        "- Treat unverified repair preferences such as 'keep the patch narrow' or "
        "'avoid broadening parsing' as hypotheses unless the latest turn cites "
        "direct evidence that they are required for correctness.\n"
        "- Decisions should describe committed edits or verified choices, not "
        "speculative implementation advice before code changes.\n"
        "- Keep each item content under 35 words unless a test input/output detail "
        "would be lost.\n\n"
        f"Original task:\n{task}\n\n"
        f"Existing active VACTH items:\n{existing}\n\n"
        f"Latest turn from {sender}:\n{latest_turn}\n\n"
        "VACTH JSON delta:"
    )


def _compact_existing_item(item: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item.get(key)
        for key in (
            "item_id",
            "slot",
            "content",
            "merge_key",
            "epistemic_status",
            "priority",
            "source_agent",
            "created_at_turn",
        )
        if key in item
    }


def parse_extraction_output(
    raw_text: str,
    *,
    sender: str,
    source_messages: list[dict[str, Any]],
    turn_id: int,
    next_item_index: int,
) -> tuple[list[VACTHStateItem], list[VACTHEdge]]:
    value = _extract_json_object(raw_text)
    raw_items = value.get("items", [])
    raw_edges = value.get("edges", [])
    if not isinstance(raw_items, list):
        raise ValueError("VACTH extraction 'items' must be a list")
    if not isinstance(raw_edges, list):
        raw_edges = []

    items: list[VACTHStateItem] = []
    for index, raw_item in enumerate(raw_items, start=next_item_index):
        if not isinstance(raw_item, dict):
            continue
        items.append(
            _item_from_mapping(
                raw_item,
                sender=sender,
                source_messages=source_messages,
                turn_id=turn_id,
                fallback_item_id=f"v{index}",
            )
        )

    edges: list[VACTHEdge] = []
    item_ids = {item.item_id for item in items}
    for raw_edge in raw_edges:
        if not isinstance(raw_edge, dict):
            continue
        source = str(raw_edge.get("source", "")).strip()
        target = str(raw_edge.get("target", "")).strip()
        relation = str(raw_edge.get("relation", "supports")).strip()
        if (
            not source
            or not target
            or relation not in VACTHEdge.model_fields["relation"].annotation.__args__
        ):  # type: ignore[attr-defined]
            continue
        if source not in item_ids and target not in item_ids:
            continue
        edges.append(
            VACTHEdge(
                source=source,
                relation=relation,  # type: ignore[arg-type]
                target=target,
                confidence=_clamp_float(raw_edge.get("confidence", 0.5)),
            )
        )
    return items, edges


def heuristic_items_from_messages(
    *,
    sender: str,
    source_messages: list[dict[str, Any]],
    turn_id: int,
    next_item_index: int,
) -> list[VACTHStateItem]:
    items: list[VACTHStateItem] = []
    for message in source_messages:
        tool_records = _tool_execution_records(message)
        if tool_records:
            for record in tool_records:
                item = _item_from_tool_record(
                    record,
                    sender=sender,
                    message=message,
                    turn_id=turn_id,
                    item_id=f"v{next_item_index + len(items)}",
                )
                if item is not None:
                    items.append(item)
            continue

        content = str(message.get("content", "")).strip()
        if not content:
            continue
        lower = content.lower()
        slot: StateSlot = "fact"
        status = "observed"
        priority: Priority = "medium"
        if "constraint" in lower or "do not" in lower:
            slot = "constraint"
            status = "active_constraint"
            priority = "high"
        elif "fail" in lower or "error" in lower or "traceback" in lower:
            slot = "unresolved_issue"
            status = "observed"
            priority = "high"
        elif "pass" in lower or "ok" in lower or "resolved" in lower:
            slot = "fact"
            status = "verified"
            priority = "high"
        elif "return a +" in lower or "patched" in lower or "implemented" in lower:
            slot = "decision"
            status = "observed"
            priority = "high"
        if message.get("role") == "event":
            slot = "tool_state_delta"
        item_id = f"v{next_item_index + len(items)}"
        summary = content if len(content) <= 500 else content[:497] + "..."
        items.append(
            VACTHStateItem(
                item_id=item_id,
                slot=slot,
                content=summary,
                normalized_predicate=None,
                arguments={},
                merge_key=_merge_key(slot, summary),
                epistemic_status=status,  # type: ignore[arg-type]
                confidence=0.55,
                priority=priority,
                source_agent=sender,
                evidence=[
                    EvidencePointer(
                        message_id=message.get("message_id"),
                        source_agent=message.get("source"),
                        turn_id=message.get("turn_id"),
                        quote=summary[:200],
                    )
                ],
                created_at_turn=turn_id,
            )
        )
    return items


def _tool_execution_records(message: dict[str, Any]) -> list[dict[str, Any]]:
    if message.get("type") != "ToolCallExecutionEvent":
        return []
    payload = message.get("payload")
    if not isinstance(payload, dict):
        return []
    records = payload.get("content")
    if not isinstance(records, list):
        return []
    return [record for record in records if isinstance(record, dict)]


def _item_from_tool_record(
    record: dict[str, Any],
    *,
    sender: str,
    message: dict[str, Any],
    turn_id: int,
    item_id: str,
) -> VACTHStateItem | None:
    tool = str(record.get("name", "")).strip()
    if not tool:
        return None
    parsed_content = _parse_tool_content(record.get("content"))
    content, status, priority, arguments = _summarize_tool_record(tool, parsed_content, record)
    if content is None:
        return None
    return VACTHStateItem(
        item_id=item_id,
        slot="tool_state_delta",
        content=content,
        normalized_predicate=f"tool.{tool}",
        arguments=arguments,
        merge_key=_merge_key("tool_state_delta", content),
        epistemic_status=status,
        confidence=0.75 if tool in {"run_tests", "test_status"} else 0.65,
        priority=priority,
        source_agent=sender,
        evidence=[
            EvidencePointer(
                message_id=message.get("message_id"),
                source_agent=message.get("source"),
                turn_id=message.get("turn_id"),
                quote=content[:200],
            )
        ],
        created_at_turn=turn_id,
    )


def _parse_tool_content(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _summarize_tool_record(
    tool: str, parsed_content: Any, record: dict[str, Any]
) -> tuple[str | None, str, Priority, dict[str, Any]]:
    arguments: dict[str, Any] = {
        "tool": tool,
        "is_error": bool(record.get("is_error", False)),
    }
    if isinstance(parsed_content, dict):
        arguments.update(
            {
                key: parsed_content.get(key)
                for key in (
                    "path",
                    "changed",
                    "passed",
                    "cached",
                    "executed",
                    "returncode",
                    "current_diff_tested",
                    "current_diff_verified",
                    "current_diff_failed",
                    "current_fingerprint",
                    "last_passed_fingerprint",
                )
                if key in parsed_content
            }
        )

    if tool == "run_tests" and isinstance(parsed_content, dict):
        passed = bool(parsed_content.get("passed", False))
        returncode = parsed_content.get("returncode")
        cached = bool(parsed_content.get("cached", False))
        output = "\n".join(
            str(parsed_content.get(key, ""))
            for key in ("stdout", "stderr", "stdout_tail", "stderr_tail")
            if parsed_content.get(key)
        )
        tail = _truncate(" ".join(output.split()), 280)
        result = "passed" if passed else "failed"
        cache_text = " cached" if cached else ""
        content = f"run_tests{cache_text} {result}"
        if returncode is not None:
            content += f" returncode={returncode}"
        if tail:
            content += f": {tail}"
        return content, "verified" if passed else "observed", "high", arguments

    if tool == "test_status" and isinstance(parsed_content, dict):
        tested = bool(parsed_content.get("current_diff_tested", False))
        verified = bool(parsed_content.get("current_diff_verified", False))
        failed = bool(parsed_content.get("current_diff_failed", False))
        changed_files = parsed_content.get("changed_files", [])
        if not isinstance(changed_files, list):
            changed_files = []
        content = (
            f"test_status current_diff_tested={tested} "
            f"current_diff_verified={verified} current_diff_failed={failed} "
            f"changed_files={','.join(str(path) for path in changed_files[:6]) or 'none'}"
        )
        return content, "verified" if verified else "observed", "high", arguments

    if tool in {"replace_text", "write_file", "edit_file", "apply_patch"} and isinstance(parsed_content, dict):
        path = parsed_content.get("path") or parsed_content.get("file") or "unknown"
        changed = bool(parsed_content.get("changed", True))
        diff = str(parsed_content.get("diff", "")).strip()
        if diff:
            arguments["diff_excerpt"] = _truncate(diff, 600)
        content = f"{tool} changed={changed} path={path}"
        if diff:
            content += f": {_truncate(' '.join(diff.split()), 280)}"
        return content, "observed", "high" if changed else "medium", arguments

    if tool == "git_diff":
        diff_text = (
            parsed_content
            if isinstance(parsed_content, str)
            else json.dumps(parsed_content, ensure_ascii=False, sort_keys=True)
        )
        diff_text = str(diff_text).strip()
        if not diff_text:
            return "git_diff shows no current workspace diff", "verified", "medium", arguments
        arguments["diff_excerpt"] = _truncate(diff_text, 600)
        return (
            f"git_diff shows current workspace diff: {_truncate(' '.join(diff_text.split()), 280)}",
            "observed",
            "high",
            arguments,
        )

    if record.get("is_error"):
        return f"{tool} returned tool error", "observed", "high", arguments
    return None, "observed", "medium", arguments


def _truncate(text: str, limit: int) -> str:
    clean = " ".join(text.split())
    if len(clean) <= limit:
        return clean
    return clean[: max(0, limit - 3)].rstrip() + "..."


def _item_from_mapping(
    raw_item: dict[str, Any],
    *,
    sender: str,
    source_messages: list[dict[str, Any]],
    turn_id: int,
    fallback_item_id: str,
) -> VACTHStateItem:
    content = str(raw_item.get("content", "")).strip()
    if not content:
        content = "(empty extracted item)"
    slot = str(raw_item.get("slot", "fact")).strip()
    if slot not in VALID_SLOTS:
        slot = "fact"
    status = str(raw_item.get("epistemic_status", "observed")).strip()
    if status not in VALID_STATUSES:
        status = "observed"
    priority = str(raw_item.get("priority", "medium")).strip()
    if priority not in VALID_PRIORITIES:
        priority = "medium"
    merge_key = str(raw_item.get("merge_key", "")).strip() or _merge_key(slot, content)
    evidence = _evidence_from_mapping(raw_item, source_messages)
    arguments = raw_item.get("arguments", {})
    if not isinstance(arguments, dict):
        arguments = {"value": arguments}
    return VACTHStateItem(
        item_id=str(raw_item.get("item_id") or fallback_item_id),
        slot=slot,  # type: ignore[arg-type]
        content=content,
        normalized_predicate=raw_item.get("normalized_predicate"),
        arguments=arguments,
        merge_key=merge_key,
        epistemic_status=status,  # type: ignore[arg-type]
        confidence=_clamp_float(raw_item.get("confidence", 0.6)),
        priority=priority,  # type: ignore[arg-type]
        source_agent=sender,
        evidence=evidence,
        created_at_turn=turn_id,
        expires_at=raw_item.get("expires_at"),
    )


def _evidence_from_mapping(
    raw_item: dict[str, Any], source_messages: list[dict[str, Any]]
) -> list[EvidencePointer]:
    raw_evidence = raw_item.get("evidence", [])
    evidence: list[EvidencePointer] = []
    if isinstance(raw_evidence, list):
        for entry in raw_evidence:
            if not isinstance(entry, dict):
                continue
            evidence.append(
                EvidencePointer(
                    message_id=entry.get("message_id"),
                    source_agent=entry.get("source_agent") or entry.get("source"),
                    turn_id=entry.get("turn_id"),
                    quote=entry.get("quote"),
                )
            )
    if evidence:
        return evidence
    if not source_messages:
        return []
    message = source_messages[-1]
    return [
        EvidencePointer(
            message_id=message.get("message_id"),
            source_agent=message.get("source"),
            turn_id=message.get("turn_id"),
            quote=str(message.get("content", ""))[:200],
        )
    ]


def _extract_json_object(text: str) -> dict[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise
        value = json.loads(text[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("VACTH extraction output must be a JSON object")
    return value


def _merge_key(slot: str, content: str) -> str:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(content)[:12]]
    return f"{slot}:{'_'.join(tokens) or 'item'}"


def _clamp_float(value: Any) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, numeric))
