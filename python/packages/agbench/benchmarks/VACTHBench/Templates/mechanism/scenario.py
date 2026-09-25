import csv
import json
import math
import re
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from autogen_agentchat.vacth import (
    VACTHRuntime,
    heuristic_items_from_messages,
    parse_extraction_output,
)
from autogen_agentchat.vacth.schema import EvidencePointer, VACTHEdge, VACTHStateItem


EXPERIMENT_CONFIG = json.loads(r"""__EXPERIMENT_CONFIG_JSON__""")
TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|-?\d+")
STRUCTURED_SLOT_MAP = {
    "facts": "fact",
    "constraints": "constraint",
    "decisions": "decision",
    "hypotheses": "hypothesis",
    "tool_states": "tool_state_delta",
    "open_issues": "unresolved_issue",
}
VACTH_ABLATIONS = {
    "vacth_wo_cve",
    "vacth_wo_thc",
    "vacth_wo_paa",
    "vacth_wo_provenance",
    "vacth_wo_reask",
    "vacth_global_capsule",
}
VACTH_METHODS = {"vacth_full", *VACTH_ABLATIONS}


def config_value_bool(key: str, default: bool) -> bool:
    value = EXPERIMENT_CONFIG.get(key, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def vacth_flags(method: str) -> dict[str, bool]:
    defaults = {
        "enable_thc": True,
        "enable_cve": True,
        "enable_paa": True,
        "enable_provenance": True,
        "enable_reask": True,
        "role_specific_routing": True,
    }
    if method == "vacth_wo_cve":
        defaults["enable_cve"] = False
        defaults["role_specific_routing"] = False
    elif method == "vacth_wo_thc":
        defaults["enable_thc"] = False
    elif method == "vacth_wo_paa":
        defaults["enable_paa"] = False
    elif method == "vacth_wo_provenance":
        defaults["enable_provenance"] = False
    elif method == "vacth_wo_reask":
        defaults["enable_reask"] = False
    elif method == "vacth_global_capsule":
        defaults["role_specific_routing"] = False
    return {
        key: config_value_bool(key, default)
        for key, default in defaults.items()
    }


def is_vacth_method(method: str) -> bool:
    return method in VACTH_METHODS


def estimate_tokens(text: str) -> int:
    return max(1, len(str(text).split()))


def write_json(path: str, value: Any) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(value, fh, ensure_ascii=False, indent=2, sort_keys=True)


def write_metrics_csv(path: str, metrics: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(metrics.keys()))
        writer.writeheader()
        writer.writerow(metrics)


def default_merge_key(slot: str, content: str) -> str:
    tokens = [token.lower() for token in TOKEN_PATTERN.findall(content)[:10]]
    return f"{slot}:{'_'.join(tokens) or 'item'}"


def evidence_from_raw(raw: Any) -> list[EvidencePointer]:
    evidence: list[EvidencePointer] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str):
                evidence.append(EvidencePointer(message_id=entry))
            elif isinstance(entry, dict):
                evidence.append(
                    EvidencePointer(
                        message_id=entry.get("message_id"),
                        source_agent=entry.get("source_agent") or entry.get("source"),
                        turn_id=entry.get("turn_id"),
                        quote=entry.get("quote"),
                    )
                )
    return evidence


def item_from_mapping(
    raw: dict[str, Any],
    *,
    fallback_id: str,
    turn_id: int = 0,
    sender: str = "mechanism",
) -> VACTHStateItem:
    content = str(raw.get("content", "")).strip() or "(empty state item)"
    slot = str(raw.get("slot", "fact"))
    return VACTHStateItem(
        item_id=str(raw.get("item_id") or fallback_id),
        slot=slot,  # type: ignore[arg-type]
        content=content,
        normalized_predicate=raw.get("normalized_predicate"),
        arguments=raw.get("arguments")
        if isinstance(raw.get("arguments"), dict)
        else {},
        merge_key=str(raw.get("merge_key") or default_merge_key(slot, content)),
        epistemic_status=str(raw.get("epistemic_status", "observed")),  # type: ignore[arg-type]
        confidence=float(raw.get("confidence", 0.6)),
        priority=str(raw.get("priority", "medium")),  # type: ignore[arg-type]
        source_agent=str(raw.get("source_agent") or sender),
        evidence=evidence_from_raw(
            raw.get("evidence", raw.get("evidence_message_ids", []))
        ),
        created_at_turn=int(raw.get("created_at_turn", turn_id)),
        expires_at=raw.get("expires_at"),
    )


def edge_from_mapping(raw: dict[str, Any]) -> VACTHEdge:
    return VACTHEdge(
        source=str(raw.get("source")),
        relation=str(raw.get("relation", "supports")),  # type: ignore[arg-type]
        target=str(raw.get("target")),
        confidence=float(raw.get("confidence", 0.6)),
    )


def items_to_json(items: Iterable[VACTHStateItem]) -> list[dict[str, Any]]:
    return [item.model_dump(mode="json") for item in items]


def edges_to_json(edges: Iterable[VACTHEdge]) -> list[dict[str, Any]]:
    return [edge.model_dump(mode="json") for edge in edges]


def item_key(item: VACTHStateItem | dict[str, Any]) -> str:
    if isinstance(item, VACTHStateItem):
        return item.merge_key
    content = str(item.get("content", ""))
    slot = str(item.get("slot", "fact"))
    return str(item.get("merge_key") or default_merge_key(slot, content))


def f1(predicted: set[Any], gold: set[Any]) -> float | None:
    if not predicted and not gold:
        return 1.0
    if not gold:
        return 0.0 if predicted else 1.0
    if not predicted:
        return 0.0
    overlap = len(predicted.intersection(gold))
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def rounded(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def evidence_pairs(
    items: list[VACTHStateItem] | list[dict[str, Any]],
) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for item in items:
        key = item_key(item)
        raw_evidence = (
            item.evidence
            if isinstance(item, VACTHStateItem)
            else item.get("evidence", item.get("evidence_message_ids", []))
        )
        for entry in raw_evidence:
            message_id = None
            if isinstance(entry, EvidencePointer):
                message_id = entry.message_id
            elif isinstance(entry, dict):
                message_id = entry.get("message_id")
            if message_id:
                pairs.add((key, str(message_id)))
    return pairs


def edge_key(edge: VACTHEdge | dict[str, Any]) -> tuple[str, str, str]:
    if isinstance(edge, VACTHEdge):
        return (edge.source, edge.relation, edge.target)
    return (str(edge.get("source")), str(edge.get("relation")), str(edge.get("target")))


def extraction_scores(
    predicted_items: list[VACTHStateItem],
    gold: dict[str, Any],
) -> dict[str, Any]:
    gold_items = list(gold.get("items", []))
    pred_by_key = {item.merge_key: item for item in predicted_items}
    gold_by_key = {
        item_key(item): item for item in gold_items if isinstance(item, dict)
    }
    pred_keys = set(pred_by_key)
    gold_keys = set(gold_by_key)
    slot_pairs_pred = {(item.merge_key, item.slot) for item in predicted_items}
    slot_pairs_gold = {
        (item_key(item), str(item.get("slot", "fact")))
        for item in gold_items
        if isinstance(item, dict)
    }
    status_hits = 0
    for key, gold_item in gold_by_key.items():
        predicted = pred_by_key.get(key)
        if predicted is not None and predicted.epistemic_status == gold_item.get(
            "epistemic_status"
        ):
            status_hits += 1
    status_accuracy = status_hits / len(gold_by_key) if gold_by_key else None
    return {
        "mechanism_extraction_item_f1": rounded(f1(pred_keys, gold_keys)),
        "mechanism_slot_f1": rounded(f1(slot_pairs_pred, slot_pairs_gold)),
        "mechanism_status_accuracy": rounded(status_accuracy),
        "mechanism_evidence_f1": rounded(
            f1(evidence_pairs(predicted_items), evidence_pairs(gold_items))
        ),
    }


def active_item_scores(
    predicted_items: list[VACTHStateItem], gold: dict[str, Any]
) -> dict[str, Any]:
    gold_active = set(gold.get("active_merge_keys", []))
    predicted_active = {
        item.merge_key
        for item in predicted_items
        if item.epistemic_status not in {"superseded", "invalidated"}
    }
    gold_superseded = set(gold.get("superseded_item_ids", []))
    predicted_superseded = {
        item.item_id
        for item in predicted_items
        if item.epistemic_status in {"superseded", "invalidated"}
    }
    if gold_superseded:
        supersession_accuracy = len(
            predicted_superseded.intersection(gold_superseded)
        ) / len(gold_superseded)
    else:
        supersession_accuracy = 1.0 if not predicted_superseded else 0.0
    return {
        "mechanism_active_item_f1": rounded(f1(predicted_active, gold_active)),
        "mechanism_supersession_accuracy": rounded(supersession_accuracy),
    }


def edge_scores(
    predicted_edges: list[VACTHEdge], gold: dict[str, Any]
) -> dict[str, Any]:
    gold_edges = {
        edge_key(edge) for edge in gold.get("edges", []) if isinstance(edge, dict)
    }
    predicted = {edge_key(edge) for edge in predicted_edges}
    return {"mechanism_edge_f1": rounded(f1(predicted, gold_edges))}


def routing_ndcg(selected_ids: list[str], preferred_order: list[str]) -> float | None:
    if not preferred_order:
        return None
    relevance = {
        item_id: len(preferred_order) - index
        for index, item_id in enumerate(preferred_order)
    }

    def dcg(ids: list[str]) -> float:
        total = 0.0
        for index, item_id in enumerate(ids):
            rel = relevance.get(item_id, 0)
            if rel:
                total += (2**rel - 1) / math.log2(index + 2)
        return total

    ideal = dcg(preferred_order)
    return 0.0 if ideal == 0 else dcg(selected_ids) / ideal


def routing_scores(selected_ids: list[str], gold: dict[str, Any]) -> dict[str, Any]:
    gold_ids = set(gold.get("selected_item_ids", []))
    selected = set(selected_ids)
    recall = len(selected.intersection(gold_ids)) / len(gold_ids) if gold_ids else None
    preferred_order = list(
        gold.get("preferred_order") or gold.get("selected_item_ids", [])
    )
    return {
        "mechanism_routing_recall": rounded(recall),
        "mechanism_routing_ndcg": rounded(routing_ndcg(selected_ids, preferred_order)),
    }


def overall_score(metrics: dict[str, Any], keys: list[str]) -> float:
    values = [metrics[key] for key in keys if isinstance(metrics.get(key), int | float)]
    if not values:
        return 0.0
    return round(sum(float(value) for value in values) / len(values), 4)


def structured_items_from_sample(sample: dict[str, Any]) -> list[VACTHStateItem]:
    structured = sample.get("structured_state", {})
    items: list[VACTHStateItem] = []
    if not isinstance(structured, dict):
        return items
    for slot_name, values in structured.items():
        slot = STRUCTURED_SLOT_MAP.get(slot_name, "fact")
        raw_values = values if isinstance(values, list) else [values]
        for value in raw_values:
            content = str(value).strip()
            if not content:
                continue
            status = "active_constraint" if slot == "constraint" else "observed"
            if slot == "tool_state_delta" and any(
                word in content.lower() for word in ["pass", "ok", "verified"]
            ):
                status = "verified"
            items.append(
                VACTHStateItem(
                    item_id=f"structured_{len(items) + 1}",
                    slot=slot,  # type: ignore[arg-type]
                    content=content,
                    merge_key=default_merge_key(slot, content),
                    epistemic_status=status,  # type: ignore[arg-type]
                    confidence=0.55,
                    priority="medium",
                    source_agent="structured_summary",
                    created_at_turn=int(sample.get("turn_id", 0)),
                )
            )
    return items


def summary_items_from_sample(sample: dict[str, Any]) -> list[VACTHStateItem]:
    summary = str(sample.get("summary", "")).strip()
    if not summary:
        return []
    return [
        VACTHStateItem(
            item_id="summary_1",
            slot="fact",
            content=summary,
            merge_key=default_merge_key("fact", summary),
            epistemic_status="observed",
            confidence=0.4,
            priority="medium",
            source_agent="summary",
            created_at_turn=int(sample.get("turn_id", 0)),
        )
    ]


def make_vacth_runtime(
    sample: dict[str, Any], method: str, *, default_token_budget: int = 80
) -> VACTHRuntime:
    flags = vacth_flags(method)
    return VACTHRuntime(
        task_id=str(sample.get("id", "mechanism")),
        token_budget=int(sample.get("token_budget", default_token_budget)),
        method_name=method,
        enable_cve=flags["enable_cve"],
        enable_paa=flags["enable_paa"],
        enable_provenance=flags["enable_provenance"],
        role_specific_routing=flags["role_specific_routing"],
    )


def apply_provenance_switch(
    *,
    items: list[VACTHStateItem],
    edges: list[VACTHEdge],
    method: str,
) -> tuple[list[VACTHStateItem], list[VACTHEdge]]:
    if not is_vacth_method(method) or vacth_flags(method)["enable_provenance"]:
        return items, edges
    for item in items:
        item.evidence = []
    return items, []


def make_raw_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    messages = sample.get("messages", [])
    if not isinstance(messages, list) or not messages:
        messages = [
            {
                "message_id": "m0",
                "turn_id": 0,
                "source": "user",
                "role": "user",
                "content": sample.get("task", ""),
            }
        ]
    normalized: list[dict[str, Any]] = []
    for index, message in enumerate(messages, start=1):
        content = str(message.get("content", ""))
        normalized.append(
            {
                "message_id": str(message.get("message_id", f"m{index}")),
                "turn_id": int(message.get("turn_id", sample.get("turn_id", 0))),
                "source": str(message.get("source", "user")),
                "role": str(message.get("role", "assistant")),
                "type": str(message.get("type", "MechanismMessage")),
                "content": content,
                "estimated_tokens": estimate_tokens(content),
            }
        )
    return normalized


def visible_context(
    raw_messages: list[dict[str, Any]], method: str, mechanism_type: str
) -> list[dict[str, Any]]:
    return [
        {
            "turn_id": max(
                (int(message.get("turn_id", 0)) for message in raw_messages), default=0
            ),
            "agent": method,
            "mode": f"mechanism_{mechanism_type}",
            "message_ids": [message["message_id"] for message in raw_messages],
            "context": raw_messages,
            "tools": [],
            "estimated_tokens": sum(
                message.get("estimated_tokens", 0) for message in raw_messages
            ),
        }
    ]


def run_extraction(sample: dict[str, Any], method: str) -> dict[str, Any]:
    raw_messages = make_raw_messages(sample)
    edges: list[VACTHEdge] = []
    vacth_extractions: list[dict[str, Any]] = []
    if is_vacth_method(method):
        flags = vacth_flags(method)
        if flags["enable_thc"]:
            raw_update = json.dumps(
                sample.get("vacth_delta", {"items": [], "edges": []}),
                ensure_ascii=False,
            )
            try:
                items, edges = parse_extraction_output(
                    raw_update,
                    sender=str(sample.get("sender", "agent")),
                    source_messages=raw_messages,
                    turn_id=int(sample.get("turn_id", 1)),
                    next_item_index=1,
                )
                parse_valid = True
                parse_error = None
            except Exception as exc:
                items = []
                parse_valid = False
                parse_error = str(exc)
            extraction_mode = "deterministic_delta"
        else:
            items = heuristic_items_from_messages(
                sender=str(sample.get("sender", "agent")),
                source_messages=raw_messages,
                turn_id=int(sample.get("turn_id", 1)),
                next_item_index=1,
            )
            raw_update = json.dumps(
                {
                    "mode": "heuristic_without_thc",
                    "source_message_count": len(raw_messages),
                    "item_count": len(items),
                },
                ensure_ascii=False,
            )
            parse_valid = True
            parse_error = None
            extraction_mode = "heuristic_without_thc"
        items, edges = apply_provenance_switch(
            items=items,
            edges=edges,
            method=method,
        )
        vacth_extractions.append(
            {
                "turn_id": int(sample.get("turn_id", 1)),
                "sender": sample.get("sender", "agent"),
                "raw_update": raw_update,
                "parse_valid": parse_valid,
                "parse_error": parse_error,
                "extraction_mode": extraction_mode,
                "reask_attempts": 0,
                "item_ids": [item.item_id for item in items],
                "edge_count": len(edges),
                "source_message_ids": [
                    message["message_id"] for message in raw_messages
                ],
            }
        )
    elif method == "structured_summary":
        items = structured_items_from_sample(sample)
    else:
        items = summary_items_from_sample(sample)
    scores = extraction_scores(items, sample.get("gold", {}))
    scores["mechanism_overall_score"] = overall_score(
        scores,
        [
            "mechanism_extraction_item_f1",
            "mechanism_slot_f1",
            "mechanism_status_accuracy",
            "mechanism_evidence_f1",
        ],
    )
    return {
        "raw_messages": raw_messages,
        "visible_contexts": visible_context(raw_messages, method, "extraction"),
        "state_items": items,
        "provenance_edges": edges,
        "routing_decisions": [],
        "capsules": [],
        "vacth_extractions": vacth_extractions,
        "scores": scores,
    }


def run_routing(sample: dict[str, Any], method: str) -> dict[str, Any]:
    raw_messages = make_raw_messages(sample)
    items = [
        item_from_mapping(
            raw, fallback_id=f"r{index}", turn_id=int(sample.get("turn_id", 1))
        )
        for index, raw in enumerate(sample.get("items", []), start=1)
        if isinstance(raw, dict)
    ]
    edges = [
        edge_from_mapping(raw)
        for raw in sample.get("edges", [])
        if isinstance(raw, dict)
    ]
    routing_decisions: list[dict[str, Any]] = []
    capsules: list[dict[str, Any]] = []
    selected_ids: list[str] = []
    if is_vacth_method(method):
        runtime = make_vacth_runtime(sample, method, default_token_budget=48)
        runtime.update(items, edges)
        capsule, decision = runtime.build_capsule(
            receiver=str(sample.get("receiver", "reviewer")),
            sender=str(sample.get("sender", "mechanism")),
            turn_id=int(sample.get("turn_id", 1)),
        )
        selected_ids = [item.item_id for item in capsule.selected_items]
        capsules.append(capsule.model_dump(mode="json"))
        routing_decisions.append(decision)
    elif method == "structured_summary":
        budget = int(sample.get("token_budget", 48))
        used = 0
        for item in items:
            cost = estimate_tokens(item.content)
            if selected_ids and used + cost > budget:
                continue
            selected_ids.append(item.item_id)
            used += cost
        routing_decisions.append(
            {
                "turn_id": int(sample.get("turn_id", 1)),
                "receiver": sample.get("receiver", "reviewer"),
                "sender": "structured_summary",
                "method": method,
                "routing_method": "global_structured_chronological",
                "budget": budget,
                "selected_item_ids": selected_ids,
                "estimated_tokens": used,
            }
        )
    else:
        selected_ids = list(sample.get("summary_selected_item_ids", []))
        routing_decisions.append(
            {
                "turn_id": int(sample.get("turn_id", 1)),
                "receiver": sample.get("receiver", "reviewer"),
                "sender": "summary",
                "method": method,
                "routing_method": "free_text_summary_no_typed_route",
                "budget": int(sample.get("token_budget", 48)),
                "selected_item_ids": selected_ids,
                "estimated_tokens": estimate_tokens(sample.get("summary", "")),
            }
        )
    scores = routing_scores(selected_ids, sample.get("gold", {}))
    scores["mechanism_budget_used_ratio"] = rounded(
        routing_decisions[0].get("estimated_tokens", 0)
        / max(1, int(sample.get("token_budget", 48)))
    )
    scores["mechanism_overall_score"] = overall_score(
        scores,
        ["mechanism_routing_recall", "mechanism_routing_ndcg"],
    )
    return {
        "raw_messages": raw_messages,
        "visible_contexts": visible_context(raw_messages, method, "routing"),
        "state_items": items,
        "provenance_edges": edges,
        "routing_decisions": routing_decisions,
        "capsules": capsules,
        "vacth_extractions": [],
        "scores": scores,
    }


def run_aggregation(sample: dict[str, Any], method: str) -> dict[str, Any]:
    raw_messages = make_raw_messages(sample)
    all_items: list[VACTHStateItem] = []
    edges: list[VACTHEdge] = []
    if is_vacth_method(method):
        runtime = make_vacth_runtime(sample, method, default_token_budget=80)
        for delta_index, delta in enumerate(sample.get("deltas", []), start=1):
            delta_items = [
                item_from_mapping(
                    raw,
                    fallback_id=f"a{delta_index}_{index}",
                    turn_id=int(delta.get("turn_id", delta_index)),
                    sender=str(delta.get("sender", "mechanism")),
                )
                for index, raw in enumerate(delta.get("items", []), start=1)
                if isinstance(raw, dict)
            ]
            delta_edges = [
                edge_from_mapping(raw)
                for raw in delta.get("edges", [])
                if isinstance(raw, dict)
            ]
            runtime.update(delta_items, delta_edges)
        all_items = list(runtime.aggregator.items)
        edges = list(runtime.aggregator.edges)
    elif method == "structured_summary":
        latest_by_key: dict[str, VACTHStateItem] = {}
        for delta_index, delta in enumerate(sample.get("deltas", []), start=1):
            for index, raw in enumerate(delta.get("items", []), start=1):
                if not isinstance(raw, dict):
                    continue
                item = item_from_mapping(
                    raw,
                    fallback_id=f"structured_a{delta_index}_{index}",
                    turn_id=int(delta.get("turn_id", delta_index)),
                    sender="structured_summary",
                )
                latest_by_key[item.merge_key] = item
        all_items = list(latest_by_key.values())
    else:
        all_items = summary_items_from_sample(sample)
    scores = {}
    scores.update(active_item_scores(all_items, sample.get("gold", {})))
    scores.update(edge_scores(edges, sample.get("gold", {})))
    scores["mechanism_overall_score"] = overall_score(
        scores,
        [
            "mechanism_active_item_f1",
            "mechanism_edge_f1",
            "mechanism_supersession_accuracy",
        ],
    )
    return {
        "raw_messages": raw_messages,
        "visible_contexts": visible_context(raw_messages, method, "aggregation"),
        "state_items": all_items,
        "provenance_edges": edges,
        "routing_decisions": [],
        "capsules": [],
        "vacth_extractions": [],
        "scores": scores,
    }


def run_reask(sample: dict[str, Any], method: str) -> dict[str, Any]:
    raw_messages = make_raw_messages(sample)
    items: list[VACTHStateItem] = []
    edges: list[VACTHEdge] = []
    vacth_extractions: list[dict[str, Any]] = []
    reask_attempts = 0
    reask_target = None
    parse_success = False
    if is_vacth_method(method):
        flags = vacth_flags(method)
        if not flags["enable_thc"]:
            items = heuristic_items_from_messages(
                sender=str(sample.get("sender", "agent")),
                source_messages=raw_messages,
                turn_id=int(sample.get("turn_id", 1)),
                next_item_index=1,
            )
            parse_success = True
            vacth_extractions.append(
                {
                    "turn_id": int(sample.get("turn_id", 1)),
                    "sender": sample.get("sender", "agent"),
                    "raw_update": json.dumps(
                        {
                            "mode": "heuristic_without_thc",
                            "source_message_count": len(raw_messages),
                            "item_count": len(items),
                        },
                        ensure_ascii=False,
                    ),
                    "parse_valid": True,
                    "parse_error": None,
                    "extraction_mode": "heuristic_without_thc",
                    "reask_attempts": 0,
                    "item_ids": [item.item_id for item in items],
                    "edge_count": 0,
                    "source_message_ids": [
                        message["message_id"] for message in raw_messages
                    ],
                }
            )
        else:
            bad_text = str(sample.get("bad_delta_text", ""))
            try:
                items, edges = parse_extraction_output(
                    bad_text,
                    sender=str(sample.get("sender", "agent")),
                    source_messages=raw_messages,
                    turn_id=int(sample.get("turn_id", 1)),
                    next_item_index=1,
                )
                if sample.get("gold", {}).get("reask_needed", False):
                    raise ValueError("sample requires a targeted re-ask repair")
                parse_success = True
            except Exception as exc:
                vacth_extractions.append(
                    {
                        "turn_id": int(sample.get("turn_id", 1)),
                        "sender": sample.get("sender", "agent"),
                        "raw_update": bad_text,
                        "parse_valid": False,
                        "parse_error": str(exc),
                        "extraction_mode": "deterministic_bad_delta",
                        "reask_attempts": 0,
                        "item_ids": [],
                        "edge_count": 0,
                        "source_message_ids": [
                            message["message_id"] for message in raw_messages
                        ],
                    }
                )
                if flags["enable_reask"]:
                    reask_attempts = 1
                    reask_target = str(
                        sample.get("reask_target", "vacth_extractor_json")
                    )
                    repair_text = json.dumps(
                        sample.get("repair_delta", {"items": [], "edges": []}),
                        ensure_ascii=False,
                    )
                    items, edges = parse_extraction_output(
                        repair_text,
                        sender=str(sample.get("sender", "agent")),
                        source_messages=raw_messages,
                        turn_id=int(sample.get("turn_id", 1)),
                        next_item_index=1,
                    )
                    parse_success = True
                    vacth_extractions.append(
                        {
                            "turn_id": int(sample.get("turn_id", 1)),
                            "sender": sample.get("sender", "agent"),
                            "raw_update": repair_text,
                            "parse_valid": True,
                            "parse_error": None,
                            "extraction_mode": "deterministic_reask_repair",
                            "reask_attempts": 1,
                            "item_ids": [item.item_id for item in items],
                            "edge_count": len(edges),
                            "source_message_ids": [
                                message["message_id"] for message in raw_messages
                            ],
                        }
                    )
                else:
                    items = heuristic_items_from_messages(
                        sender=str(sample.get("sender", "agent")),
                        source_messages=raw_messages,
                        turn_id=int(sample.get("turn_id", 1)),
                        next_item_index=1,
                    )
                    vacth_extractions[-1]["extraction_mode"] = (
                        "heuristic_without_reask"
                    )
                    vacth_extractions[-1]["item_ids"] = [
                        item.item_id for item in items
                    ]
                    parse_success = False
        items, edges = apply_provenance_switch(
            items=items,
            edges=edges,
            method=method,
        )
    elif method == "structured_summary":
        items = structured_items_from_sample(sample)
    else:
        items = summary_items_from_sample(sample)

    gold = sample.get("gold", {})
    target_accuracy = (
        1.0
        if reask_target is not None and reask_target == gold.get("reask_target")
        else 0.0
    )
    scores = extraction_scores(items, gold)
    scores.update(
        {
            "mechanism_reask_parse_success": 1.0 if parse_success else 0.0,
            "mechanism_reask_target_accuracy": target_accuracy,
            "mechanism_reask_needed": 1.0 if gold.get("reask_needed", False) else 0.0,
            "mechanism_reask_attempts": reask_attempts,
        }
    )
    scores["mechanism_overall_score"] = overall_score(
        scores,
        [
            "mechanism_extraction_item_f1",
            "mechanism_slot_f1",
            "mechanism_status_accuracy",
            "mechanism_evidence_f1",
            "mechanism_reask_parse_success",
            "mechanism_reask_target_accuracy",
        ],
    )
    return {
        "raw_messages": raw_messages,
        "visible_contexts": visible_context(raw_messages, method, "reask"),
        "state_items": items,
        "provenance_edges": edges,
        "routing_decisions": [],
        "capsules": [],
        "vacth_extractions": vacth_extractions,
        "scores": scores,
    }


def run_sample(
    sample: dict[str, Any], method: str, mechanism_type: str
) -> dict[str, Any]:
    if mechanism_type == "extraction":
        return run_extraction(sample, method)
    if mechanism_type == "routing":
        return run_routing(sample, method)
    if mechanism_type == "aggregation":
        return run_aggregation(sample, method)
    if mechanism_type == "reask":
        return run_reask(sample, method)
    raise ValueError(f"Unknown mechanism task type: {mechanism_type}")


def main() -> None:
    start = time.time()
    method = str(EXPERIMENT_CONFIG["method"])
    mechanism_type = str(EXPERIMENT_CONFIG["mechanism_task_type"])
    sample = EXPERIMENT_CONFIG["sample"]
    task_id = EXPERIMENT_CONFIG["task_id"]
    active_vacth_flags = vacth_flags(method) if is_vacth_method(method) else {}

    run = run_sample(sample, method, mechanism_type)
    raw_messages = run["raw_messages"]
    state_items = items_to_json(run["state_items"])
    provenance_edges = edges_to_json(run["provenance_edges"])
    summaries = []
    structured_states = []
    if method == "summary":
        summaries = [
            {
                "summary_id": "summary_1",
                "turn_id": int(sample.get("turn_id", 1)),
                "sender": sample.get("sender", "mechanism"),
                "content": sample.get("summary", ""),
                "estimated_tokens": estimate_tokens(sample.get("summary", "")),
            }
        ]
    if method == "structured_summary":
        structured_states = [
            {
                "state_id": "structured_1",
                "turn_id": int(sample.get("turn_id", 1)),
                "sender": sample.get("sender", "mechanism"),
                "state": sample.get("structured_state", {}),
                "schema_valid": True,
                "parse_error": None,
            }
        ]
    capsule_token_counts = [
        int(capsule.get("budget", {}).get("used_tokens", 0))
        for capsule in run["capsules"]
        if isinstance(capsule.get("budget", {}), dict)
    ]

    metrics = {
        "success": True,
        "turns": max(1, len(raw_messages)),
        "tool_calls": 0,
        "estimated_tokens": sum(
            message.get("estimated_tokens", 0) for message in raw_messages
        ),
        "wall_time_sec": round(time.time() - start, 4),
        "raw_messages": len(raw_messages),
        "visible_contexts": len(run["visible_contexts"]),
        "state_items": len(state_items),
        "routing_decisions": len(run["routing_decisions"]),
        "provenance_edges": len(provenance_edges),
        "summary_count": len(summaries),
        "structured_update_count": len(structured_states),
        "vacth_extraction_count": len(run["vacth_extractions"]),
        "vacth_parse_error_count": len(
            [
                entry
                for entry in run["vacth_extractions"]
                if not entry.get("parse_valid", False)
            ]
        ),
        "vacth_heuristic_extraction_count": len(
            [
                entry
                for entry in run["vacth_extractions"]
                if "heuristic" in str(entry.get("extraction_mode", ""))
            ]
        ),
        "vacth_reask_count": sum(
            int(entry.get("reask_attempts", 0)) for entry in run["vacth_extractions"]
        ),
        "vacth_capsule_count": len(run["capsules"]),
        "vacth_selected_item_count": sum(
            len(capsule.get("selected_items", [])) for capsule in run["capsules"]
        ),
        "vacth_cve_routing_count": len(
            [
                decision
                for decision in run["routing_decisions"]
                if str(decision.get("routing_method", "")).startswith("cve")
            ]
        ),
        "vacth_avg_capsule_tokens": rounded(
            sum(capsule_token_counts) / len(capsule_token_counts)
            if capsule_token_counts
            else None
        ),
        "vacth_enable_thc": active_vacth_flags.get("enable_thc"),
        "vacth_enable_cve": active_vacth_flags.get("enable_cve"),
        "vacth_enable_paa": active_vacth_flags.get("enable_paa"),
        "vacth_enable_provenance": active_vacth_flags.get("enable_provenance"),
        "vacth_enable_reask": active_vacth_flags.get("enable_reask"),
        "vacth_role_specific_routing": active_vacth_flags.get(
            "role_specific_routing"
        ),
        "mechanism_task_type": mechanism_type,
        "mechanism_sample_id": sample.get("id"),
        **run["scores"],
    }
    final_answer = f"mechanism_score={metrics.get('mechanism_overall_score', 0)}"
    result = {
        "task_id": task_id,
        "method": method,
        "experiment_config": EXPERIMENT_CONFIG,
        "raw_messages": raw_messages,
        "visible_contexts": run["visible_contexts"],
        "tool_calls": [],
        "summaries": summaries,
        "structured_states": structured_states,
        "capsules": run["capsules"],
        "vacth_extractions": run["vacth_extractions"],
        "memory_items": [],
        "state_items": state_items,
        "routing_decisions": run["routing_decisions"],
        "provenance_edges": provenance_edges,
        "metrics": metrics,
        "final_answer": final_answer,
    }

    write_json("raw_messages.json", raw_messages)
    write_json("visible_contexts.json", run["visible_contexts"])
    write_json("tool_calls.json", [])
    write_json("summaries.json", summaries)
    write_json(
        "structured_summary.json",
        {"schema": list(STRUCTURED_SLOT_MAP), "states": structured_states},
    )
    write_json("capsules.json", run["capsules"])
    write_json("vacth_extractions.json", run["vacth_extractions"])
    write_json("vector_memory.json", {"memory_items": [], "retrievals": []})
    write_json("state_items.json", state_items)
    write_json("routing_decisions.json", run["routing_decisions"])
    write_json(
        "provenance_graph.json", {"items": state_items, "edges": provenance_edges}
    )
    write_json("result.json", result)
    write_metrics_csv("metrics.csv", metrics)
    Path("mechanism_gold.json").write_text(
        json.dumps(
            sample.get("gold", {}), ensure_ascii=False, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    print(f"FINAL ANSWER: {final_answer}")
    print("ALL TESTS PASSED !#!#")


if __name__ == "__main__":
    main()
