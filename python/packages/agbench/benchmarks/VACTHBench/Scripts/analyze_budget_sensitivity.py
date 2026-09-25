from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
DATA_PATH = BENCHMARK_DIR / "data" / "routing.jsonl"


PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}
PRIORITY_VALUE = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
STATUS_ORDER = {
    "superseded": 0,
    "hypothesis": 1,
    "observed": 2,
    "active_constraint": 3,
    "verified": 4,
    "invalidated": 4,
}
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


@dataclass
class EvidencePointer:
    message_id: str | None = None
    source_agent: str | None = None
    turn_id: int | None = None
    quote: str | None = None


@dataclass
class VACTHStateItem:
    item_id: str
    slot: str
    content: str
    merge_key: str
    epistemic_status: str
    confidence: float
    priority: str
    source_agent: str
    evidence: list[EvidencePointer]
    created_at_turn: int
    normalized_predicate: str | None = None
    arguments: dict[str, Any] | None = None
    expires_at: str | None = None


@dataclass
class Selection:
    selected_ids: list[str]
    used_tokens: int


def estimate_render_tokens(item: VACTHStateItem) -> int:
    evidence_text = " ".join(
        " ".join(
            part
            for part in [
                pointer.source_agent or "",
                str(pointer.turn_id or ""),
                pointer.message_id or "",
                pointer.quote or "",
            ]
            if part
        )
        for pointer in item.evidence
    )
    rendered = (
        f"slot {item.slot} status {item.epistemic_status} priority {item.priority} "
        f"source {item.source_agent} merge_key {item.merge_key} "
        f"content {item.content} evidence {evidence_text}"
    )
    return max(1, len(rendered.split()))


def score_item(item: VACTHStateItem, *, receiver: str, turn_id: int) -> float:
    role_values = ROLE_SLOT_VALUE.get(receiver, {})
    role_utility = role_values.get(item.slot, 0.45)
    recency = 1.0 / max(1, turn_id - item.created_at_turn + 1)
    priority = PRIORITY_VALUE.get(item.priority, 0.5)
    status = STATUS_VALUE.get(item.epistemic_status, 0.55)
    fidelity = (status + item.confidence) / 2
    evidence_strength = min(1.0, 0.2 * len(item.evidence))
    lower_content = item.content.lower()
    test_signal = 0.35 if any(term in lower_content for term in TEST_SIGNAL_TERMS) else 0.0
    cost_penalty = 0.002 * len(item.content.split())
    return (
        1.0 * role_utility
        + 0.45 * recency
        + 0.6 * priority
        + 0.7 * fidelity
        + 0.25 * evidence_strength
        + test_signal
        - cost_penalty
    )


def load_samples() -> list[dict[str, Any]]:
    samples = []
    with DATA_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def evidence_from_raw(raw: Any) -> list[EvidencePointer]:
    evidence = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, dict):
                evidence.append(
                    EvidencePointer(
                        message_id=entry.get("message_id"),
                        source_agent=entry.get("source_agent") or entry.get("source"),
                        turn_id=entry.get("turn_id"),
                        quote=entry.get("quote"),
                    )
                )
            elif isinstance(entry, str):
                evidence.append(EvidencePointer(message_id=entry))
    return evidence


def item_from_mapping(raw: dict[str, Any], fallback_id: str) -> VACTHStateItem:
    return VACTHStateItem(
        item_id=str(raw.get("item_id") or fallback_id),
        slot=str(raw.get("slot", "fact")),
        content=str(raw.get("content", "") or "(empty state item)"),
        normalized_predicate=raw.get("normalized_predicate"),
        arguments=raw.get("arguments") if isinstance(raw.get("arguments"), dict) else {},
        merge_key=str(raw.get("merge_key") or fallback_id),
        epistemic_status=str(raw.get("epistemic_status", "observed")),
        confidence=float(raw.get("confidence", 0.5)),
        priority=str(raw.get("priority", "medium")),
        source_agent=str(raw.get("source_agent") or "agent"),
        evidence=evidence_from_raw(raw.get("evidence", [])),
        created_at_turn=int(raw.get("created_at_turn", 0)),
        expires_at=raw.get("expires_at"),
    )


def expand_content(content: str, target_tokens: int) -> str:
    words = content.split()
    if len(words) >= target_tokens:
        return content
    filler = (
        "background note stale branch local observation candidate patch note "
        "previous attempt unrelated module historical context"
    ).split()
    expanded = list(words)
    index = 0
    while len(expanded) < target_tokens:
        expanded.append(filler[index % len(filler)])
        index += 1
    return " ".join(expanded)


def distractor_item(
    raw: dict[str, Any],
    *,
    item_id: str,
    sample_turn: int,
    recent: bool,
    min_tokens: int,
) -> VACTHStateItem:
    cloned = dict(raw)
    cloned["item_id"] = item_id
    cloned["merge_key"] = f"distractor:{item_id}"
    cloned["content"] = expand_content(str(raw.get("content", "irrelevant state")), min_tokens)
    cloned["priority"] = "medium" if recent else "low"
    cloned["confidence"] = min(0.7, float(raw.get("confidence", 0.5)))
    cloned["created_at_turn"] = sample_turn if recent else max(1, sample_turn - 12)
    if recent and cloned.get("slot") in {"constraint", "tool_state_delta", "unresolved_issue"}:
        cloned["epistemic_status"] = "observed"
    return item_from_mapping(cloned, item_id)


def expanded_items_for_sample(
    sample: dict[str, Any],
    all_samples: list[dict[str, Any]],
    *,
    distractors: int,
    min_state_tokens: int,
    min_distractor_tokens: int,
    seed: int,
) -> list[VACTHStateItem]:
    items = []
    for index, raw in enumerate(sample.get("items", []), start=1):
        if not isinstance(raw, dict):
            continue
        cloned = dict(raw)
        cloned["content"] = expand_content(str(raw.get("content", "")), min_state_tokens)
        items.append(item_from_mapping(cloned, f"item_{index}"))
    if distractors <= 0:
        return items

    rng = random.Random(f"{seed}:{sample.get('id')}")
    pool = [
        raw
        for other in all_samples
        if other.get("id") != sample.get("id")
        for raw in other.get("items", [])
        if isinstance(raw, dict)
    ]
    sample_turn = int(sample.get("turn_id", 1))
    for index in range(distractors):
        raw = rng.choice(pool)
        recent = index % 3 != 0
        items.append(
            distractor_item(
                raw,
                item_id=f"{sample.get('id')}_distractor_{index + 1}",
                sample_turn=sample_turn,
                recent=recent,
                min_tokens=min_distractor_tokens,
            )
        )
    return items


def f1_like_recall(selected_ids: list[str], gold_ids: set[str]) -> tuple[float, float]:
    if not gold_ids:
        return 1.0, 1.0
    selected = set(selected_ids)
    recall = len(selected & gold_ids) / len(gold_ids)
    precision = len(selected & gold_ids) / len(selected) if selected else 0.0
    return recall, precision


def ndcg(selected_ids: list[str], preferred_order: list[str]) -> float:
    if not preferred_order:
        return 1.0
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


def budget_value(label: str) -> int | None:
    if label == "unlimited":
        return None
    return int(label)


def select_in_order(items: list[VACTHStateItem], budget: int | None) -> Selection:
    selected = []
    used = 0
    for item in items:
        cost = estimate_render_tokens(item)
        if budget is not None and selected and used + cost > budget:
            continue
        selected.append(item.item_id)
        used += cost
    return Selection(selected, used)


def select_value_aware(
    items: list[VACTHStateItem],
    *,
    receiver: str,
    turn_id: int,
    budget: int | None,
) -> Selection:
    scored = [
        (
            score_item(item, receiver=receiver, turn_id=turn_id),
            estimate_render_tokens(item),
            item,
        )
        for item in items
        if item.epistemic_status != "superseded"
    ]
    scored.sort(key=lambda row: row[0] / max(1, row[1]), reverse=True)
    return select_in_order([item for _, _, item in scored], budget)


def select_recent(items: list[VACTHStateItem], budget: int | None) -> Selection:
    ordered = sorted(
        items,
        key=lambda item: (
            item.created_at_turn,
            PRIORITY_ORDER.get(item.priority, 0),
            STATUS_ORDER.get(item.epistemic_status, 0),
        ),
        reverse=True,
    )
    return select_in_order(ordered, budget)


def select_random(
    items: list[VACTHStateItem],
    budget: int | None,
    *,
    seed: str,
) -> Selection:
    ordered = list(items)
    random.Random(seed).shuffle(ordered)
    return select_in_order(ordered, budget)


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["method"], row["budget"]), []).append(row)
    summary = []
    for (method, budget), group in sorted(grouped.items()):
        summary.append(
            {
                "method": method,
                "budget": budget,
                "runs": len(group),
                "routing_recall": sum(row["routing_recall"] for row in group) / len(group),
                "routing_precision": sum(row["routing_precision"] for row in group) / len(group),
                "routing_ndcg": sum(row["routing_ndcg"] for row in group) / len(group),
                "all_gold_selected_rate": sum(row["all_gold_selected"] for row in group) / len(group),
                "avg_selected_items": sum(row["selected_items"] for row in group) / len(group),
                "avg_used_tokens": sum(row["used_tokens"] for row in group) / len(group),
            }
        )
    budget_order = {
        "128": 0,
        "256": 1,
        "512": 2,
        "1024": 3,
        "2048": 4,
        "4096": 5,
        "unlimited": 6,
    }
    method_order = {
        "vacth_value_aware": 0,
        "recent_state": 1,
        "random_state": 2,
        "all_state_upper_bound": 3,
    }
    summary.sort(
        key=lambda row: (
            method_order.get(row["method"], 99),
            budget_order.get(row["budget"], 99),
        )
    )
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "budget",
        "runs",
        "routing_recall",
        "routing_precision",
        "routing_ndcg",
        "all_gold_selected_rate",
        "avg_selected_items",
        "avg_used_tokens",
    ]
    lines = [
        "# Budget Sensitivity Summary",
        "",
        "Routing-only stress test built from `data/routing.jsonl` with deterministic cross-task distractors.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" if column in {"method", "budget"} else "---:" for column in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row[column]) for column in columns) + " |")
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `routing_recall` measures how many gold state items are preserved under the capsule budget.",
            "- `all_gold_selected_rate` is the fraction of runs where every required state item survived.",
            "- Random selection is averaged over repeated deterministic shuffles.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--budgets",
        nargs="+",
        default=["128", "256", "512", "1024", "2048", "4096", "unlimited"],
    )
    parser.add_argument("--distractors", type=int, default=48)
    parser.add_argument("--min-state-tokens", type=int, default=80)
    parser.add_argument("--min-distractor-tokens", type=int, default=90)
    parser.add_argument("--random-repeats", type=int, default=30)
    parser.add_argument("--seed", type=int, default=20260630)
    args = parser.parse_args()

    samples = load_samples()
    rows = []
    for sample in samples:
        items = expanded_items_for_sample(
            sample,
            samples,
            distractors=args.distractors,
            min_state_tokens=args.min_state_tokens,
            min_distractor_tokens=args.min_distractor_tokens,
            seed=args.seed,
        )
        gold_ids = set(sample.get("gold", {}).get("selected_item_ids", []))
        preferred_order = list(sample.get("gold", {}).get("preferred_order", gold_ids))
        receiver = str(sample.get("receiver", "reviewer"))
        turn_id = int(sample.get("turn_id", 1))
        for budget_label in args.budgets:
            budget = budget_value(budget_label)
            selections = [
                ("vacth_value_aware", select_value_aware(items, receiver=receiver, turn_id=turn_id, budget=budget)),
                ("recent_state", select_recent(items, budget)),
            ]
            if budget_label == "unlimited":
                selections.append(("all_state_upper_bound", select_in_order(items, None)))
            for method, selection in selections:
                recall, precision = f1_like_recall(selection.selected_ids, gold_ids)
                rows.append(
                    {
                        "sample_id": sample.get("id"),
                        "method": method,
                        "budget": budget_label,
                        "repeat": 0,
                        "routing_recall": recall,
                        "routing_precision": precision,
                        "routing_ndcg": ndcg(selection.selected_ids, preferred_order),
                        "all_gold_selected": 1.0 if recall == 1.0 else 0.0,
                        "selected_items": len(selection.selected_ids),
                        "used_tokens": selection.used_tokens,
                    }
                )
            for repeat in range(args.random_repeats):
                selection = select_random(
                    items,
                    budget,
                    seed=f"{args.seed}:{sample.get('id')}:{budget_label}:{repeat}",
                )
                recall, precision = f1_like_recall(selection.selected_ids, gold_ids)
                rows.append(
                    {
                        "sample_id": sample.get("id"),
                        "method": "random_state",
                        "budget": budget_label,
                        "repeat": repeat,
                        "routing_recall": recall,
                        "routing_precision": precision,
                        "routing_ndcg": ndcg(selection.selected_ids, preferred_order),
                        "all_gold_selected": 1.0 if recall == 1.0 else 0.0,
                        "selected_items": len(selection.selected_ids),
                        "used_tokens": selection.used_tokens,
                    }
                )

    summary = summarize(rows)
    write_csv(rows, RESULTS_DIR / "budget_sensitivity_raw.csv")
    write_csv(summary, RESULTS_DIR / "budget_sensitivity_summary.csv")
    write_markdown(summary, RESULTS_DIR / "budget_sensitivity_summary.md")
    print(f"Wrote {RESULTS_DIR / 'budget_sensitivity_summary.md'}")


if __name__ == "__main__":
    main()
