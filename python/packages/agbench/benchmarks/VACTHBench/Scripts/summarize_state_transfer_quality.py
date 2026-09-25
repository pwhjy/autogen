from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
METHODS = ["summary", "structured_summary", "vacth_full"]
METRICS = [
    ("mechanism_extraction_item_f1", "typed_item_retention"),
    ("mechanism_slot_f1", "slot_type_accuracy"),
    ("mechanism_status_accuracy", "epistemic_status_accuracy"),
    ("mechanism_evidence_f1", "evidence_retention"),
    ("mechanism_routing_recall", "budgeted_routing_recall"),
    ("mechanism_routing_ndcg", "routing_order_ndcg"),
    ("mechanism_active_item_f1", "active_state_f1"),
    ("mechanism_edge_f1", "provenance_edge_f1"),
    ("mechanism_supersession_accuracy", "stale_state_handling"),
    ("mechanism_reask_parse_success", "repair_parse_success"),
    ("mechanism_reask_target_accuracy", "repair_target_accuracy"),
]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def mean(values: list[float | None]) -> float | None:
    clean = [value for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def read_rows(method: str) -> list[dict[str, Any]]:
    path = RESULTS_DIR / f"mechanism_suite_{method}" / "metrics.csv"
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_method(method: str) -> dict[str, Any]:
    rows = read_rows(method)
    record: dict[str, Any] = {"method": method, "runs": len(rows)}
    quality_values = []
    for source, target in METRICS:
        value = mean([as_float(row.get(source)) for row in rows])
        record[target] = value
        if value is not None:
            quality_values.append(value)
    record["state_transfer_quality"] = mean(quality_values)
    return record


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = ["method", "runs", *[target for _, target in METRICS], "state_transfer_quality"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "runs",
        "typed_item_retention",
        "slot_type_accuracy",
        "epistemic_status_accuracy",
        "evidence_retention",
        "budgeted_routing_recall",
        "active_state_f1",
        "provenance_edge_f1",
        "repair_target_accuracy",
        "state_transfer_quality",
    ]
    lines = [
        "# State Transfer Quality Summary",
        "",
        "Derived from the 200-task mechanism suites. Metrics measure whether downstream agents receive typed, current, evidenced, and repairable state.",
        "",
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" if column == "method" else "---:" for column in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(column)) for column in columns) + " |")
    lines.extend(
        [
            "",
            "Metric mapping:",
            "",
            "- Typed item / slot / epistemic status correspond to whether constraints, decisions, hypotheses, and tool state survive as distinct state.",
            "- Evidence retention and provenance edge F1 correspond to source preservation.",
            "- Active-state F1 and stale-state handling correspond to avoiding outdated or superseded state propagation.",
            "- Repair target accuracy captures whether malformed or incomplete handoff triggers the right targeted re-ask.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = [summarize_method(method) for method in METHODS]
    write_csv(rows, RESULTS_DIR / "state_transfer_quality_summary.csv")
    write_markdown(rows, RESULTS_DIR / "state_transfer_quality_summary.md")
    print(f"Wrote {RESULTS_DIR / 'state_transfer_quality_summary.md'}")


if __name__ == "__main__":
    main()
