from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from mechanism_score_calibration import calibrated_row

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
METHODS = [
    "vacth_full",
    "vacth_wo_cve",
    "vacth_global_capsule",
    "vacth_wo_thc",
    "vacth_wo_paa",
    "vacth_wo_provenance",
    "vacth_wo_reask",
]
TASK_TYPES = ["all", "extraction", "routing", "aggregation", "reask"]
METRIC_COLUMNS = [
    ("mechanism_overall_score", "overall"),
    ("mechanism_extraction_item_f1", "extract_item_f1"),
    ("mechanism_slot_f1", "slot_f1"),
    ("mechanism_status_accuracy", "status_acc"),
    ("mechanism_evidence_f1", "evidence_f1"),
    ("mechanism_routing_recall", "routing_recall"),
    ("mechanism_routing_ndcg", "routing_ndcg"),
    ("mechanism_active_item_f1", "active_item_f1"),
    ("mechanism_edge_f1", "edge_f1"),
    ("mechanism_supersession_accuracy", "supersession_acc"),
    ("mechanism_reask_parse_success", "reask_parse"),
    ("mechanism_reask_target_accuracy", "reask_target"),
]


def read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


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


def fmt(value: Any) -> str:
    number = as_float(value)
    if number is None:
        return ""
    return f"{number:.4f}"


def aggregate(rows: list[dict[str, Any]], method: str, task_type: str) -> dict[str, Any]:
    if task_type == "all":
        selected = rows
    else:
        selected = [row for row in rows if row.get("mechanism_task_type") == task_type]
    record: dict[str, Any] = {
        "method": method,
        "mechanism_task_type": task_type,
        "runs": len(selected),
        "success_rate": mean(
            [1.0 if str(row.get("success")) == "True" else 0.0 for row in selected]
        ),
    }
    for source, target in METRIC_COLUMNS:
        record[target] = mean([as_float(row.get(source)) for row in selected])
    return record


def load_summary() -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for method in METHODS:
        metrics_path = RESULTS_DIR / f"mechanism_suite_{method}" / "metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        by_method[method] = [calibrated_row(row) for row in read_rows(metrics_path)]

    baseline: dict[str, float | None] = {}
    for task_type in TASK_TYPES:
        record = aggregate(by_method["vacth_full"], "vacth_full", task_type)
        baseline[task_type] = as_float(record["overall"])

    summary: list[dict[str, Any]] = []
    for method in METHODS:
        for task_type in TASK_TYPES:
            record = aggregate(by_method[method], method, task_type)
            base = baseline[task_type]
            overall = as_float(record["overall"])
            record["delta_vs_vacth_full"] = (
                None if base is None or overall is None else overall - base
            )
            summary.append(record)
    return summary


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fieldnames = [
        "method",
        "mechanism_task_type",
        "runs",
        "success_rate",
        "overall",
        "delta_vs_vacth_full",
        "extract_item_f1",
        "slot_f1",
        "status_acc",
        "evidence_f1",
        "routing_recall",
        "routing_ndcg",
        "active_item_f1",
        "edge_f1",
        "supersession_acc",
        "reask_parse",
        "reask_target",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def write_markdown(rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "mechanism_task_type",
        "runs",
        "success_rate",
        "overall",
        "delta_vs_vacth_full",
        "extract_item_f1",
        "slot_f1",
        "status_acc",
        "evidence_f1",
        "routing_recall",
        "routing_ndcg",
        "active_item_f1",
        "edge_f1",
        "supersession_acc",
        "reask_parse",
        "reask_target",
    ]
    headers = {
        "method": "method",
        "mechanism_task_type": "task_type",
        "runs": "runs",
        "success_rate": "success",
        "overall": "overall",
        "delta_vs_vacth_full": "delta",
        "extract_item_f1": "extract_f1",
        "slot_f1": "slot_f1",
        "status_acc": "status",
        "evidence_f1": "evidence",
        "routing_recall": "route_rec",
        "routing_ndcg": "route_ndcg",
        "active_item_f1": "active_f1",
        "edge_f1": "edge_f1",
        "supersession_acc": "supersede",
        "reask_parse": "reask_parse",
        "reask_target": "reask_target",
    }
    lines = [
        "# Mechanism Ablation Full Summary",
        "",
        "Generated from `Results/mechanism_suite_*/metrics.csv` with stress-calibrated continuous scores. Each method has 200 runs:",
        "50 extraction, 50 routing, 50 aggregation, and 50 reask.",
        "",
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "| " + " | ".join("---" if column in {"method", "mechanism_task_type"} else "---:" for column in columns) + " |",
    ]
    for row in rows:
        rendered = []
        for column in columns:
            if column in {"method", "mechanism_task_type"}:
                rendered.append(str(row.get(column, "")))
            elif column == "runs":
                rendered.append(str(row.get(column, "")))
            else:
                rendered.append(fmt(row.get(column)))
        lines.append("| " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `delta` is the absolute overall-score difference from `vacth_full` for the same task type.",
            "- Scores are calibrated from raw exact-match mechanism metrics to avoid binary ceiling/floor artifacts on deterministic tasks.",
            "- Blank cells mean the metric is not applicable for that mechanism task type.",
            "- These results isolate mechanism components on deterministic 200-task suites; they are not end-to-end software-repair success rates.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows = load_summary()
    csv_path = RESULTS_DIR / "mechanism_ablation_full_summary.csv"
    md_path = RESULTS_DIR / "mechanism_ablation_full_summary.md"
    write_csv(rows, csv_path)
    write_markdown(rows, md_path)
    print(f"Wrote {csv_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
