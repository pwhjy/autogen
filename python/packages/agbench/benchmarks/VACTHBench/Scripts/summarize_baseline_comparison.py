import csv
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
SOFTWARE_METHODS = [
    ("E1", "autogen_broadcast"),
    ("E2", "summary"),
    ("E3", "sliding_window"),
    ("E4", "vector_memory"),
    ("E5", "structured_summary"),
    ("E6", "vacth_full"),
]
MECHANISM_METHODS = [
    "summary",
    "structured_summary",
    "vacth_full",
]
MECHANISM_TASK_TYPES = ["all", "extraction", "routing", "aggregation", "reask"]


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def software_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for unit, method in SOFTWARE_METHODS:
        metrics_path = RESULTS_DIR / f"software_repair_{method}" / "metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        metrics_rows = read_csv_rows(metrics_path)
        if len(metrics_rows) != 1:
            raise ValueError(f"Expected one smoke row in {metrics_path}")
        row = metrics_rows[0]
        rows.append(
            {
                "unit": unit,
                "method": method,
                "runs": 1,
                "success": 1.0 if str(row.get("success")) == "True" else 0.0,
                "turns": as_float(row.get("turns")),
                "tool_calls": as_float(row.get("tool_calls")),
                "estimated_tokens": as_float(row.get("estimated_tokens")),
                "wall_time_sec": as_float(row.get("wall_time_sec")),
                "summary_count": as_float(row.get("summary_count")),
                "structured_update_count": as_float(
                    row.get("structured_update_count")
                ),
                "memory_item_count": as_float(row.get("memory_item_count")),
                "retrieval_count": as_float(row.get("retrieval_count")),
                "vacth_extraction_count": as_float(
                    row.get("vacth_extraction_count")
                ),
                "vacth_capsule_count": as_float(row.get("vacth_capsule_count")),
                "vacth_selected_item_count": as_float(
                    row.get("vacth_selected_item_count")
                ),
                "constraint_violations": as_float(row.get("constraint_violations")),
                "invalid_patch_count": as_float(row.get("invalid_patch_count")),
            }
        )
    return rows


def mechanism_rows() -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for method in MECHANISM_METHODS:
        metrics_path = RESULTS_DIR / f"mechanism_suite_{method}" / "metrics.csv"
        if not metrics_path.is_file():
            raise FileNotFoundError(metrics_path)
        by_method[method] = read_csv_rows(metrics_path)

    baseline: dict[str, float | None] = {}
    for task_type in MECHANISM_TASK_TYPES:
        selected = (
            by_method["vacth_full"]
            if task_type == "all"
            else [
                row
                for row in by_method["vacth_full"]
                if row.get("mechanism_task_type") == task_type
            ]
        )
        baseline[task_type] = mean(
            [as_float(row.get("mechanism_overall_score")) for row in selected]
        )

    rows: list[dict[str, Any]] = []
    for method in MECHANISM_METHODS:
        for task_type in MECHANISM_TASK_TYPES:
            selected = (
                by_method[method]
                if task_type == "all"
                else [
                    row
                    for row in by_method[method]
                    if row.get("mechanism_task_type") == task_type
                ]
            )
            overall = mean(
                [as_float(row.get("mechanism_overall_score")) for row in selected]
            )
            base = baseline[task_type]
            rows.append(
                {
                    "method": method,
                    "task_type": task_type,
                    "runs": len(selected),
                    "success": mean(
                        [
                            1.0 if str(row.get("success")) == "True" else 0.0
                            for row in selected
                        ]
                    ),
                    "overall": overall,
                    "delta_vs_vacth_full": (
                        None if overall is None or base is None else overall - base
                    ),
                    "extraction_item_f1": mean(
                        [
                            as_float(row.get("mechanism_extraction_item_f1"))
                            for row in selected
                        ]
                    ),
                    "routing_recall": mean(
                        [
                            as_float(row.get("mechanism_routing_recall"))
                            for row in selected
                        ]
                    ),
                    "routing_ndcg": mean(
                        [
                            as_float(row.get("mechanism_routing_ndcg"))
                            for row in selected
                        ]
                    ),
                    "active_item_f1": mean(
                        [
                            as_float(row.get("mechanism_active_item_f1"))
                            for row in selected
                        ]
                    ),
                    "edge_f1": mean(
                        [
                            as_float(row.get("mechanism_edge_f1"))
                            for row in selected
                        ]
                    ),
                    "reask_parse_success": mean(
                        [
                            as_float(row.get("mechanism_reask_parse_success"))
                            for row in selected
                        ]
                    ),
                    "reask_target_accuracy": mean(
                        [
                            as_float(row.get("mechanism_reask_target_accuracy"))
                            for row in selected
                        ]
                    ),
                }
            )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    software: list[dict[str, Any]], mechanism: list[dict[str, Any]], path: Path
) -> None:
    software_columns = [
        "unit",
        "method",
        "runs",
        "success",
        "turns",
        "tool_calls",
        "estimated_tokens",
        "summary_count",
        "structured_update_count",
        "memory_item_count",
        "retrieval_count",
        "vacth_extraction_count",
        "vacth_capsule_count",
        "constraint_violations",
        "invalid_patch_count",
    ]
    software_headers = {
        "unit": "unit",
        "method": "method",
        "runs": "runs",
        "success": "success",
        "turns": "turns",
        "tool_calls": "tools",
        "estimated_tokens": "tokens",
        "summary_count": "summaries",
        "structured_update_count": "structured",
        "memory_item_count": "memory",
        "retrieval_count": "retrievals",
        "vacth_extraction_count": "vacth_extract",
        "vacth_capsule_count": "capsules",
        "constraint_violations": "violations",
        "invalid_patch_count": "bad_patch",
    }
    mechanism_columns = [
        "method",
        "task_type",
        "runs",
        "success",
        "overall",
        "delta_vs_vacth_full",
        "extraction_item_f1",
        "routing_recall",
        "routing_ndcg",
        "active_item_f1",
        "edge_f1",
        "reask_parse_success",
        "reask_target_accuracy",
    ]
    mechanism_headers = {
        "method": "method",
        "task_type": "task_type",
        "runs": "runs",
        "success": "success",
        "overall": "overall",
        "delta_vs_vacth_full": "delta",
        "extraction_item_f1": "extract_f1",
        "routing_recall": "route_rec",
        "routing_ndcg": "route_ndcg",
        "active_item_f1": "active_f1",
        "edge_f1": "edge_f1",
        "reask_parse_success": "reask_parse",
        "reask_target_accuracy": "reask_target",
    }

    lines = [
        "# Baseline Comparison Summary",
        "",
        "This file separates two comparison levels:",
        "",
        "1. software-repair smoke comparison for E1-E6 baselines",
        "2. 200-task mechanism-suite comparison for communication mechanisms",
        "",
        "The software-repair table is a one-task sanity comparison, not the SWE/BugsInPy",
        "main experiment. It is useful for checking that every communication protocol can",
        "complete the same calculator repair task under the shared AutoGen harness.",
        "",
        "## Software Repair Smoke",
        "",
        "| "
        + " | ".join(software_headers[column] for column in software_columns)
        + " |",
        "| "
        + " | ".join(
            "---" if column in {"unit", "method"} else "---:"
            for column in software_columns
        )
        + " |",
    ]
    for row in software:
        rendered = []
        for column in software_columns:
            if column in {"unit", "method"}:
                rendered.append(str(row.get(column, "")))
            elif column == "runs":
                rendered.append(str(row.get(column, "")))
            else:
                rendered.append(fmt(row.get(column)))
        lines.append("| " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "## Mechanism Suite",
            "",
            "| "
            + " | ".join(mechanism_headers[column] for column in mechanism_columns)
            + " |",
            "| "
            + " | ".join(
                "---" if column in {"method", "task_type"} else "---:"
                for column in mechanism_columns
            )
            + " |",
        ]
    )
    for row in mechanism:
        rendered = []
        for column in mechanism_columns:
            if column in {"method", "task_type"}:
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
            "- `delta` is relative to `vacth_full` within the same mechanism task type.",
            "- Blank cells mean the method does not use that intermediate artifact.",
            "- The software smoke has one run per method, so it should not be used as a",
            "  statistical claim about end-to-end repair quality.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    software = software_rows()
    mechanism = mechanism_rows()

    write_csv(software, RESULTS_DIR / "software_repair_baseline_comparison.csv")
    write_csv(mechanism, RESULTS_DIR / "mechanism_baseline_comparison.csv")
    write_markdown(
        software,
        mechanism,
        RESULTS_DIR / "baseline_comparison_summary.md",
    )
    print(f"Wrote {RESULTS_DIR / 'software_repair_baseline_comparison.csv'}")
    print(f"Wrote {RESULTS_DIR / 'mechanism_baseline_comparison.csv'}")
    print(f"Wrote {RESULTS_DIR / 'baseline_comparison_summary.md'}")


if __name__ == "__main__":
    main()
