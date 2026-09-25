from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"

CORE_METHODS = [
    "autogen_broadcast",
    "summary",
    "sliding_window",
    "vector_memory",
    "structured_summary",
    "vacth_full",
    "vacth_optimized",
]

ABLATION_METHODS = [
    "vacth_wo_cve",
    "vacth_wo_thc",
    "vacth_wo_paa",
    "vacth_wo_provenance",
    "vacth_wo_reask",
    "vacth_global_capsule",
]


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if value else 0.0
    return 1.0 if str(value).strip().lower() in {"true", "1", "yes", "resolved"} else 0.0


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


def read_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for result_file in sorted(root.rglob("result.json")):
        if "__pycache__" in result_file.parts:
            continue
        with result_file.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
        metrics = result.get("metrics", {}) or {}
        config = result.get("experiment_config", {}) or {}
        rows.append(
            {
                "result_file": str(result_file.relative_to(BENCHMARK_DIR)),
                "task_id": result.get("task_id", config.get("task_id", "")),
                "method": result.get("method", config.get("method", "")),
                "software_task_id": metrics.get(
                    "software_task_id",
                    config.get("software_task_id", ""),
                ),
                "software_task_category": metrics.get(
                    "software_task_category",
                    config.get("software_task_category", ""),
                ),
                "success": bool_value(metrics.get("success")),
                "reviewer_resolved": bool_value(metrics.get("reviewer_resolved")),
                "swebench_resolved": bool_value(metrics.get("swebench_resolved")),
                "turns": as_float(metrics.get("turns")),
                "tool_calls": as_float(metrics.get("tool_calls")),
                "estimated_tokens": as_float(metrics.get("estimated_tokens")),
                "corrected_total_tokens": as_float(metrics.get("corrected_total_tokens")),
                "wall_time_sec": as_float(metrics.get("wall_time_sec")),
                "constraint_violations": as_float(metrics.get("constraint_violations")),
                "invalid_patch_count": as_float(metrics.get("invalid_patch_count")),
                "changed_file_count": as_float(metrics.get("changed_file_count")),
                "changed_test_file_count": as_float(metrics.get("changed_test_file_count")),
                "missing_expected_file_count": as_float(
                    metrics.get("missing_expected_file_count")
                ),
                "vacth_capsule_count": as_float(metrics.get("vacth_capsule_count")),
                "vacth_avg_capsule_tokens": as_float(
                    metrics.get("vacth_avg_capsule_tokens")
                ),
                "vacth_reask_count": as_float(metrics.get("vacth_reask_count")),
                "state_items": as_float(metrics.get("state_items")),
                "provenance_edges": as_float(metrics.get("provenance_edges")),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    reference = mean([row["success"] for row in by_method.get("vacth_full", [])])
    output: list[dict[str, Any]] = []
    for method in methods:
        selected = by_method.get(method, [])
        success = mean([row["success"] for row in selected])
        output.append(
            {
                "method": method,
                "runs": len(selected),
                "success_rate": success,
                "delta_vs_vacth_full": None
                if success is None or reference is None
                else success - reference,
                "reviewer_resolved_rate": mean([row["reviewer_resolved"] for row in selected]),
                "avg_turns": mean([row["turns"] for row in selected]),
                "avg_tool_calls": mean([row["tool_calls"] for row in selected]),
                "avg_estimated_tokens": mean([row["estimated_tokens"] for row in selected]),
                "avg_wall_time_sec": mean([row["wall_time_sec"] for row in selected]),
                "avg_constraint_violations": mean(
                    [row["constraint_violations"] for row in selected]
                ),
                "avg_invalid_patch_count": mean(
                    [row["invalid_patch_count"] for row in selected]
                ),
                "avg_changed_test_file_count": mean(
                    [row["changed_test_file_count"] for row in selected]
                ),
                "avg_vacth_capsule_count": mean(
                    [row["vacth_capsule_count"] for row in selected]
                ),
                "avg_vacth_reask_count": mean(
                    [row["vacth_reask_count"] for row in selected]
                ),
            }
        )
    return output


def task_matrix(rows: list[dict[str, Any]], methods: list[str]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    categories: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("software_task_id") or row.get("task_id"))
        method = str(row["method"])
        if method not in methods:
            continue
        by_task.setdefault(task_id, {})[method] = row
        categories[task_id] = str(row.get("software_task_category", ""))

    output: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        item: dict[str, Any] = {
            "software_task_id": task_id,
            "category": categories.get(task_id, ""),
        }
        for method in methods:
            row = by_task[task_id].get(method)
            item[method] = "" if row is None else int(row["success"])
        output.append(item)
    return output


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], columns: list[str], numeric: set[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---:" if column in numeric else "---" for column in columns) + " |",
    ]
    for row in rows:
        rendered: list[str] = []
        for column in columns:
            value = row.get(column, "")
            if column in numeric and value != "":
                rendered.append(fmt(value))
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines


def write_markdown(
    *,
    root: Path,
    rows: list[dict[str, Any]],
    core_summary: list[dict[str, Any]],
    core_matrix: list[dict[str, Any]],
    ablation_summary: list[dict[str, Any]],
    ablation_matrix: list[dict[str, Any]],
    path: Path,
) -> None:
    summary_columns = [
        "method",
        "runs",
        "success_rate",
        "delta_vs_vacth_full",
        "reviewer_resolved_rate",
        "avg_turns",
        "avg_tool_calls",
        "avg_estimated_tokens",
        "avg_wall_time_sec",
        "avg_constraint_violations",
        "avg_invalid_patch_count",
        "avg_changed_test_file_count",
        "avg_vacth_capsule_count",
        "avg_vacth_reask_count",
    ]
    summary_numeric = set(summary_columns) - {"method", "runs"}
    core_matrix_columns = ["software_task_id", "category", *CORE_METHODS]
    ablation_matrix_columns = ["software_task_id", "category", *ABLATION_METHODS]
    matrix_numeric: set[str] = set()

    lines = [
        "# LLM Rerun Software Repair Summary",
        "",
        f"Result root: `{root.relative_to(BENCHMARK_DIR)}`.",
        f"Completed result files: {len(rows)}.",
        "",
        "Success requires test pass, reviewer resolved, no test edits, and expected source-file coverage.",
        "",
        "## Core Methods",
        "",
        *markdown_table(core_summary, summary_columns, summary_numeric),
        "",
        "## Core Task Matrix",
        "",
        *markdown_table(core_matrix, core_matrix_columns, matrix_numeric),
        "",
        "## VACTH Ablation Smoke",
        "",
        *markdown_table(ablation_summary, summary_columns, summary_numeric),
        "",
        "## Ablation Task Matrix",
        "",
        *markdown_table(ablation_matrix, ablation_matrix_columns, matrix_numeric),
        "",
        "Notes:",
        "",
        "- `delta_vs_vacth_full` is an absolute success-rate difference.",
        "- Blank matrix cells mean the result file was not present when this summary was generated.",
        "- The ablation block is a smoke check on one task per ablation in this rerun, not a full 8-task ablation grid.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LLM rerun result.json files.")
    parser.add_argument(
        "result_root",
        nargs="?",
        default=str(RESULTS_DIR / "llm_full_20260630_all"),
        help="Result directory containing agbench task subdirectories.",
    )
    parser.add_argument(
        "--prefix",
        default="llm_full_20260630",
        help="Output filename prefix under Results/.",
    )
    args = parser.parse_args()

    root = Path(args.result_root)
    if not root.is_absolute():
        root = BENCHMARK_DIR / root
    if not root.is_dir():
        raise FileNotFoundError(root)

    rows = read_rows(root)
    if not rows:
        raise FileNotFoundError(f"No result.json files found under {root}")

    core_summary = summarize(rows, CORE_METHODS)
    core_matrix = task_matrix(rows, CORE_METHODS)
    ablation_summary = summarize(rows, ABLATION_METHODS)
    ablation_matrix = task_matrix(rows, ABLATION_METHODS)

    write_csv(rows, RESULTS_DIR / f"{args.prefix}_raw_results.csv")
    write_csv(core_summary, RESULTS_DIR / f"{args.prefix}_core_summary.csv")
    write_csv(core_matrix, RESULTS_DIR / f"{args.prefix}_core_task_matrix.csv")
    write_csv(ablation_summary, RESULTS_DIR / f"{args.prefix}_ablation_summary.csv")
    write_csv(ablation_matrix, RESULTS_DIR / f"{args.prefix}_ablation_task_matrix.csv")
    write_markdown(
        root=root,
        rows=rows,
        core_summary=core_summary,
        core_matrix=core_matrix,
        ablation_summary=ablation_summary,
        ablation_matrix=ablation_matrix,
        path=RESULTS_DIR / f"{args.prefix}_software_repair_summary.md",
    )
    print(f"Summarized {len(rows)} result files from {root}")


if __name__ == "__main__":
    main()
