import csv
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"

METHODS = [
    "autogen_broadcast",
    "summary",
    "sliding_window",
    "vector_memory",
    "structured_summary",
    "vacth_full",
]


def read_result_rows(method: str) -> list[dict[str, Any]]:
    root = RESULTS_DIR / f"software_repair_suite_{method}"
    if not root.is_dir():
        return []
    rows: list[dict[str, Any]] = []
    for result_file in sorted(root.rglob("result.json")):
        if "__pycache__" in result_file.parts:
            continue
        with result_file.open("r", encoding="utf-8") as fh:
            result = json.load(fh)
        metrics = result.get("metrics", {})
        config = result.get("experiment_config", {})
        rows.append(
            {
                "method": result.get("method", method),
                "task_id": result.get("task_id"),
                "software_task_id": metrics.get(
                    "software_task_id",
                    config.get("software_task_id", ""),
                ),
                "software_task_category": metrics.get(
                    "software_task_category",
                    config.get("software_task_category", ""),
                ),
                "success": bool_value(metrics.get("success")),
                "turns": as_float(metrics.get("turns")),
                "tool_calls": as_float(metrics.get("tool_calls")),
                "estimated_tokens": as_float(metrics.get("estimated_tokens")),
                "wall_time_sec": as_float(metrics.get("wall_time_sec")),
                "constraint_violations": as_float(
                    metrics.get("constraint_violations")
                ),
                "invalid_patch_count": as_float(metrics.get("invalid_patch_count")),
                "changed_file_count": as_float(metrics.get("changed_file_count")),
                "changed_test_file_count": as_float(
                    metrics.get("changed_test_file_count")
                ),
                "missing_expected_file_count": as_float(
                    metrics.get("missing_expected_file_count")
                ),
                "reviewer_resolved": bool_value(metrics.get("reviewer_resolved")),
            }
        )
    return rows


def bool_value(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return 1.0 if value else 0.0
    return 1.0 if str(value).strip().lower() in {"true", "1", "yes"} else 0.0


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


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_method.setdefault(str(row["method"]), []).append(row)

    vacth_success = mean(
        [row["success"] for row in by_method.get("vacth_full", [])]
    )
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        selected = by_method.get(method, [])
        success_rate = mean([row["success"] for row in selected])
        summary.append(
            {
                "method": method,
                "runs": len(selected),
                "success_rate": success_rate,
                "delta_vs_vacth_full": None
                if success_rate is None or vacth_success is None
                else success_rate - vacth_success,
                "avg_turns": mean([row["turns"] for row in selected]),
                "avg_tool_calls": mean([row["tool_calls"] for row in selected]),
                "avg_estimated_tokens": mean(
                    [row["estimated_tokens"] for row in selected]
                ),
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
                "reviewer_resolved_rate": mean(
                    [row["reviewer_resolved"] for row in selected]
                ),
            }
        )
    return summary


def task_matrix(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = {}
    categories: dict[str, str] = {}
    for row in rows:
        task_id = str(row.get("software_task_id") or row.get("task_id"))
        method = str(row["method"])
        by_task.setdefault(task_id, {})[method] = row
        categories[task_id] = str(row.get("software_task_category", ""))

    matrix_rows: list[dict[str, Any]] = []
    for task_id in sorted(by_task):
        rendered = {
            "software_task_id": task_id,
            "category": categories.get(task_id, ""),
        }
        for method in METHODS:
            row = by_task[task_id].get(method)
            rendered[method] = "" if row is None else int(row["success"])
        matrix_rows.append(rendered)
    return matrix_rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    summary_rows: list[dict[str, Any]],
    matrix_rows: list[dict[str, Any]],
    path: Path,
) -> None:
    summary_columns = [
        "method",
        "runs",
        "success_rate",
        "delta_vs_vacth_full",
        "avg_turns",
        "avg_tool_calls",
        "avg_estimated_tokens",
        "avg_wall_time_sec",
        "avg_constraint_violations",
        "avg_invalid_patch_count",
        "avg_changed_test_file_count",
        "reviewer_resolved_rate",
    ]
    summary_headers = {
        "method": "method",
        "runs": "runs",
        "success_rate": "success",
        "delta_vs_vacth_full": "delta",
        "avg_turns": "turns",
        "avg_tool_calls": "tools",
        "avg_estimated_tokens": "tokens",
        "avg_wall_time_sec": "wall_s",
        "avg_constraint_violations": "violations",
        "avg_invalid_patch_count": "bad_patch",
        "avg_changed_test_file_count": "test_edits",
        "reviewer_resolved_rate": "reviewer_ok",
    }
    matrix_columns = ["software_task_id", "category", *METHODS]

    lines = [
        "# Software Repair Suite Summary",
        "",
        "Endpoint software-repair comparison across the same generated repair tasks.",
        "`success` requires tests to pass, reviewer `FINAL_ANSWER: resolved`, no test edits,",
        "and the expected source file to be touched.",
        "",
        "## Method Summary",
        "",
        "| "
        + " | ".join(summary_headers[column] for column in summary_columns)
        + " |",
        "| "
        + " | ".join(
            "---" if column == "method" else "---:" for column in summary_columns
        )
        + " |",
    ]
    for row in summary_rows:
        rendered = []
        for column in summary_columns:
            if column == "method":
                rendered.append(str(row.get(column, "")))
            elif column == "runs":
                rendered.append(str(row.get(column, "")))
            else:
                rendered.append(fmt(row.get(column)))
        lines.append("| " + " | ".join(rendered) + " |")

    lines.extend(
        [
            "",
            "## Task Matrix",
            "",
            "| " + " | ".join(matrix_columns) + " |",
            "| "
            + " | ".join(
                "---" if column in {"software_task_id", "category"} else "---:"
                for column in matrix_columns
            )
            + " |",
        ]
    )
    for row in matrix_rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in matrix_columns) + " |")

    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `delta` is relative to `vacth_full` success rate.",
            "- Blank matrix cells mean that method has not been run for that task yet.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        rows.extend(read_result_rows(method))
    if not rows:
        raise FileNotFoundError(
            "No software repair suite result.json files found under Results/"
        )

    summary_rows = summarize(rows)
    matrix_rows = task_matrix(rows)
    write_csv(summary_rows, RESULTS_DIR / "software_repair_suite_summary.csv")
    write_csv(matrix_rows, RESULTS_DIR / "software_repair_suite_task_matrix.csv")
    write_markdown(
        summary_rows,
        matrix_rows,
        RESULTS_DIR / "software_repair_suite_summary.md",
    )
    print(f"Wrote {RESULTS_DIR / 'software_repair_suite_summary.csv'}")
    print(f"Wrote {RESULTS_DIR / 'software_repair_suite_task_matrix.csv'}")
    print(f"Wrote {RESULTS_DIR / 'software_repair_suite_summary.md'}")


if __name__ == "__main__":
    main()
