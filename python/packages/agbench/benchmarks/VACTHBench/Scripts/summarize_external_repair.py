import argparse
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


def read_rows(benchmark: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        root = RESULTS_DIR / f"external_{benchmark}_{method}"
        if not root.is_dir():
            continue
        for result_file in sorted(root.rglob("result.json")):
            if "__pycache__" in result_file.parts:
                continue
            with result_file.open("r", encoding="utf-8") as fh:
                result = json.load(fh)
            metrics = result.get("metrics", {})
            config = result.get("experiment_config", {})
            software_task = config.get("software_task", {})

            # SWE-bench evaluation fields
            swebench_resolved = bool_value(metrics.get("swebench_resolved"))
            swebench_outcome = str(metrics.get("swebench_outcome", ""))

            # Docker evaluation (set by run_swebench_docker_eval.py)
            docker_eval = result.get("docker_eval", {})
            docker_resolved = (
                bool_value(docker_eval.get("resolved"))
                if docker_eval
                else None
            )
            docker_outcome = str(docker_eval.get("outcome", "")) if docker_eval else ""

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
                    "invalid_patch_count": as_float(
                        metrics.get("invalid_patch_count")
                    ),
                    "changed_test_file_count": as_float(
                        metrics.get("changed_test_file_count")
                    ),
                    "reviewer_resolved": bool_value(
                        metrics.get("reviewer_resolved")
                    ),
                    # SWE-bench metrics
                    "swebench_resolved": swebench_resolved,
                    "swebench_outcome": swebench_outcome,
                    "swebench_f2p_pass": as_float(metrics.get("swebench_f2p_pass")),
                    "swebench_f2p_total": as_float(metrics.get("swebench_f2p_total")),
                    "swebench_p2p_pass": as_float(metrics.get("swebench_p2p_pass")),
                    "swebench_p2p_total": as_float(metrics.get("swebench_p2p_total")),
                    # Docker evaluation
                    "docker_resolved": docker_resolved,
                    "docker_outcome": docker_outcome,
                    "metadata": software_task.get("benchmark_metadata", {}),
                }
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    vacth_success = mean(
        [row["success"] for row in rows if row["method"] == "vacth_full"]
    )
    vacth_swebench = mean(
        [row["swebench_resolved"] for row in rows if row["method"] == "vacth_full"]
    )
    summary: list[dict[str, Any]] = []
    for method in METHODS:
        selected = [row for row in rows if row["method"] == method]
        success_rate = mean([row["success"] for row in selected])
        swebench_resolve_rate = mean(
            [row["swebench_resolved"] for row in selected]
        )
        docker_resolve_rate = mean(
            [
                row["docker_resolved"]
                for row in selected
                if row["docker_resolved"] is not None
            ]
        )

        # Outcome distribution counts
        outcome_counts: dict[str, int] = {}
        for row in selected:
            outcome = str(row.get("swebench_outcome", ""))
            if outcome:
                outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1

        summary.append(
            {
                "method": method,
                "runs": len(selected),
                "success_rate": success_rate,
                "delta_vs_vacth_full": None
                if success_rate is None or vacth_success is None
                else success_rate - vacth_success,
                # SWE-bench evaluation
                "swebench_resolve_rate": swebench_resolve_rate,
                "swebench_delta_vs_vacth_full": None
                if swebench_resolve_rate is None or vacth_swebench is None
                else swebench_resolve_rate - vacth_swebench,
                "docker_resolve_rate": docker_resolve_rate
                if docker_resolve_rate != 0.0
                else None,
                "swebench_resolved": outcome_counts.get("resolved", 0),
                "swebench_breaking_resolved": outcome_counts.get(
                    "breaking_resolved", 0
                ),
                "swebench_partially_resolved": outcome_counts.get(
                    "partially_resolved", 0
                ),
                "swebench_work_in_progress": outcome_counts.get(
                    "work_in_progress", 0
                ),
                "swebench_no_op": outcome_counts.get("no_op", 0),
                "swebench_regression": outcome_counts.get("regression", 0),
                # Traditional metrics
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


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(benchmark: str, summary_rows: list[dict[str, Any]], path: Path) -> None:
    columns = [
        "method",
        "runs",
        "swebench_resolve_rate",
        "swebench_delta_vs_vacth_full",
        "docker_resolve_rate",
        "swebench_resolved",
        "swebench_breaking_resolved",
        "swebench_partially_resolved",
        "swebench_work_in_progress",
        "swebench_no_op",
        "swebench_regression",
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
    headers = {
        "method": "method",
        "runs": "runs",
        "swebench_resolve_rate": "SWE-resolve",
        "swebench_delta_vs_vacth_full": "SWE-delta",
        "docker_resolve_rate": "docker",
        "swebench_resolved": "resolved",
        "swebench_breaking_resolved": "break",
        "swebench_partially_resolved": "partial",
        "swebench_work_in_progress": "wip",
        "swebench_no_op": "noop",
        "swebench_regression": "regr",
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
    lines = [
        f"# External Repair Summary: {benchmark}",
        "",
        "| " + " | ".join(headers[column] for column in columns) + " |",
        "| "
        + " | ".join("---" if column == "method" else "---:" for column in columns)
        + " |",
    ]
    for row in summary_rows:
        rendered = []
        for column in columns:
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
            "Notes:",
            "",
            "- `SWE-resolve`: SWE-bench resolution rate (all F2P + P2P tests pass).",
            "- `SWE-delta`: relative to `vacth_full` SWE-bench resolution rate.",
            "- `docker`: resolution rate from Docker re-evaluation (if available).",
            "- Outcome columns: resolved / breaking / partial / wip / noop / regr.",
            "- `success` and `delta`: legacy metric (returncode + reviewer + constraints).",
            "- Blank rows indicate that method has not been run yet.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize external repair runs.")
    parser.add_argument("benchmark", help="Example: swebench_lite or bugsinpy")
    args = parser.parse_args()

    rows = read_rows(args.benchmark)
    if not rows:
        raise FileNotFoundError(
            f"No result.json files found under Results/external_{args.benchmark}_*"
        )
    summary_rows = summarize(rows)
    write_csv(summary_rows, RESULTS_DIR / f"external_{args.benchmark}_summary.csv")
    write_markdown(
        args.benchmark,
        summary_rows,
        RESULTS_DIR / f"external_{args.benchmark}_summary.md",
    )
    print(f"Wrote {RESULTS_DIR / f'external_{args.benchmark}_summary.csv'}")
    print(f"Wrote {RESULTS_DIR / f'external_{args.benchmark}_summary.md'}")


if __name__ == "__main__":
    main()
