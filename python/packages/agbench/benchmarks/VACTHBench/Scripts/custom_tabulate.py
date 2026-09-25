import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import tabulate as tb


EXCLUDE_DIR_NAMES = {"__pycache__"}


def _iter_result_files(runlogs: Path) -> list[Path]:
    result_files: list[Path] = []
    for task_dir in sorted(runlogs.iterdir(), key=lambda path: path.name):
        if task_dir.name in EXCLUDE_DIR_NAMES or not task_dir.is_dir():
            continue
        for instance_dir in sorted(task_dir.iterdir(), key=lambda path: path.name):
            if not instance_dir.is_dir():
                continue
            result_file = instance_dir / "result.json"
            if result_file.is_file():
                result_files.append(result_file)
    return result_files


def _flatten_result(result_file: Path) -> dict[str, Any]:
    with result_file.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    metrics = result.get("metrics", {})
    return {
        "task_id": result.get("task_id"),
        "method": result.get("method"),
        "trial": result_file.parent.name,
        "success": metrics.get("success"),
        "software_task_id": metrics.get("software_task_id"),
        "software_task_category": metrics.get("software_task_category"),
        "turns": metrics.get("turns"),
        "tool_calls": metrics.get("tool_calls"),
        "estimated_tokens": metrics.get("estimated_tokens"),
        "wall_time_sec": metrics.get("wall_time_sec"),
        "summary_count": metrics.get("summary_count"),
        "summary_tokens": metrics.get("summary_tokens"),
        "summary_compression_ratio": metrics.get("summary_compression_ratio"),
        "structured_update_count": metrics.get("structured_update_count"),
        "structured_state_item_count": metrics.get("structured_state_item_count"),
        "structured_tokens": metrics.get("structured_tokens"),
        "structured_parse_error_count": metrics.get("structured_parse_error_count"),
        "structured_schema_valid": metrics.get("structured_schema_valid"),
        "context_window_messages": metrics.get("context_window_messages"),
        "dropped_message_count": metrics.get("dropped_message_count"),
        "max_visible_context_messages": metrics.get("max_visible_context_messages"),
        "memory_item_count": metrics.get("memory_item_count"),
        "retrieval_count": metrics.get("retrieval_count"),
        "retrieval_top_k": metrics.get("retrieval_top_k"),
        "retrieved_tokens": metrics.get("retrieved_tokens"),
        "avg_retrieval_score": metrics.get("avg_retrieval_score"),
        "vacth_extraction_count": metrics.get("vacth_extraction_count"),
        "vacth_parse_error_count": metrics.get("vacth_parse_error_count"),
        "vacth_heuristic_extraction_count": metrics.get(
            "vacth_heuristic_extraction_count"
        ),
        "vacth_reask_count": metrics.get("vacth_reask_count"),
        "vacth_capsule_count": metrics.get("vacth_capsule_count"),
        "vacth_selected_item_count": metrics.get("vacth_selected_item_count"),
        "vacth_cve_routing_count": metrics.get("vacth_cve_routing_count"),
        "vacth_avg_capsule_tokens": metrics.get("vacth_avg_capsule_tokens"),
        "vacth_enable_thc": metrics.get("vacth_enable_thc"),
        "vacth_enable_cve": metrics.get("vacth_enable_cve"),
        "vacth_enable_paa": metrics.get("vacth_enable_paa"),
        "vacth_enable_provenance": metrics.get("vacth_enable_provenance"),
        "vacth_enable_reask": metrics.get("vacth_enable_reask"),
        "vacth_role_specific_routing": metrics.get("vacth_role_specific_routing"),
        "mechanism_task_type": metrics.get("mechanism_task_type"),
        "mechanism_sample_id": metrics.get("mechanism_sample_id"),
        "mechanism_overall_score": metrics.get("mechanism_overall_score"),
        "mechanism_extraction_item_f1": metrics.get("mechanism_extraction_item_f1"),
        "mechanism_slot_f1": metrics.get("mechanism_slot_f1"),
        "mechanism_status_accuracy": metrics.get("mechanism_status_accuracy"),
        "mechanism_evidence_f1": metrics.get("mechanism_evidence_f1"),
        "mechanism_routing_recall": metrics.get("mechanism_routing_recall"),
        "mechanism_routing_ndcg": metrics.get("mechanism_routing_ndcg"),
        "mechanism_budget_used_ratio": metrics.get("mechanism_budget_used_ratio"),
        "mechanism_active_item_f1": metrics.get("mechanism_active_item_f1"),
        "mechanism_edge_f1": metrics.get("mechanism_edge_f1"),
        "mechanism_supersession_accuracy": metrics.get(
            "mechanism_supersession_accuracy"
        ),
        "mechanism_reask_parse_success": metrics.get("mechanism_reask_parse_success"),
        "mechanism_reask_target_accuracy": metrics.get(
            "mechanism_reask_target_accuracy"
        ),
        "mechanism_reask_needed": metrics.get("mechanism_reask_needed"),
        "mechanism_reask_attempts": metrics.get("mechanism_reask_attempts"),
        "constraint_violations": metrics.get("constraint_violations"),
        "invalid_patch_count": metrics.get("invalid_patch_count"),
        "changed_file_count": metrics.get("changed_file_count"),
        "changed_test_file_count": metrics.get("changed_test_file_count"),
        "expected_changed_file_count": metrics.get("expected_changed_file_count"),
        "missing_expected_file_count": metrics.get("missing_expected_file_count"),
        "reviewer_resolved": metrics.get("reviewer_resolved"),
        "result_path": str(result_file),
    }


def _write_metrics_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main(args: list[str]) -> None:
    parser = argparse.ArgumentParser(
        prog=args[0], description="Tabulate VACTHBench result.json files."
    )
    parser.add_argument(
        "runlogs", help="Path to a VACTHBench Results/<scenario> directory."
    )
    parser.add_argument(
        "-c", "--csv", action="store_true", help="Print per-run rows as CSV."
    )
    parser.add_argument("-e", "--excel", help="Write per-run rows to an Excel file.")
    parsed = parser.parse_args(args[1:])

    runlogs = Path(parsed.runlogs)
    if not runlogs.is_dir():
        raise FileNotFoundError(f"Run log directory not found: {runlogs}")

    rows = [_flatten_result(path) for path in _iter_result_files(runlogs)]
    if not rows:
        print(f"No result.json files found under {runlogs}", file=sys.stderr)
        return

    df = pd.DataFrame(rows)
    metrics_csv = runlogs / "metrics.csv"
    _write_metrics_csv(rows, metrics_csv)

    if parsed.excel:
        df.to_excel(parsed.excel, index=False)

    if parsed.csv:
        print(df.to_csv(index=False))
    else:
        print(tb.tabulate(df, headers="keys", tablefmt="simple", showindex=False))

        summary = (
            df.groupby("method", dropna=False)
            .agg(
                runs=("task_id", "count"),
                success_rate=("success", "mean"),
                avg_turns=("turns", "mean"),
                avg_tool_calls=("tool_calls", "mean"),
                avg_estimated_tokens=("estimated_tokens", "mean"),
                avg_wall_time_sec=("wall_time_sec", "mean"),
                avg_summary_count=("summary_count", "mean"),
                avg_summary_tokens=("summary_tokens", "mean"),
                avg_summary_compression_ratio=("summary_compression_ratio", "mean"),
                avg_structured_update_count=("structured_update_count", "mean"),
                avg_structured_state_item_count=(
                    "structured_state_item_count",
                    "mean",
                ),
                avg_structured_tokens=("structured_tokens", "mean"),
                structured_schema_valid_rate=("structured_schema_valid", "mean"),
                avg_dropped_message_count=("dropped_message_count", "mean"),
                avg_max_visible_context_messages=(
                    "max_visible_context_messages",
                    "mean",
                ),
                avg_memory_item_count=("memory_item_count", "mean"),
                avg_retrieval_count=("retrieval_count", "mean"),
                avg_retrieved_tokens=("retrieved_tokens", "mean"),
                avg_retrieval_score=("avg_retrieval_score", "mean"),
                avg_vacth_extraction_count=("vacth_extraction_count", "mean"),
                avg_vacth_parse_error_count=("vacth_parse_error_count", "mean"),
                avg_vacth_heuristic_extraction_count=(
                    "vacth_heuristic_extraction_count",
                    "mean",
                ),
                avg_vacth_reask_count=("vacth_reask_count", "mean"),
                avg_vacth_capsule_count=("vacth_capsule_count", "mean"),
                avg_vacth_selected_item_count=("vacth_selected_item_count", "mean"),
                avg_vacth_cve_routing_count=("vacth_cve_routing_count", "mean"),
                avg_vacth_capsule_tokens=("vacth_avg_capsule_tokens", "mean"),
                thc_enabled_rate=("vacth_enable_thc", "mean"),
                cve_enabled_rate=("vacth_enable_cve", "mean"),
                paa_enabled_rate=("vacth_enable_paa", "mean"),
                provenance_enabled_rate=("vacth_enable_provenance", "mean"),
                reask_enabled_rate=("vacth_enable_reask", "mean"),
                role_specific_routing_rate=("vacth_role_specific_routing", "mean"),
                avg_mechanism_overall_score=("mechanism_overall_score", "mean"),
                avg_mechanism_extraction_item_f1=(
                    "mechanism_extraction_item_f1",
                    "mean",
                ),
                avg_mechanism_slot_f1=("mechanism_slot_f1", "mean"),
                avg_mechanism_status_accuracy=(
                    "mechanism_status_accuracy",
                    "mean",
                ),
                avg_mechanism_evidence_f1=("mechanism_evidence_f1", "mean"),
                avg_mechanism_routing_recall=("mechanism_routing_recall", "mean"),
                avg_mechanism_routing_ndcg=("mechanism_routing_ndcg", "mean"),
                avg_mechanism_active_item_f1=("mechanism_active_item_f1", "mean"),
                avg_mechanism_edge_f1=("mechanism_edge_f1", "mean"),
                avg_mechanism_supersession_accuracy=(
                    "mechanism_supersession_accuracy",
                    "mean",
                ),
                avg_mechanism_reask_parse_success=(
                    "mechanism_reask_parse_success",
                    "mean",
                ),
                avg_mechanism_reask_target_accuracy=(
                    "mechanism_reask_target_accuracy",
                    "mean",
                ),
                avg_changed_file_count=("changed_file_count", "mean"),
                avg_changed_test_file_count=("changed_test_file_count", "mean"),
                avg_missing_expected_file_count=(
                    "missing_expected_file_count",
                    "mean",
                ),
                reviewer_resolved_rate=("reviewer_resolved", "mean"),
            )
            .reset_index()
        )
        print("\nSummary Statistics\n")
        print(tb.tabulate(summary, headers="keys", tablefmt="simple", showindex=False))
        mechanism_df = df[df["mechanism_task_type"].notna()]
        if not mechanism_df.empty:
            mechanism_summary = (
                mechanism_df.groupby(["mechanism_task_type", "method"], dropna=False)
                .agg(
                    runs=("task_id", "count"),
                    success_rate=("success", "mean"),
                    overall=("mechanism_overall_score", "mean"),
                    extraction_item_f1=("mechanism_extraction_item_f1", "mean"),
                    slot_f1=("mechanism_slot_f1", "mean"),
                    status_accuracy=("mechanism_status_accuracy", "mean"),
                    evidence_f1=("mechanism_evidence_f1", "mean"),
                    routing_recall=("mechanism_routing_recall", "mean"),
                    routing_ndcg=("mechanism_routing_ndcg", "mean"),
                    active_item_f1=("mechanism_active_item_f1", "mean"),
                    edge_f1=("mechanism_edge_f1", "mean"),
                    supersession_accuracy=(
                        "mechanism_supersession_accuracy",
                        "mean",
                    ),
                    reask_parse_success=("mechanism_reask_parse_success", "mean"),
                    reask_target_accuracy=(
                        "mechanism_reask_target_accuracy",
                        "mean",
                    ),
                )
                .reset_index()
            )
            print("\nMechanism Scores\n")
            print(
                tb.tabulate(
                    mechanism_summary,
                    headers="keys",
                    tablefmt="simple",
                    showindex=False,
                )
            )
        print(f"\nWrote {metrics_csv}")


if __name__ == "__main__" and __package__ is None:
    main(sys.argv)
