"""Summarize aligned SWE-bench official results with VACTHBench run metrics."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"

# OpenAI public API rates for GPT-5.4 at the time this experiment was summarized.
# These are configurable because the local run used an OpenAI-compatible proxy.
DEFAULT_INPUT_PRICE_PER_1M = 2.50
DEFAULT_OUTPUT_PRICE_PER_1M = 15.00
LONG_CONTEXT_PROMPT_THRESHOLD = 272_000
AGENT_MODEL_NAMES = {"planner", "retriever", "coder", "tester", "reviewer"}


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def resolve_metadata_path(path_value: Any) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.exists():
        return path
    parts = path.parts
    if "VACTHBench" in parts:
        index = parts.index("VACTHBench")
        candidate = BENCHMARK_DIR.joinpath(*parts[index + 1 :])
        if candidate.exists():
            return candidate
    return path


def iter_summary_report_statuses(report: dict[str, Any]) -> list[tuple[str, str]]:
    statuses: list[tuple[str, str]] = []
    for status, key in (
        ("resolved", "resolved_ids"),
        ("unresolved", "unresolved_ids"),
        ("empty_patch", "empty_patch_ids"),
        ("error", "error_ids"),
        ("incomplete", "incomplete_ids"),
    ):
        for instance_id in report.get(key, []):
            statuses.append((status, str(instance_id)))
    known_ids = {instance_id for _, instance_id in statuses}
    for instance_id in report.get("completed_ids", []):
        instance_id = str(instance_id)
        if instance_id not in known_ids:
            statuses.append(("completed_unknown", instance_id))
    if statuses:
        return statuses
    submitted = report.get("submitted_ids", [])
    return [("unknown", str(instance_id)) for instance_id in submitted]


def per_instance_status(report: dict[str, Any]) -> tuple[str, str]:
    if not report:
        return "unknown", ""
    instance_id, details = next(iter(report.items()))
    if not isinstance(details, dict):
        return "unknown", str(instance_id)
    if bool(details.get("resolved")):
        return "resolved", str(instance_id)
    if bool(details.get("patch_is_None")) or details.get("patch_exists") is False:
        return "empty_patch", str(instance_id)
    if details.get("patch_successfully_applied") is False:
        return "error", str(instance_id)
    return "unresolved", str(instance_id)


def load_official_reports(reports_dir: Path, run_logs_dir: Path | None = None) -> dict[tuple[str, str], dict[str, Any]]:
    statuses: dict[tuple[str, str], dict[str, Any]] = {}
    for path in sorted(reports_dir.glob("*.json")):
        if path.name.startswith("._"):
            continue
        report = read_json(path)
        method = path.name.split(".", 1)[0]
        for status, instance_id in iter_summary_report_statuses(report):
            if not instance_id:
                continue
            statuses[(method, instance_id)] = {
                "official_status": status,
                "official_resolved": status == "resolved",
                "official_report": str(path),
            }
    if run_logs_dir is not None and run_logs_dir.is_dir():
        for path in sorted(run_logs_dir.glob("*/*/*/report.json")):
            if any(part.startswith("._") for part in path.parts):
                continue
            report = read_json(path)
            method = path.parent.parent.name
            status, instance_id = per_instance_status(report)
            if not instance_id:
                continue
            statuses[(method, instance_id)] = {
                "official_status": status,
                "official_resolved": status == "resolved",
                "official_report": str(path),
            }
    return statuses


def _empty_usage() -> dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "max_request_prompt_tokens": 0,
        "long_context_request_count": 0,
        "usage_message_count": 0,
        "estimated_aux_tokens": 0,
        "agent_model_tokens": 0,
        "aux_model_tokens": 0,
        "extractor_model_tokens": 0,
        "raw_message_model_tokens": 0,
        "corrected_total_tokens": 0,
        "token_source": "missing",
    }


def token_usage_from_records(records: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    prompt_tokens = 0
    completion_tokens = 0
    agent_model_tokens = 0
    aux_model_tokens = 0
    extractor_model_tokens = 0
    max_prompt = 0
    long_context = 0
    usage_count = 0
    for record in records:
        usage = record.get("models_usage") or record
        if not usage:
            continue
        prompt = int(usage.get("prompt_tokens") or 0)
        completion = int(usage.get("completion_tokens") or 0)
        if prompt == 0 and completion == 0:
            continue
        usage_count += 1
        prompt_tokens += prompt
        completion_tokens += completion
        max_prompt = max(max_prompt, prompt)
        if prompt > LONG_CONTEXT_PROMPT_THRESHOLD:
            long_context += 1
        total = prompt + completion
        agent = str(record.get("agent", ""))
        if agent in AGENT_MODEL_NAMES:
            agent_model_tokens += total
        elif source == "model_usage":
            aux_model_tokens += total
        if agent == "vacth_extractor":
            extractor_model_tokens += total
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "corrected_total_tokens": prompt_tokens + completion_tokens,
        "max_request_prompt_tokens": max_prompt,
        "long_context_request_count": long_context,
        "usage_message_count": usage_count,
        "estimated_aux_tokens": 0,
        "agent_model_tokens": agent_model_tokens,
        "aux_model_tokens": aux_model_tokens,
        "extractor_model_tokens": extractor_model_tokens,
        "raw_message_model_tokens": 0,
        "token_source": source,
    }


def token_usage_from_run(run_dir: Path, metrics: dict[str, Any]) -> dict[str, Any]:
    raw_messages_path = run_dir / "raw_messages.json"
    raw_usage = (
        token_usage_from_records(read_json(raw_messages_path), source="raw_messages")
        if raw_messages_path.is_file()
        else _empty_usage()
    )
    model_usage_path = run_dir / "model_usage.json"
    if model_usage_path.is_file():
        usage = token_usage_from_records(read_json(model_usage_path), source="model_usage")
        usage["raw_message_model_tokens"] = raw_usage["total_tokens"]
        return usage

    if not raw_messages_path.is_file():
        return {
            **_empty_usage(),
        }
    usage = raw_usage
    usage["raw_message_model_tokens"] = raw_usage["total_tokens"]
    metric_total = as_int(metrics.get("estimated_tokens"))
    if metric_total > int(usage["total_tokens"]):
        missing = metric_total - int(usage["total_tokens"])
        usage["prompt_tokens"] = int(usage["prompt_tokens"]) + missing
        usage["total_tokens"] = metric_total
        usage["corrected_total_tokens"] = metric_total
        usage["estimated_aux_tokens"] = missing
        usage["aux_model_tokens"] = missing
        usage["token_source"] = "raw_messages_plus_metrics_delta"
    return usage


def load_tool_counts(path: Path) -> dict[str, int]:
    counts = {
        "run_tests_count": 0,
        "run_tests_cached_count": 0,
        "edit_count": 0,
        "git_diff_count": 0,
        "tool_error_count": 0,
    }
    if not path.is_file():
        return counts
    for call in read_json(path):
        tool = str(call.get("tool", ""))
        result = call.get("result") or {}
        if tool == "run_tests":
            content = result.get("content")
            parsed_content = {}
            if isinstance(content, str):
                try:
                    parsed_content = json.loads(content)
                except json.JSONDecodeError:
                    parsed_content = {}
            if isinstance(parsed_content, dict) and parsed_content.get("cached"):
                counts["run_tests_cached_count"] += 1
            else:
                counts["run_tests_count"] += 1
        if tool in {"replace_text", "write_file", "edit_file", "apply_patch"}:
            counts["edit_count"] += 1
        if tool == "git_diff":
            counts["git_diff_count"] += 1
        if result.get("is_error"):
            counts["tool_error_count"] += 1
    return counts


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def rows_from_metadata(
    *,
    metadata_path: Path,
    official: dict[tuple[str, str], dict[str, Any]],
    input_price_per_1m: float,
    output_price_per_1m: float,
) -> list[dict[str, Any]]:
    metadata = read_json(metadata_path)
    rows: list[dict[str, Any]] = []
    for item in metadata["runs"]:
        method = item["method"]
        instance_id = item["instance_id"]
        run_dir = resolve_metadata_path(item["run_dir"])
        result_path = resolve_metadata_path(item["result_path"]) if item.get("result_path") else None
        result = read_json(result_path) if result_path and result_path.is_file() else None
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}

        usage = token_usage_from_run(run_dir, metrics)

        tool_counts = load_tool_counts(run_dir / "tool_calls.json")
        status = official.get(
            (method, instance_id),
            {
                "official_status": "missing_report",
                "official_resolved": False,
                "official_report": "",
            },
        )
        official_resolved = bool(status["official_resolved"])
        has_result = bool(item.get("has_result"))
        agent_error = (
            ((not has_result) and str(item.get("status", "")).startswith("agent_error"))
            or bool(metrics.get("agent_error"))
        )
        incomplete_run_patch = str(item.get("status", "")) == "patch_from_incomplete_run"
        local_positive = bool(
            metrics.get("success")
            or metrics.get("swebench_resolved")
            or metrics.get("reviewer_resolved")
        )
        cascade_error_proxy = bool(local_positive and not official_resolved)

        cost_usd = (
            usage["prompt_tokens"] * input_price_per_1m
            + usage["completion_tokens"] * output_price_per_1m
        ) / 1_000_000
        rows.append(
            {
                "method": method,
                "instance_id": instance_id,
                "official_status": status["official_status"],
                "official_resolved": int(official_resolved),
                "export_status": item.get("status", ""),
                "has_result": int(has_result),
                "agent_error": int(agent_error),
                "incomplete_run_patch": int(incomplete_run_patch),
                "patch_bytes": item.get("patch_bytes", 0),
                "prompt_tokens": usage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"],
                "total_tokens": usage["total_tokens"],
                "corrected_total_tokens": usage["corrected_total_tokens"],
                "agent_model_tokens": usage["agent_model_tokens"],
                "aux_model_tokens": usage["aux_model_tokens"],
                "extractor_model_tokens": usage["extractor_model_tokens"],
                "raw_message_model_tokens": usage["raw_message_model_tokens"],
                "usage_message_count": usage["usage_message_count"],
                "max_request_prompt_tokens": usage["max_request_prompt_tokens"],
                "long_context_request_count": usage["long_context_request_count"],
                "estimated_aux_tokens": usage["estimated_aux_tokens"],
                "token_source": usage["token_source"],
                "estimated_cost_usd_standard": round(cost_usd, 6),
                "turns": as_int(metrics.get("turns")),
                "tool_calls": as_int(metrics.get("tool_calls")),
                "run_tests_count": tool_counts["run_tests_count"],
                "edit_count": tool_counts["edit_count"],
                "git_diff_count": tool_counts["git_diff_count"],
                "tool_error_count": tool_counts["tool_error_count"],
                "rework_count": max(0, tool_counts["run_tests_count"] - 1),
                "reviewer_resolved": int(bool(metrics.get("reviewer_resolved"))),
                "local_swebench_resolved": int(bool(metrics.get("swebench_resolved"))),
                "local_swebench_outcome": str(metrics.get("swebench_outcome", "")),
                "local_success": int(bool(metrics.get("success"))),
                "local_official_false_positive": int(cascade_error_proxy),
                "cascade_error_proxy": int(cascade_error_proxy),
                "constraint_violations": as_int(metrics.get("constraint_violations")),
                "invalid_patch_count": as_int(metrics.get("invalid_patch_count")),
                "changed_file_count": as_int(metrics.get("changed_file_count")),
                "changed_test_file_count": as_int(metrics.get("changed_test_file_count")),
                "missing_expected_file_count": as_int(metrics.get("missing_expected_file_count")),
                "summary_count": as_int(metrics.get("summary_count")),
                "summary_tokens": as_int(metrics.get("summary_tokens")),
                "structured_update_count": as_int(metrics.get("structured_update_count")),
                "structured_parse_error_count": as_int(metrics.get("structured_parse_error_count")),
                "vacth_extraction_count": as_int(metrics.get("vacth_extraction_count")),
                "vacth_parse_error_count": as_int(metrics.get("vacth_parse_error_count")),
                "vacth_heuristic_extraction_count": as_int(metrics.get("vacth_heuristic_extraction_count")),
                "vacth_reask_count": as_int(metrics.get("vacth_reask_count")),
                "vacth_capsule_count": as_int(metrics.get("vacth_capsule_count")),
                "vacth_cve_routing_count": as_int(metrics.get("vacth_cve_routing_count")),
                "retrieval_count": as_int(metrics.get("retrieval_count")),
                "retrieved_tokens": as_int(metrics.get("retrieved_tokens")),
                "wall_time_sec": round(as_float(metrics.get("wall_time_sec")), 4),
                "run_dir": str(run_dir),
                "result_path": str(result_path) if result_path else "",
                "official_report": status.get("official_report", ""),
            }
        )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_method[row["method"]].append(row)

    summaries: list[dict[str, Any]] = []
    for method, selected in sorted(by_method.items()):
        resolved = sum(row["official_resolved"] for row in selected)
        total_tokens = sum(row["total_tokens"] for row in selected)
        corrected_total_tokens = sum(row["corrected_total_tokens"] for row in selected)
        total_cost = sum(row["estimated_cost_usd_standard"] for row in selected)
        known_token_runs = sum(1 for row in selected if row["total_tokens"] > 0)
        summary = {
            "method": method,
            "instances": len(selected),
            "resolved": resolved,
            "unresolved": sum(1 for row in selected if row["official_status"] == "unresolved"),
            "empty_patch": sum(1 for row in selected if row["official_status"] == "empty_patch"),
            "errors": sum(1 for row in selected if row["official_status"] == "error"),
            "missing_reports": sum(1 for row in selected if row["official_status"] == "missing_report"),
            "agent_error_runs": sum(row["agent_error"] for row in selected),
            "incomplete_run_patch_count": sum(row["incomplete_run_patch"] for row in selected),
            "known_token_runs": known_token_runs,
            "unknown_token_runs": len(selected) - known_token_runs,
            "prompt_tokens": sum(row["prompt_tokens"] for row in selected),
            "completion_tokens": sum(row["completion_tokens"] for row in selected),
            "total_tokens": total_tokens,
            "corrected_total_tokens": corrected_total_tokens,
            "agent_model_tokens": sum(row["agent_model_tokens"] for row in selected),
            "aux_model_tokens": sum(row["aux_model_tokens"] for row in selected),
            "extractor_model_tokens": sum(row["extractor_model_tokens"] for row in selected),
            "raw_message_model_tokens": sum(row["raw_message_model_tokens"] for row in selected),
            "estimated_aux_tokens": sum(row["estimated_aux_tokens"] for row in selected),
            "tokens_per_resolved": round(corrected_total_tokens / resolved, 2) if resolved else "",
            "tokens_per_resolved_known_only": round(corrected_total_tokens / resolved, 2) if resolved else "",
            "estimated_cost_usd_standard": round(total_cost, 6),
            "cost_per_resolved_usd_known_only": round(total_cost / resolved, 6) if resolved else "",
            "tool_calls": sum(row["tool_calls"] for row in selected),
            "run_tests_count": sum(row["run_tests_count"] for row in selected),
            "edit_count": sum(row["edit_count"] for row in selected),
            "rework_count": sum(row["rework_count"] for row in selected),
            "tool_error_count": sum(row["tool_error_count"] for row in selected),
            "cascade_error_proxy": sum(row["cascade_error_proxy"] for row in selected),
            "constraint_violations": sum(row["constraint_violations"] for row in selected),
            "invalid_patch_count": sum(row["invalid_patch_count"] for row in selected),
            "changed_test_file_count": sum(row["changed_test_file_count"] for row in selected),
            "summary_count": sum(row["summary_count"] for row in selected),
            "summary_tokens": sum(row["summary_tokens"] for row in selected),
            "structured_update_count": sum(row["structured_update_count"] for row in selected),
            "structured_parse_error_count": sum(row["structured_parse_error_count"] for row in selected),
            "vacth_extraction_count": sum(row["vacth_extraction_count"] for row in selected),
            "vacth_parse_error_count": sum(row["vacth_parse_error_count"] for row in selected),
            "vacth_heuristic_extraction_count": sum(row["vacth_heuristic_extraction_count"] for row in selected),
            "vacth_reask_count": sum(row["vacth_reask_count"] for row in selected),
            "vacth_capsule_count": sum(row["vacth_capsule_count"] for row in selected),
            "vacth_cve_routing_count": sum(row["vacth_cve_routing_count"] for row in selected),
            "retrieval_count": sum(row["retrieval_count"] for row in selected),
            "retrieved_tokens": sum(row["retrieved_tokens"] for row in selected),
            "long_context_request_count": sum(row["long_context_request_count"] for row in selected),
            "wall_time_sec": round(sum(row["wall_time_sec"] for row in selected), 4),
        }
        summaries.append(summary)
    return summaries


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, summaries: list[dict[str, Any]]) -> None:
    headers = [
        "method",
        "instances",
        "resolved",
        "unresolved",
        "empty_patch",
        "agent_error_runs",
        "incomplete_run_patch_count",
        "unknown_token_runs",
        "total_tokens",
        "corrected_total_tokens",
        "agent_model_tokens",
        "aux_model_tokens",
        "extractor_model_tokens",
        "estimated_aux_tokens",
        "tokens_per_resolved",
        "tokens_per_resolved_known_only",
        "estimated_cost_usd_standard",
        "cost_per_resolved_usd_known_only",
        "rework_count",
        "cascade_error_proxy",
    ]
    lines = [
        "# Aligned SWE-bench Extended Metrics",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] + ["---:"] * (len(headers) - 1)) + " |",
    ]
    for row in summaries:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    totals = {header: "" for header in headers}
    totals["method"] = "**TOTAL**"
    for header in headers[1:]:
        values = [row.get(header) for row in summaries if isinstance(row.get(header), (int, float))]
        if values:
            totals[header] = round(sum(values), 6)
    total_resolved = sum(row["resolved"] for row in summaries)
    total_tokens = sum(row["corrected_total_tokens"] for row in summaries)
    total_cost = sum(row["estimated_cost_usd_standard"] for row in summaries)
    totals["tokens_per_resolved"] = (
        round(total_tokens / total_resolved, 2) if total_resolved else ""
    )
    totals["tokens_per_resolved_known_only"] = (
        round(total_tokens / total_resolved, 2) if total_resolved else ""
    )
    totals["cost_per_resolved_usd_known_only"] = (
        round(total_cost / total_resolved, 6) if total_resolved else ""
    )
    lines.append("| " + " | ".join(str(totals.get(header, "")) for header in headers) + " |")
    lines.extend(
        [
            "",
            "Definitions:",
            "",
            "- `total_tokens` uses `model_usage.json` when available; for legacy "
            "runs it sums `raw_messages.json` and adds the metrics-level "
            "auxiliary-token delta when present.",
            "- `estimated_aux_tokens` is the legacy fallback delta for LLM calls "
            "that were not saved in `raw_messages.json`.",
            "- `estimated_cost_usd_standard` uses configurable per-1M prompt/output "
            "prices and excludes cached-input discounts and long-context surcharges.",
            "- `rework_count = max(run_tests_count - 1, 0)` per instance.",
            "- `cascade_error_proxy` counts local positive outcomes (`success`, "
            "`reviewer_resolved`, or local `swebench_resolved`) that official "
            "SWE-bench did not resolve.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        default=str(RESULTS_DIR / "swebench_predictions_aligned6" / "summary.json"),
    )
    parser.add_argument(
        "--reports-dir",
        default=str(RESULTS_DIR / "swebench_official_eval" / "aligned6" / "reports"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "swebench_official_eval" / "aligned6"),
    )
    parser.add_argument(
        "--run-logs-dir",
        default="",
        help="Optional SWE-bench run_logs directory; per-instance report.json files override summary reports.",
    )
    parser.add_argument("--input-price-per-1m", type=float, default=DEFAULT_INPUT_PRICE_PER_1M)
    parser.add_argument("--output-price-per-1m", type=float, default=DEFAULT_OUTPUT_PRICE_PER_1M)
    args = parser.parse_args()

    metadata_path = Path(args.metadata).expanduser().resolve()
    reports_dir = Path(args.reports_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    run_logs_dir = Path(args.run_logs_dir).expanduser().resolve() if args.run_logs_dir else None

    official = load_official_reports(reports_dir, run_logs_dir=run_logs_dir)
    instance_rows = rows_from_metadata(
        metadata_path=metadata_path,
        official=official,
        input_price_per_1m=args.input_price_per_1m,
        output_price_per_1m=args.output_price_per_1m,
    )
    summary_rows = summarize(instance_rows)

    write_csv(output_dir / "extended_instance_metrics.csv", instance_rows)
    write_csv(output_dir / "extended_summary_metrics.csv", summary_rows)
    write_markdown(output_dir / "extended_summary_metrics.md", summary_rows)
    (output_dir / "extended_summary_metrics.json").write_text(
        json.dumps(
            {
                "input_price_per_1m": args.input_price_per_1m,
                "output_price_per_1m": args.output_price_per_1m,
                "notes": {
                    "cost": (
                        "Standard estimate from known prompt/completion tokens; "
                        "crashed runs may have unknown token usage."
                    ),
                    "rework_count": "max(run_tests_count - 1, 0) per instance.",
                    "cascade_error_proxy": (
                        "Local positive outcome but official SWE-bench unresolved/"
                        "empty/error."
                    ),
                },
                "methods": summary_rows,
                "instances": instance_rows,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary_rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
