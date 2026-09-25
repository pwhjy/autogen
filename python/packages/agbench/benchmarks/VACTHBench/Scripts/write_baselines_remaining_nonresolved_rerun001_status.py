#!/usr/bin/env python3
"""Write status for the remaining baseline non-resolved rerun."""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any


RUN_NAME = "swebench_lite_full_baselines_remaining_nonresolved_rerun001"
BENCHMARK_PREFIX = f"external_{RUN_NAME}"
METHODS = ("summary", "vector_memory", "autogen_broadcast", "sliding_window", "structured_summary", "vacth_full")
ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "Results"
MANIFEST_PATH = RESULTS_ROOT / f"{RUN_NAME}_manifest.json"
LOCAL_ROOT = RESULTS_ROOT / RUN_NAME
PREDICTION_DIR = RESULTS_ROOT / f"swebench_predictions_{RUN_NAME}"
OFFICIAL_ROOT = RESULTS_ROOT / "swebench_official_eval" / RUN_NAME
STATUS_PATH = RESULTS_ROOT / f"{RUN_NAME}_status.md"


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain an object")
    return value


def manifest() -> dict[str, Any]:
    return read_json(MANIFEST_PATH)


def method_info(method: str) -> dict[str, Any]:
    methods = manifest().get("methods", {})
    if not isinstance(methods, dict):
        raise TypeError("manifest methods must be an object")
    info = methods.get(method, {})
    if not isinstance(info, dict):
        raise TypeError(f"manifest method info for {method} must be an object")
    return info


def selected_instances(method: str) -> list[str]:
    value = method_info(method).get("selected_instance_ids", [])
    if not isinstance(value, list):
        raise TypeError(f"selected_instance_ids for {method} must be a list")
    return [str(item) for item in value]


def expected_for_method(method: str) -> int:
    return int(method_info(method).get("expected_tasks", 0))


def result_path(method: str, instance_id: str) -> Path:
    return (
        LOCAL_ROOT
        / f"{BENCHMARK_PREFIX}_{method}"
        / f"{BENCHMARK_PREFIX}__{instance_id}_{method}"
        / "0"
        / "result.json"
    )


def local_results(method: str) -> list[Path]:
    return [path for instance_id in selected_instances(method) if (path := result_path(method, instance_id)).is_file()]


def result_outcomes(method: str) -> str:
    counts: Counter[str] = Counter()
    for path in local_results(method):
        try:
            data = read_json(path)
        except Exception as exc:  # noqa: BLE001 - status should be robust to partial files.
            counts[f"bad_json:{type(exc).__name__}"] += 1
            continue
        counts[str(data.get("final_answer") or data.get("status") or "has_result")] += 1
    return ", ".join(f"{key}: {value}" for key, value in sorted(counts.items()))


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for line in fh if line.strip())


def report_count(report_dir: Path, method: str) -> int:
    if not report_dir.is_dir():
        return 0
    return sum(1 for path in report_dir.glob(f"{method}.*.json") if not path.name.startswith("._"))


def process_count(pattern: str) -> int:
    try:
        completed = subprocess.run(["ps", "axo", "command="], check=False, capture_output=True, text=True)
    except OSError:
        return 0
    return sum(1 for line in completed.stdout.splitlines() if pattern in line and "write_baselines_remaining" not in line)


def local_run_baseline_count() -> int:
    try:
        completed = subprocess.run(["ps", "axo", "command="], check=False, capture_output=True, text=True)
    except OSError:
        return 0
    count = 0
    for line in completed.stdout.splitlines():
        if "run_baseline.py" not in line or RUN_NAME not in line:
            continue
        if "uv run" in line or "write_baselines_remaining" in line or "/bin/zsh -lc" in line:
            continue
        count += 1
    return count


def inferred_phase(data: dict[str, Any]) -> str:
    active_local = local_run_baseline_count()
    active_official = process_count("run_swebench_official_patched.py")
    total_expected = 0
    total_results = 0
    total_predictions = 0
    corrected_methods = 0
    method_count = 0
    for method in METHODS:
        if method not in data.get("methods", {}):
            continue
        method_count += 1
        expected = expected_for_method(method)
        total_expected += expected
        total_results += len(local_results(method))
        total_predictions += line_count(PREDICTION_DIR / f"{method}.jsonl")
        if report_count(OFFICIAL_ROOT / "reports_corrected", method) > 0:
            corrected_methods += 1

    if (RESULTS_ROOT / f"{RUN_NAME}_flip_analysis.json").is_file():
        return "complete"
    if method_count and corrected_methods == method_count:
        return "official-eval-complete"
    if active_official:
        return "official-eval-running"
    if total_predictions:
        return "predictions-exported-or-official-eval-running"
    if total_expected and total_results == total_expected:
        return "local-complete"
    if active_local:
        return "local-running"
    if total_results:
        return "local-running-or-partial"
    return "prepared-not-started"


def main() -> int:
    data = manifest()
    total_expected = 0
    total_results = 0
    total_predictions = 0

    lines = [
        "# SWE-bench Lite Remaining Baselines Non-Resolved Rerun001 Status",
        "",
        f"Updated: {now()}",
        "",
        f"- phase: {inferred_phase(data)}",
        f"- selection: {data.get('selection', '')}",
        f"- total tasks: {data.get('total_tasks', 0)}",
        "- source status counts: "
        + ", ".join(f"{key}: {value}" for key, value in sorted(data.get("status_counts", {}).items())),
        f"- active local run_baseline jobs: {local_run_baseline_count()}",
        f"- active official eval jobs: {process_count('run_swebench_official_patched.py')}",
        "",
        "## Counts",
        "",
        "| method | expected | result.json | predictions | raw reports | corrected reports | source statuses | result outcomes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]

    for method in METHODS:
        if method not in data.get("methods", {}):
            continue
        expected = expected_for_method(method)
        results = len(local_results(method))
        predictions = line_count(PREDICTION_DIR / f"{method}.jsonl")
        raw = report_count(OFFICIAL_ROOT / "reports", method)
        corrected = report_count(OFFICIAL_ROOT / "reports_corrected", method)
        source_statuses = ", ".join(
            f"{key}: {value}" for key, value in sorted(method_info(method).get("source_status_counts", {}).items())
        )
        total_expected += expected
        total_results += results
        total_predictions += predictions
        lines.append(
            f"| {method} | {expected} | {results} | {predictions} | {raw} | {corrected} | "
            f"{source_statuses} | {result_outcomes(method)} |"
        )

    lines.extend(
        [
            f"| **TOTAL** | {total_expected} | {total_results} | {total_predictions} |  |  |  |  |",
            "",
            "## Paths",
            "",
            f"- manifest: `{MANIFEST_PATH.relative_to(ROOT)}`",
            f"- combined task file: `{data.get('combined_task_file', '')}`",
            f"- local results: `{LOCAL_ROOT.relative_to(ROOT)}/`",
            f"- predictions: `{PREDICTION_DIR.relative_to(ROOT)}/`",
            f"- official eval: `{OFFICIAL_ROOT.relative_to(ROOT)}/`",
        ]
    )

    if isinstance(data.get("rows"), list):
        lines.extend(
            [
                "",
                "## Remaining Rows",
                "",
                "| source official run | method | instance | source status | export status | patch bytes |",
                "| --- | --- | --- | --- | --- | ---: |",
            ]
        )
        for row in data["rows"]:
            lines.append(
                f"| {row['source_official_run']} | {row['method']} | {row['instance_id']} | "
                f"{row['source_official_status']} | {row['source_export_status']} | {row['source_patch_bytes']} |"
            )

    STATUS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {STATUS_PATH.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
