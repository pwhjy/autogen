"""Write live status for the baseline non-resolved rerun."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from pathlib import Path


RUN_NAME = "swebench_lite_full_baselines_nonresolved_rerun001"
METHODS = (
    "summary",
    "vector_memory",
    "autogen_broadcast",
    "sliding_window",
    "structured_summary",
)
RESULTS_ROOT = Path("Results")
MANIFEST_PATH = RESULTS_ROOT / f"{RUN_NAME}_manifest.json"
STATUS_PATH = RESULTS_ROOT / f"{RUN_NAME}_status.md"
LOCAL_ROOT = RESULTS_ROOT / RUN_NAME
PREDICTION_DIR = RESULTS_ROOT / f"swebench_predictions_{RUN_NAME}"
OFFICIAL_ROOT = RESULTS_ROOT / "swebench_official_eval" / RUN_NAME


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def shell_output(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except Exception as exc:  # noqa: BLE001 - status reporting should not fail hard.
        return str(exc)


def process_lines() -> list[str]:
    output = shell_output(["ps", "ax", "-o", "pid=,ppid=,etime=,%cpu=,%mem=,command="])
    needles = (
        RUN_NAME,
        "baselines-nonresolved-rerun",
        "run_swebench_official_patched",
        "export_swebench_predictions",
        "summarize_swebench",
    )
    return [line for line in output.splitlines() if any(needle in line for needle in needles)]


def tmux_lines() -> list[str]:
    output = shell_output(["tmux", "ls"])
    if "no server running" in output:
        return []
    return [line for line in output.splitlines() if line.strip()]


def report_count(report_dir: Path, method: str) -> int:
    if not report_dir.is_dir():
        return 0
    return sum(1 for path in report_dir.glob(f"{method}.*.json") if not path.name.startswith("._"))


def latest_event(run_dir: Path) -> tuple[float, str]:
    event_path = run_dir / "model_events.jsonl"
    if not event_path.is_file():
        return 0.0, ""
    mtime = event_path.stat().st_mtime
    latest = ""
    with event_path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.strip():
                latest = line.strip()
    return mtime, latest


def method_status(method: str, expected: int) -> dict[str, object]:
    root = LOCAL_ROOT / f"external_{RUN_NAME}_{method}"
    run_dirs = sorted(path for path in root.glob("*/0") if path.is_dir()) if root.is_dir() else []
    result_dirs = [path for path in run_dirs if (path / "result.json").is_file()]
    active_dirs = [path for path in run_dirs if not (path / "result.json").is_file()]
    latest_mtime = 0.0
    latest_dir = ""
    latest_line = ""
    outcome_counter: Counter[str] = Counter()

    for run_dir in run_dirs:
        result_path = run_dir / "result.json"
        if result_path.is_file():
            try:
                result = read_json(result_path)
                if isinstance(result, dict):
                    outcome = result.get("status") or result.get("final_answer") or "unknown"
                    outcome_counter[str(outcome)] += 1
            except Exception:
                outcome_counter["unreadable"] += 1
        mtime, line = latest_event(run_dir)
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_dir = run_dir.parent.name
            latest_line = line

    return {
        "expected": expected,
        "run_dirs": len(run_dirs),
        "results": len(result_dirs),
        "active": len(active_dirs),
        "predictions": line_count(PREDICTION_DIR / f"{method}.jsonl"),
        "raw_reports": report_count(OFFICIAL_ROOT / "reports", method),
        "corrected_reports": report_count(OFFICIAL_ROOT / "reports_corrected", method),
        "result_outcome_counts": dict(sorted(outcome_counter.items())),
        "active_dirs": [path.parent.name for path in active_dirs[:5]],
        "latest_event_age_sec": round(time.time() - latest_mtime, 1) if latest_mtime else "",
        "latest_event_dir": latest_dir,
        "latest_event": latest_line[:260],
    }


def phase_for(statuses: dict[str, dict[str, object]]) -> str:
    if (OFFICIAL_ROOT / "extended_summary_metrics.csv").is_file():
        return "complete"
    if any(int(status["corrected_reports"]) for status in statuses.values()):
        return "official-eval-running-or-partial"
    if any(int(status["predictions"]) for status in statuses.values()):
        return "predictions-exported-or-partial"
    if all(int(status["results"]) >= int(status["expected"]) for status in statuses.values()):
        return "local-complete"
    return "local-running"


def main() -> None:
    manifest = read_json(MANIFEST_PATH)
    if not isinstance(manifest, dict):
        raise TypeError(f"{MANIFEST_PATH} must contain an object")
    methods_info = manifest.get("methods", {})
    if not isinstance(methods_info, dict):
        raise TypeError("manifest methods must be an object")

    statuses: dict[str, dict[str, object]] = {}
    for method in METHODS:
        info = methods_info.get(method, {})
        expected = int(info.get("expected_tasks", 0)) if isinstance(info, dict) else 0
        statuses[method] = method_status(method, expected)

    total_expected = sum(int(status["expected"]) for status in statuses.values())
    total_results = sum(int(status["results"]) for status in statuses.values())
    total_predictions = sum(int(status["predictions"]) for status in statuses.values())
    processes = process_lines()
    tmux = tmux_lines()
    phase = phase_for(statuses)

    lines: list[str] = []
    lines.append("# SWE-bench Lite Baselines Non-Resolved Rerun001 Status")
    lines.append("")
    lines.append(f"Updated: {time.strftime('%Y-%m-%d %H:%M:%S %Z')}")
    lines.append("")
    lines.append(f"- phase: {phase}")
    lines.append("- selection: official_status != resolved from swebench_lite_full_baselines300, per method")
    lines.append(f"- methods: {','.join(METHODS)}")
    lines.append(f"- tmux sessions: {'; '.join(tmux) if tmux else 'none'}")
    lines.append(f"- matching processes: {len(processes)}")
    lines.append(f"- total result.json: {total_results}/{total_expected}")
    lines.append(f"- total predictions: {total_predictions}/{total_expected}")
    lines.append("")
    lines.append("## Counts")
    lines.append("")
    lines.append(
        "| method | expected | result.json | run dirs | active dirs | predictions | raw reports | corrected reports | result outcomes |"
    )
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for method in METHODS:
        status = statuses[method]
        result_outcomes = ", ".join(
            f"{key}: {value}" for key, value in dict(status["result_outcome_counts"]).items()
        )
        lines.append(
            f"| {method} | {status['expected']} | {status['results']} | {status['run_dirs']} | "
            f"{status['active']} | {status['predictions']} | {status['raw_reports']} | "
            f"{status['corrected_reports']} | {result_outcomes} |"
        )
    lines.append(
        f"| **TOTAL** | {total_expected} | {total_results} |  |  | {total_predictions} |  |  |  |"
    )
    lines.append("")
    lines.append("## Active Cases")
    lines.append("")
    lines.append("| method | active examples | latest event age | latest event dir | latest event |")
    lines.append("| --- | --- | ---: | --- | --- |")
    for method in METHODS:
        status = statuses[method]
        active_examples = ", ".join(status["active_dirs"]) if status["active_dirs"] else ""
        latest_event = str(status["latest_event"]).replace("|", "\\|")
        lines.append(
            f"| {method} | {active_examples} | {status['latest_event_age_sec']} | "
            f"{status['latest_event_dir']} | `{latest_event}` |"
        )
    lines.append("")
    lines.append("## Paths")
    lines.append("")
    lines.append(f"- manifest: `{MANIFEST_PATH}`")
    lines.append(f"- local results: `{LOCAL_ROOT}/`")
    lines.append(f"- predictions: `{PREDICTION_DIR}/`")
    lines.append(f"- official eval: `{OFFICIAL_ROOT}/`")
    lines.append(f"- logs: `{LOCAL_ROOT / 'logs'}/`")
    lines.append("")
    lines.append("## Matching Processes")
    lines.append("")
    if processes:
        lines.extend(f"- `{line.strip()}`" for line in processes[:20])
    else:
        lines.append("- none")
    lines.append("")

    STATUS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {STATUS_PATH}")


if __name__ == "__main__":
    main()
