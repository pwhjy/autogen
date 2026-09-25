from __future__ import annotations

import json
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

BENCH_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCH_DIR / "Results"
RUN_PARENT = "swebench_lite_full_baselines300"
BENCHMARK_NAME = "swebench_lite_full_baselines300"
METHODS = [
    "summary",
    "vector_memory",
    "autogen_broadcast",
    "sliding_window",
    "structured_summary",
]


def count_paths(pattern: str) -> int:
    return sum(1 for _ in BENCH_DIR.glob(pattern))


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def command_output(args: list[str]) -> str:
    completed = subprocess.run(args, capture_output=True, text=True, check=False)
    return completed.stdout.strip()


def tmux_session_running(name: str) -> bool:
    completed = subprocess.run(
        ["tmux", "has-session", "-t", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    return completed.returncode == 0


def process_lines() -> list[str]:
    output = command_output(["ps", "-axo", "pid,ppid,stat,%cpu,%mem,etime,command"])
    needles = (
        "run_swebench_lite_full_baselines300_supervisor.sh",
        "run_baseline.py Tasks/external_swebench_lite_full_baselines300",
        "swebench_lite_full_baselines300__",
        "run_swebench_official_patched.py",
        "export_swebench_predictions_aligned.py",
        "summarize_swebench_aligned_metrics.py",
    )
    return [
        line
        for line in output.splitlines()
        if any(needle in line for needle in needles)
    ]


def active_local_job_count() -> int:
    count = 0
    for method in METHODS:
        completed = subprocess.run(
            ["pgrep", "-f", f"run_baseline.py Tasks/external_{BENCHMARK_NAME}_{method}.jsonl"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if completed.returncode == 0:
            count += 1
    return count


def current_task(method: str) -> str:
    root = RESULTS_DIR / RUN_PARENT / f"external_{BENCHMARK_NAME}_{method}"
    if not root.is_dir():
        return ""
    task_dirs = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.name.startswith("._")
    )
    if not task_dirs:
        return ""
    active = [
        path.name
        for path in task_dirs
        if (path / "0" / "console_log.txt").is_file()
        and not (path / "0" / "result.json").is_file()
    ]
    return active[-1] if active else task_dirs[-1].name


def prediction_count(method: str) -> int:
    path = RESULTS_DIR / f"swebench_predictions_{BENCHMARK_NAME}" / f"{method}.jsonl"
    return line_count(path)


def official_summary_exists() -> bool:
    return (
        RESULTS_DIR
        / "swebench_official_eval"
        / BENCHMARK_NAME
        / "extended_summary_metrics.csv"
    ).is_file()


def official_report_count(method: str) -> int:
    reports = RESULTS_DIR / "swebench_official_eval" / BENCHMARK_NAME / "reports_corrected"
    if not reports.is_dir():
        return 0
    return len(list(reports.glob(f"{method}.*.corrected.json")))


def main() -> None:
    now = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S %Z")
    status_path = RESULTS_DIR / f"{RUN_PARENT}_status.md"
    supervisor_log = RESULTS_DIR / RUN_PARENT / "logs" / "supervisor.log"
    controller_log = RESULTS_DIR / RUN_PARENT / "logs" / "controller.log"
    supervisor_running = tmux_session_running("baselines300")
    controller_running = tmux_session_running("baselines300-controller")
    monitor_running = tmux_session_running("baselines300-monitor")
    sliding_running = tmux_session_running("baselines300-sliding")
    structured_running = tmux_session_running("baselines300-structured")
    active_jobs = active_local_job_count()
    phase = "complete" if official_summary_exists() else "running"
    if not (supervisor_running or controller_running or active_jobs) and not official_summary_exists():
        phase = "stopped-or-between-phases"

    lines = [
        "# SWE-bench Lite Full Baselines300 Status",
        "",
        f"Updated: {now}",
        "",
        f"- phase: {phase}",
        f"- tmux baselines300: {'running' if supervisor_running else 'not running'}",
        f"- tmux baselines300-controller: {'running' if controller_running else 'not running'}",
        f"- tmux baselines300-monitor: {'running' if monitor_running else 'not running'}",
        f"- tmux baselines300-sliding: {'running' if sliding_running else 'not running'}",
        f"- tmux baselines300-structured: {'running' if structured_running else 'not running'}",
        f"- active local run_baseline jobs: {active_jobs}",
        "- methods: summary,vector_memory,autogen_broadcast,sliding_window,structured_summary",
        "- vacth_full: excluded",
        "",
        "## Local Result Counts",
        "",
        "| method | result.json | task dirs | task file rows | current/last task | predictions | official reports |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for method in METHODS:
        root = RESULTS_DIR / RUN_PARENT / f"external_{BENCHMARK_NAME}_{method}"
        task_file = BENCH_DIR / "Tasks" / f"external_{BENCHMARK_NAME}_{method}.jsonl"
        result_count = len(list(root.glob("*/0/result.json"))) if root.is_dir() else 0
        task_dirs = len(
            [
                path
                for path in root.iterdir()
                if path.is_dir() and not path.name.startswith("._")
            ]
        ) if root.is_dir() else 0
        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(result_count),
                    str(task_dirs),
                    str(line_count(task_file)),
                    current_task(method) or "",
                    str(prediction_count(method)),
                    str(official_report_count(method)),
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Active Processes",
            "",
        ]
    )
    proc = process_lines()
    if proc:
        lines.extend(["```text", *proc[:30], "```"])
    else:
        lines.append("No matching baselines300 processes found.")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            f"- supervisor log: `{supervisor_log.relative_to(BENCH_DIR)}`",
            f"- controller log: `{controller_log.relative_to(BENCH_DIR)}`",
            f"- local results: `Results/{RUN_PARENT}/`",
            f"- predictions: `Results/swebench_predictions_{BENCHMARK_NAME}/`",
            f"- official eval: `Results/swebench_official_eval/{BENCHMARK_NAME}/`",
        ]
    )
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(status_path)


if __name__ == "__main__":
    main()
