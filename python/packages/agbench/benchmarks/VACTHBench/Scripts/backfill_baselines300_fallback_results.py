from __future__ import annotations

import json
import subprocess
from pathlib import Path

import run_baseline

BENCH_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BENCH_DIR / "Results"
RUN_PARENT = "swebench_lite_full_baselines300"
BENCHMARK_NAME = "swebench_lite_full_baselines300"
METHODS = (
    "summary",
    "vector_memory",
    "autogen_broadcast",
    "sliding_window",
    "structured_summary",
)


def active_command_text() -> str:
    completed = subprocess.run(
        ["ps", "-axo", "command"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def load_tasks(method: str) -> dict[str, dict]:
    path = BENCH_DIR / "Tasks" / f"external_{BENCHMARK_NAME}_{method}.jsonl"
    tasks: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            task = json.loads(line)
            tasks[str(task["id"])] = task
    return tasks


def should_backfill(run_dir: Path, active_text: str) -> bool:
    if (run_dir / "result.json").is_file():
        return False
    if str(run_dir / "scenario.py") in active_text:
        return False
    console_log = run_dir / "console_log.txt"
    if not console_log.is_file():
        return False
    text = console_log.read_text(encoding="utf-8", errors="replace")
    return "TIMEOUT after" in text or "SCENARIO.PY RUNTIME:" in text


def main() -> None:
    active_text = active_command_text()
    backfilled = 0
    for method in METHODS:
        tasks = load_tasks(method)
        root = RESULTS_DIR / RUN_PARENT / f"external_{BENCHMARK_NAME}_{method}"
        if not root.is_dir():
            continue
        for task_dir in sorted(path for path in root.iterdir() if path.is_dir() and not path.name.startswith("._")):
            run_dir = task_dir / "0"
            task = tasks.get(task_dir.name)
            if task is None or not run_dir.is_dir():
                continue
            if not should_backfill(run_dir, active_text):
                continue
            run_baseline.write_fallback_result(
                run_dir,
                task,
                reason="task_timeout_backfill",
                message="Backfilled unresolved result after run_baseline task timeout left no result.json.",
                elapsed_sec=float(run_baseline.TASK_TIMEOUT_SEC),
            )
            backfilled += 1
            print(f"backfilled {task_dir.name}")
    print(f"backfilled_count={backfilled}")


if __name__ == "__main__":
    main()
