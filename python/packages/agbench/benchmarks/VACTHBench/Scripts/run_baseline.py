"""Direct runner for VACTHBench baseline experiments.

Bypasses agbench's Docker/venv infrastructure. Uses the uv-managed venv
directly (which already has all autogen packages installed), expands the
task template, and runs scenario.py with a larger timeout.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import select
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
TEMPLATES_DIR = BENCHMARK_DIR / "Templates"
RESULTS_DIR = BENCHMARK_DIR / "Results"

# agbench uses 120 min per task; keep 180 min by default, but allow
# experiment runs to tighten it without editing this helper.
TASK_TIMEOUT_SEC = int(os.environ.get("VACTHBENCH_TASK_TIMEOUT_SEC", str(60 * 180)))


METHOD_SUFFIXES = (
    "autogen_broadcast",
    "sliding_window",
    "structured_summary",
    "summary",
    "vacth_full",
    "vacth_optimized",
    "vector_memory",
)


def read_json_file(path: Path, default):
    if not path.is_file():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def read_jsonl_file(path: Path) -> list[dict]:
    records: list[dict] = []
    if not path.is_file():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    return records


def experiment_config_from_task(task: dict) -> dict:
    raw_config = (
        task.get("substitutions", {})
        .get("scenario.py", {})
        .get("__EXPERIMENT_CONFIG_JSON__", "")
    )
    if raw_config:
        try:
            value = json.loads(raw_config)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass

    task_id = str(task.get("id", ""))
    method = ""
    for suffix in METHOD_SUFFIXES:
        marker = f"_{suffix}"
        if task_id.endswith(marker):
            method = suffix
            break
    return {
        "task_id": task_id,
        "method": method,
    }


def write_metrics_csv(path: Path, metrics: dict) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=sorted(metrics))
        writer.writeheader()
        writer.writerow(metrics)


def write_fallback_result(
    instance_dir: Path,
    task: dict,
    *,
    reason: str,
    message: str,
    elapsed_sec: float,
) -> None:
    result_path = instance_dir / "result.json"
    if result_path.is_file():
        return

    config = experiment_config_from_task(task)
    task_id = str(config.get("task_id") or task.get("id", ""))
    method = str(config.get("method") or "")

    raw_messages = read_json_file(instance_dir / "raw_messages.json", [])
    visible_contexts = read_json_file(instance_dir / "visible_contexts.json", [])
    model_usage = read_json_file(instance_dir / "model_usage.json", [])
    model_events = read_json_file(instance_dir / "model_events.json", None)
    if not isinstance(model_events, list):
        model_events = read_jsonl_file(instance_dir / "model_events.jsonl")
    tool_calls = read_json_file(instance_dir / "tool_calls.json", [])

    metrics = {
        "agent_error": True,
        "agent_error_reason": reason,
        "agent_error_message": message[-1000:],
        "changed_file_count": 0,
        "changed_test_file_count": 0,
        "constraint_violations": 0,
        "edit_count": 0,
        "invalid_patch_count": 1,
        "missing_expected_file_count": 0,
        "reviewer_resolved": False,
        "run_tests_count": 0,
        "success": False,
        "swebench_outcome": "agent_error",
        "swebench_resolved": False,
        "timeout_sec": TASK_TIMEOUT_SEC if reason == "task_timeout" else 0,
        "tool_calls": len(tool_calls) if isinstance(tool_calls, list) else 0,
        "turns": len(raw_messages) if isinstance(raw_messages, list) else 0,
        "wall_time_sec": round(elapsed_sec, 3),
    }
    result = {
        "task_id": task_id,
        "method": method,
        "experiment_config": config,
        "raw_messages": raw_messages if isinstance(raw_messages, list) else [],
        "visible_contexts": visible_contexts if isinstance(visible_contexts, list) else [],
        "model_usage": model_usage if isinstance(model_usage, list) else [],
        "model_events": model_events,
        "tool_calls": tool_calls if isinstance(tool_calls, list) else [],
        "summaries": read_json_file(instance_dir / "summaries.json", []),
        "structured_states": read_json_file(instance_dir / "structured_summary.json", {}).get("states", []),
        "capsules": read_json_file(instance_dir / "capsules.json", []),
        "vacth_extractions": read_json_file(instance_dir / "vacth_extractions.json", []),
        "memory_items": read_json_file(instance_dir / "vector_memory.json", {}).get("memory_items", []),
        "state_items": read_json_file(instance_dir / "state_items.json", []),
        "routing_decisions": read_json_file(instance_dir / "routing_decisions.json", []),
        "provenance_edges": read_json_file(instance_dir / "provenance_graph.json", {}).get("edges", []),
        "metrics": metrics,
        "final_answer": "unresolved",
    }
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_metrics_csv(instance_dir / "metrics.csv", metrics)


def expand_task(task: dict, output_dir: Path, config_path: Path) -> None:
    """Expand a VACTHBench task into a runnable directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Copy common template files
    common_dir = TEMPLATES_DIR / "common"
    if common_dir.is_dir():
        shutil.copytree(common_dir, output_dir, dirs_exist_ok=True)

    # Copy domain template files
    template = task.get("template", [])
    for entry in template:
        src = TEMPLATES_DIR / entry
        if src.is_dir():
            shutil.copytree(src, output_dir, dirs_exist_ok=True)
        elif src.is_file():
            shutil.copy2(src, output_dir / src.name)

    # Copy config.yaml
    shutil.copy2(config_path, output_dir / "config.yaml")

    # Apply substitutions
    substitutions = task.get("substitutions", {})
    for filename, replacements in substitutions.items():
        target = output_dir / filename
        if not target.is_file():
            continue
        content = target.read_text(encoding="utf-8")
        for placeholder, value in replacements.items():
            content = content.replace(placeholder, value)
        target.write_text(content, encoding="utf-8")


def run_one_task(
    task: dict,
    results_dir: Path,
    config_path: Path,
    env: dict[str, str],
) -> tuple[bool, str]:
    """Run a single task and return (success, error_message)."""
    task_id = task["id"]
    instance_dir = results_dir / task_id / "0"

    # Skip if already completed
    if (instance_dir / "result.json").is_file():
        print(f"  SKIP {task_id} — already done")
        return True, "SKIP"

    print(f"  RUN  {task_id}")
    start_time = time.time()

    try:
        # Expand template
        expand_task(task, instance_dir, config_path)

        # Run scenario.py
        workspace = instance_dir
        console_log = workspace / "console_log.txt"

        python_bin = sys.executable  # use current Python (has all packages)

        # Add gnubin to PATH for timeout command
        run_env = os.environ.copy()
        run_env.update(env)
        gnubin = "/opt/homebrew/opt/coreutils/libexec/gnubin"
        if gnubin not in run_env.get("PATH", ""):
            run_env["PATH"] = f"{gnubin}:{run_env.get('PATH', '')}"
        existing_pythonpath = run_env.get("PYTHONPATH", "")
        run_env["PYTHONPATH"] = (
            f"{workspace.resolve()}{os.pathsep}{existing_pythonpath}"
            if existing_pythonpath
            else str(workspace.resolve())
        )

        # Write initial timestamp
        (workspace / "timestamp.txt").write_text(
            datetime.now(timezone.utc).isoformat()
        )

        cmd = [
            python_bin,
            str(workspace / "scenario.py"),
        ]

        with open(console_log, "w", encoding="utf-8") as log_fh:
            log_fh.write(
                f"SCENARIO.PY STARTING !#!#\n"
                f"Command: {' '.join(cmd)}\n"
                f"CWD: {workspace}\n\n"
            )
            log_fh.flush()

            try:
                proc = subprocess.Popen(
                    cmd,
                    cwd=str(workspace),
                    env=run_env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                )
                deadline = time.monotonic() + TASK_TIMEOUT_SEC
                assert proc.stdout is not None
                while True:
                    if proc.poll() is not None:
                        remaining = proc.stdout.read()
                        if remaining:
                            log_fh.write(remaining)
                            log_fh.flush()
                            print(remaining, end="", flush=True)
                        break

                    remaining_sec = deadline - time.monotonic()
                    if remaining_sec <= 0:
                        raise subprocess.TimeoutExpired(cmd, TASK_TIMEOUT_SEC)

                    readable, _, _ = select.select([proc.stdout], [], [], min(1.0, remaining_sec))
                    if not readable:
                        continue
                    line = proc.stdout.readline()
                    if line:
                        log_fh.write(line)
                        log_fh.flush()
                        # Also echo to our stdout
                        print(line, end="", flush=True)
                proc.wait()
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                log_fh.write(
                    f"\nTIMEOUT after {TASK_TIMEOUT_SEC}s !#!#\n"
                )
                log_fh.flush()
                elapsed = time.time() - start_time
                message = f"timeout after {elapsed:.0f}s"
                write_fallback_result(
                    instance_dir,
                    task,
                    reason="task_timeout",
                    message=message,
                    elapsed_sec=elapsed,
                )
                print(f"  TIMEOUT {task_id} ({elapsed:.0f}s)")
                return True, "FALLBACK_TIMEOUT"

            elapsed = time.time() - start_time
            log_fh.write(
                f"\nSCENARIO.PY RUNTIME: {elapsed:.0f}s !#!#\n"
            )

        # Write elapsed time
        with open(instance_dir / "timestamp.txt", "a") as tf:
            tf.write(f"\nelapsed_sec: {elapsed:.0f}\n")

        # Check for result.json
        if not (instance_dir / "result.json").is_file():
            write_fallback_result(
                instance_dir,
                task,
                reason="no_result",
                message="scenario.py exited without producing result.json",
                elapsed_sec=elapsed,
            )
            print(f"  FALLBACK {task_id}: no result.json produced")
            return True, "FALLBACK_NO_RESULT"

        print(f"  DONE  {task_id} ({elapsed:.0f}s)")
        return True, ""

    except Exception as exc:
        elapsed = time.time() - start_time
        write_fallback_result(
            instance_dir,
            task,
            reason="runner_exception",
            message=str(exc),
            elapsed_sec=elapsed,
        )
        print(f"  FAIL  {task_id}: {exc}")
        return True, "FALLBACK_EXCEPTION"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run VACTHBench baseline experiments."
    )
    parser.add_argument(
        "task_file",
        help="Path to the task JSONL file.",
    )
    parser.add_argument(
        "--results-dir",
        default=None,
        help="Override results directory name (default: derived from task file).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Only run the first N tasks (0 = all).",
    )
    parser.add_argument(
        "--instances",
        default="",
        help="Comma-separated instance_id filter.",
    )
    args = parser.parse_args()

    task_file = Path(args.task_file).resolve()
    if not task_file.is_file():
        print(f"Error: {task_file} not found")
        sys.exit(1)

    # Load tasks
    tasks: list[dict] = []
    with task_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            tasks.append(json.loads(line))

    instance_filter = set(args.instances.split(",")) if args.instances else set()
    if instance_filter:
        tasks = [
            t for t in tasks
            if any(iid in t["id"] for iid in instance_filter)
        ]

    if args.limit > 0:
        tasks = tasks[: args.limit]

    print(f"Loaded {len(tasks)} tasks from {task_file.name}")

    # Determine results directory name
    if args.results_dir:
        results_name = args.results_dir
    else:
        results_name = task_file.stem  # e.g., "external_swebench_lite_autogen_broadcast"

    results_dir = RESULTS_DIR / results_name
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results -> {results_dir}")

    # Config path
    config_path = BENCHMARK_DIR / "config.yaml"
    if not config_path.is_file():
        print(f"Error: {config_path} not found")
        sys.exit(1)

    # Environment variables for the model
    env = {}
    for key, value in os.environ.items():
        if key == "OPENAI_API_KEY" or key.startswith("VACTHBENCH_"):
            env[key] = value

    # Run tasks sequentially
    ok = 0
    failed = 0
    skipped = 0
    for task in tasks:
        success, error = run_one_task(task, results_dir, config_path, env)
        if success:
            if "SKIP" not in error:
                ok += 1
            else:
                skipped += 1
        else:
            failed += 1

    print(f"\nDone: {ok} ok, {failed} failed, {skipped} skipped "
          f"out of {len(tasks)}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
