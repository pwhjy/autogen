#!/usr/bin/env python3
"""Safe local chunk supervisor for the baseline non-resolved rerun.

This intentionally avoids the bash controller's mutable loop-variable state.
It only launches a chunk when all three are true:

* the chunk is not complete,
* the exact method/chunk instance list is not already active, and
* the active chunk count is below the configured local concurrency.
"""

from __future__ import annotations

import datetime as dt
import os
import shlex
import subprocess
import time
from pathlib import Path


RUN_NAME = "swebench_lite_full_baselines_nonresolved_rerun001"
BENCHMARK_PREFIX = f"external_{RUN_NAME}"
METHODS = ["summary", "vector_memory", "autogen_broadcast", "sliding_window", "structured_summary"]
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "Results" / RUN_NAME
CHUNK_DIR = RESULTS / "chunks"
LOG_DIR = RESULTS / "logs"
PYTHON_DIR = ROOT / "../../../.."
MAX_JOBS = int(os.environ.get("VACTHBENCH_BASELINE_RERUN_MAX_JOBS", "4"))
POLL_SEC = int(os.environ.get("VACTHBENCH_BASELINE_RERUN_SUPERVISOR_POLL_SEC", "300"))


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (LOG_DIR / "local_safe_supervisor.log").open("a", encoding="utf-8") as fh:
        fh.write(f"[{now()}] {message}\n")
    print(f"[{now()}] {message}", flush=True)


def job_order() -> list[tuple[str, str]]:
    jobs: list[tuple[str, str]] = []
    for line in (CHUNK_DIR / "job_order.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        method, chunk, _path = line.split("\t")
        jobs.append((method, chunk))
    return jobs


def chunk_instances(method: str, chunk: str) -> str:
    return (CHUNK_DIR / f"{method}_{chunk}.instances").read_text(encoding="utf-8").strip()


def result_path(method: str, instance_id: str) -> Path:
    return (
        RESULTS
        / f"{BENCHMARK_PREFIX}_{method}"
        / f"{BENCHMARK_PREFIX}__{instance_id}_{method}"
        / "0"
        / "result.json"
    )


def chunk_complete(method: str, chunk: str) -> bool:
    instances = [item for item in chunk_instances(method, chunk).split(",") if item]
    return bool(instances) and all(result_path(method, instance_id).is_file() for instance_id in instances)


def active_chunks() -> set[tuple[str, str]]:
    output = subprocess.run(["ps", "ax", "-o", "command="], check=True, text=True, capture_output=True).stdout
    active: set[tuple[str, str]] = set()
    for line in output.splitlines():
        if "run_baseline.py" not in line or RUN_NAME not in line or "--instances " not in line:
            continue
        method = next(
            (candidate for candidate in METHODS if f"Tasks/{BENCHMARK_PREFIX}_{candidate}.jsonl" in line),
            None,
        )
        if method is None:
            continue
        instance_arg = line.split("--instances ", 1)[1].strip().strip("'\"")
        for chunk in ["chunk0", "chunk1", "chunk2", "chunk3"]:
            if instance_arg == chunk_instances(method, chunk):
                active.add((method, chunk))
                break
    return active


def launch_chunk(method: str, chunk: str) -> None:
    instances = chunk_instances(method, chunk)
    session = f"bnr1-{method.replace('_', '-')}-{chunk}"
    log_path = LOG_DIR / f"local_{method}_{chunk}_safe_supervisor.log"
    cmd = (
        f"cd {shlex.quote(str(ROOT))} && "
        f"instances={shlex.quote(instances)} && "
        "{ "
        f"echo \"[$(date '+%Y-%m-%d %H:%M:%S %Z')] START local {method} {chunk} safe_supervisor\"; "
        f"echo task_file=Tasks/{BENCHMARK_PREFIX}_{method}.jsonl; "
        f"echo results_arg={RUN_NAME}/{BENCHMARK_PREFIX}_{method}; "
        "echo \"instances=${instances}\"; "
        "AGBENCH_ALLOW_NATIVE=Yes "
        "VACTHBENCH_TASK_TIMEOUT_SEC=5400 "
        "VACTHBENCH_MODEL_CALL_TIMEOUT_SEC=900 "
        "VACTHBENCH_MODEL_CALL_RETRIES=2 "
        "VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE=1 "
        "VACTHBENCH_DOCKER_PULL=0 "
        "VACTHBENCH_GIT_FETCH_RETRIES=2 "
        "VACTHBENCH_DOCKER_PROXY= "
        "uv run --no-project --offline --isolated "
        f"--with {shlex.quote(str((PYTHON_DIR / 'packages/agbench').resolve()))} "
        f"--with {shlex.quote(str((PYTHON_DIR / 'packages/autogen-agentchat').resolve()))} "
        f"--with {shlex.quote(str((PYTHON_DIR / 'packages/autogen-core').resolve()))} "
        f"--with {shlex.quote(str((PYTHON_DIR / 'packages/autogen-ext[openai]').resolve()))} "
        f"python Scripts/run_baseline.py Tasks/{BENCHMARK_PREFIX}_{method}.jsonl "
        f"--results-dir {RUN_NAME}/{BENCHMARK_PREFIX}_{method} "
        "--instances \"${instances}\"; "
        "exit_status=$?; "
        f"echo \"[$(date '+%Y-%m-%d %H:%M:%S %Z')] END local {method} {chunk} safe_supervisor status=${{exit_status}}\"; "
        "exit ${exit_status}; "
        f"}} >> {shlex.quote(str(log_path))} 2>&1"
    )
    subprocess.run(["tmux", "new-session", "-d", "-s", session, cmd], check=True)
    log(f"launched {method} {chunk} session={session}")


def main() -> int:
    os.chdir(ROOT)
    log(f"safe supervisor start max_jobs={MAX_JOBS} poll_sec={POLL_SEC}")
    jobs = job_order()
    while True:
        active = active_chunks()
        incomplete = [(method, chunk) for method, chunk in jobs if not chunk_complete(method, chunk)]
        log(f"active={len(active)} incomplete_chunks={len(incomplete)}")
        if not incomplete:
            log("all local chunks complete")
            return 0
        for method, chunk in jobs:
            if len(active) >= MAX_JOBS:
                break
            if (method, chunk) in active or chunk_complete(method, chunk):
                continue
            launch_chunk(method, chunk)
            active.add((method, chunk))
        time.sleep(POLL_SEC)


if __name__ == "__main__":
    raise SystemExit(main())
