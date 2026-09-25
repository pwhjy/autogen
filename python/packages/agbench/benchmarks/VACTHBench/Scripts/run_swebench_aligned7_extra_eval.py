"""Run official SWE-bench evaluation for aligned7 extra instances."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"

DEFAULT_METHODS = (
    "autogen_broadcast",
    "sliding_window",
    "structured_summary",
    "summary",
    "vacth_full",
    "vector_memory",
)


def safe_id(value: str) -> str:
    return value.replace("/", "_").replace("-", "_").replace("__", "_")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instance-id", default="django__django-10924")
    parser.add_argument(
        "--prediction-dir",
        default=str(RESULTS_DIR / "swebench_predictions_aligned7"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "swebench_official_eval" / "aligned7"),
    )
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--timeout", default="1800")
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    reports_dir = output_dir / "reports"
    logs_dir = output_dir / "logs"
    reports_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    failures: list[tuple[str, int]] = []
    for method in methods:
        prediction_file = prediction_dir / f"{method}.jsonl"
        if not prediction_file.is_file():
            print(f"missing prediction file for {method}: {prediction_file}", flush=True)
            failures.append((method, 2))
            continue

        run_id = f"aligned7_extra_{method}_{safe_id(args.instance_id)}_20260520"
        log_path = logs_dir / f"{method}.{safe_id(args.instance_id)}.log"
        cmd = [
            "uvx",
            "--python",
            "3.12",
            "--from",
            "swebench",
            "python",
            "-m",
            "swebench.harness.run_evaluation",
            "-d",
            "SWE-bench/SWE-bench_Lite",
            "-s",
            "test",
            "-i",
            args.instance_id,
            "-p",
            str(prediction_file),
            "--max_workers",
            "1",
            "-t",
            args.timeout,
            "--cache_level",
            "instance",
            "--clean",
            "False",
            "-id",
            run_id,
            "--report_dir",
            str(reports_dir),
        ]
        print(f"=== {method} {args.instance_id} ===", flush=True)
        print(" ".join(cmd), flush=True)
        with log_path.open("w", encoding="utf-8") as log_fh:
            log_fh.write(" ".join(cmd) + "\n\n")
            log_fh.flush()
            completed = subprocess.run(
                cmd,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                text=True,
            )
        print(f"{method} rc={completed.returncode} log={log_path}", flush=True)
        if completed.returncode:
            failures.append((method, completed.returncode))

    if failures:
        print("Failures:", failures, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
