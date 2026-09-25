#!/usr/bin/env python3
"""Finalize the SWE-bench Lite baseline non-resolved rerun after local results complete."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import time
from pathlib import Path


RUN_NAME = "swebench_lite_full_baselines_nonresolved_rerun001"
BENCHMARK_PREFIX = f"external_{RUN_NAME}"
METHODS = ("summary", "vector_memory", "autogen_broadcast", "sliding_window", "structured_summary")
ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = ROOT / "Results"
LOCAL_ROOT = RESULTS_ROOT / RUN_NAME
LOG_DIR = LOCAL_ROOT / "logs"
MANIFEST_PATH = RESULTS_ROOT / f"{RUN_NAME}_manifest.json"
PREDICTION_DIR = RESULTS_ROOT / f"swebench_predictions_{RUN_NAME}"
OFFICIAL_ROOT = RESULTS_ROOT / "swebench_official_eval" / RUN_NAME
ORIGINAL_OFFICIAL_ROOT = RESULTS_ROOT / "swebench_official_eval" / "swebench_lite_full_baselines300"


def now() -> str:
    return dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def log(message: str) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{now()}] {message}"
    with (LOG_DIR / "finalizer.log").open("a", encoding="utf-8") as fh:
        fh.write(f"{line}\n")
    print(line, flush=True)


def read_json(path: Path) -> object:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def manifest() -> dict[str, object]:
    data = read_json(MANIFEST_PATH)
    if not isinstance(data, dict):
        raise TypeError(f"{MANIFEST_PATH} must contain an object")
    return data


def method_info(method: str) -> dict[str, object]:
    methods = manifest().get("methods", {})
    if not isinstance(methods, dict) or method not in methods or not isinstance(methods[method], dict):
        raise KeyError(f"missing manifest method info for {method}")
    return methods[method]


def expected_for_method(method: str) -> int:
    return int(method_info(method)["expected_tasks"])


def selected_instances(method: str) -> list[str]:
    selected = method_info(method)["selected_instance_ids"]
    if not isinstance(selected, list):
        raise TypeError(f"selected_instance_ids for {method} must be a list")
    return [str(item) for item in selected]


def local_result_path(method: str, instance_id: str) -> Path:
    return (
        LOCAL_ROOT
        / f"{BENCHMARK_PREFIX}_{method}"
        / f"{BENCHMARK_PREFIX}__{instance_id}_{method}"
        / "0"
        / "result.json"
    )


def local_counts() -> dict[str, tuple[int, int]]:
    counts: dict[str, tuple[int, int]] = {}
    for method in METHODS:
        selected = selected_instances(method)
        done = sum(local_result_path(method, instance_id).is_file() for instance_id in selected)
        counts[method] = (done, len(selected))
    return counts


def local_complete() -> bool:
    return all(done == expected for done, expected in local_counts().values())


def line_count(path: Path) -> int:
    if not path.is_file():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        return sum(1 for _ in fh)


def prediction_counts() -> dict[str, tuple[int, int]]:
    return {method: (line_count(PREDICTION_DIR / f"{method}.jsonl"), expected_for_method(method)) for method in METHODS}


def predictions_complete() -> bool:
    return all(done == expected for done, expected in prediction_counts().values())


def report_count(report_dir: Path, method: str) -> int:
    if not report_dir.is_dir():
        return 0
    return sum(1 for path in report_dir.glob(f"{method}.*.json") if not path.name.startswith("._"))


def update_status() -> None:
    subprocess.run(
        ["uv", "run", "--no-project", "--offline", "--isolated", "python", "Scripts/write_baselines_nonresolved_rerun001_status.py"],
        cwd=ROOT,
        check=False,
    )


def run_logged(command: list[str], log_path: Path, *, env: dict[str, str] | None = None) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(f"\n[{now()}] COMMAND {' '.join(command)}\n")
        fh.flush()
        process = subprocess.run(command, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT, env=env, check=False)
        fh.write(f"[{now()}] STATUS {process.returncode}\n")
    return process.returncode


def export_predictions() -> None:
    if predictions_complete():
        log("predictions already complete")
        return
    log("exporting predictions")
    status = run_logged(
        [
            "uv",
            "run",
            "--no-project",
            "--offline",
            "--isolated",
            "python",
            "Scripts/export_swebench_predictions_aligned.py",
            "--results-dir",
            f"Results/{RUN_NAME}",
            "--output-dir",
            f"Results/swebench_predictions_{RUN_NAME}",
            "--methods",
            ",".join(METHODS),
            "--benchmark-prefixes",
            BENCHMARK_PREFIX,
            "--task-prefixes",
            BENCHMARK_PREFIX,
        ],
        LOG_DIR / "export_predictions_finalizer.log",
    )
    update_status()
    if status != 0 or not predictions_complete():
        raise RuntimeError(f"prediction export incomplete: status={status}, counts={prediction_counts()}")
    log("predictions complete")


def copy_tree_contents(src: Path, dst: Path) -> None:
    if not src.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        if item.name.startswith("._"):
            continue
        target = dst / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def copy_method_official_artifacts(method: str) -> None:
    method_root = OFFICIAL_ROOT / "per_method" / method
    for report_name in ("reports", "reports_corrected", "logs"):
        src = method_root / report_name
        dst = OFFICIAL_ROOT / report_name
        dst.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            for item in src.iterdir():
                if item.is_file() and not item.name.startswith("._"):
                    shutil.copy2(item, dst / item.name)
    copy_tree_contents(method_root / "run_logs", OFFICIAL_ROOT / "run_logs")


def official_report_complete(method: str) -> bool:
    return report_count(OFFICIAL_ROOT / "reports_corrected", method) > 0


def run_official_eval() -> None:
    if not predictions_complete():
        raise RuntimeError("cannot run official eval before predictions are complete")
    docker_config = Path("/tmp/vacthbench_docker_config")
    docker_config.mkdir(parents=True, exist_ok=True)
    (docker_config / "config.json").write_text("{}", encoding="utf-8")
    env = os.environ.copy()
    env["VACTHBENCH_DOCKER_PROXY"] = ""
    env["DOCKER_CONFIG"] = str(docker_config)
    for method in METHODS:
        if official_report_complete(method):
            log(f"official eval already complete for {method}")
            continue
        log(f"running official eval for {method}")
        status = run_logged(
            [
                "uv",
                "run",
                "--no-project",
                "--isolated",
                "--with",
                "swebench",
                "--with",
                "docker",
                "--with",
                "datasets",
                "--with",
                "pandas",
                "python",
                "Scripts/run_swebench_official_patched.py",
                "--prediction-dir",
                f"Results/swebench_predictions_{RUN_NAME}",
                "--output-dir",
                f"Results/swebench_official_eval/{RUN_NAME}/per_method/{method}",
                "--run-id-prefix",
                f"{RUN_NAME}_{method}",
                "--methods",
                method,
                "--instance-ids",
                ",".join(selected_instances(method)),
                "--timeout",
                "1800",
                "--namespace",
                "none",
                "--no-offline",
            ],
            LOG_DIR / f"official_eval_{method}_finalizer.log",
            env=env,
        )
        copy_method_official_artifacts(method)
        update_status()
        if status != 0 or not official_report_complete(method):
            raise RuntimeError(f"official eval failed for {method}: status={status}")
        log(f"official eval complete for {method}")


def summarize_metrics() -> None:
    if (OFFICIAL_ROOT / "extended_summary_metrics.csv").is_file():
        log("extended metrics already complete")
        return
    log("summarizing extended metrics")
    status = run_logged(
        [
            "uv",
            "run",
            "--no-project",
            "--offline",
            "--isolated",
            "python",
            "Scripts/summarize_swebench_aligned_metrics.py",
            "--metadata",
            f"Results/swebench_predictions_{RUN_NAME}/summary.json",
            "--reports-dir",
            f"Results/swebench_official_eval/{RUN_NAME}/reports_corrected",
            "--run-logs-dir",
            f"Results/swebench_official_eval/{RUN_NAME}/run_logs",
            "--output-dir",
            f"Results/swebench_official_eval/{RUN_NAME}",
        ],
        LOG_DIR / "summarize_finalizer.log",
    )
    update_status()
    if status != 0:
        raise RuntimeError(f"extended summary failed: status={status}")
    log("extended metrics complete")


def corrected_report(root: Path, method: str) -> Path:
    candidates = [path for path in (root / "reports_corrected").glob(f"{method}.*.corrected.json") if not path.name.startswith("._")]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one corrected report for {method} under {root}, found {len(candidates)}")
    return candidates[0]


def report_ids(root: Path, method: str, key: str) -> set[str]:
    data = read_json(corrected_report(root, method))
    if not isinstance(data, dict):
        raise TypeError(f"corrected report for {method} must be an object")
    values = data.get(key, [])
    if not isinstance(values, list):
        raise TypeError(f"{key} in corrected report for {method} must be a list")
    return {str(item) for item in values}


def write_flip_analysis() -> None:
    rows: list[dict[str, object]] = []
    total_expected = 0
    total_flips = 0
    total_original_resolved = 0
    for method in METHODS:
        selected = set(selected_instances(method))
        original_resolved = report_ids(ORIGINAL_OFFICIAL_ROOT, method, "resolved_ids")
        rerun_resolved = report_ids(OFFICIAL_ROOT, method, "resolved_ids")
        rerun_empty = report_ids(OFFICIAL_ROOT, method, "empty_patch_ids")
        rerun_errors = report_ids(OFFICIAL_ROOT, method, "error_ids")
        unexpected_original_resolved = sorted(selected & original_resolved)
        flips = selected & rerun_resolved
        expected = len(selected)
        original_count = len(original_resolved)
        best_of = original_count + len(flips)
        row = {
            "method": method,
            "rerun_expected": expected,
            "flip_to_resolved": len(flips),
            "flip_rate_pct": round(len(flips) / expected * 100, 2) if expected else 0.0,
            "still_not_resolved": expected - len(flips),
            "rerun_empty_patch": len(selected & rerun_empty),
            "rerun_errors": len(selected & rerun_errors),
            "original_resolved": original_count,
            "best_of_resolved": best_of,
            "best_of_resolved_pct": round(best_of / 300 * 100, 2),
            "unexpected_original_resolved_in_selection": unexpected_original_resolved,
        }
        rows.append(row)
        total_expected += expected
        total_flips += len(flips)
        total_original_resolved += original_count

    output = {
        "run_name": RUN_NAME,
        "source_original": "swebench_lite_full_baselines300",
        "methods": rows,
        "total": {
            "rerun_expected": total_expected,
            "flip_to_resolved": total_flips,
            "flip_rate_pct": round(total_flips / total_expected * 100, 2) if total_expected else 0.0,
            "best_of_resolved_sum": total_original_resolved + total_flips,
        },
    }
    (RESULTS_ROOT / f"{RUN_NAME}_flip_analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# SWE-bench Lite Baseline Non-Resolved Rerun Flip Analysis",
        "",
        f"Updated: {now()}",
        "",
        "| method | rerun N | flip to resolved | flip % | still not resolved | rerun empty patch | rerun errors | original resolved | best-of resolved | best-of % |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['method']} | {row['rerun_expected']} | {row['flip_to_resolved']} | "
            f"{row['flip_rate_pct']:.2f} | {row['still_not_resolved']} | {row['rerun_empty_patch']} | "
            f"{row['rerun_errors']} | {row['original_resolved']} | {row['best_of_resolved']} | "
            f"{row['best_of_resolved_pct']:.2f} |"
        )
    lines.extend(
        [
            f"| **TOTAL** | {total_expected} | {total_flips} | {output['total']['flip_rate_pct']:.2f} | "
            f"{total_expected - total_flips} |  |  | {total_original_resolved} | "
            f"{total_original_resolved + total_flips} |  |",
            "",
            "Notes:",
            "",
            "- Selection is the original official non-resolved set for each baseline method.",
            "- `best-of resolved` is original full-300 resolved plus rerun flips for that same method.",
        ]
    )
    (RESULTS_ROOT / f"{RUN_NAME}_flip_analysis.md").write_text("\n".join(lines), encoding="utf-8")
    log("flip analysis complete")


def wait_for_local_complete(poll_sec: int, once: bool) -> bool:
    while True:
        counts = local_counts()
        log("local counts " + ", ".join(f"{method}={done}/{expected}" for method, (done, expected) in counts.items()))
        update_status()
        if all(done == expected for done, expected in counts.values()):
            log("local results complete")
            return True
        if once:
            return False
        time.sleep(poll_sec)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-sec", type=int, default=600)
    parser.add_argument("--once", action="store_true", help="Check once and exit before finalization if local results are incomplete.")
    args = parser.parse_args()

    log(f"finalizer start poll_sec={args.poll_sec} once={args.once}")
    if not wait_for_local_complete(args.poll_sec, args.once):
        log("local results incomplete; no finalization attempted")
        return 0
    export_predictions()
    run_official_eval()
    summarize_metrics()
    write_flip_analysis()
    update_status()
    log("finalizer complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
