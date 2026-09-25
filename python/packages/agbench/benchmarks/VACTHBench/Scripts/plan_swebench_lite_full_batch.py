"""Plan an incremental VACTH optimized batch for full SWE-bench Lite.

The planner reads the 300-instance SWE-bench Lite index, skips instances that
already have official VACTH optimized metrics, writes a VACTHBench task JSONL
for the next batch, and records exact follow-up commands for run/export/official
eval/summary. It does not run model calls by itself.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
from pathlib import Path
from typing import Any

from build_external_repair_tasks import build_swebench_task, read_jsonl, task_record, write_jsonl

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
TASKS_DIR = BENCHMARK_DIR / "Tasks"
RESULTS_DIR = BENCHMARK_DIR / "Results"
INDEX_PATH = BENCHMARK_DIR / "data" / "external" / "swebench_lite" / "index_test.jsonl"
DEFAULT_METHOD = "vacth_optimized"
COMPLETED_OFFICIAL_STATUSES = frozenset({"resolved", "unresolved", "empty_patch", "error"})
PYTHON_DIR = BENCHMARK_DIR.parents[3]
LOCAL_PACKAGE_SPECS = (
    str(PYTHON_DIR / "packages" / "agbench"),
    str(PYTHON_DIR / "packages" / "autogen-agentchat"),
    str(PYTHON_DIR / "packages" / "autogen-core"),
    f"{PYTHON_DIR / 'packages' / 'autogen-ext'}[openai]",
)


def read_index() -> list[dict[str, Any]]:
    return read_jsonl(INDEX_PATH)


def official_metric_paths() -> list[Path]:
    official_root = RESULTS_DIR / "swebench_official_eval"
    if not official_root.is_dir():
        return []
    return sorted(
        path
        for path in official_root.glob("*/extended_instance_metrics.csv")
        if not path.name.startswith("._")
    )


def completed_from_official(paths: list[Path], *, method: str) -> dict[str, str]:
    completed: dict[str, str] = {}
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("method") != method:
                    continue
                instance_id = str(row.get("instance_id", "")).strip()
                status = str(row.get("official_status", "")).strip() or "unknown"
                if instance_id and status in COMPLETED_OFFICIAL_STATUSES:
                    completed[instance_id] = status
    return completed


def completed_from_local_results(paths: list[Path], *, method: str) -> set[str]:
    completed: set[str] = set()
    suffix = f"_{method}"
    for root in paths:
        if not root.is_dir():
            continue
        for result_path in root.glob(f"**/*{suffix}/0/result.json"):
            task_dir = result_path.parent.parent.name
            if not task_dir.endswith(suffix):
                continue
            instance_id = task_dir[: -len(suffix)].split("__", 1)[-1]
            if instance_id:
                completed.add(instance_id)
    return completed


def select_rows(
    rows: list[dict[str, Any]],
    *,
    completed: set[str],
    batch_size: int,
    offset: int,
) -> list[dict[str, Any]]:
    remaining = [row for row in rows if str(row["instance_id"]) not in completed]
    return remaining[offset : offset + batch_size]


def command_block(
    *,
    benchmark_name: str,
    run_results_parent: str,
    task_file: Path,
    prediction_dir: str,
    official_dir: str,
    instance_ids: list[str],
) -> dict[str, str]:
    benchmark_prefix = f"external_{benchmark_name}"
    results_parent_path = f"Results/{run_results_parent}"
    instance_arg = ",".join(instance_ids)
    agent_uv = [
        "uv",
        "run",
        "--no-project",
        "--offline",
        "--isolated",
    ]
    for package_spec in LOCAL_PACKAGE_SPECS:
        agent_uv.extend(["--with", package_spec])
    agent_uv.extend(["python", "Scripts/run_baseline.py"])
    agent_uv_text = " ".join(shlex.quote(part) for part in agent_uv)
    return {
        "prebuild_images": (
            "mkdir -p /tmp/vacthbench_docker_config && printf '{}' > /tmp/vacthbench_docker_config/config.json && "
            "DOCKER_CONFIG=/tmp/vacthbench_docker_config "
            "uv run --no-project --isolated --with swebench --with docker --with datasets python "
            "Scripts/prebuild_swebench_images.py "
            f"--task-file {task_file.relative_to(BENCHMARK_DIR)} "
            f"--output Results/{benchmark_name}_image_prebuild_status.json "
            "--namespace none --max-workers 1"
        ),
        "run_agent_batch": (
            "AGBENCH_ALLOW_NATIVE=Yes "
            "VACTHBENCH_TASK_TIMEOUT_SEC=5400 "
            "VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE=1 "
            "VACTHBENCH_DOCKER_PULL=0 "
            "VACTHBENCH_GIT_FETCH_RETRIES=2 "
            f"{agent_uv_text} "
            f"{task_file.relative_to(BENCHMARK_DIR)} "
            f"--results-dir {run_results_parent}/{benchmark_prefix}_{DEFAULT_METHOD}"
        ),
        "export_predictions": (
            "uv run --no-project --offline --isolated python "
            "Scripts/export_swebench_predictions_aligned.py "
            f"--results-dir {results_parent_path} "
            f"--output-dir Results/{prediction_dir} "
            f"--methods {DEFAULT_METHOD} "
            f"--benchmark-prefixes {benchmark_prefix} "
            f"--task-prefixes {benchmark_prefix}"
        ),
        "official_eval": (
            "mkdir -p /tmp/vacthbench_docker_config && printf '{}' > /tmp/vacthbench_docker_config/config.json && "
            "DOCKER_CONFIG=/tmp/vacthbench_docker_config "
            "uv run --no-project --isolated --with swebench --with docker --with datasets --with pandas python "
            "Scripts/run_swebench_official_patched.py "
            f"--prediction-dir Results/{prediction_dir} "
            f"--output-dir Results/swebench_official_eval/{official_dir} "
            f"--run-id-prefix {official_dir} "
            f"--methods {DEFAULT_METHOD} "
            f"--instance-ids {instance_arg} "
            "--timeout 1800 --namespace none --no-offline"
        ),
        "summarize": (
            "uv run --no-project --offline --isolated python "
            "Scripts/summarize_swebench_aligned_metrics.py "
            f"--metadata Results/{prediction_dir}/summary.json "
            f"--reports-dir Results/swebench_official_eval/{official_dir}/reports_corrected "
            f"--run-logs-dir Results/swebench_official_eval/{official_dir}/run_logs "
            f"--output-dir Results/swebench_official_eval/{official_dir}"
        ),
    }


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-name", required=True, help="Example: swebench_lite_full_vopt_n30_b001")
    parser.add_argument("--batch-size", type=int, default=30)
    parser.add_argument("--offset", type=int, default=0, help="Offset after removing completed instances.")
    parser.add_argument("--with-setup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--completed-csv",
        action="append",
        default=[],
        help="Extra extended_instance_metrics.csv to use for skip decisions.",
    )
    parser.add_argument(
        "--skip-local-result-root",
        action="append",
        default=[],
        help="Optional local Results root to skip instances with result.json even before official eval.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing batch task/manifest even if the batch has local results.",
    )
    parser.add_argument("--print-commands", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    rows = read_index()
    official_paths = official_metric_paths() + [Path(value) for value in args.completed_csv]
    completed_status = completed_from_official(official_paths, method=DEFAULT_METHOD)
    completed = set(completed_status)
    local_completed = completed_from_local_results(
        [Path(value).expanduser().resolve() for value in args.skip_local_result_root],
        method=DEFAULT_METHOD,
    )
    completed.update(local_completed)

    selected = select_rows(rows, completed=completed, batch_size=args.batch_size, offset=args.offset)
    benchmark_name = args.batch_name
    task_file = TASKS_DIR / f"external_{benchmark_name}_{DEFAULT_METHOD}.jsonl"
    all_file = TASKS_DIR / f"external_{benchmark_name}_all.jsonl"
    manifest_path = RESULTS_DIR / f"{args.batch_name}_manifest.json"
    run_results_parent = args.batch_name
    if (
        not args.force
        and (task_file.exists() or manifest_path.exists())
        and (RESULTS_DIR / run_results_parent).exists()
    ):
        raise SystemExit(
            f"Refusing to overwrite active batch {args.batch_name}; "
            "pass --force only if you intentionally want to regenerate it."
        )
    software_tasks = [
        build_swebench_task(row, benchmark_name, with_setup=args.with_setup)
        for row in selected
    ]
    task_rows = [task_record(DEFAULT_METHOD, task) for task in software_tasks]
    write_jsonl(task_file, task_rows)
    write_jsonl(all_file, task_rows)

    instance_ids = [str(row["instance_id"]) for row in selected]
    prediction_dir = f"swebench_predictions_{args.batch_name}"
    official_dir = args.batch_name
    commands = command_block(
        benchmark_name=benchmark_name,
        run_results_parent=run_results_parent,
        task_file=task_file,
        prediction_dir=prediction_dir,
        official_dir=official_dir,
        instance_ids=instance_ids,
    )
    manifest = {
        "batch_name": args.batch_name,
        "batch_size": args.batch_size,
        "offset": args.offset,
        "method": DEFAULT_METHOD,
        "index_path": str(INDEX_PATH),
        "total_index_instances": len(rows),
        "completed_official_count": len(completed_status),
        "completed_local_count": len(local_completed),
        "selected_count": len(selected),
        "selected_instance_ids": instance_ids,
        "task_file": str(task_file),
        "all_task_file": str(all_file),
        "run_results_parent": str(RESULTS_DIR / run_results_parent),
        "prediction_dir": str(RESULTS_DIR / prediction_dir),
        "official_dir": str(RESULTS_DIR / "swebench_official_eval" / official_dir),
        "commands": commands,
    }
    write_manifest(manifest_path, manifest)

    print(f"completed official: {len(completed_status)}")
    print(f"completed local: {len(local_completed)}")
    print(f"selected: {len(selected)}")
    for index, instance_id in enumerate(instance_ids, start=1):
        print(f"{index:02d} {instance_id}")
    print(f"wrote {task_file}")
    print(f"wrote {manifest_path}")
    if args.print_commands:
        print("\nCommands:")
        for name, command in commands.items():
            print(f"\n# {name}\n{command}")


if __name__ == "__main__":
    main()
