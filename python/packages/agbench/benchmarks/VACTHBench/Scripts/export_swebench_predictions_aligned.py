"""Export aligned SWE-bench predictions from VACTHBench run directories.

The older exporter only emitted rows for runs with a completed result.json and
a non-empty source patch. That is useful for debugging patches, but it biases a
method comparison because failed/no-patch runs disappear from the denominator.

This exporter scans the run directories directly, writes every discovered
method/instance pair, strips pre-applied SWE-bench test patches, and represents
no-patch runs with an empty ``model_patch`` so the official harness can count
them as unsolved/empty-patch cases.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"

DEFAULT_METHODS = (
    "autogen_broadcast",
    "sliding_window",
    "structured_summary",
    "summary",
    "vacth_full",
    "vacth_optimized",
    "vector_memory",
)

PATCH_CACHE_FILE = "source_patch_cache.json"
PATCH_CACHE_VERSION = 1


@dataclass(frozen=True)
class RunRecord:
    method: str
    instance_id: str
    run_dir: Path
    result_path: Path | None
    workspace: Path | None
    model_patch: str
    status: str


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def read_source_patch_cache(run_dir: Path) -> tuple[str, str] | None:
    cache_path = run_dir / PATCH_CACHE_FILE
    if not cache_path.is_file():
        return None
    try:
        cache = read_json(cache_path)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if cache.get("cache_version") != PATCH_CACHE_VERSION:
        return None
    patch = cache.get("model_patch")
    status = cache.get("status")
    if not isinstance(patch, str) or not isinstance(status, str):
        return None
    return patch, status


def write_source_patch_cache(run_dir: Path, patch: str, status: str) -> None:
    cache_path = run_dir / PATCH_CACHE_FILE
    payload = {
        "cache_version": PATCH_CACHE_VERSION,
        "model_patch": patch,
        "patch_bytes": len(patch.encode("utf-8")),
        "status": status,
    }
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def patch_paths(patch: str) -> set[str]:
    paths: set[str] = set()
    for line in patch.splitlines():
        if line.startswith("--- a/") or line.startswith("+++ b/"):
            path = line[6:].strip()
            if path and path != "/dev/null":
                paths.add(path)
    return paths


def is_test_path(path: str) -> bool:
    parts = Path(path).parts
    name = Path(path).name
    return (
        "tests" in parts
        or "testing" in parts
        or name.startswith("test_")
        or name.endswith("_test.py")
    )


def split_git_diff(diff_text: str) -> list[tuple[str, str]]:
    chunks: list[tuple[str, str]] = []
    current: list[str] = []
    current_path = ""
    for line in diff_text.splitlines(keepends=True):
        if line.startswith("diff --git "):
            if current and current_path:
                chunks.append((current_path, "".join(current)))
            current = [line]
            current_path = path_from_diff_git_line(line)
            continue
        if current:
            current.append(line)
    if current and current_path:
        chunks.append((current_path, "".join(current)))
    return chunks


def path_from_diff_git_line(line: str) -> str:
    match = re.match(r"diff --git a/(.*?) b/(.*)$", line.strip())
    if not match:
        return ""
    return match.group(2)


def git_diff(workspace: Path) -> str:
    pack_dir = workspace / ".git" / "objects" / "pack"
    if pack_dir.is_dir():
        for sidecar in pack_dir.glob("._*"):
            sidecar.unlink(missing_ok=True)
    completed = subprocess.run(
        ["git", "-c", "core.fileMode=false", "-C", str(workspace), "diff", "--binary", "--no-ext-diff"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def test_patch_for_run(run_dir: Path, result_path: Path | None) -> str:
    patch_file = run_dir / "swebench_test_patch.diff"
    if patch_file.is_file():
        return patch_file.read_text(encoding="utf-8", errors="replace")
    if result_path is not None:
        result = read_json(result_path)
        software_task = result.get("experiment_config", {}).get("software_task", {})
        repo_source = software_task.get("repo_source", {})
        return str(repo_source.get("test_patch", ""))
    return ""


def source_patch_for_run(run_dir: Path, result_path: Path | None) -> tuple[str, str]:
    workspace = run_dir / "workspace"
    if not workspace.is_dir():
        cached = read_source_patch_cache(run_dir)
        if cached is not None:
            patch, status = cached
            return patch, f"{status}_cached"
        return "", "missing_workspace"

    test_patch_paths = patch_paths(test_patch_for_run(run_dir, result_path))
    chunks: list[str] = []
    for path, chunk in split_git_diff(git_diff(workspace)):
        if path in test_patch_paths:
            continue
        if is_test_path(path):
            continue
        chunks.append(chunk)

    patch = "".join(chunks).strip()
    if not patch:
        if result_path is None:
            return "", "agent_error_no_result_empty_patch"
        status = "empty_source_patch"
        write_source_patch_cache(run_dir, "", status)
        return "", status
    if result_path is None:
        return patch + "\n", "patch_from_incomplete_run"
    patch = patch + "\n"
    status = "patch_from_result_run"
    write_source_patch_cache(run_dir, patch, status)
    return patch, status


def parsed_dir_method(
    name: str,
    methods: tuple[str, ...],
    benchmark_prefix: str,
) -> tuple[str, str] | None:
    prefix = f"{benchmark_prefix}__"
    if not name.startswith(prefix):
        return None

    matched_method = ""
    for candidate in methods:
        suffix = f"_{candidate}"
        if name.endswith(suffix) and len(candidate) > len(matched_method):
            matched_method = candidate
    if not matched_method:
        return None
    suffix = f"_{matched_method}"
    return name[len(prefix) : -len(suffix)], matched_method


def instance_id_from_dir(
    name: str,
    method: str,
    benchmark_prefix: str,
    methods: tuple[str, ...],
) -> str | None:
    parsed = parsed_dir_method(name, methods, benchmark_prefix)
    if parsed is None:
        return None
    instance_id, parsed_method = parsed
    if parsed_method != method:
        return None
    return instance_id


def iter_runs(
    results_dir: Path,
    methods: tuple[str, ...],
    benchmark_prefixes: tuple[str, ...],
    task_prefixes: tuple[str, ...] | None = None,
) -> list[RunRecord]:
    runs: list[RunRecord] = []
    if task_prefixes is None:
        task_prefixes = benchmark_prefixes
    if len(task_prefixes) != len(benchmark_prefixes):
        raise ValueError("--task-prefixes must have the same item count as --benchmark-prefixes")
    for method in methods:
        for benchmark_prefix, task_prefix in zip(benchmark_prefixes, task_prefixes):
            roots: list[Path] = []
            nested_root = results_dir / f"{benchmark_prefix}_{method}"
            if nested_root.is_dir():
                roots.append(nested_root)
            if results_dir.is_dir() and any(
                instance_id_from_dir(path.name, method, task_prefix, methods)
                for path in results_dir.iterdir()
                if path.is_dir()
            ):
                roots.append(results_dir)
            for root in dict.fromkeys(roots):
                for task_dir in sorted(path for path in root.iterdir() if path.is_dir()):
                    instance_id = instance_id_from_dir(task_dir.name, method, task_prefix, methods)
                    if not instance_id:
                        continue
                    run_dir = task_dir / "0"
                    result_path = run_dir / "result.json"
                    result_path_or_none = result_path if result_path.is_file() else None
                    workspace = run_dir / "workspace"
                    patch, status = source_patch_for_run(run_dir, result_path_or_none)
                    runs.append(
                        RunRecord(
                            method=method,
                            instance_id=instance_id,
                            run_dir=run_dir,
                            result_path=result_path_or_none,
                            workspace=workspace if workspace.is_dir() else None,
                            model_patch=patch,
                            status=status,
                        )
                    )
    return runs


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export aligned official SWE-bench predictions from VACTHBench runs."
    )
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "swebench_predictions_aligned6"),
    )
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated methods to export.",
    )
    parser.add_argument(
        "--benchmark-prefixes",
        default="external_swebench_lite",
        help=(
            "Comma-separated result/task prefixes to scan, for example "
            "'external_swebench_lite,external_swebench_lite_extra2'."
        ),
    )
    parser.add_argument(
        "--task-prefixes",
        default="",
        help=(
            "Optional comma-separated prefixes used inside task directory names. "
            "Defaults to --benchmark-prefixes."
        ),
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    methods = tuple(item.strip() for item in args.methods.split(",") if item.strip())
    benchmark_prefixes = tuple(item.strip() for item in args.benchmark_prefixes.split(",") if item.strip())
    task_prefixes = (
        tuple(item.strip() for item in args.task_prefixes.split(",") if item.strip())
        if args.task_prefixes
        else None
    )

    runs = iter_runs(results_dir, methods, benchmark_prefixes, task_prefixes)
    by_method: dict[str, list[RunRecord]] = {method: [] for method in methods}
    for run in runs:
        by_method.setdefault(run.method, []).append(run)

    metadata: list[dict[str, Any]] = []
    for method in methods:
        rows: list[dict[str, str]] = []
        for run in sorted(by_method.get(method, []), key=lambda item: item.instance_id):
            rows.append(
                {
                    "instance_id": run.instance_id,
                    "model_name_or_path": method,
                    "model_patch": run.model_patch,
                }
            )
            metadata.append(
                {
                    "method": run.method,
                    "instance_id": run.instance_id,
                    "status": run.status,
                    "has_result": run.result_path is not None,
                    "has_workspace": run.workspace is not None,
                    "patch_bytes": len(run.model_patch.encode("utf-8")),
                    "run_dir": str(run.run_dir),
                    "result_path": str(run.result_path) if run.result_path else "",
                }
            )
        write_jsonl(output_dir / f"{method}.jsonl", rows)
        print(f"wrote {output_dir / f'{method}.jsonl'} ({len(rows)} predictions)")

    summary = {
        "methods": [
            {
                "method": method,
                "predictions": len(by_method.get(method, [])),
                "non_empty_patches": sum(1 for run in by_method.get(method, []) if run.model_patch.strip()),
                "empty_patches": sum(1 for run in by_method.get(method, []) if not run.model_patch.strip()),
            }
            for method in methods
        ],
        "runs": sorted(metadata, key=lambda item: (item["method"], item["instance_id"])),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
