"""Export SWE-bench predictions from VACTHBench repair workspaces.

The VACTHBench scenarios pre-apply the SWE-bench test patch in each workspace.
Official SWE-bench evaluation applies that test patch itself, so predictions
must contain only the agent's source changes. This script reads each result.json,
uses the corresponding workspace git diff, removes files touched by the test
patch, and writes one predictions JSONL file per method.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        value = json.load(fh)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


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
            current_path = _path_from_diff_git_line(line)
            continue
        if current:
            current.append(line)
    if current and current_path:
        chunks.append((current_path, "".join(current)))
    return chunks


def _path_from_diff_git_line(line: str) -> str:
    # Format: diff --git a/path b/path
    parts = line.strip().split(" ")
    if len(parts) >= 4 and parts[3].startswith("b/"):
        return parts[3][2:]
    return ""


def git_diff(workspace: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(workspace), "diff", "--binary", "--no-ext-diff"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout


def source_patch_for_result(result: dict[str, Any], result_dir: Path) -> str:
    workspace = result_dir / "workspace"
    if not workspace.is_dir():
        return ""

    software_task = result.get("experiment_config", {}).get("software_task", {})
    repo_source = software_task.get("repo_source", {})
    test_patch = str(repo_source.get("test_patch", ""))
    test_patch_paths = patch_paths(test_patch)

    chunks: list[str] = []
    for path, chunk in split_git_diff(git_diff(workspace)):
        if path in test_patch_paths:
            continue
        if is_test_path(path):
            continue
        chunks.append(chunk)
    return "".join(chunks).strip() + ("\n" if chunks else "")


def instance_id_for_result(result: dict[str, Any]) -> str:
    software_task = result.get("experiment_config", {}).get("software_task", {})
    repo_source = software_task.get("repo_source", {})
    metadata = software_task.get("benchmark_metadata", {})
    return str(
        repo_source.get("instance_id")
        or metadata.get("instance_id")
        or software_task.get("id", "")
    )


def method_for_result(result: dict[str, Any]) -> str:
    return str(
        result.get("method")
        or result.get("experiment_config", {}).get("method")
        or "unknown"
    )


def is_swebench_result(result: dict[str, Any]) -> bool:
    software_task = result.get("experiment_config", {}).get("software_task", {})
    category = str(software_task.get("category", ""))
    return category.startswith("swebench")


def iter_result_files(results_roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    for root in results_roots:
        if root.is_file() and root.name == "result.json":
            files.append(root)
        elif root.is_dir():
            files.extend(root.rglob("result.json"))
    return sorted(set(files))


def write_jsonl(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export official SWE-bench predictions from VACTHBench results."
    )
    parser.add_argument(
        "results_roots",
        nargs="+",
        help="Result directory/directories, e.g. Results/external_swebench_lite_autogen_broadcast",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR / "swebench_predictions"),
    )
    parser.add_argument(
        "--instances",
        default="",
        help="Comma-separated instance_id filter.",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method filter.",
    )
    args = parser.parse_args()

    roots = [Path(item).expanduser().resolve() for item in args.results_roots]
    output_dir = Path(args.output_dir).expanduser().resolve()
    instance_filter = {item for item in args.instances.split(",") if item}
    method_filter = {item for item in args.methods.split(",") if item}

    rows_by_method: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: set[tuple[str, str]] = set()
    skipped_no_patch = 0
    skipped_duplicate = 0

    for result_file in iter_result_files(roots):
        result = read_json(result_file)
        if not is_swebench_result(result):
            continue
        method = method_for_result(result)
        instance_id = instance_id_for_result(result)
        if not instance_id:
            continue
        if instance_filter and instance_id not in instance_filter:
            continue
        if method_filter and method not in method_filter:
            continue
        key = (method, instance_id)
        if key in seen:
            skipped_duplicate += 1
            continue
        seen.add(key)

        patch = source_patch_for_result(result, result_file.parent)
        if not patch.strip():
            skipped_no_patch += 1
            continue
        rows_by_method[method].append(
            {
                "instance_id": instance_id,
                "model_name_or_path": method,
                "model_patch": patch,
            }
        )

    summary: list[dict[str, Any]] = []
    for method, rows in sorted(rows_by_method.items()):
        path = output_dir / f"{method}.jsonl"
        write_jsonl(path, rows)
        summary.append(
            {
                "method": method,
                "predictions": len(rows),
                "path": str(path),
            }
        )
        print(f"wrote {path} ({len(rows)} predictions)")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "methods": summary,
                "skipped_no_patch": skipped_no_patch,
                "skipped_duplicate": skipped_duplicate,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
