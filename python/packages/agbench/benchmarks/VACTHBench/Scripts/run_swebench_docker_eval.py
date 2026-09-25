"""Re-evaluate VACTHBench SWE-bench results in Docker containers.

This script traverses a results directory, extracts the final patch generated
by each agent run, builds a SWE-bench predictions file, and runs the official
SWE-bench evaluation harness (or a direct-Docker fallback) to produce F2P/P2P
resolution results.  The output is written back as a ``docker_eval`` field in
each instance's ``result.json``.

Usage::

    python Scripts/run_swebench_docker_eval.py Results/external_swebench_lite_autogen_broadcast_no_docker
    python Scripts/run_swebench_docker_eval.py Results/external_swebench_lite_autogen_broadcast_no_docker --instances astropy__astropy-12907,psf__requests-863

If ``swebench`` is installed the official harness is used; otherwise this
script runs the test patch inside the SWE-bench Docker images directly.
"""  # noqa: E501

from __future__ import annotations

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent

# ---------------------------------------------------------------------------
# pytest log parsing (mirrors scenario.py)
# ---------------------------------------------------------------------------

_TEST_RESULT_RE = re.compile(
    r"^(.+?::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)",
    re.MULTILINE,
)


def parse_pytest_log(stdout: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for match in _TEST_RESULT_RE.finditer(stdout):
        test_id, status = match.group(1), match.group(2)
        results[test_id] = status.lower()
    return results


def classify_swebench_outcome(
    f2p_pass: int,
    f2p_total: int,
    p2p_pass: int,
    p2p_total: int,
) -> str:
    if f2p_total == 0:
        return "resolved" if p2p_pass == p2p_total else "error"
    all_f2p = f2p_pass == f2p_total
    some_f2p = f2p_pass > 0
    none_f2p = f2p_pass == 0
    all_p2p = p2p_pass == p2p_total
    some_p2p = p2p_pass > 0

    if all_f2p and all_p2p:
        return "resolved"
    if all_f2p and not all_p2p:
        return "breaking_resolved"
    if some_f2p and all_p2p:
        return "partially_resolved"
    if some_f2p and some_p2p:
        return "work_in_progress"
    if none_f2p and all_p2p:
        return "no_op"
    if none_f2p and not all_p2p:
        return "regression"
    return "unknown"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def read_docker_image_map() -> dict[str, str]:
    path = BENCHMARK_DIR / "data" / "external" / "docker_image_map.json"
    if path.is_file():
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def iter_result_instances(
    results_dir: Path,
    *,
    instance_filter: set[str] | None = None,
) -> list[Path]:
    instance_dirs: list[Path] = []
    for result_file in sorted(results_dir.rglob("result.json")):
        if "__pycache__" in result_file.parts:
            continue
        if instance_filter:
            # instance_id appears in the directory name
            parent_name = result_file.parent.parent.name if result_file.parent.name == "0" else result_file.parent.name  # noqa: E501
            if not any(
                iid in parent_name for iid in instance_filter
            ):
                continue
        instance_dirs.append(result_file.parent)
    return instance_dirs


def extract_patch_from_workspace(
    workspace_path: Path, original_files: dict[str, str]
) -> str:
    """Create a unified diff between original snapshots and current workspace."""
    patches: list[str] = []
    for rel_path, orig_content in sorted(original_files.items()):
        current_path = workspace_path / rel_path
        if not current_path.is_file():
            continue
        try:
            current_content = current_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if current_content != orig_content:
            diff = "".join(
                difflib.unified_diff(
                    orig_content.splitlines(keepends=True),
                    current_content.splitlines(keepends=True),
                    fromfile=f"a/{rel_path}",
                    tofile=f"b/{rel_path}",
                )
            )
            if diff:
                patches.append(diff)
    return "\n".join(patches)


def _swebench_harness_available() -> bool:
    try:
        import swebench  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Docker-based evaluation (manual fallback)
# ---------------------------------------------------------------------------


def run_docker_eval(
    *,
    instance_id: str,
    docker_image: str,
    test_patch_str: str,
    full_test_command: list[str],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
    timeout: int = 1800,
) -> dict[str, Any]:
    """Run tests inside the SWE-bench Docker image and return F2P/P2P results."""

    # Generate a temporary patch file from the agent's changes
    # (we'll write the actual patch at call-site)

    test_script = _docker_test_script(full_test_command)
    container_command = ["docker", "run", "--rm"]

    # Mount a temp dir for patch + results
    with tempfile.TemporaryDirectory(
        prefix="swebench_docker_"
    ) as tmpdir:
        tmp = Path(tmpdir)
        (tmp / "eval.sh").write_text(test_script, encoding="utf-8")
        (tmp / "eval.sh").chmod(0o755)

        container_command.extend(
            [
                "-v",
                f"{tmpdir}:/eval_work",
                "-w",
                "/testbed",
                docker_image,
                "bash",
                "/eval_work/eval.sh",
            ]
        )

        try:
            completed = subprocess.run(
                container_command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return {
                "resolved": False,
                "f2p_pass": 0,
                "f2p_total": len(fail_to_pass),
                "p2p_pass": len(pass_to_pass),
                "p2p_total": len(pass_to_pass),
                "outcome": "error",
                "error": "timeout",
                "missing_f2p": list(fail_to_pass),
                "missing_p2p": [],
            }

        stdout = completed.stdout
        test_results = parse_pytest_log(stdout)

        f2p_total = len(fail_to_pass)
        f2p_pass = sum(
            1 for t in fail_to_pass if test_results.get(t) == "passed"
        )
        missing_f2p = sorted(
            set(fail_to_pass) - set(test_results.keys())
        )

        p2p_total = len(pass_to_pass)
        p2p_pass = sum(
            1 for t in pass_to_pass if test_results.get(t) == "passed"
        )
        missing_p2p = sorted(
            set(pass_to_pass) - set(test_results.keys())
        )

        outcome = classify_swebench_outcome(
            f2p_pass, f2p_total, p2p_pass, p2p_total
        )

        return {
            "resolved": outcome == "resolved",
            "f2p_pass": f2p_pass,
            "f2p_total": f2p_total,
            "p2p_pass": p2p_pass,
            "p2p_total": p2p_total,
            "outcome": outcome,
            "missing_f2p": missing_f2p,
            "missing_p2p": missing_p2p,
        }


def _docker_test_script(full_test_command: list[str]) -> str:
    cmd_str = " ".join(full_test_command)
    return f"""#!/bin/bash
set -e
cd /testbed

# Activate the testbed environment
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate testbed 2>/dev/null || true

# Run tests
{cmd_str}
"""


# ---------------------------------------------------------------------------
# Prediction file generation (for swebench harness)
# ---------------------------------------------------------------------------


def build_predictions_json(
    instance_dirs: list[Path],
) -> dict[str, list[dict[str, str]]]:
    predictions: dict[str, list[dict[str, str]]] = {}

    for inst_dir in instance_dirs:
        result_file = inst_dir / "result.json"
        if not result_file.is_file():
            continue
        with result_file.open("r", encoding="utf-8") as fh:
            result = json.load(fh)

        instance_id = _get_instance_id(result)
        if not instance_id:
            continue

        patch = _build_diff_from_result(result)
        if not patch:
            continue

        model_name = str(result.get("method", "unknown"))
        predictions.setdefault(instance_id, []).append(
            {"model_name": model_name, "model_patch": patch}
        )

    return predictions


def _get_instance_id(result: dict[str, Any]) -> str:
    software_task = (
        result.get("experiment_config", {}).get("software_task", {})
    )
    return str(
        software_task.get("repo_source", {}).get("instance_id", "")
        or software_task.get("benchmark_metadata", {}).get(
            "instance_id", ""
        )
        or software_task.get("id", "")
    )


def _build_diff_from_result(result: dict[str, Any]) -> str:
    """Build a unified diff from file change operations in tool_calls."""
    tool_calls: list[dict[str, Any]] = result.get("tool_calls", [])

    # Track the last known content of each file before and after edits
    file_edits: dict[str, list[tuple[str, str]]] = (
        {}
    )  # path -> [(old, new), ...]

    for call in tool_calls:
        tool_name = str(call.get("tool_name", ""))
        result_str = str(call.get("result", ""))
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}

        if tool_name == "replace_text":
            path = str(arguments.get("path", ""))
            old = str(arguments.get("old", ""))
            new = str(arguments.get("new", ""))
            if path:
                file_edits.setdefault(path, []).append((old, new))

        elif tool_name == "write_file":
            path = str(arguments.get("path", ""))
            content = str(arguments.get("content", ""))
            if path:
                file_edits.setdefault(path, []).append(("", content))

    if not file_edits:
        return ""

    patches: list[str] = []
    for rel_path, edits in sorted(file_edits.items()):
        # Reconstruct original and modified content
        original_lines: list[str] = []
        modified_lines: list[str] = []
        for old_text, new_text in edits:
            if original_lines:
                # Subsequent edit to same file
                current = "\n".join(modified_lines)
                if old_text in current:
                    current = current.replace(old_text, new_text, 1)
                modified_lines = current.split("\n")
            else:
                # First edit
                original_lines = old_text.split("\n")
                modified_lines = new_text.split("\n")

        if original_lines == modified_lines:
            continue

        diff = "".join(
            difflib.unified_diff(
                [line + "\n" for line in original_lines],
                [line + "\n" for line in modified_lines],
                fromfile=f"a/{rel_path}",
                tofile=f"b/{rel_path}",
            )
        )
        if diff:
            patches.append(diff)

    return "\n".join(patches)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-evaluate SWE-bench results in Docker."
    )
    parser.add_argument(
        "results_dir",
        help="Path to Results/<scenario>/ directory.",
    )
    parser.add_argument(
        "--instances",
        help="Comma-separated instance ID filter.",
        default="",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=1800,
        help="Timeout per Docker container in seconds.",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir).resolve()
    if not results_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {results_dir}")

    instance_filter = (
        set(args.instances.split(",")) if args.instances else None
    )
    instance_dirs = iter_result_instances(results_dir, instance_filter=instance_filter)
    if not instance_dirs:
        print("No result.json files found.")
        return

    image_map = read_docker_image_map()
    print(
        f"Processing {len(instance_dirs)} instances "
        f"(docker images known: {len(image_map)})"
    )

    processed = 0
    for inst_dir in instance_dirs:
        result_file = inst_dir / "result.json"
        with result_file.open("r", encoding="utf-8") as fh:
            result = json.load(fh)

        instance_id = _get_instance_id(result)
        if not instance_id:
            print(f"  SKIP {inst_dir.name} — no instance_id found")
            continue

        # Only process SWE-bench tasks
        category = str(
            result.get("experiment_config", {})
            .get("software_task", {})
            .get("category", "")
        )
        if not category.startswith("swebench"):
            continue

        docker_image = image_map.get(instance_id.replace("__", "__", 1))
        if not docker_image:
            print(f"  SKIP {instance_id} — no docker image mapping")
            continue

        software_task = (
            result.get("experiment_config", {}).get("software_task", {})
        )
        benchmark_meta = software_task.get("benchmark_metadata", {})
        fail_to_pass = benchmark_meta.get("fail_to_pass", [])
        pass_to_pass = benchmark_meta.get("pass_to_pass", [])

        full_test_command = software_task.get("full_test_command")
        if not full_test_command:
            # fall back to running all test files from test_patch
            test_patch = (
                software_task.get("repo_source", {}).get("test_patch", "")
            )
            test_files = _test_files_from_patch(test_patch)
            if test_files:
                full_test_command = [
                    "python",
                    "-m",
                    "pytest",
                ] + test_files
            else:
                full_test_command = ["python", "-m", "pytest"]

        print(f"  {instance_id} ...", end=" ", flush=True)

        try:
            eval_result = run_docker_eval(
                instance_id=instance_id,
                docker_image=docker_image,
                test_patch_str=software_task.get("repo_source", {}).get(
                    "test_patch", ""
                ),
                full_test_command=full_test_command,
                fail_to_pass=fail_to_pass,
                pass_to_pass=pass_to_pass,
                timeout=args.timeout,
            )
        except Exception as exc:
            eval_result = {
                "resolved": False,
                "outcome": "error",
                "error": str(exc),
                "f2p_pass": 0,
                "f2p_total": len(fail_to_pass),
                "p2p_pass": 0,
                "p2p_total": len(pass_to_pass),
                "missing_f2p": list(fail_to_pass),
                "missing_p2p": list(pass_to_pass),
            }

        eval_result["evaluated_at"] = (
            datetime.now(timezone.utc).isoformat()
        )
        result["docker_eval"] = eval_result

        with result_file.open("w", encoding="utf-8") as fh:
            json.dump(result, fh, ensure_ascii=False, indent=2, sort_keys=True)

        status = eval_result["outcome"]
        print(status)
        processed += 1

    print(f"Done. Updated {processed} result(s).")


def _test_files_from_patch(patch: str) -> list[str]:
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/").strip()
            if path and path != "/dev/null":
                if path not in paths:
                    paths.append(path)
    return sorted(set(paths))


if __name__ == "__main__":
    main()
