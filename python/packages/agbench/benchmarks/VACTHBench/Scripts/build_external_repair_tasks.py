from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from init_tasks import (
    COMMON_TEMPLATE,
    SOFTWARE_METHODS,
    TEMPLATES_DIR,
    VACTH_METHODS,
    _vacth_flags,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
TASKS_DIR = BENCHMARK_DIR / "Tasks"
EXTERNAL_DIR = BENCHMARK_DIR / "data" / "external"

# SWE-bench Docker images give deterministic checkouts and avoid GitHub fetch
# failures during repeated local experiments.
# Image naming follows the official SWE-bench harness when --namespace none is
# used: sweb.eval.x86_64.{org}__{repo_name}-{issue_number}:latest.
_DOCKER_FALLBACK_REPOS: set[str] = {
    "astropy/astropy",
    "django/django",
    "matplotlib/matplotlib",
    "mwaskom/seaborn",
    "pallets/flask",
    "psf/requests",
    "pytest-dev/pytest",
    "pydata/xarray",
    "pylint-dev/pylint",
    "scikit-learn/scikit-learn",
    "sphinx-doc/sphinx",
    "sympy/sympy",
}


def _docker_image_name(repo: str, instance_id: str) -> str:
    org, repo_name = repo.split("/")
    issue_number = instance_id.rsplit("-", 1)[-1]
    return f"sweb.eval.x86_64.{org}__{repo_name}-{issue_number}"


def _django_test_label(label: str) -> str:
    if " (" not in label or not label.endswith(")"):
        return label
    method_name, _, class_part = label.partition(" (")
    class_path = class_part[:-1]
    return f"{class_path}.{method_name}"


def _pytest_command_for_labels(labels: list[str], test_files: list[str]) -> list[str]:
    base = ["{python}", "-m", "pytest", "-vv"]
    if not labels:
        return base
    if all(".py" in label or "/" in label for label in labels):
        return base + labels
    if test_files:
        names = [label.split("[", 1)[0] for label in labels]
        return base + test_files + ["-k", " or ".join(names)]
    return base + labels


def _swebench_test_command(
    row: dict[str, Any],
    labels: list[str],
    *,
    test_files: list[str],
) -> list[str]:
    if row["repo"] == "django/django":
        return [
            "{python}",
            "tests/runtests.py",
            "--verbosity",
            "2",
            *[_django_test_label(label) for label in labels],
        ]
    return _pytest_command_for_labels(labels, test_files)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            rows.append(value)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def method_config(method: str, task_id: str, software_task: dict[str, Any]) -> dict[str, Any]:
    is_vacth = method in VACTH_METHODS
    vacth_flags = _vacth_flags(method) if is_vacth else {}
    return {
        "task_id": task_id,
        "template": "software_repair",
        "method": method,
        "model": "config.yaml",
        "max_turns": int(software_task.get("max_turns", 10)),
        "max_total_tokens": 80000,
        "max_context_tokens_per_agent": 10000,
        "context_window_messages": 4 if method == "sliding_window" else None,
        "memory_top_k": 5 if method == "vector_memory" else None,
        "structured_max_items_per_slot": 16 if method == "structured_summary" else None,
        "capsule_token_budget": 1200 if is_vacth else None,
        "seed": 7,
        "token_budget": 10000,
        "software_task": software_task,
        "software_task_id": str(software_task["id"]),
        "software_task_category": str(software_task.get("category", "external")),
        **vacth_flags,
    }


def task_record(method: str, software_task: dict[str, Any]) -> dict[str, Any]:
    task_id = f"external_{software_task['id']}_{method}"
    config = method_config(method, task_id, software_task)
    return {
        "id": task_id,
        "template": [COMMON_TEMPLATE, str(TEMPLATES_DIR / "software_repair")],
        "substitutions": {
            "scenario.py": {
                "__EXPERIMENT_CONFIG_JSON__": json.dumps(
                    config, ensure_ascii=False, sort_keys=True
                ),
            },
            "prompt.txt": {
                "__TASK__": str(software_task["task"]),
                "__CONSTRAINTS__": str(software_task["constraints"]),
                "__GOLD_STATE__": str(software_task["gold_state"]),
            },
        },
    }


def build_swebench_task(row: dict[str, Any], benchmark: str, *, with_setup: bool) -> dict[str, Any]:
    fail_to_pass = [str(item) for item in row.get("fail_to_pass", [])]
    instance_id = str(row["instance_id"])
    task_text = (
        f"Fix SWE-bench instance {instance_id} from {row['repo']}.\n\n"
        f"Issue:\n{row['problem_statement']}\n\n"
        "Use repository inspection and tests to produce a minimal source repair."
    )
    constraints = (
        "Do not modify tests except the pre-applied SWE-bench test patch. "
        "Prefer minimal source changes. Use run_tests to run the selected "
        "FAIL_TO_PASS tests."
    )
    setup_commands: list[str] = []
    if with_setup and row["repo"] not in _DOCKER_FALLBACK_REPOS:
        setup_commands.append("{python} -m pip install -e .")

    # Full test suite for final evaluation: run all test files from test_patch (F2P+P2P)
    test_patch = str(row.get("test_patch", ""))
    test_files = _test_file_paths_from_patch(test_patch)
    pass_to_pass = [str(item) for item in row.get("pass_to_pass", [])]
    test_command = _swebench_test_command(row, fail_to_pass, test_files=test_files)
    full_test_command = _swebench_test_command(
        row,
        fail_to_pass + pass_to_pass,
        test_files=test_files,
    )

    return {
        "id": f"{benchmark}__{instance_id}",
        "category": benchmark,
        "task": task_text,
        "constraints": constraints,
        "gold_state": (
            "The selected SWE-bench FAIL_TO_PASS tests pass after the repair, "
            "without changing source-irrelevant test files."
        ),
        "repo_files": {},
        "repo_source": {
            "type": "swebench",
            "repo": row["repo"],
            "base_commit": row["base_commit"],
            "instance_id": instance_id,
            "apply_test_patch": True,
            "git_timeout_sec": 600,
            "test_patch": row.get("test_patch", ""),
            **(
                {"docker_image": _docker_image_name(row["repo"], instance_id)}
                if row["repo"] in _DOCKER_FALLBACK_REPOS
                else {}
            ),
        },
        "test_command": test_command,
        "full_test_command": full_test_command,
        "setup_commands": setup_commands,
        "expected_changed_files": patch_paths(row.get("patch", "")),
        "forbid_test_changes": True,
        "require_expected_changes": False,
        "reviewer_success_criteria": (
            "The selected SWE-bench tests pass, the fix is source-level and "
            "minimal, and no additional test files are modified."
        ),
        "benchmark_metadata": {
            "repo": row["repo"],
            "instance_id": instance_id,
            "fail_to_pass": fail_to_pass,
            "pass_to_pass": pass_to_pass,
            "difficulty": row.get("difficulty", ""),
        },
    }


def _test_file_paths_from_patch(patch: str) -> list[str]:
    """Extract unique test file paths referenced in a unified diff."""
    paths: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/").strip()
            if path and path != "/dev/null":
                trimmed = path.strip()
                if trimmed not in paths:
                    paths.append(trimmed)
    return sorted(set(paths))


def patch_paths(patch: Any) -> list[str]:
    paths: list[str] = []
    for line in str(patch or "").splitlines():
        if line.startswith("+++ b/"):
            path = line.removeprefix("+++ b/").strip()
            if path and path != "/dev/null":
                paths.append(path)
    return sorted(set(paths))


def build_bugsinpy_task(row: dict[str, Any], *, compile_project: bool) -> dict[str, Any]:
    project = str(row["project"])
    bug_id = str(row["bug_id"])
    run_test = str(row.get("run_test", "")).strip()
    task_text = (
        f"Fix BugsInPy bug {project} #{bug_id}.\n\n"
        f"Relevant test command from BugsInPy:\n{run_test or '(use bugsinpy-test)'}\n\n"
        "Use repository inspection and tests to produce a minimal source repair."
    )
    constraints = (
        "Do not modify test files. Preserve public APIs. Use run_tests for the "
        "BugsInPy relevant tests."
    )
    return {
        "id": f"bugsinpy__{project}__{bug_id}",
        "category": "bugsinpy",
        "task": task_text,
        "constraints": constraints,
        "gold_state": "The BugsInPy relevant test command passes after the repair.",
        "repo_files": {},
        "repo_source": {
            "type": "bugsinpy",
            "root": str((EXTERNAL_DIR / "BugsInPy").resolve()),
            "project": project,
            "bug_id": bug_id,
            "version": 0,
            "compile": compile_project,
            "compile_mode": "native_venv",
        },
        "test_command": [
            "bash",
            "-lc",
            "source env/bin/activate && bash bugsinpy_run_test.sh",
        ],
        "expected_changed_files": [],
        "forbid_test_changes": True,
        "require_expected_changes": False,
        "reviewer_success_criteria": (
            "The BugsInPy relevant tests pass and no test files are modified."
        ),
        "benchmark_metadata": {
            "project": project,
            "bug_id": bug_id,
            "github_url": row.get("github_url", ""),
            "python_version": row.get("python_version", ""),
            "buggy_commit_id": row.get("buggy_commit_id", ""),
            "fixed_commit_id": row.get("fixed_commit_id", ""),
            "test_file": row.get("test_file", ""),
            "run_test": run_test,
        },
    }


def selected_swebench_tasks(
    benchmark: str,
    split: str,
    limit: int,
    *,
    with_setup: bool,
    repos: set[str] | None,
    instance_ids: set[str] | None,
    task_benchmark_name: str | None = None,
) -> list[dict[str, Any]]:
    index = EXTERNAL_DIR / benchmark / f"index_{split}.jsonl"
    rows = read_jsonl(index)
    if repos:
        rows = [row for row in rows if str(row.get("repo")) in repos]
    if instance_ids:
        rows = [
            row
            for row in rows
            if str(row.get("instance_id")) in instance_ids
        ]
    return [
        build_swebench_task(row, task_benchmark_name or benchmark, with_setup=with_setup)
        for row in rows[:limit]
    ]


def selected_bugsinpy_tasks(limit: int, *, compile_project: bool) -> list[dict[str, Any]]:
    index = EXTERNAL_DIR / "BugsInPy" / "index.jsonl"
    rows = [
        row
        for row in read_jsonl(index)
        if str(row.get("status", "")).upper() in {"", "OK"}
    ]
    preferred = [
        row
        for row in rows
        if row.get("project") in {"PySnooper", "tqdm", "cookiecutter", "black"}
    ]
    selected = (preferred or rows)[:limit]
    return [
        build_bugsinpy_task(row, compile_project=compile_project)
        for row in selected
    ]


def write_method_files(
    *, benchmark_name: str, software_tasks: list[dict[str, Any]], methods: list[str]
) -> None:
    combined: list[dict[str, Any]] = []
    for method in methods:
        records = [task_record(method, task) for task in software_tasks]
        write_jsonl(TASKS_DIR / f"external_{benchmark_name}_{method}.jsonl", records)
        combined.extend(records)
        print(
            f"wrote {TASKS_DIR / f'external_{benchmark_name}_{method}.jsonl'} "
            f"({len(records)} records)"
        )
    write_jsonl(TASKS_DIR / f"external_{benchmark_name}_all.jsonl", combined)
    print(
        f"wrote {TASKS_DIR / f'external_{benchmark_name}_all.jsonl'} "
        f"({len(combined)} records)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build VACTHBench task JSONL files from external benchmarks."
    )
    parser.add_argument(
        "--benchmark",
        choices=["swebench_lite", "swebench_verified", "bugsinpy"],
        required=True,
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument(
        "--repo",
        action="append",
        default=[],
        help="SWE-bench repo filter, e.g. --repo psf/requests. Can repeat.",
    )
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="SWE-bench instance_id filter. Can repeat.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=SOFTWARE_METHODS,
        help="Methods to generate; defaults to all E1-E6 methods.",
    )
    parser.add_argument(
        "--output-benchmark-name",
        default="",
        help=(
            "Override the benchmark prefix used in generated task ids/files. "
            "Useful for non-overlapping SWE-bench batches."
        ),
    )
    parser.add_argument(
        "--with-setup",
        action="store_true",
        help="For SWE-bench, add a generic `pip install -e .` setup command.",
    )
    parser.add_argument(
        "--compile-bugsinpy",
        action="store_true",
        help="Run BugsInPy compile during scenario setup.",
    )
    args = parser.parse_args()
    output_benchmark_name = args.output_benchmark_name or args.benchmark

    if args.benchmark.startswith("swebench"):
        tasks = selected_swebench_tasks(
            args.benchmark,
            args.split,
            args.limit,
            with_setup=args.with_setup,
            repos=set(args.repo) or None,
            instance_ids=set(args.instance_id) or None,
            task_benchmark_name=output_benchmark_name,
        )
    else:
        tasks = selected_bugsinpy_tasks(
            args.limit,
            compile_project=args.compile_bugsinpy,
        )
    write_method_files(
        benchmark_name=output_benchmark_name,
        software_tasks=tasks,
        methods=args.methods,
    )


if __name__ == "__main__":
    main()
