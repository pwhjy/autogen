import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
EXTERNAL_DIR = BENCHMARK_DIR / "data" / "external"

SWE_DATASETS = {
    "swebench_lite": {
        "repo_id": "SWE-bench/SWE-bench_Lite",
        "files": [
            "README.md",
            "data/dev-00000-of-00001.parquet",
            "data/test-00000-of-00001.parquet",
        ],
    },
    "swebench_verified": {
        "repo_id": "SWE-bench/SWE-bench_Verified",
        "files": [
            "README.md",
            "eval.yaml",
            "data/test-00000-of-00001.parquet",
        ],
    },
}

BUGSINPY_REPO = "https://github.com/soarsmu/BugsInPy.git"


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def download_swebench() -> None:
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    for name, spec in SWE_DATASETS.items():
        target = EXTERNAL_DIR / name
        target.mkdir(parents=True, exist_ok=True)
        repo_id = str(spec["repo_id"])
        print(f"Downloading {repo_id} -> {target}")
        for filename in spec["files"]:
            local = hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=str(filename),
                local_dir=target,
            )
            print(f"  {filename} -> {local}")
        build_swebench_indexes(name, target)


def build_swebench_indexes(name: str, target: Path) -> None:
    for parquet in sorted((target / "data").glob("*.parquet")):
        split = parquet.stem.split("-")[0]
        df = pd.read_parquet(parquet)
        rows = [
            {
                "benchmark": name,
                "split": split,
                "repo": str(row["repo"]),
                "instance_id": str(row["instance_id"]),
                "base_commit": str(row["base_commit"]),
                "problem_statement": str(row["problem_statement"]),
                "hints_text": str(row.get("hints_text", "")),
                "created_at": str(row.get("created_at", "")),
                "version": str(row.get("version", "")),
                "fail_to_pass": parse_swe_list(row.get("FAIL_TO_PASS", [])),
                "pass_to_pass": parse_swe_list(row.get("PASS_TO_PASS", [])),
                "environment_setup_commit": str(
                    row.get("environment_setup_commit", "")
                ),
                "difficulty": str(row.get("difficulty", "")),
                "patch": str(row.get("patch", "")),
                "test_patch": str(row.get("test_patch", "")),
            }
            for _, row in df.iterrows()
        ]
        write_jsonl(target / f"index_{split}.jsonl", rows)
        print(f"  wrote {target / f'index_{split}.jsonl'} ({len(rows)} rows)")


def parse_swe_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [value] if value.strip() else []
    return []


def clone_or_update_bugsinpy() -> None:
    target = EXTERNAL_DIR / "BugsInPy"
    if (target / ".git").is_dir():
        print(f"Updating BugsInPy -> {target}")
        subprocess.run(["git", "-C", str(target), "pull", "--ff-only"], check=True)
    else:
        print(f"Cloning {BUGSINPY_REPO} -> {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", BUGSINPY_REPO, str(target)],
            check=True,
        )
    build_bugsinpy_index(target)


def parse_info_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped or "=" not in stripped:
                continue
            key, raw_value = stripped.split("=", 1)
            values[key.strip()] = raw_value.strip().strip('"')
    return values


def build_bugsinpy_index(root: Path) -> None:
    rows: list[dict[str, Any]] = []
    for project_dir in sorted((root / "projects").iterdir(), key=lambda path: path.name):
        bugs_dir = project_dir / "bugs"
        if not bugs_dir.is_dir():
            continue
        project_info = parse_info_file(project_dir / "project.info")
        for bug_dir in sorted(
            [path for path in bugs_dir.iterdir() if path.is_dir()],
            key=lambda path: int(path.name) if path.name.isdigit() else path.name,
        ):
            bug_info = parse_info_file(bug_dir / "bug.info")
            run_test = ""
            run_test_file = bug_dir / "run_test.sh"
            if run_test_file.is_file():
                run_test = run_test_file.read_text(
                    encoding="utf-8", errors="replace"
                ).strip()
            rows.append(
                {
                    "benchmark": "bugsinpy",
                    "project": project_dir.name,
                    "bug_id": bug_dir.name,
                    "instance_id": f"{project_dir.name}-{bug_dir.name}",
                    "github_url": bug_info.get(
                        "github_url", project_info.get("github_url", "")
                    ),
                    "python_version": bug_info.get("python_version", ""),
                    "buggy_commit_id": bug_info.get("buggy_commit_id", ""),
                    "fixed_commit_id": bug_info.get("fixed_commit_id", ""),
                    "test_file": bug_info.get("test_file", ""),
                    "status": bug_info.get("status", ""),
                    "cause": bug_info.get("cause", ""),
                    "run_test": run_test,
                    "has_requirements": (bug_dir / "requirements.txt").is_file(),
                    "has_setup": (bug_dir / "setup.sh").is_file(),
                    "bug_dir": str(bug_dir.relative_to(root)),
                }
            )
    write_jsonl(root / "index.jsonl", rows)
    print(f"  wrote {root / 'index.jsonl'} ({len(rows)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download external repair benchmarks and build local indexes."
    )
    parser.add_argument(
        "--skip-swebench",
        action="store_true",
        help="Do not download SWE-bench Lite/Verified.",
    )
    parser.add_argument(
        "--skip-bugsinpy",
        action="store_true",
        help="Do not clone/update BugsInPy.",
    )
    args = parser.parse_args()

    if not args.skip_swebench:
        download_swebench()
    if not args.skip_bugsinpy:
        clone_or_update_bugsinpy()


if __name__ == "__main__":
    main()
