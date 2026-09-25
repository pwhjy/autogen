"""Preflight checks for the incremental SWE-bench Lite VACTH run."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from clean_macos_sidecars import DEFAULT_TARGETS, clean_path

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
INDEX_PATH = BENCHMARK_DIR / "data" / "external" / "swebench_lite" / "index_test.jsonl"
PYTHON_DIR = BENCHMARK_DIR.parents[3]
LOCAL_PACKAGE_SPECS = (
    str(PYTHON_DIR / "packages" / "agbench"),
    str(PYTHON_DIR / "packages" / "autogen-agentchat"),
    str(PYTHON_DIR / "packages" / "autogen-core"),
    f"{PYTHON_DIR / 'packages' / 'autogen-ext'}[openai]",
)


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    command: list[str] = field(default_factory=list)


def run_command(command: list[str], *, timeout: int = 60) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, str(exc)
    output = "\n".join(part for part in (completed.stdout.strip(), completed.stderr.strip()) if part)
    return completed.returncode == 0, output[-4000:]


def check_docker() -> list[Check]:
    checks: list[Check] = []
    command = ["docker", "version", "--format", "{{.Client.Version}} / {{.Server.Version}}"]
    ok, detail = run_command(command)
    checks.append(Check("docker_version", ok, detail, command))

    command = ["docker", "info", "--format", "{{.OSType}}/{{.Architecture}} {{.Name}}"]
    ok, detail = run_command(command)
    checks.append(Check("docker_info", ok, detail, command))

    command = ["docker", "image", "inspect", "sweb.base.py.x86_64:latest"]
    ok, detail = run_command(command)
    checks.append(
        Check(
            "swebench_base_image",
            True,
            "present" if ok else f"missing; official eval with --namespace none can build it ({detail})",
            command,
        )
    )
    if ok:
        command = ["docker", "run", "--rm", "sweb.base.py.x86_64:latest", "python", "--version"]
        ok, detail = run_command(command, timeout=120)
        checks.append(Check("swebench_base_container", ok, detail, command))
    return checks


def check_uv_swebench(*, offline: bool) -> Check:
    command = [
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
        "-c",
        (
            "import docker, swebench, datasets, pandas; "
            "print('docker_ping', docker.from_env().ping()); "
            "print('swebench', bool(swebench)); "
            "print('datasets', datasets.__version__); "
            "print('pandas', pandas.__version__)"
        ),
    ]
    if offline:
        command.insert(3, "--offline")
    ok, detail = run_command(command, timeout=180)
    return Check("uv_swebench_runtime", ok, detail, command)


def check_agent_runtime(*, offline: bool) -> Check:
    command = [
        "uv",
        "run",
        "--no-project",
        "--isolated",
    ]
    if offline:
        command.append("--offline")
    for package_spec in LOCAL_PACKAGE_SPECS:
        command.extend(["--with", package_spec])
    command.extend(
        [
            "python",
            "-c",
            (
                "import agbench, autogen_agentchat, autogen_core, autogen_ext, tiktoken; "
                "from autogen_ext.models.openai import OpenAIChatCompletionClient; "
                "print('agent_runtime_imports_ok', OpenAIChatCompletionClient.__name__, tiktoken.__version__)"
            ),
        ]
    )
    ok, detail = run_command(command, timeout=180)
    rendered = " ".join(shlex.quote(part) for part in command)
    return Check("uv_agent_runtime", ok, f"{detail}\n{rendered}", command)


def check_index() -> Check:
    if not INDEX_PATH.is_file():
        return Check("swebench_lite_index", False, f"missing {INDEX_PATH}")
    count = 0
    repos: dict[str, int] = {}
    with INDEX_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            count += 1
            repo = str(row.get("repo", ""))
            repos[repo] = repos.get(repo, 0) + 1
    ok = count == 300
    return Check("swebench_lite_index", ok, f"rows={count} repos={len(repos)}")


def check_model_config() -> Check:
    config_path = BENCHMARK_DIR / "config.yaml"
    if not config_path.is_file():
        return Check("model_config", False, f"missing {config_path}")
    configured_env = {
        key: bool(os.environ.get(key))
        for key in (
            "VACTHBENCH_MODEL",
            "VACTHBENCH_OPENAI_BASE_URL",
            "VACTHBENCH_OPENAI_API_KEY",
        )
    }
    return Check("model_config", True, f"config={config_path} env={configured_env}")


def clean_sidecars(paths: list[Path], *, delete: bool) -> Check:
    files = 0
    dirs = 0
    errors = 0
    for path in paths:
        stats = clean_path(path.resolve(), delete=delete, quiet=True)
        files += stats.files
        dirs += stats.dirs
        errors += stats.errors
    action = "removed" if delete else "would_remove"
    return Check("macos_sidecars", errors == 0, f"{action}: files={files} dirs={dirs} errors={errors}")


def write_report(path: Path, checks: list[Check]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ok": all(check.ok for check in checks),
                "checks": [
                    {
                        "name": check.name,
                        "ok": check.ok,
                        "detail": check.detail,
                        "command": check.command,
                    }
                    for check in checks
                ],
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-sidecars",
        action="store_true",
        help="Delete macOS sidecar files from default experiment paths before checking.",
    )
    parser.add_argument(
        "--sidecar-path",
        action="append",
        default=[],
        help="Additional path to scan for sidecars. Can repeat.",
    )
    parser.add_argument(
        "--uv-online",
        action="store_true",
        help="Allow uv to fetch missing preflight dependencies. Default uses --offline.",
    )
    parser.add_argument(
        "--report",
        default=str(RESULTS_DIR / "swebench_lite_full_preflight.json"),
    )
    args = parser.parse_args()

    sidecar_paths = [Path(path) for path in DEFAULT_TARGETS] + [Path(path) for path in args.sidecar_path]
    checks: list[Check] = [
        clean_sidecars(sidecar_paths, delete=args.clean_sidecars),
        *check_docker(),
        check_uv_swebench(offline=not args.uv_online),
        check_agent_runtime(offline=not args.uv_online),
        check_index(),
        check_model_config(),
    ]

    for check in checks:
        status = "OK" if check.ok else "FAIL"
        print(f"[{status}] {check.name}: {check.detail}")

    report_path = Path(args.report).expanduser().resolve()
    write_report(report_path, checks)
    print(f"wrote {report_path}")
    if not all(check.ok for check in checks):
        sys.exit(1)


if __name__ == "__main__":
    main()
