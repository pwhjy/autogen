"""Run SWE-bench official evaluation using cached datasets and prebuilt images.

This helper is intentionally narrow for VACTHBench experiments. It avoids
network fetches for requirements files when prebuilt ``swebench/sweb.eval.*``
instance images are available, writes per-method logs, and also emits corrected
method reports from per-instance ``report.json`` files because some swebench
harness versions misclassify completed instances in the outer summary report.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time
import traceback
from pathlib import Path


DEFAULT_METHODS = (
    "autogen_broadcast",
    "sliding_window",
    "structured_summary",
    "summary",
    "vacth_full",
    "vacth_optimized",
    "vector_memory",
)


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def optional_namespace(value: str) -> str | None:
    """Match SWE-bench CLI semantics: "none" means build local images."""
    normalized = value.strip()
    if normalized.lower() in {"", "none", "null"}:
        return None
    return normalized


def wait_for_docker_daemon(timeout_sec: int | None = None) -> None:
    """Wait for Docker Desktop/daemon instead of failing before evaluation starts."""
    import docker

    timeout = (
        int(os.environ.get("VACTHBENCH_DOCKER_WAIT_SEC", "180"))
        if timeout_sec is None
        else int(timeout_sec)
    )
    if timeout <= 0:
        return

    deadline = time.monotonic() + timeout
    last_error = ""
    while True:
        try:
            client = docker.from_env()
            client.ping()
            return
        except Exception as exc:  # noqa: BLE001 - preserve docker-specific failures in message.
            last_error = str(exc)
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Docker daemon is not reachable. Start Docker Desktop and retry; "
                    f"last error: {last_error}"
                ) from exc
            time.sleep(5)


def patch_python_dockerfile_for_cn_mirrors() -> None:
    """Use stable mirrors for local SWE-bench Python image builds."""
    from swebench.harness import dockerfiles

    proxy = os.environ.get("VACTHBENCH_DOCKER_PROXY", "http://host.docker.internal:10808").strip()
    base = dockerfiles._DOCKERFILE_BASE["py"]
    if "mirrors.tuna.tsinghua.edu.cn" not in base:
        base = base.replace(
            "ENV TZ=Etc/UTC\n",
            (
                "ENV TZ=Etc/UTC\n"
                "ENV PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple\n"
                "ENV PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn\n"
                "RUN sed -i "
                "'s|http://archive.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g; "
                "s|http://security.ubuntu.com/ubuntu|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|g' "
                "/etc/apt/sources.list\n"
            ),
        )
        base = base.replace(
            "https://repo.anaconda.com/miniconda/",
            "https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/",
        )
        base = base.replace(
            "RUN conda config --append channels conda-forge",
            (
                "RUN printf '%s\\n' "
                "'channels:' "
                "'  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r' "
                "'  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main' "
                "'  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge' "
                "'default_channels: []' "
                "'show_channel_urls: true' "
                "> /root/.condarc"
            ),
        )
        dockerfiles._DOCKERFILE_BASE["py"] = base

    if proxy:
        instance = dockerfiles._DOCKERFILE_INSTANCE["py"]
        if "VACTHBENCH_DOCKER_PROXY" not in instance:
            proxy_block = (
                f"ENV VACTHBENCH_DOCKER_PROXY={proxy}\n"
                f"ENV HTTP_PROXY={proxy}\n"
                f"ENV HTTPS_PROXY={proxy}\n"
                f"ENV http_proxy={proxy}\n"
                f"ENV https_proxy={proxy}\n"
                "ENV NO_PROXY=localhost,127.0.0.1,host.docker.internal,"
                "mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn\n"
                "ENV no_proxy=localhost,127.0.0.1,host.docker.internal,"
                "mirrors.tuna.tsinghua.edu.cn,pypi.tuna.tsinghua.edu.cn\n"
            )
            instance = instance.replace("\nCOPY ./setup_repo.sh /root/\n", f"\n{proxy_block}\nCOPY ./setup_repo.sh /root/\n")
            dockerfiles._DOCKERFILE_INSTANCE["py"] = instance


def patch_python_repo_setup_for_partial_clone() -> None:
    """Avoid full-history repository clones when SWE-bench builds local instance images."""
    from swebench.harness.test_spec import create_scripts, python as py_spec, test_spec

    original_make_repo_script_list_py = py_spec.make_repo_script_list_py

    def add_apt_retry_fallback(command: str) -> str:
        if "apt-get" not in command:
            return command
        retry_command = command.replace("apt-get ", "apt-get -o Acquire::Retries=5 ")
        retry_command = retry_command.replace("apt-get install ", "apt-get install --fix-missing ")
        fallback_command = command.replace(
            "apt-get -y update",
            (
                "sed -i "
                "'s|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|http://archive.ubuntu.com/ubuntu|g; "
                "s|http://mirrors.tuna.tsinghua.edu.cn/ubuntu|http://security.ubuntu.com/ubuntu|g' "
                "/etc/apt/sources.list || true; apt-get clean; apt-get -o Acquire::Retries=5 -y update"
            ),
        )
        fallback_command = fallback_command.replace("apt-get ", "apt-get -o Acquire::Retries=5 ")
        fallback_command = fallback_command.replace("apt-get install ", "apt-get install --fix-missing ")
        return f"( {retry_command} ) || ( {fallback_command} )"

    def add_pip_install_fallbacks(command: str) -> str:
        if "pip install" not in command:
            return command

        candidates = [command]
        if "--no-use-pep517" in command:
            candidates.append(
                command.replace("--no-use-pep517 ", "").replace(" --no-use-pep517", "")
            )
        for candidate in list(candidates):
            if " -e " in candidate:
                candidates.append(candidate.replace(" -e ", " "))

        deduplicated: list[str] = []
        for candidate in candidates:
            if candidate not in deduplicated:
                deduplicated.append(candidate)
        if len(deduplicated) == 1:
            return command
        return " || ".join(f"( {candidate} )" for candidate in deduplicated)

    def patch_repo_setup_command(command: str) -> str:
        return add_pip_install_fallbacks(add_apt_retry_fallback(command))

    def make_repo_script_list_py_partial(specs, repo, repo_directory, base_commit, env_name) -> list:
        original_commands = original_make_repo_script_list_py(specs, repo, repo_directory, base_commit, env_name)
        fetch_commit = (
            "if ! git -c protocol.version=2 fetch --filter=blob:none --no-tags --depth 1 "
            f"origin {base_commit}; then "
            "git -c protocol.version=2 fetch --filter=blob:none --no-tags "
            f"origin {base_commit}; "
            "fi"
        )
        partial_clone_commands = [
            f"git init {repo_directory}",
            f"chmod -R 777 {repo_directory}",  # So nonroot user can run tests
            f"cd {repo_directory}",
            f"git config --global --add safe.directory {repo_directory}",
            f"git remote add origin https://github.com/{repo}",
            fetch_commit,
            "git checkout --force FETCH_HEAD",
            f"git reset --hard {base_commit}",
        ]

        def is_original_checkout_command(command: str) -> bool:
            stripped = command.strip()
            exact_checkout_commands = {
                f"git init {repo_directory}",
                f"chmod -R 777 {repo_directory}",
                f"cd {repo_directory}",
                f"git config --global --add safe.directory {repo_directory}",
                f"git reset --hard {base_commit}",
            }
            if stripped in exact_checkout_commands:
                return True
            if stripped.startswith("git clone ") or stripped.startswith("git remote add origin "):
                return True
            if stripped.startswith("git checkout ") or stripped.startswith("git switch "):
                return True
            return " fetch " in stripped and f" {base_commit}" in stripped

        remaining_commands = [
            patch_repo_setup_command(command)
            for command in original_commands
            if not is_original_checkout_command(str(command))
        ]
        return partial_clone_commands + remaining_commands

    def make_repo_script_list_partial(specs, repo, repo_directory, base_commit, env_name) -> list:
        if create_scripts.MAP_REPO_TO_EXT[repo] == "py":
            return make_repo_script_list_py_partial(specs, repo, repo_directory, base_commit, env_name)
        return create_scripts.make_repo_script_list_common(specs, repo, repo_directory, base_commit, env_name)

    py_spec.make_repo_script_list_py = make_repo_script_list_py_partial
    create_scripts.make_repo_script_list_py = make_repo_script_list_py_partial
    create_scripts.make_repo_script_list = make_repo_script_list_partial
    test_spec.make_repo_script_list = make_repo_script_list_partial


def patch_python_eval_for_pytest_self_tests() -> None:
    """Keep pytest repo self-tests from failing before the submitted patch is exercised."""
    from swebench.harness.test_spec import create_scripts, python as py_spec, test_spec

    original_make_eval_script_list_py = py_spec.make_eval_script_list_py

    def make_eval_script_list_py_patched(
        instance, specs, env_name, repo_directory, base_commit, test_patch
    ) -> list:
        commands = original_make_eval_script_list_py(
            instance, specs, env_name, repo_directory, base_commit, test_patch
        )
        if instance.get("repo") != "pytest-dev/pytest":
            return commands

        version_export = (
            "export SETUPTOOLS_SCM_PRETEND_VERSION=8.0.0 "
            "SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST=8.0.0"
        )
        patched: list[str] = []
        for command in commands:
            patched.append(command)
            if command == f"conda activate {env_name}":
                patched.append(version_export)
        return patched

    py_spec.make_eval_script_list_py = make_eval_script_list_py_patched
    create_scripts.make_eval_script_list_py = make_eval_script_list_py_patched
    test_spec.make_eval_script_list_py = make_eval_script_list_py_patched


def patch_python_env_for_fast_local_builds() -> None:
    """Avoid slow repository dependency solves when local SWE-bench images are rebuilt."""
    from swebench.harness.test_spec import python as py_spec

    def _minimal_environment_yml(env_name: str, python_version: str) -> str:
        version = str(python_version or "3.10").strip()
        return f"name: {env_name}\ndependencies:\n  - python={version}\n"

    def _python_version_for_instance(instance: dict) -> str:
        repo = str(instance.get("repo", ""))
        version = str(instance.get("version", ""))
        specs = py_spec.MAP_REPO_VERSION_TO_SPECS.get(repo, {}).get(version, {})
        return str(specs.get("python") or "3.10")

    def _python_version_for_repo(repo: str) -> str:
        versions = {
            str(spec.get("python"))
            for spec in py_spec.MAP_REPO_VERSION_TO_SPECS.get(repo, {}).values()
            if spec.get("python")
        }
        if len(versions) == 1:
            return next(iter(versions))
        return "3.10"

    py_spec.get_requirements_by_commit = lambda repo, commit: ""
    py_spec.get_requirements = lambda instance: ""
    py_spec.get_environment_yml_by_commit = (
        lambda repo, commit, env_name: _minimal_environment_yml(
            env_name, _python_version_for_repo(str(repo))
        )
    )
    py_spec.get_environment_yml = (
        lambda instance, env_name: _minimal_environment_yml(
            env_name, _python_version_for_instance(instance)
        )
    )


def aggregate_corrected_reports(
    reports_dir: Path,
    run_logs_dir: Path,
    corrected_dir: Path,
    expected_instance_ids: list[str] | None = None,
) -> None:
    corrected_dir.mkdir(parents=True, exist_ok=True)
    expected = [str(instance_id) for instance_id in expected_instance_ids or []]
    for raw_path in sorted(reports_dir.glob("*.json")):
        if raw_path.name.startswith("._"):
            continue
        raw = read_json(raw_path)
        method = raw_path.name.split(".", 1)[0]
        run_id = raw_path.name.split(".", 1)[1].removesuffix(".json")

        submitted = expected or [str(instance_id) for instance_id in raw.get("submitted_ids", [])]
        status_by_id = {instance_id: "unknown" for instance_id in submitted}
        for key, status in (
            ("resolved_ids", "resolved"),
            ("unresolved_ids", "unresolved"),
            ("empty_patch_ids", "empty_patch"),
            ("error_ids", "error"),
            ("incomplete_ids", "incomplete"),
        ):
            for instance_id in raw.get(key, []):
                instance_id = str(instance_id)
                if not submitted or instance_id in status_by_id:
                    status_by_id[instance_id] = status

        for report_path in sorted(run_logs_dir.glob(f"*/{method}/*/report.json")):
            report = read_json(report_path)
            if not report:
                continue
            instance_id, detail = next(iter(report.items()))
            instance_id = str(instance_id)
            if submitted and instance_id not in status_by_id:
                continue
            if detail.get("resolved"):
                status = "resolved"
            elif detail.get("patch_is_None") or detail.get("patch_exists") is False:
                status = "empty_patch"
            elif detail.get("patch_successfully_applied") is False:
                status = "error"
            else:
                status = "unresolved"
            status_by_id[instance_id] = status

        submitted = submitted or sorted(status_by_id)
        resolved = [instance_id for instance_id in submitted if status_by_id.get(instance_id) == "resolved"]
        unresolved = [instance_id for instance_id in submitted if status_by_id.get(instance_id) == "unresolved"]
        empty = [instance_id for instance_id in submitted if status_by_id.get(instance_id) == "empty_patch"]
        errors = [instance_id for instance_id in submitted if status_by_id.get(instance_id) == "error"]
        incomplete = [
            instance_id
            for instance_id in submitted
            if status_by_id.get(instance_id) in {"incomplete", "unknown"}
        ]
        completed = resolved + unresolved
        corrected = {
            "total_instances": len(submitted),
            "submitted_instances": len(submitted),
            "completed_instances": len(completed),
            "resolved_instances": len(resolved),
            "unresolved_instances": len(unresolved),
            "empty_patch_instances": len(empty),
            "error_instances": len(errors),
            "completed_ids": completed,
            "incomplete_ids": incomplete,
            "empty_patch_ids": empty,
            "submitted_ids": submitted,
            "resolved_ids": resolved,
            "unresolved_ids": unresolved,
            "error_ids": errors,
            "source": "raw method report plus per-instance run_logs report.json overrides",
            "schema_version": 2,
        }
        out_path = corrected_dir / f"{method}.{run_id}.corrected.json"
        out_path.write_text(json.dumps(corrected, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        print(
            f"corrected {method}: total={len(submitted)} resolved={len(resolved)} "
            f"unresolved={len(unresolved)} empty={len(empty)} errors={len(errors)}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-id-prefix", required=True)
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--instance-ids", required=True, help="Comma-separated SWE-bench instance ids.")
    parser.add_argument("--dataset-name", default="SWE-bench/SWE-bench_Lite")
    parser.add_argument("--split", default="test")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument(
        "--namespace",
        default="swebench",
        help='Docker image namespace. Use "none" to build local instance images instead of pulling remote ones.',
    )
    parser.add_argument("--offline", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    prediction_dir = Path(args.prediction_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    namespace = optional_namespace(args.namespace)
    reports_dir = output_dir / "reports"
    corrected_dir = output_dir / "reports_corrected"
    logs_dir = output_dir / "logs"
    run_logs_dir = output_dir / "run_logs"
    for path in (reports_dir, corrected_dir, logs_dir, run_logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    if args.offline:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"

    wait_for_docker_daemon()

    from swebench.harness import run_evaluation as re
    patch_python_dockerfile_for_cn_mirrors()
    patch_python_repo_setup_for_partial_clone()
    patch_python_eval_for_pytest_self_tests()
    patch_python_env_for_fast_local_builds()
    re.RUN_EVALUATION_LOG_DIR = run_logs_dir

    methods = [item.strip() for item in args.methods.split(",") if item.strip()]
    instance_ids = [item.strip() for item in args.instance_ids.split(",") if item.strip()]
    failures: list[dict[str, str]] = []
    os.chdir(reports_dir)
    for method in methods:
        prediction_file = prediction_dir / f"{method}.jsonl"
        run_id = f"{args.run_id_prefix}_{method}"
        log_path = logs_dir / f"{method}.patched_official.log"
        print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] START official-patched {method}", flush=True)
        start = time.time()
        with log_path.open("w", encoding="utf-8") as fh, contextlib.redirect_stdout(fh), contextlib.redirect_stderr(fh):
            print(f"method={method} run_id={run_id} predictions={prediction_file}", flush=True)
            try:
                report_path = re.main(
                    dataset_name=args.dataset_name,
                    split=args.split,
                    instance_ids=instance_ids,
                    predictions_path=str(prediction_file),
                    max_workers=1,
                    force_rebuild=False,
                    cache_level="instance",
                    clean=False,
                    open_file_limit=4096,
                    run_id=run_id,
                    timeout=args.timeout,
                    namespace=namespace,
                    rewrite_reports=False,
                    modal=False,
                    instance_image_tag="latest",
                    env_image_tag="latest",
                    report_dir=str(reports_dir),
                )
                print(f"report_path={report_path}", flush=True)
            except Exception:
                traceback.print_exc()
                failures.append({"method": method, "log": str(log_path)})
        elapsed = time.time() - start
        print(
            f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] END official-patched {method} "
            f"elapsed_min={elapsed / 60:.1f} log={log_path}",
            flush=True,
        )

    aggregate_corrected_reports(reports_dir, run_logs_dir, corrected_dir, instance_ids)
    if failures:
        print("FAILURES", failures, flush=True)
        raise SystemExit(1)
    print("OFFICIAL_PATCHED_EVAL_DONE", reports_dir, flush=True)


if __name__ == "__main__":
    main()
