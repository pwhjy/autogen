"""Prebuild local SWE-bench instance images for VACTHBench batches.

This is an environment-stability helper for large SWE-bench Lite runs.  It
builds the same local ``sweb.eval.*`` images used by the official harness with
``--namespace none`` so agent-side workspace setup can copy ``/testbed`` from
Docker instead of repeatedly fetching source repositories from GitHub.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from pathlib import Path
from typing import Any

from build_external_repair_tasks import read_jsonl
from run_swebench_official_patched import (
    optional_namespace,
    patch_python_dockerfile_for_cn_mirrors,
    patch_python_env_for_fast_local_builds,
    patch_python_repo_setup_for_partial_clone,
    wait_for_docker_daemon,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
RESULTS_DIR = BENCHMARK_DIR / "Results"
INDEX_PATH = BENCHMARK_DIR / "data" / "external" / "swebench_lite" / "index_test.jsonl"


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def instance_ids_from_task_file(path: Path) -> list[str]:
    instance_ids: list[str] = []
    for row in read_jsonl(path):
        substitutions = row.get("substitutions", {})
        scenario_subs = substitutions.get("scenario.py", {}) if isinstance(substitutions, dict) else {}
        config_raw = scenario_subs.get("__EXPERIMENT_CONFIG_JSON__", "")
        if not config_raw:
            continue
        config = json.loads(str(config_raw))
        software_task = config.get("software_task", {})
        repo_source = software_task.get("repo_source", {}) if isinstance(software_task, dict) else {}
        instance_id = str(repo_source.get("instance_id", "")).strip()
        if instance_id and instance_id not in instance_ids:
            instance_ids.append(instance_id)
    return instance_ids


def selected_rows(index_path: Path, instance_ids: list[str]) -> list[dict[str, Any]]:
    wanted = set(instance_ids)
    rows = [row for row in read_jsonl(index_path) if str(row.get("instance_id")) in wanted]
    found = {str(row["instance_id"]) for row in rows}
    missing = sorted(wanted - found)
    if missing:
        raise SystemExit(f"Instance ids not found in {index_path}: {', '.join(missing)}")
    order = {instance_id: index for index, instance_id in enumerate(instance_ids)}
    return sorted(rows, key=lambda row: order[str(row["instance_id"])])


def image_status(client: Any, image_refs: list[str]) -> dict[str, bool]:
    status: dict[str, bool] = {}
    for image_ref in image_refs:
        try:
            client.images.get(image_ref)
        except Exception:
            status[image_ref] = False
        else:
            status[image_ref] = True
    return status


def write_status(path: Path, status: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_rows(rows: list[dict[str, Any]], *, namespace: str | None, force_rebuild: bool, max_workers: int) -> None:
    patch_python_dockerfile_for_cn_mirrors()
    patch_python_repo_setup_for_partial_clone()
    patch_python_env_for_fast_local_builds()

    import docker
    from swebench.harness.docker_build import build_instance_images

    client = docker.from_env()
    build_instance_images(
        client,
        rows,
        force_rebuild=force_rebuild,
        max_workers=max(1, int(max_workers)),
        namespace=namespace,
        tag="latest",
        env_image_tag="latest",
    )


def _build_rows_child(
    rows: list[dict[str, Any]],
    namespace: str | None,
    force_rebuild: bool,
    max_workers: int,
    queue: Any,
) -> None:
    try:
        build_rows(rows, namespace=namespace, force_rebuild=force_rebuild, max_workers=max_workers)
    except BaseException as exc:
        queue.put(
            {
                "ok": False,
                "error_type": type(exc).__name__,
                "error": str(exc)[-2000:],
                "traceback": traceback.format_exc()[-6000:],
            }
        )
        raise
    queue.put({"ok": True})


def run_build_with_timeout(
    rows: list[dict[str, Any]],
    *,
    namespace: str | None,
    force_rebuild: bool,
    max_workers: int,
    timeout_sec: int,
) -> dict[str, Any]:
    started = time.monotonic()
    if timeout_sec <= 0:
        try:
            build_rows(rows, namespace=namespace, force_rebuild=force_rebuild, max_workers=max_workers)
        except BaseException as exc:
            return {
                "status": "error",
                "elapsed_sec": round(time.monotonic() - started, 3),
                "error_type": type(exc).__name__,
                "error": str(exc)[-2000:],
                "traceback": traceback.format_exc()[-6000:],
            }
        return {"status": "ok", "elapsed_sec": round(time.monotonic() - started, 3)}

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_build_rows_child,
        args=(rows, namespace, force_rebuild, max_workers, queue),
    )
    process.start()
    process.join(timeout_sec)
    if process.is_alive():
        process.terminate()
        process.join(30)
        if process.is_alive():
            process.kill()
            process.join(10)
        return {
            "status": "timeout",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "timeout_sec": timeout_sec,
            "exitcode": process.exitcode,
        }

    message: dict[str, Any] = {}
    while not queue.empty():
        message = queue.get()
    if process.exitcode == 0 and message.get("ok", False):
        return {
            "status": "ok",
            "elapsed_sec": round(time.monotonic() - started, 3),
            "exitcode": process.exitcode,
        }
    return {
        "status": "error",
        "elapsed_sec": round(time.monotonic() - started, 3),
        "exitcode": process.exitcode,
        **{key: value for key, value in message.items() if key != "ok"},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-file", help="VACTHBench task JSONL to prebuild.")
    parser.add_argument("--instance-ids", help="Comma-separated SWE-bench instance ids.")
    parser.add_argument("--index-path", default=str(INDEX_PATH))
    parser.add_argument("--namespace", default="none")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--per-instance",
        action="store_true",
        help="Build each missing instance image in a separate child process and write status after each attempt.",
    )
    parser.add_argument(
        "--build-timeout-sec",
        type=int,
        default=0,
        help="Optional timeout for each build subprocess. Use with --per-instance to isolate slow env solves.",
    )
    parser.add_argument(
        "--stop-on-timeout",
        action="store_true",
        help="Stop after the first timed-out per-instance build attempt.",
    )
    parser.add_argument(
        "--retry-missing-per-instance",
        action="store_true",
        help=(
            "After a batch build, retry any still-missing images one at a time. "
            "This lowers Docker memory pressure after env image build failures."
        ),
    )
    parser.add_argument(
        "--retry-build-timeout-sec",
        type=int,
        default=1800,
        help="Timeout for each automatic missing-image retry.",
    )
    parser.add_argument(
        "--output",
        default=str(RESULTS_DIR / "swebench_image_prebuild_status.json"),
        help="JSON status file to write.",
    )
    args = parser.parse_args()

    instance_ids: list[str] = []
    if args.instance_ids:
        instance_ids = _split_csv(args.instance_ids)
    elif args.task_file:
        instance_ids = instance_ids_from_task_file(Path(args.task_file))
    if not instance_ids:
        raise SystemExit("Provide --task-file or --instance-ids.")

    patch_python_dockerfile_for_cn_mirrors()
    patch_python_repo_setup_for_partial_clone()
    patch_python_env_for_fast_local_builds()
    wait_for_docker_daemon()

    import docker
    from swebench.harness.docker_build import build_instance_images, make_test_spec

    rows = selected_rows(Path(args.index_path), instance_ids)
    namespace = optional_namespace(args.namespace)
    specs = [make_test_spec(row, namespace=namespace) for row in rows]
    image_refs = [spec.instance_image_key for spec in specs]

    client = docker.from_env()
    before = image_status(client, image_refs)
    pending = [image for image, present in before.items() if not present]
    output_path = Path(args.output)
    status: dict[str, Any] = {
        "instance_ids": instance_ids,
        "image_refs": image_refs,
        "present_before": before,
        "present_after": before,
        "missing_after": [image for image, present in before.items() if not present],
        "namespace": namespace,
        "max_workers": max(1, int(args.max_workers)),
        "per_instance": bool(args.per_instance),
        "build_timeout_sec": int(args.build_timeout_sec),
        "retry_missing_per_instance": bool(args.retry_missing_per_instance),
        "retry_build_timeout_sec": int(args.retry_build_timeout_sec),
        "attempts": [],
        "phase": "starting",
    }
    write_status(output_path, status)

    print(f"instances: {len(rows)}", flush=True)
    print(f"present before: {sum(before.values())}", flush=True)
    print(f"pending before: {len(pending)}", flush=True)
    for image in pending:
        print(f"  build {image}", flush=True)

    if args.per_instance:
        status["phase"] = "building"
        write_status(output_path, status)
        for row, image_ref in zip(rows, image_refs, strict=True):
            if not args.force_rebuild and image_status(client, [image_ref]).get(image_ref, False):
                attempt = {
                    "instance_id": str(row["instance_id"]),
                    "image_ref": image_ref,
                    "status": "already_present",
                    "elapsed_sec": 0,
                }
            else:
                print(f"building {row['instance_id']} -> {image_ref}", flush=True)
                result = run_build_with_timeout(
                    [row],
                    namespace=namespace,
                    force_rebuild=args.force_rebuild,
                    max_workers=max(1, int(args.max_workers)),
                    timeout_sec=max(0, int(args.build_timeout_sec)),
                )
                present_now = image_status(client, [image_ref]).get(image_ref, False)
                if result.get("status") == "ok" and not present_now:
                    result = {
                        **result,
                        "status": "missing_image",
                        "error": "build subprocess exited successfully but target image is absent",
                    }
                attempt = {
                    "instance_id": str(row["instance_id"]),
                    "image_ref": image_ref,
                    "present_after_attempt": present_now,
                    **result,
                }
                print(
                    f"  {attempt['status']} present={present_now} elapsed={attempt.get('elapsed_sec')}",
                    flush=True,
                )
            status["attempts"].append(attempt)
            after = image_status(client, image_refs)
            status["present_after"] = after
            status["missing_after"] = [image for image, present in after.items() if not present]
            status["phase"] = "interrupted" if attempt["status"] == "timeout" and args.stop_on_timeout else "building"
            write_status(output_path, status)
            if attempt["status"] == "timeout" and args.stop_on_timeout:
                print("stopping after timeout", flush=True)
                break
    else:
        status["phase"] = "building"
        write_status(output_path, status)
        result = run_build_with_timeout(
            rows,
            namespace=namespace,
            force_rebuild=args.force_rebuild,
            max_workers=max(1, int(args.max_workers)),
            timeout_sec=max(0, int(args.build_timeout_sec)),
        )
        status["attempts"].append({"instance_id": "*", "image_ref": "*", **result})
        if result["status"] != "ok":
            status["phase"] = "interrupted" if result["status"] == "timeout" else "error"
            after = image_status(client, image_refs)
            status["present_after"] = after
            status["missing_after"] = [image for image, present in after.items() if not present]
            write_status(output_path, status)
            if not args.retry_missing_per_instance:
                raise SystemExit(f"Image prebuild failed: {result['status']}")

    after = image_status(client, image_refs)
    status["present_after"] = after
    status["missing_after"] = [image for image, present in after.items() if not present]
    if status["missing_after"] and args.retry_missing_per_instance:
        status["phase"] = "retrying_missing"
        write_status(output_path, status)
        for row, image_ref in zip(rows, image_refs, strict=True):
            if image_status(client, [image_ref]).get(image_ref, False):
                continue
            print(f"retrying missing {row['instance_id']} -> {image_ref}", flush=True)
            result = run_build_with_timeout(
                [row],
                namespace=namespace,
                force_rebuild=args.force_rebuild,
                max_workers=1,
                timeout_sec=max(0, int(args.retry_build_timeout_sec)),
            )
            present_now = image_status(client, [image_ref]).get(image_ref, False)
            if result.get("status") == "ok" and not present_now:
                result = {
                    **result,
                    "status": "missing_image",
                    "error": "retry subprocess exited successfully but target image is absent",
                }
            attempt = {
                "instance_id": str(row["instance_id"]),
                "image_ref": image_ref,
                "retry": True,
                "present_after_attempt": present_now,
                **result,
            }
            status["attempts"].append(attempt)
            after = image_status(client, image_refs)
            status["present_after"] = after
            status["missing_after"] = [image for image, present in after.items() if not present]
            write_status(output_path, status)
            print(
                f"  retry {attempt['status']} present={present_now} elapsed={attempt.get('elapsed_sec')}",
                flush=True,
            )

    after = image_status(client, image_refs)
    status["present_after"] = after
    status["missing_after"] = [image for image, present in after.items() if not present]
    status["phase"] = "complete" if not status["missing_after"] else status.get("phase", "incomplete")
    if status["phase"] == "building":
        status["phase"] = "incomplete"
    write_status(output_path, status)
    print(f"present after: {sum(after.values())}", flush=True)
    print(f"wrote {output_path}", flush=True)
    if status["missing_after"]:
        raise SystemExit("Some images are still missing after build.")


if __name__ == "__main__":
    main()
