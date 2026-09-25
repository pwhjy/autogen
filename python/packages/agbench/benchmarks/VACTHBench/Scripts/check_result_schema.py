import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_RESULT_KEYS = {
    "task_id",
    "method",
    "experiment_config",
    "raw_messages",
    "visible_contexts",
    "tool_calls",
    "state_items",
    "routing_decisions",
    "provenance_edges",
    "metrics",
    "final_answer",
}

REQUIRED_METRICS_KEYS = {
    "success",
    "turns",
    "tool_calls",
    "estimated_tokens",
    "wall_time_sec",
}

SWEBENCH_METRICS_KEYS = {
    "swebench_resolved",
    "swebench_outcome",
    "swebench_f2p_pass",
    "swebench_f2p_total",
    "swebench_p2p_pass",
    "swebench_p2p_total",
}

REQUIRED_SIDE_FILES = {
    "raw_messages.json",
    "visible_contexts.json",
    "tool_calls.json",
    "state_items.json",
    "routing_decisions.json",
    "provenance_graph.json",
    "metrics.csv",
}


def _iter_instance_dirs(path: Path) -> list[Path]:
    if (path / "result.json").is_file():
        return [path]

    instance_dirs: list[Path] = []
    for result_file in path.rglob("result.json"):
        if "__pycache__" in result_file.parts:
            continue
        instance_dirs.append(result_file.parent)
    return sorted(instance_dirs)


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _check_list(value: Any, key: str) -> None:
    _expect(isinstance(value, list), f"{key} must be a list")


def check_instance(instance_dir: Path) -> None:
    result_file = instance_dir / "result.json"
    _expect(result_file.is_file(), f"Missing result.json in {instance_dir}")
    with result_file.open("r", encoding="utf-8") as fh:
        result = json.load(fh)

    missing = REQUIRED_RESULT_KEYS.difference(result)
    _expect(not missing, f"{result_file} missing keys: {sorted(missing)}")

    for key in [
        "raw_messages",
        "visible_contexts",
        "tool_calls",
        "state_items",
        "routing_decisions",
        "provenance_edges",
    ]:
        _check_list(result[key], key)

    _expect(
        isinstance(result["experiment_config"], dict),
        "experiment_config must be an object",
    )
    _expect(isinstance(result["metrics"], dict), "metrics must be an object")
    metric_missing = REQUIRED_METRICS_KEYS.difference(result["metrics"])
    _expect(
        not metric_missing,
        f"{result_file} missing metric keys: {sorted(metric_missing)}",
    )

    # SWE-bench tasks must include SWE-bench evaluation metrics
    category = str(
        result.get("experiment_config", {})
        .get("software_task", {})
        .get("category", "")
    )
    if category.startswith("swebench"):
        swebench_missing = SWEBENCH_METRICS_KEYS.difference(result["metrics"])
        _expect(
            not swebench_missing,
            f"{result_file} missing SWE-bench metric keys: "
            f"{sorted(swebench_missing)}",
        )

    # Optional docker_eval validation
    docker_eval = result.get("docker_eval")
    if docker_eval is not None:
        _expect(isinstance(docker_eval, dict), "docker_eval must be an object")
        _expect(
            "resolved" in docker_eval,
            "docker_eval must contain 'resolved'",
        )
        _expect(
            "outcome" in docker_eval,
            "docker_eval must contain 'outcome'",
        )

    for side_file in REQUIRED_SIDE_FILES:
        _expect(
            (instance_dir / side_file).is_file(),
            f"Missing side file {side_file} in {instance_dir}",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate VACTHBench result.json contracts."
    )
    parser.add_argument(
        "path", help="A result instance directory or a Results/<scenario> directory."
    )
    args = parser.parse_args()

    root = Path(args.path)
    instance_dirs = _iter_instance_dirs(root)
    _expect(bool(instance_dirs), f"No result.json files found under {root}")

    for instance_dir in instance_dirs:
        check_instance(instance_dir)
        print(f"OK {instance_dir}")

    print(f"Validated {len(instance_dirs)} result instance(s).")


if __name__ == "__main__":
    main()
