from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from init_tasks import TASKS_DIR, VACTH_ABLATIONS, _software_smoke_record


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def config_from_record(record: dict[str, Any]) -> dict[str, Any]:
    raw = record["substitutions"]["scenario.py"]["__EXPERIMENT_CONFIG_JSON__"]
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid experiment config for {record.get('id')}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an 8-task LLM software-repair grid for VACTH ablations."
    )
    parser.add_argument(
        "--source",
        default=str(TASKS_DIR / "software_repair_suite_vacth_full.jsonl"),
        help="Task file whose software_task payloads define the repair suite.",
    )
    parser.add_argument(
        "--run-id",
        default="llm_full_20260630_ablation_full",
        help="Prefix for generated task IDs and task files.",
    )
    args = parser.parse_args()

    source = Path(args.source)
    if not source.is_absolute():
        source = Path.cwd() / source
    suite_rows = read_jsonl(source)

    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in VACTH_ABLATIONS}
    combined: list[dict[str, Any]] = []

    for base_record in suite_rows:
        config = config_from_record(base_record)
        software_task = config.get("software_task")
        if not isinstance(software_task, dict):
            raise ValueError(f"{base_record.get('id')} does not contain software_task")
        software_task_id = str(software_task["id"])
        for method in VACTH_ABLATIONS:
            task_id = f"{args.run_id}_software_repair_{software_task_id}_{method}"
            record = _software_smoke_record(
                method,
                task_id=task_id,
                software_task=software_task,
            )
            by_method[method].append(record)
            combined.append(record)

    for method, records in by_method.items():
        path = TASKS_DIR / f"{args.run_id}_{method}.jsonl"
        write_jsonl(path, records)
        print(f"{path}: {len(records)}")

    combined_path = TASKS_DIR / f"{args.run_id}_all.jsonl"
    write_jsonl(combined_path, combined)
    print(f"{combined_path}: {len(combined)}")


if __name__ == "__main__":
    main()
