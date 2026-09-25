from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
TASKS_DIR = BENCHMARK_DIR / "Tasks"


DEFAULT_INPUTS = [
    "software_repair_suite_autogen_broadcast",
    "software_repair_suite_summary",
    "software_repair_suite_sliding_window",
    "software_repair_suite_vector_memory",
    "software_repair_suite_structured_summary",
    "software_repair_suite_vacth_full",
    "software_repair_suite_vacth_optimized",
    "software_repair_vacth_ablations",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            fh.write("\n")


def rewrite_record(record: dict[str, Any], run_id: str) -> dict[str, Any]:
    updated = json.loads(json.dumps(record, ensure_ascii=False))
    old_id = str(updated["id"])
    new_id = f"{run_id}_{old_id}"
    updated["id"] = new_id

    scenario_subs = updated["substitutions"]["scenario.py"]
    config = json.loads(scenario_subs["__EXPERIMENT_CONFIG_JSON__"])
    config["task_id"] = new_id
    config["llm_rerun_id"] = run_id
    scenario_subs["__EXPERIMENT_CONFIG_JSON__"] = json.dumps(
        config, ensure_ascii=False, sort_keys=True
    )
    return updated


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    args = parser.parse_args()

    combined: list[dict[str, Any]] = []
    written = []
    for stem in args.inputs:
        input_path = TASKS_DIR / f"{stem}.jsonl"
        rows = [rewrite_record(row, args.run_id) for row in read_jsonl(input_path)]
        output_path = TASKS_DIR / f"{args.run_id}_{stem}.jsonl"
        write_jsonl(output_path, rows)
        written.append((output_path, len(rows)))
        combined.extend(rows)

    combined_path = TASKS_DIR / f"{args.run_id}_all.jsonl"
    write_jsonl(combined_path, combined)
    written.append((combined_path, len(combined)))

    for path, count in written:
        print(f"{path}: {count}")


if __name__ == "__main__":
    main()
