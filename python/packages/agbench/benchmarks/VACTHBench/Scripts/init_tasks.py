from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
TEMPLATES_DIR = BENCHMARK_DIR / "Templates"
TASKS_DIR = BENCHMARK_DIR / "Tasks"
DATA_DIR = BENCHMARK_DIR / "data"
COMMON_TEMPLATE = str(TEMPLATES_DIR / "common")

SOFTWARE_METHODS = [
    "autogen_broadcast",
    "summary",
    "sliding_window",
    "vector_memory",
    "structured_summary",
    "vacth_full",
    "vacth_optimized",
]

VACTH_ABLATIONS = [
    "vacth_wo_cve",
    "vacth_wo_thc",
    "vacth_wo_paa",
    "vacth_wo_provenance",
    "vacth_wo_reask",
    "vacth_global_capsule",
]
VACTH_METHODS = {"vacth_full", "vacth_optimized", *VACTH_ABLATIONS}
MECHANISM_TYPES = ["extraction", "routing", "aggregation", "reask"]
MECHANISM_METHODS = [
    "summary",
    "structured_summary",
    "vacth_full",
    "vacth_optimized",
    *VACTH_ABLATIONS,
]

SOFTWARE_REPAIR_SUITE: list[dict[str, Any]] = [
    {
        "id": "calc_add",
        "category": "operator_bug",
        "task": "Fix calculator.add so that it returns the arithmetic sum of two numbers.",
        "constraints": "Keep add(a, b) as the public API. Do not modify tests.",
        "gold_state": "add(2, 3) returns 5 and add(-2, 3) returns 1.",
        "repo_files": {
            "calculator.py": (
                "def add(a, b):\n"
                "    # Return the sum of a and b.\n"
                "    return a - b\n"
            ),
            "test_calculator.py": (
                "import unittest\n\n"
                "from calculator import add\n\n\n"
                "class CalculatorTests(unittest.TestCase):\n"
                "    def test_positive_addition(self):\n"
                "        self.assertEqual(add(2, 3), 5)\n\n"
                "    def test_mixed_sign_addition(self):\n"
                "        self.assertEqual(add(-2, 3), 1)\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["calculator.py"],
        "reviewer_success_criteria": (
            "The tests pass, add performs arithmetic addition, and test files are unchanged."
        ),
    },
    {
        "id": "csv_quoted_commas",
        "category": "parser_edge_case",
        "task": "Fix csv_utils.parse_line so quoted commas stay inside the same field.",
        "constraints": "Keep parse_line(line) as the public API. Do not modify tests.",
        "gold_state": "Quoted comma fields such as 'alpha,\"beta,gamma\",delta' parse into three fields.",
        "repo_files": {
            "csv_utils.py": (
                "def parse_line(line):\n"
                "    return [part.strip() for part in line.split(',')]\n"
            ),
            "test_csv_utils.py": (
                "import unittest\n\n"
                "from csv_utils import parse_line\n\n\n"
                "class CsvUtilsTests(unittest.TestCase):\n"
                "    def test_quoted_comma(self):\n"
                "        self.assertEqual(parse_line('alpha,\"beta,gamma\",delta'), "
                "['alpha', 'beta,gamma', 'delta'])\n\n"
                "    def test_two_quoted_fields(self):\n"
                "        self.assertEqual(parse_line('\"a,b\",\"c\"'), ['a,b', 'c'])\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["csv_utils.py"],
        "reviewer_success_criteria": (
            "The tests pass, quoted CSV fields are parsed correctly, and test files are unchanged."
        ),
    },
    {
        "id": "slug_whitespace",
        "category": "normalization",
        "task": "Fix slugs.make_slug so it strips and collapses all whitespace into single hyphens.",
        "constraints": "Keep make_slug(title) as the public API. Do not modify tests.",
        "gold_state": "Whitespace-only differences collapse; '  Hello   World  ' becomes 'hello-world'.",
        "repo_files": {
            "slugs.py": (
                "def make_slug(title):\n"
                "    return title.lower().replace(' ', '-')\n"
            ),
            "test_slugs.py": (
                "import unittest\n\n"
                "from slugs import make_slug\n\n\n"
                "class SlugTests(unittest.TestCase):\n"
                "    def test_collapse_spaces(self):\n"
                "        self.assertEqual(make_slug('  Hello   World  '), 'hello-world')\n\n"
                "    def test_tabs(self):\n"
                "        self.assertEqual(make_slug('Release\\tCandidate'), 'release-candidate')\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["slugs.py"],
        "reviewer_success_criteria": (
            "The tests pass, all whitespace is normalized into single hyphens, and tests are unchanged."
        ),
    },
    {
        "id": "refund_boundary",
        "category": "boundary_condition",
        "task": "Fix refunds.is_refundable so purchases are refundable through day 30 inclusive.",
        "constraints": "Keep is_refundable(days_since_purchase) as the public API. Do not modify tests.",
        "gold_state": "Day 0 and day 30 return True; day 31 returns False.",
        "repo_files": {
            "refunds.py": (
                "def is_refundable(days_since_purchase):\n"
                "    return days_since_purchase < 30\n"
            ),
            "test_refunds.py": (
                "import unittest\n\n"
                "from refunds import is_refundable\n\n\n"
                "class RefundTests(unittest.TestCase):\n"
                "    def test_boundary_inclusive(self):\n"
                "        self.assertTrue(is_refundable(30))\n\n"
                "    def test_after_window(self):\n"
                "        self.assertFalse(is_refundable(31))\n\n"
                "    def test_purchase_day(self):\n"
                "        self.assertTrue(is_refundable(0))\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["refunds.py"],
        "reviewer_success_criteria": (
            "The tests pass, the refund boundary is inclusive at day 30, and tests are unchanged."
        ),
    },
    {
        "id": "safe_divide_zero",
        "category": "exception_handling",
        "task": "Fix ratios.safe_divide so division by zero returns None instead of raising.",
        "constraints": "Keep safe_divide(numerator, denominator) as the public API. Do not modify tests.",
        "gold_state": "safe_divide(6, 3) returns 2.0 and safe_divide(6, 0) returns None.",
        "repo_files": {
            "ratios.py": (
                "def safe_divide(numerator, denominator):\n"
                "    return numerator / denominator\n"
            ),
            "test_ratios.py": (
                "import unittest\n\n"
                "from ratios import safe_divide\n\n\n"
                "class RatioTests(unittest.TestCase):\n"
                "    def test_normal_division(self):\n"
                "        self.assertEqual(safe_divide(6, 3), 2.0)\n\n"
                "    def test_zero_denominator(self):\n"
                "        self.assertIsNone(safe_divide(6, 0))\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["ratios.py"],
        "reviewer_success_criteria": (
            "The tests pass, zero denominator returns None, and test files are unchanged."
        ),
    },
    {
        "id": "dedupe_order",
        "category": "data_structure_semantics",
        "task": "Fix collections_ext.dedupe so it removes duplicates while preserving first-seen order.",
        "constraints": "Keep dedupe(values) as the public API. Do not modify tests.",
        "gold_state": "dedupe(['b', 'a', 'b', 'c', 'a']) returns ['b', 'a', 'c'].",
        "repo_files": {
            "collections_ext.py": (
                "def dedupe(values):\n"
                "    return list(set(values))\n"
            ),
            "test_collections_ext.py": (
                "import unittest\n\n"
                "from collections_ext import dedupe\n\n\n"
                "class DedupeTests(unittest.TestCase):\n"
                "    def test_preserves_first_seen_order(self):\n"
                "        self.assertEqual(dedupe(['b', 'a', 'b', 'c', 'a']), ['b', 'a', 'c'])\n\n"
                "    def test_empty(self):\n"
                "        self.assertEqual(dedupe([]), [])\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["collections_ext.py"],
        "reviewer_success_criteria": (
            "The tests pass, first-seen ordering is preserved, and tests are unchanged."
        ),
    },
    {
        "id": "iso_date_format",
        "category": "format_bug",
        "task": "Fix dates.parse_iso_date so it accepts ISO dates in YYYY-MM-DD format.",
        "constraints": "Keep parse_iso_date(value) as the public API. Do not modify tests.",
        "gold_state": "parse_iso_date('2026-05-03') returns date(2026, 5, 3).",
        "repo_files": {
            "dates.py": (
                "from datetime import datetime\n\n\n"
                "def parse_iso_date(value):\n"
                "    return datetime.strptime(value, '%m-%d-%Y').date()\n"
            ),
            "test_dates.py": (
                "import unittest\n"
                "from datetime import date\n\n"
                "from dates import parse_iso_date\n\n\n"
                "class DateTests(unittest.TestCase):\n"
                "    def test_iso_date(self):\n"
                "        self.assertEqual(parse_iso_date('2026-05-03'), date(2026, 5, 3))\n\n"
                "    def test_iso_new_year(self):\n"
                "        self.assertEqual(parse_iso_date('2027-01-01'), date(2027, 1, 1))\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["dates.py"],
        "reviewer_success_criteria": (
            "The tests pass, ISO YYYY-MM-DD dates parse correctly, and tests are unchanged."
        ),
    },
    {
        "id": "inventory_equal_stock",
        "category": "boundary_condition",
        "task": "Fix inventory.can_reserve so a request equal to available stock is allowed.",
        "constraints": "Keep can_reserve(stock, requested) as the public API. Do not modify tests.",
        "gold_state": "can_reserve(5, 5) and can_reserve(5, 4) are True; can_reserve(5, 6) is False.",
        "repo_files": {
            "inventory.py": (
                "def can_reserve(stock, requested):\n"
                "    return stock > requested\n"
            ),
            "test_inventory.py": (
                "import unittest\n\n"
                "from inventory import can_reserve\n\n\n"
                "class InventoryTests(unittest.TestCase):\n"
                "    def test_equal_stock_allowed(self):\n"
                "        self.assertTrue(can_reserve(5, 5))\n\n"
                "    def test_less_than_stock_allowed(self):\n"
                "        self.assertTrue(can_reserve(5, 4))\n\n"
                "    def test_more_than_stock_rejected(self):\n"
                "        self.assertFalse(can_reserve(5, 6))\n\n\n"
                "if __name__ == '__main__':\n"
                "    unittest.main()\n"
            ),
        },
        "expected_changed_files": ["inventory.py"],
        "reviewer_success_criteria": (
            "The tests pass, equal stock is allowed, and test files are unchanged."
        ),
    },
]


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} must be a JSON object")
            records.append(value)
    return records


def _software_smoke_record(
    method: str,
    task_id: str | None = None,
    software_task: dict[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_task_id = task_id or f"software_smoke_{method}"
    is_vacth = method in VACTH_METHODS
    vacth_flags = _vacth_flags(method) if is_vacth else {}
    task_prompt = {
        "task": "Fix calculator.add so that it returns the arithmetic sum of two numbers.",
        "constraints": "Keep the public function name add(a, b). Do not change the tests.",
        "gold_state": (
            "calculator.add(2, 3) must return 5 and calculator.add(-2, 3) "
            "must return 1."
        ),
    }
    if software_task is not None:
        task_prompt = {
            "task": str(software_task["task"]),
            "constraints": str(software_task["constraints"]),
            "gold_state": str(software_task["gold_state"]),
        }
    config = {
        "task_id": resolved_task_id,
        "template": "software_repair",
        "method": method,
        "model": "config.yaml"
        if method
        in {
            "autogen_broadcast",
            "summary",
            "sliding_window",
            "vector_memory",
            "structured_summary",
            *VACTH_METHODS,
        }
        else "deterministic-smoke",
        "max_turns": int(software_task.get("max_turns", 8))
        if software_task is not None
        else 8,
        "max_total_tokens": 50000,
        "max_context_tokens_per_agent": 6000,
        "context_window_messages": 4 if method == "sliding_window" else None,
        "memory_top_k": 4 if method == "vector_memory" else None,
        "structured_max_items_per_slot": 12 if method == "structured_summary" else None,
        "capsule_token_budget": 900 if is_vacth else None,
        "seed": 7,
        "token_budget": 6000,
        **vacth_flags,
    }
    if software_task is not None:
        config["software_task"] = software_task
        config["software_task_id"] = str(software_task["id"])
        config["software_task_category"] = str(
            software_task.get("category", "unspecified")
        )
    return {
        "id": resolved_task_id,
        "template": [COMMON_TEMPLATE, str(TEMPLATES_DIR / "software_repair")],
        "substitutions": {
            "scenario.py": {
                "__EXPERIMENT_CONFIG_JSON__": json.dumps(
                    config, ensure_ascii=False, sort_keys=True
                ),
            },
            "prompt.txt": {
                "__TASK__": task_prompt["task"],
                "__CONSTRAINTS__": task_prompt["constraints"],
                "__GOLD_STATE__": task_prompt["gold_state"],
            },
        },
    }


def _vacth_flags(method: str) -> dict[str, Any]:
    flags: dict[str, Any] = {
        "enable_thc": True,
        "enable_cve": True,
        "enable_paa": True,
        "enable_provenance": True,
        "enable_reask": True,
        "role_specific_routing": True,
        "capsule_scope": "role_specific",
    }
    if method == "vacth_wo_cve":
        flags["enable_cve"] = False
        flags["role_specific_routing"] = False
        flags["capsule_scope"] = "global_chronological"
    elif method == "vacth_wo_thc":
        flags["enable_thc"] = False
    elif method == "vacth_wo_paa":
        flags["enable_paa"] = False
    elif method == "vacth_wo_provenance":
        flags["enable_provenance"] = False
    elif method == "vacth_wo_reask":
        flags["enable_reask"] = False
    elif method == "vacth_global_capsule":
        flags["role_specific_routing"] = False
        flags["capsule_scope"] = "global_cve"
    elif method == "vacth_optimized":
        flags.update(
            {
                "vacth_skip_planner_capsule": True,
                "vacth_skip_extraction_agents": ["planner", "reviewer"],
                "vacth_compact_downstream_task": True,
                "vacth_compact_capsule": True,
                "vacth_hybrid_extraction": True,
                "capsule_token_budget": 700,
                "vacth_role_token_budgets": {
                    "retriever": 350,
                    "coder": 550,
                    "tester": 500,
                    "reviewer": 650,
                },
                "thc_source_token_budget": 1800,
                "thc_existing_item_limit": 12,
                "thc_max_items_per_turn": 6,
                "vacth_capsule_max_item_chars": 180,
                "vacth_source_patch_guard_max_attempts": 2,
            }
        )
    return flags


def _generic_record(domain: str, method: str) -> dict[str, Any]:
    task_id = f"{domain}_smoke_{method}"
    config = {
        "task_id": task_id,
        "template": domain,
        "method": method,
        "model": "deterministic-smoke",
        "max_turns": 4,
        "max_total_tokens": 20000,
        "max_context_tokens_per_agent": 4000,
        "seed": 7,
        "token_budget": 4000,
    }
    return {
        "id": task_id,
        "template": [COMMON_TEMPLATE, str(TEMPLATES_DIR / domain)],
        "substitutions": {
            "scenario.py": {
                "__EXPERIMENT_CONFIG_JSON__": json.dumps(
                    config, ensure_ascii=False, sort_keys=True
                ),
            },
            "prompt.txt": {
                "__TASK__": f"Run the deterministic {domain} smoke task.",
                "__CONSTRAINTS__": "Produce a valid VACTHBench result.json.",
                "__GOLD_STATE__": "The smoke scenario should complete successfully.",
            },
        },
    }


def _mechanism_record(
    *, mechanism_type: str, method: str, sample: dict[str, Any]
) -> dict[str, Any]:
    sample_id = str(sample["id"])
    task_id = f"mechanism_{mechanism_type}_{sample_id}_{method}"
    is_vacth = method in VACTH_METHODS
    vacth_flags = _vacth_flags(method) if is_vacth else {}
    config = {
        "task_id": task_id,
        "template": "mechanism",
        "method": method,
        "model": "deterministic-mechanism",
        "mechanism_task_type": mechanism_type,
        "mechanism_sample_id": sample_id,
        "max_turns": 1,
        "max_total_tokens": 20000,
        "seed": 7,
        "token_budget": sample.get("token_budget", 80),
        "sample": sample,
        **vacth_flags,
    }
    return {
        "id": task_id,
        "template": [str(TEMPLATES_DIR / "mechanism")],
        "substitutions": {
            "scenario.py": {
                "__EXPERIMENT_CONFIG_JSON__": json.dumps(
                    config, ensure_ascii=False, sort_keys=True
                ),
            },
            "prompt.txt": {
                "__MECHANISM_TYPE__": mechanism_type,
                "__SAMPLE_ID__": sample_id,
                "__TASK__": str(sample.get("task", "")),
            },
        },
    }


def _generate_mechanism_tasks() -> None:
    samples_by_type = {
        mechanism_type: _read_jsonl(DATA_DIR / f"{mechanism_type}.jsonl")
        for mechanism_type in MECHANISM_TYPES
    }
    for method in MECHANISM_METHODS:
        suite_records: list[dict[str, Any]] = []
        scaleup_smoke_records: list[dict[str, Any]] = []
        for mechanism_type, samples in samples_by_type.items():
            records = [
                _mechanism_record(
                    mechanism_type=mechanism_type,
                    method=method,
                    sample=sample,
                )
                for sample in samples
            ]
            _write_jsonl(
                TASKS_DIR / f"mechanism_{mechanism_type}_{method}.jsonl",
                records,
            )
            suite_records.extend(records)
            generated_samples = [
                sample
                for sample in samples
                if str(sample.get("id", "")).startswith("gen_")
            ]
            if generated_samples:
                scaleup_smoke_records.append(
                    _mechanism_record(
                        mechanism_type=mechanism_type,
                        method=method,
                        sample=generated_samples[0],
                    )
                )
        _write_jsonl(TASKS_DIR / f"mechanism_suite_{method}.jsonl", suite_records)
        if scaleup_smoke_records:
            _write_jsonl(
                TASKS_DIR / f"mechanism_scaleup_smoke_{method}.jsonl",
                scaleup_smoke_records,
            )


def _generate_software_repair_suite_tasks() -> None:
    combined_records: list[dict[str, Any]] = []
    for method in SOFTWARE_METHODS:
        method_records = [
            _software_smoke_record(
                method,
                task_id=f"software_repair_{task['id']}_{method}",
                software_task=task,
            )
            for task in SOFTWARE_REPAIR_SUITE
        ]
        _write_jsonl(
            TASKS_DIR / f"software_repair_suite_{method}.jsonl",
            method_records,
        )
        _write_jsonl(
            TASKS_DIR / f"software_repair_suite_smoke_{method}.jsonl",
            method_records[:2],
        )
        combined_records.extend(method_records)
    _write_jsonl(TASKS_DIR / "software_repair_suite_all.jsonl", combined_records)


def main() -> None:
    TASKS_DIR.mkdir(parents=True, exist_ok=True)

    for method in SOFTWARE_METHODS:
        _write_jsonl(
            TASKS_DIR / f"software_repair_{method}.jsonl",
            [_software_smoke_record(method)],
        )

    ablation_records = [_software_smoke_record(method) for method in VACTH_ABLATIONS]
    _write_jsonl(TASKS_DIR / "software_repair_vacth_ablations.jsonl", ablation_records)

    for domain in ["fact_verification", "tool_use"]:
        for method in ["autogen_broadcast", "structured_summary", "vacth_full"]:
            _write_jsonl(
                TASKS_DIR / f"{domain}_{method}.jsonl",
                [_generic_record(domain, method)],
            )

    _generate_mechanism_tasks()
    _generate_software_repair_suite_tasks()

    generated = sorted(
        os.path.relpath(path, BENCHMARK_DIR) for path in TASKS_DIR.glob("*.jsonl")
    )
    print("Generated VACTHBench task files:")
    for path in generated:
        print(f"  {path}")


if __name__ == "__main__" and __package__ is None:
    main()
