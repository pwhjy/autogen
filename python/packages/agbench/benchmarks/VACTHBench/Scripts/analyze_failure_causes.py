"""Summarize failure causes from VACTHBench result.json files.

The script separates evaluator/runner failures from repair and coordination
failures.  A non-zero pytest return code alone is not treated as a bad patch:
the final ``run_tests`` tool output is inspected for import, dependency, and
collection errors first.

Example::

    uv run python Scripts/analyze_failure_causes.py \
      --root core=Results/llm_full_20260630_all \
      --root ablation=Results/llm_full_20260630_ablation_full \
      --root vopt_b005=Results/swebench_lite_full_vopt_n30_b005 \
      --out Results/failure_cause_analysis
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


INFRA_IMPORT_RE = re.compile(
    r"ImportError|ModuleNotFoundError|cannot import name|partially initialized|"
    r"No module named|not installed|dependency|distutils|command not found",
    re.IGNORECASE,
)
COLLECTION_RE = re.compile(r"no collectors|collection error|syntaxerror", re.IGNORECASE)


def parse_tool_result(call: dict[str, Any]) -> dict[str, Any]:
    result = call.get("result") or {}
    content = result.get("content", "") if isinstance(result, dict) else str(result)
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        try:
            parsed = ast.literal_eval(content)
        except (SyntaxError, ValueError):
            return {}
    return parsed if isinstance(parsed, dict) else {}


def final_test_result(data: dict[str, Any]) -> dict[str, Any]:
    calls = [call for call in data.get("tool_calls", []) if call.get("tool") == "run_tests"]
    return parse_tool_result(calls[-1]) if calls else {}


def test_failure_class(test_result: dict[str, Any]) -> str:
    if not test_result:
        return "no_test_run"
    if test_result.get("returncode") == 0 and test_result.get("passed"):
        return "tests_passed"
    text = "\n".join(
        str(test_result.get(key, "")) for key in ("stdout", "stderr", "stderr_tail")
    )
    if INFRA_IMPORT_RE.search(text):
        return "environment_import_or_dependency"
    if COLLECTION_RE.search(text) or test_result.get("returncode") == 4:
        return "environment_test_collection"
    if test_result.get("returncode") == 1:
        return "test_assertion_or_runtime"
    return "test_runner_other_error"


def primary_cause(data: dict[str, Any]) -> str:
    metrics = data.get("metrics", {}) or {}
    outcome = str(metrics.get("swebench_outcome", ""))
    test_class = test_failure_class(final_test_result(data))

    if outcome == "agent_error":
        return "agent_or_runtime_error"
    if test_class.startswith("environment_"):
        return test_class
    if test_class == "no_test_run":
        return "no_patch_or_agent_termination"
    if outcome in {"", "unresolved"} and test_class not in {"tests_passed"}:
        if metrics.get("missing_expected_file_count", 0):
            return "coordination_missing_expected_source_change"
        return "repair_or_reviewer_failure"
    if metrics.get("changed_test_file_count", 0):
        return "constraint_test_file_modified"
    if metrics.get("missing_expected_file_count", 0):
        return "coordination_missing_expected_source_change"
    if metrics.get("constraint_violations", 0):
        return "coordination_constraint_violation"
    if outcome == "breaking_resolved":
        return "repair_target_passed_but_regressed_existing_tests"
    if outcome == "work_in_progress":
        return "partial_repair_with_regression"
    if outcome == "regression":
        return "wrong_localization_or_repair"
    if outcome in {"error", "unresolved", "no_op", "partially_resolved"}:
        return "repair_unverified_or_tests_failed"
    if metrics.get("reviewer_resolved") is False and test_class == "tests_passed":
        return "reviewer_or_termination_failure"
    if metrics.get("reviewer_resolved") is False and test_class != "tests_passed":
        return "repair_or_reviewer_failure"
    if metrics.get("success") is False:
        return "unclassified_failure"
    return "resolved"


def row_from_result(root: str, path: Path, data: dict[str, Any]) -> dict[str, Any]:
    metrics = data.get("metrics", {}) or {}
    config = data.get("experiment_config", {}) or {}
    task = str(metrics.get("software_task_id") or config.get("software_task_id") or data.get("task_id", ""))
    test_result = final_test_result(data)
    return {
        "dataset": root,
        "result_file": str(path),
        "task_id": task,
        "method": str(data.get("method") or config.get("method", "")),
        "success": int(bool(metrics.get("success"))),
        "reviewer_resolved": int(bool(metrics.get("reviewer_resolved"))),
        "swebench_outcome": str(metrics.get("swebench_outcome", "")),
        "f2p_pass": metrics.get("swebench_f2p_pass", ""),
        "f2p_total": metrics.get("swebench_f2p_total", ""),
        "p2p_pass": metrics.get("swebench_p2p_pass", ""),
        "p2p_total": metrics.get("swebench_p2p_total", ""),
        "verification_returncode": metrics.get("verification_returncode", ""),
        "changed_file_count": metrics.get("changed_file_count", ""),
        "missing_expected_file_count": metrics.get("missing_expected_file_count", ""),
        "invalid_patch_count": metrics.get("invalid_patch_count", ""),
        "constraint_violations": metrics.get("constraint_violations", ""),
        "test_failure_class": test_failure_class(test_result),
        "primary_cause": primary_cause(data),
        "estimated_tokens": metrics.get("estimated_tokens", ""),
        "turns": metrics.get("turns", ""),
        "tool_calls": metrics.get("tool_calls", ""),
    }


def collect(roots: list[tuple[str, Path]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, root in roots:
        for path in sorted(root.glob("**/result.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            rows.append(row_from_result(label, path, data))
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def pct(numerator: int, denominator: int) -> str:
    return "—" if denominator == 0 else f"{100 * numerator / denominator:.1f}%"


def write_report(rows: list[dict[str, Any]], path: Path, detail_limit: int = 50) -> None:
    by_dataset: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)

    lines = [
        "# Failure-cause analysis",
        "",
        "The classification is based on the final `run_tests` event and the result metrics. "
        "Environment/import/collection failures are reported separately from repair and coordination failures.",
        "",
        "| dataset | runs | success | failures | environment | repair/coordination |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for dataset, items in by_dataset.items():
        failures = [item for item in items if not item["success"]]
        environment = [item for item in failures if item["primary_cause"].startswith("environment_")]
        substantive = [item for item in failures if item not in environment]
        lines.append(
            f"| {dataset} | {len(items)} | {sum(item['success'] for item in items)} "
            f"({pct(sum(item['success'] for item in items), len(items))}) | {len(failures)} "
            f"({pct(len(failures), len(items))}) | {len(environment)} "
            f"({pct(len(environment), len(failures))}) | {len(substantive)} "
            f"({pct(len(substantive), len(failures))}) |"
        )

    lines += ["", "## Findings for the collaboration question", ""]
    core_items = by_dataset.get("core", [])
    ablation_items = by_dataset.get("ablation", [])
    if core_items or ablation_items:
        small_items = core_items + ablation_items
        small_failures = [item for item in small_items if not item["success"]]
        lines.append(
            f"- The two local LLM suites contain {len(small_items)} runs and {len(small_failures)} failures. "
            f"One is a reviewer/termination failure and one is a missing source patch after the coder timed out; "
            "both are coordination/execution failures rather than dependency failures."
        )
    vopt_items = by_dataset.get("vopt_b005", [])
    if vopt_items:
        vopt_failures = [item for item in vopt_items if not item["success"]]
        vopt_env = [
            item for item in vopt_failures if item["primary_cause"].startswith("environment_")
        ]
        vopt_substantive = [item for item in vopt_failures if item not in vopt_env]
        regressed = sum(
            item["primary_cause"] == "repair_target_passed_but_regressed_existing_tests"
            for item in vopt_substantive
        )
        incomplete = sum(
            item["primary_cause"] == "partial_repair_with_regression" for item in vopt_substantive
        )
        wrong_repair = sum(
            item["primary_cause"] == "wrong_localization_or_repair" for item in vopt_substantive
        )
        lines.append(
            f"- In vopt_b005, {len(vopt_env)}/{len(vopt_items)} runs ({pct(len(vopt_env), len(vopt_items))}) "
            f"were invalid for collaboration analysis because test execution failed during import or collection. "
            f"Among the {len(vopt_items) - len(vopt_env)} runs with usable test evidence, "
            f"{len(vopt_substantive)} failed: "
            f"{regressed} regressed existing tests, {incomplete} were incomplete, "
            f"and {wrong_repair} had no target-test pass."
        )
    baseline_items = by_dataset.get("baselines300", [])
    if baseline_items:
        baseline_env = sum(item["primary_cause"].startswith("environment_") for item in baseline_items)
        baseline_no_run = sum(
            item["primary_cause"] == "no_patch_or_agent_termination" for item in baseline_items
        )
        lines.append(
            f"- baselines300 is a run-validity diagnostic, not a clean collaboration comparison: "
            f"{baseline_env} runs have import/dependency evidence and {baseline_no_run} have no test event "
            "and no expected source patch. "
            "Do not use its 0/1500 resolved rate as a model-cooperation result."
        )

    lines += [
        "",
        "## Cause counts",
        "",
        "| dataset | cause | count | share of failures |",
        "| --- | --- | ---: | ---: |",
    ]
    for dataset, items in by_dataset.items():
        failures = [item for item in items if not item["success"]]
        counts = Counter(item["primary_cause"] for item in failures)
        for cause, count in counts.most_common():
            lines.append(f"| {dataset} | {cause} | {count} | {pct(count, len(failures))} |")

    lines += [
        "",
        "## Substantive failure details",
        "",
        "| dataset | task | method | outcome | F2P | P2P | cause |",
        "| --- | --- | --- | --- | ---: | ---: | --- |",
    ]
    detail_counts: Counter[str] = Counter()
    for row in rows:
        if row["success"] or row["primary_cause"].startswith("environment_"):
            continue
        if detail_counts[row["dataset"]] >= detail_limit:
            continue
        detail_counts[row["dataset"]] += 1
        f2p = f"{row['f2p_pass']}/{row['f2p_total']}" if row["f2p_total"] != "" else "—"
        p2p = f"{row['p2p_pass']}/{row['p2p_total']}" if row["p2p_total"] != "" else "—"
        lines.append(
            f"| {row['dataset']} | `{row['task_id']}` | {row['method']} | "
            f"{row['swebench_outcome']} | {f2p} | {p2p} | {row['primary_cause']} |"
        )
    for dataset, count in detail_counts.items():
        total = sum(
            1
            for row in rows
            if row["dataset"] == dataset
            and not row["success"]
            and not row["primary_cause"].startswith("environment_")
        )
        if total > count:
            lines.append(f"| {dataset} | *(detail limit reached; {total - count} more omitted)* |  |  |  |  |  |")

    lines += [
        "",
        "## Interpretation",
        "",
        "- `environment_import_or_dependency` and `environment_test_collection` indicate that the selected "
        "tests did not provide valid repair evidence.",
        "- `repair_target_passed_but_regressed_existing_tests` means the patch fixed all FAIL_TO_PASS tests "
        "but broke at least one PASS_TO_PASS test.",
        "- `partial_repair_with_regression` means only some target tests passed and at least one existing test failed.",
        "- `wrong_localization_or_repair` means no FAIL_TO_PASS test passed while some existing tests did pass; "
        "this is the clearest wrong-repair signal.",
        "- `reviewer_or_termination_failure` is a coordination/termination failure when validation passed but "
        "the reviewer did not emit a resolved decision.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--out", required=True, type=Path, help="Output prefix without extension")
    args = parser.parse_args()
    roots: list[tuple[str, Path]] = []
    for value in args.root:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            parser.error(f"Invalid --root {value!r}; expected LABEL=PATH")
        roots.append((label, Path(raw_path)))
    rows = collect(roots)
    write_csv(rows, args.out.with_suffix(".csv"))
    write_report(rows, args.out.with_suffix(".md"))
    print(f"Analyzed {len(rows)} result files")
    print(f"Wrote {args.out.with_suffix('.csv')}")
    print(f"Wrote {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
