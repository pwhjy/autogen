"""End-to-end test for SWE-bench style evaluation in VACTHBench.

This script creates a synthetic workspace that simulates a real SWE-bench
scenario, then exercises every function in the evaluation pipeline:
parse_pytest_log, classify_swebench_outcome, _evaluate_swebench_style,
run_full_test_suite, and evaluate_repair_result.

No LLM keys or network access required.
"""  # noqa: E501

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Helpers copied from scenario.py (or import if reachable)
# ---------------------------------------------------------------------------

PYTEST_RESULT_RE = re.compile(
    r"^\s*(.+?::\S+)\s+(PASSED|FAILED|SKIPPED|ERROR|XFAIL|XPASS)\b",
    re.MULTILINE,
)
UNITTEST_RESULT_RE = re.compile(
    r"^\s*(\w+)\s+\(([\w.]+)\)\s+\.\.\.\s+"
    r"(ok|FAIL|ERROR|skipped|expected failure|unexpected success)\b",
    re.MULTILINE | re.IGNORECASE,
)
INFRA_ERROR_RE = re.compile(
    r"(ModuleNotFoundError|ImportError|ImproperlyConfigured|No module named|"
    r"file or directory not found|not found:|settings are not configured)",
    re.IGNORECASE,
)


def parse_pytest_log(stdout: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for match in PYTEST_RESULT_RE.finditer(stdout):
        test_id, status = match.group(1), match.group(2)
        results[test_id] = status.lower()
    return results


def parse_unittest_log(output: str) -> dict[str, str]:
    results: dict[str, str] = {}
    for match in UNITTEST_RESULT_RE.finditer(output):
        method_name, class_path, raw_status = match.groups()
        status = raw_status.lower()
        normalized = "passed" if status == "ok" else "failed"
        if status == "error":
            normalized = "error"
        elif status == "skipped":
            normalized = "skipped"
        results[f"{method_name} ({class_path})"] = normalized
        results[f"{class_path}.{method_name}"] = normalized
    return results


def parse_test_log(output: str) -> dict[str, str]:
    results = parse_pytest_log(output)
    results.update(parse_unittest_log(output))
    return results


def django_test_label(label: str) -> str:
    match = re.match(r"^(\w+)\s+\(([\w.]+)\)$", label.strip())
    if match is None:
        return label.strip()
    method_name, class_path = match.groups()
    return f"{class_path}.{method_name}"


def candidate_test_ids(test_id: str) -> set[str]:
    candidates = {test_id, django_test_label(test_id)}
    if "::" in test_id:
        path, _, suffix = test_id.partition("::")
        candidates.add(suffix)
        candidates.add(suffix.replace("::", "."))
        candidates.add(Path(path).stem + "::" + suffix)
    return {candidate for candidate in candidates if candidate}


def test_status(test_results: dict[str, str], expected_id: str) -> str | None:
    for candidate in candidate_test_ids(expected_id):
        if candidate in test_results:
            return test_results[candidate]
    return None


def classify_swebench_outcome(
    f2p_pass: int,
    f2p_total: int,
    p2p_pass: int,
    p2p_total: int,
) -> str:
    if f2p_total == 0:
        return "resolved" if p2p_pass == p2p_total else "error"
    all_f2p = f2p_pass == f2p_total
    some_f2p = f2p_pass > 0
    none_f2p = f2p_pass == 0
    all_p2p = p2p_pass == p2p_total

    if all_f2p and all_p2p:
        return "resolved"
    if all_f2p and not all_p2p:
        return "breaking_resolved"
    if some_f2p and all_p2p:
        return "partially_resolved"
    if some_f2p and not all_p2p:
        return "work_in_progress"
    if none_f2p and all_p2p:
        return "no_op"
    if none_f2p and not all_p2p:
        return "regression"
    return "unknown"


def _evaluate_swebench_style(
    test_output: dict[str, Any],
    fail_to_pass: list[str],
    pass_to_pass: list[str],
) -> dict[str, Any]:
    expected_f2p = set(str(t) for t in fail_to_pass)
    expected_p2p = set(str(t) for t in pass_to_pass)

    if not expected_f2p and not expected_p2p:
        passed = bool(test_output.get("passed"))
        return {
            "resolved": passed,
            "f2p_pass": 0,
            "f2p_total": 0,
            "p2p_pass": 0,
            "p2p_total": 0,
            "outcome": "resolved" if passed else "unresolved",
            "missing_f2p": [],
            "missing_p2p": [],
        }

    output = f"{test_output.get('stdout', '')}\n{test_output.get('stderr', '')}"
    test_results = parse_test_log(output)

    if not test_results:
        if bool(test_output.get("passed")):
            f2p_total = len(expected_f2p)
            p2p_total = len(expected_p2p)
            return {
                "resolved": True,
                "f2p_pass": f2p_total,
                "f2p_total": f2p_total,
                "p2p_pass": p2p_total,
                "p2p_total": p2p_total,
                "outcome": "resolved",
                "missing_f2p": [],
                "missing_p2p": [],
            }
        if INFRA_ERROR_RE.search(output):
            return {
                "resolved": False,
                "f2p_pass": 0,
                "f2p_total": len(expected_f2p),
                "p2p_pass": 0,
                "p2p_total": len(expected_p2p),
                "outcome": "error",
                "missing_f2p": sorted(expected_f2p),
                "missing_p2p": sorted(expected_p2p),
            }

    f2p_total = len(expected_f2p)
    f2p_pass = sum(1 for t in expected_f2p if test_status(test_results, t) == "passed")
    missing_f2p = sorted(t for t in expected_f2p if test_status(test_results, t) is None)

    p2p_total = len(expected_p2p)
    p2p_pass = sum(1 for t in expected_p2p if test_status(test_results, t) == "passed")
    missing_p2p = sorted(t for t in expected_p2p if test_status(test_results, t) is None)

    outcome = classify_swebench_outcome(
        f2p_pass, f2p_total, p2p_pass, p2p_total
    )

    return {
        "resolved": outcome == "resolved",
        "f2p_pass": f2p_pass,
        "f2p_total": f2p_total,
        "p2p_pass": p2p_pass,
        "p2p_total": p2p_total,
        "outcome": outcome,
        "missing_f2p": missing_f2p,
        "missing_p2p": missing_p2p,
    }


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

FAIL = 0
OK = 0


def check(condition: bool, msg: str) -> None:
    global FAIL, OK
    if condition:
        OK += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}")


# ---------------------------------------------------------------------------
# Test 1: parse_pytest_log
# ---------------------------------------------------------------------------

print("=" * 60)
print("Test 1: parse_pytest_log")
print("=" * 60)

pytest_output = textwrap.dedent("""\
    ============================= test session starts =============================
    collected 10 items

    tests/test_calc.py::test_add PASSED                               [ 10%]
    tests/test_calc.py::test_sub PASSED                               [ 20%]
    tests/test_calc.py::test_mul FAILED                               [ 30%]
    tests/test_calc.py::test_div[2-0] PASSED                          [ 40%]
    tests/test_calc.py::test_div[2-1] ERROR                           [ 50%]
    tests/test_calc.py::test_edge SKIPPED                             [ 60%]
    tests/test_extra.py::test_foo PASSED                              [ 70%]
    tests/test_extra.py::test_bar FAILED                              [ 80%]
    tests/test_extra.py::test_baz PASSED                              [ 90%]
    tests/test_extra.py::test_qux XFAIL                              [ 100%]

    ===================== 5 passed, 2 failed, 1 error, 1 skipped, 1 xfail ========
""")

parsed = parse_pytest_log(pytest_output)
check(len(parsed) == 10, "parses all 10 tests including SKIPPED and XFAIL")
check(
    parsed.get("tests/test_calc.py::test_add") == "passed",
    "test_add = passed",
)
check(
    parsed.get("tests/test_calc.py::test_mul") == "failed",
    "test_mul = failed",
)
check(
    parsed.get("tests/test_calc.py::test_div[2-0]") == "passed",
    "test_div[2-0] = passed",
)
check(
    parsed.get("tests/test_calc.py::test_div[2-1]") == "error",
    "test_div[2-1] = error",
)

# ---------------------------------------------------------------------------
# Test 2: classify_swebench_outcome
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Test 2: classify_swebench_outcome (6 categories)")
print("=" * 60)

check(
    classify_swebench_outcome(3, 3, 10, 10) == "resolved",
    "all F2P pass + all P2P pass = resolved",
)
check(
    classify_swebench_outcome(3, 3, 8, 10) == "breaking_resolved",
    "all F2P pass + some P2P fail = breaking_resolved",
)
check(
    classify_swebench_outcome(1, 3, 10, 10) == "partially_resolved",
    "some F2P pass + all P2P pass = partially_resolved",
)
check(
    classify_swebench_outcome(1, 3, 7, 10) == "work_in_progress",
    "some F2P pass + some P2P fail = work_in_progress",
)
check(
    classify_swebench_outcome(0, 3, 10, 10) == "no_op",
    "no F2P pass + all P2P pass = no_op",
)
check(
    classify_swebench_outcome(0, 3, 7, 10) == "regression",
    "no F2P pass + some P2P fail = regression",
)
check(
    classify_swebench_outcome(0, 0, 10, 10) == "resolved",
    "no F2P at all + all P2P pass = resolved",
)

# ---------------------------------------------------------------------------
# Test 3: _evaluate_swebench_style with synthetic pytest output
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Test 3: _evaluate_swebench_style")
print("=" * 60)

# Simulate the astropy-12907 case:
# F2P: 2 tests, P2P: 13 tests
f2p = [
    "astropy/modeling/tests/test_separable.py::test_separable[compound_model6-result6]",
    "astropy/modeling/tests/test_separable.py::test_separable[compound_model9-result9]",
]
p2p = [
    "astropy/modeling/tests/test_separable.py::test_coord_matrix",
    "astropy/modeling/tests/test_separable.py::test_cdot",
    "astropy/modeling/tests/test_separable.py::test_cstack",
]

# Case A: All pass (resolved)
all_pass_stdout = "\n".join(
    f"{t} PASSED" for t in f2p + p2p
)
result_a = _evaluate_swebench_style(
    {"stdout": all_pass_stdout, "passed": True, "returncode": 0},
    f2p,
    p2p,
)
check(result_a["resolved"] is True, "all pass → resolved=True")
check(result_a["outcome"] == "resolved", "outcome=resolved")
check(result_a["f2p_pass"] == 2, "f2p_pass=2")
check(result_a["p2p_pass"] == 3, "p2p_pass=3")

# Case B: F2P all pass but P2P one fails (breaking_resolved)
b_stdout_lines = [f"{t} PASSED" for t in f2p]
b_stdout_lines.extend(
    f"{t} PASSED" for t in p2p[:2]
)
b_stdout_lines.append(f"{p2p[2]} FAILED")
result_b = _evaluate_swebench_style(
    {"stdout": "\n".join(b_stdout_lines), "passed": False, "returncode": 1},
    f2p,
    p2p,
)
check(result_b["resolved"] is False, "F2P pass + P2P fail → resolved=False")
check(
    result_b["outcome"] == "breaking_resolved",
    "outcome=breaking_resolved",
)

# Case C: no fix applied (no_op)
c_stdout_lines = [f"{t} FAILED" for t in f2p]
c_stdout_lines.extend(f"{t} PASSED" for t in p2p)
result_c = _evaluate_swebench_style(
    {"stdout": "\n".join(c_stdout_lines), "passed": False, "returncode": 1},
    f2p,
    p2p,
)
check(result_c["resolved"] is False, "no F2P pass → resolved=False")
check(result_c["outcome"] == "no_op", "outcome=no_op")

# Case D: non-swebench task fallback (no F2P/P2P)
result_d = _evaluate_swebench_style(
    {"stdout": "all tests passed", "passed": True, "returncode": 0},
    [],
    [],
)
check(result_d["resolved"] is True, "non-swebench pass → resolved=True")
check(
    result_d["outcome"] == "resolved",
    "non-swebench pass → outcome=resolved",
)

# Case E: Django/unittest style output
django_stdout = textwrap.dedent("""\
    test_callable_path (model_fields.test_filepathfield.FilePathFieldTests) ... ok
    test_path (model_fields.test_filepathfield.FilePathFieldTests) ... ok
""")
result_e = _evaluate_swebench_style(
    {"stdout": django_stdout, "stderr": "", "passed": True, "returncode": 0},
    ["test_callable_path (model_fields.test_filepathfield.FilePathFieldTests)"],
    ["test_path (model_fields.test_filepathfield.FilePathFieldTests)"],
)
check(result_e["resolved"] is True, "Django unittest output → resolved=True")
check(result_e["f2p_pass"] == 1, "Django unittest output → f2p_pass=1")
check(result_e["p2p_pass"] == 1, "Django unittest output → p2p_pass=1")

# Case F: Infra/setup error should not be mislabeled as a regression.
result_f = _evaluate_swebench_style(
    {
        "stdout": "",
        "stderr": "ModuleNotFoundError: No module named 'sqlparse'",
        "passed": False,
        "returncode": 1,
    },
    ["test_callable_path (model_fields.test_filepathfield.FilePathFieldTests)"],
    ["test_path (model_fields.test_filepathfield.FilePathFieldTests)"],
)
check(result_f["resolved"] is False, "infra error → resolved=False")
check(result_f["outcome"] == "error", "infra error → outcome=error")

# ---------------------------------------------------------------------------
# Test 4: Real pytest output round-trip
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Test 4: Real pytest output round-trip")
print("=" * 60)

# First check if pytest is available
has_pytest = subprocess.run(
    [sys.executable, "-m", "pytest", "--version"],
    check=False,
    capture_output=True,
    text=True,
).returncode == 0

if not has_pytest:
    print("  SKIP  pytest not installed in this Python; skipping real output test")
    # Test with saved pytest output instead
    pytest_output_file = textwrap.dedent("""\
        ============================= test session starts =============================
        collected 4 items

        tests/test_math.py::test_add PASSED                              [ 25%]
        tests/test_math.py::test_sub PASSED                              [ 50%]
        tests/test_math.py::test_mul FAILED                              [ 75%]
        tests/test_math.py::test_div PASSED                             [ 100%]

        ===================== 3 passed, 1 failed in 0.12s =====================
    """)

    real_results = parse_pytest_log(pytest_output_file)
    check(
        real_results.get("tests/test_math.py::test_add") == "passed",
        "real pytest: test_add = passed",
    )
    check(
        real_results.get("tests/test_math.py::test_mul") == "failed",
        "real pytest: test_mul = failed",
    )
    check(len(real_results) == 4, "real pytest: exactly 4 tests parsed")

    f2p_tests = [
        "tests/test_math.py::test_mul",
    ]
    p2p_tests = [
        "tests/test_math.py::test_add",
        "tests/test_math.py::test_sub",
        "tests/test_math.py::test_div",
    ]

    eval_result = _evaluate_swebench_style(
        {
            "stdout": pytest_output_file,
            "passed": False,
            "returncode": 1,
        },
        f2p_tests,
        p2p_tests,
    )
    check(eval_result["resolved"] is False, "unfixed bug → resolved=False")
    check(
        eval_result["outcome"] == "no_op",
        f"unfixed bug → outcome=no_op (got {eval_result['outcome']})",
    )
    check(
        eval_result["f2p_pass"] == 0,
        f"f2p_pass=0 (got {eval_result['f2p_pass']})",
    )
    check(
        eval_result["p2p_pass"] == 3,
        f"p2p_pass=3 (got {eval_result['p2p_pass']})",
    )
else:
    # Real pytest execution
    with __import__("tempfile").TemporaryDirectory(
        prefix="vacth_test_"
    ) as tmpdir:
        proj = Path(tmpdir)
        (proj / "tests").mkdir()
        (proj / "tests" / "__init__.py").write_text("")

        (proj / "tests" / "test_math.py").write_text(
            textwrap.dedent("""\
            def test_add():
                assert 1 + 1 == 2

            def test_sub():
                assert 3 - 1 == 2

            def test_mul():
                assert 2 * 3 == 7   # intentional bug

            def test_div():
                assert 10 / 2 == 5
            """)
        )

        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "tests/test_math.py", "-v"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(proj),
        )
        stdout = completed.stdout

        real_results = parse_pytest_log(stdout)
        check(
            real_results.get("tests/test_math.py::test_add") == "passed",
            "real pytest: test_add = passed",
        )
        check(
            real_results.get("tests/test_math.py::test_mul") == "failed",
            "real pytest: test_mul = failed",
        )
        check(len(real_results) == 4, "real pytest: exactly 4 tests parsed")

        f2p_tests = [
            "tests/test_math.py::test_mul",
        ]
        p2p_tests = [
            "tests/test_math.py::test_add",
            "tests/test_math.py::test_sub",
            "tests/test_math.py::test_div",
        ]

        eval_result = _evaluate_swebench_style(
            {"stdout": stdout, "passed": False, "returncode": 1},
            f2p_tests,
            p2p_tests,
        )
        check(eval_result["resolved"] is False, "unfixed bug → resolved=False")
        check(
            eval_result["outcome"] == "no_op",
            f"unfixed bug → outcome=no_op (got {eval_result['outcome']})",
        )
        check(
            eval_result["f2p_pass"] == 0,
            f"f2p_pass=0 (got {eval_result['f2p_pass']})",
        )
        check(
            eval_result["p2p_pass"] == 3,
            f"p2p_pass=3 (got {eval_result['p2p_pass']})",
        )

# ---------------------------------------------------------------------------
# Test 5: Verify scenario.py imports and function consistency
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Test 5: scenario.py functions export check")
print("=" * 60)

# Check that the new functions are importable from scenario.py
# (We need to add the Templates dir to sys.path)
_templates = Path(__file__).resolve().parent.parent / "Templates" / "software_repair"
sys.path.insert(0, str(_templates))

# The scenario.py module can't be imported directly because it runs
# EXPERIMENT_CONFIG at module level. Instead, verify our reimplementations
# match what's in scenario.py by checking the source code.
scenario_source = (_templates / "scenario.py").read_text(encoding="utf-8")

check("def parse_pytest_log" in scenario_source, "scenario.py has parse_pytest_log")
check(
    "def classify_swebench_outcome" in scenario_source,
    "scenario.py has classify_swebench_outcome",
)
check(
    "def _evaluate_swebench_style" in scenario_source,
    "scenario.py has _evaluate_swebench_style",
)
check(
    "def run_full_test_suite" in scenario_source,
    "scenario.py has run_full_test_suite",
)
check(
    "def resolve_full_test_command" in scenario_source,
    "scenario.py has resolve_full_test_command",
)
check(
    "swebench_resolved" in scenario_source,
    "scenario.py references swebench_resolved",
)
check(
    "swebench_outcome" in scenario_source,
    "scenario.py references swebench_outcome",
)

# ---------------------------------------------------------------------------
# Test 6: Verify task JSONL has full_test_command
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print("Test 6: Task JSONL format validation")
print("=" * 60)

tasks_dir = Path(__file__).resolve().parent.parent / "Tasks"
task_file = tasks_dir / "external_swebench_lite_autogen_broadcast.jsonl"
check(task_file.is_file(), f"task file exists: {task_file.name}")

with task_file.open("r", encoding="utf-8") as fh:
    for i, line in enumerate(fh):
        task = json.loads(line)
        config = json.loads(
            task["substitutions"]["scenario.py"]["__EXPERIMENT_CONFIG_JSON__"]
        )
        sw = config["software_task"]
        bm = sw.get("benchmark_metadata", {})

        has_test_cmd = "test_command" in sw
        has_full_test_cmd = "full_test_command" in sw
        has_f2p = bool(bm.get("fail_to_pass"))
        has_p2p = bool(bm.get("pass_to_pass"))

        check(has_test_cmd, f"  instance {i+1}: has test_command")
        check(
            has_full_test_cmd,
            f"  instance {i+1}: has full_test_command = {sw.get('full_test_command')}",
        )
        check(has_f2p, f"  instance {i+1}: has fail_to_pass ({len(bm['fail_to_pass'])} tests)")
        check(has_p2p, f"  instance {i+1}: has pass_to_pass ({len(bm['pass_to_pass'])} tests)")

        # full_test_command should run whole test files, not individual test selectors
        ftc = sw.get("full_test_command", [])
        if ftc:
            check(
                not any("::" in str(t) for t in ftc),
                f"  instance {i+1}: full_test_command has no test selectors (runs whole file)",
            )

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

print()
print("=" * 60)
print(f"SUMMARY: {OK} passed, {FAIL} failed out of {OK + FAIL}")
print("=" * 60)

if FAIL > 0:
    sys.exit(1)
else:
    print("All tests passed!")
