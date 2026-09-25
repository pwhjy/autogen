from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any

import pytest

SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
PYTHON_DIR = BENCHMARK_DIR.parents[3]


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


summarizer = _load_module(
    SCRIPT_DIR / "summarize_swebench_aligned_metrics.py",
    "summarize_swebench_aligned_metrics_test",
)


def test_model_usage_token_split_counts_agent_aux_and_extractor() -> None:
    usage = summarizer.token_usage_from_records(
        [
            {"agent": "planner", "prompt_tokens": 10, "completion_tokens": 2},
            {"agent": "coder", "prompt_tokens": 20, "completion_tokens": 3},
            {"agent": "vacth_extractor", "prompt_tokens": 7, "completion_tokens": 1},
            {"agent": "summary_runtime", "prompt_tokens": 4, "completion_tokens": 2},
        ],
        source="model_usage",
    )

    assert usage["total_tokens"] == 49
    assert usage["corrected_total_tokens"] == 49
    assert usage["agent_model_tokens"] == 35
    assert usage["extractor_model_tokens"] == 8
    assert usage["aux_model_tokens"] == 14


def test_token_usage_from_run_prefers_model_usage_over_legacy_metrics(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "raw_messages.json").write_text(
        json.dumps(
            [
                {
                    "models_usage": {
                        "prompt_tokens": 100,
                        "completion_tokens": 20,
                    }
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "model_usage.json").write_text(
        json.dumps(
            [
                {"agent": "coder", "prompt_tokens": 30, "completion_tokens": 5},
                {"agent": "vacth_extractor", "prompt_tokens": 8, "completion_tokens": 2},
            ]
        ),
        encoding="utf-8",
    )

    usage = summarizer.token_usage_from_run(run_dir, {"estimated_tokens": 999})

    assert usage["token_source"] == "model_usage"
    assert usage["total_tokens"] == 45
    assert usage["corrected_total_tokens"] == 45
    assert usage["raw_message_model_tokens"] == 120
    assert usage["estimated_aux_tokens"] == 0
    assert usage["agent_model_tokens"] == 35
    assert usage["extractor_model_tokens"] == 10


def test_tokens_per_resolved_uses_corrected_total_tokens() -> None:
    row = {
        "method": "vacth_optimized",
        "official_resolved": 1,
        "official_status": "resolved",
        "agent_error": 0,
        "incomplete_run_patch": 0,
        "total_tokens": 10,
        "corrected_total_tokens": 20,
        "estimated_cost_usd_standard": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "agent_model_tokens": 0,
        "aux_model_tokens": 0,
        "extractor_model_tokens": 0,
        "raw_message_model_tokens": 0,
        "estimated_aux_tokens": 0,
        "tool_calls": 0,
        "run_tests_count": 0,
        "edit_count": 0,
        "rework_count": 0,
        "tool_error_count": 0,
        "cascade_error_proxy": 0,
        "constraint_violations": 0,
        "invalid_patch_count": 0,
        "changed_test_file_count": 0,
        "summary_count": 0,
        "summary_tokens": 0,
        "structured_update_count": 0,
        "structured_parse_error_count": 0,
        "vacth_extraction_count": 0,
        "vacth_parse_error_count": 0,
        "vacth_heuristic_extraction_count": 0,
        "vacth_reask_count": 0,
        "vacth_capsule_count": 0,
        "vacth_cve_routing_count": 0,
        "retrieval_count": 0,
        "retrieved_tokens": 0,
        "long_context_request_count": 0,
        "wall_time_sec": 0,
    }

    summary = summarizer.summarize([row])[0]

    assert summary["tokens_per_resolved"] == 20


def _load_scenario_module() -> Any:
    for path in (
        PYTHON_DIR / "packages" / "autogen-agentchat" / "src",
        PYTHON_DIR / "packages" / "autogen-core" / "src",
        BENCHMARK_DIR / "Templates" / "common",
    ):
        sys.path.insert(0, str(path))
    source_path = BENCHMARK_DIR / "Templates" / "software_repair" / "scenario.py"
    source = source_path.read_text(encoding="utf-8").replace(
        "__EXPERIMENT_CONFIG_JSON__",
        json.dumps({"task_id": "unit", "method": "vacth_optimized"}),
    )
    module = types.ModuleType("vacth_optimized_scenario_test")
    module.__file__ = str(source_path)
    try:
        exec(compile(source, str(source_path), "exec"), module.__dict__)
    except ModuleNotFoundError as exc:
        pytest.skip(f"scenario dependencies are unavailable: {exc}")
    return module


def test_adaptive_repair_defaults_are_progress_gated(tmp_path: Path) -> None:
    scenario = _load_scenario_module()

    assert scenario.adaptive_repair_max_attempts({"method": "vacth_optimized"}) == 2
    assert scenario.adaptive_repair_max_attempts({"method": "summary"}) == 0
    assert scenario.source_patch_guard_max_attempts({"method": "vacth_optimized"}) == 2
    assert scenario.source_patch_guard_max_attempts({"method": "summary"}) == 0
    assert scenario.adaptive_repair_progress_gate_enabled({"method": "vacth_optimized"}) is True
    assert scenario.test_failure_score({"passed": True, "stdout": "", "stderr": ""}) == 0
    assert scenario.test_failure_score({"passed": False, "stdout": "FAILED (failures=2, errors=20)"}) == 22
    assert scenario.test_failure_score({"passed": False, "stdout": "==== 1 failed, 15 passed in 0.1s ===="}) == 1

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "source.py").write_text("value = 1\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(workspace)
    tools._last_test_result = {"passed": False, "stdout": "FAILED (errors=3)", "stderr": ""}
    tools._last_test_fingerprint = tools.workspace_fingerprint()
    tools._last_test_command = tools.test_command
    assert scenario.should_attempt_adaptive_repair(config={"method": "vacth_optimized"}, repair_tools=tools) is True

    tools._last_test_result = {"passed": False, "stdout": "ImportError: no module named optional_dep", "stderr": ""}
    assert scenario.should_attempt_adaptive_repair(config={"method": "vacth_optimized"}, repair_tools=tools) is False


def test_macos_sidecar_cleanup_removes_git_pack_artifacts(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    pack_dir = workspace / ".git" / "objects" / "pack"
    pack_dir.mkdir(parents=True)
    sidecar = pack_dir / "._pack-example.idx"
    normal = pack_dir / "pack-example.idx"
    sidecar.write_text("appledouble", encoding="utf-8")
    normal.write_text("pack", encoding="utf-8")

    stats = scenario._remove_macos_sidecars(workspace)

    assert stats["files"] == 1
    assert stats["dirs"] == 0
    assert stats["errors"] == 0
    assert not sidecar.exists()
    assert normal.exists()


def test_source_patch_guard_stops_after_exhausted_invalid_patch(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "source.py"
    source.write_text("value = 1\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(workspace)
    config = {"method": "vacth_optimized"}

    assert scenario.source_patch_guard_reason(tools) == "git_diff has no non-test source changes yet."
    assert (
        scenario.should_stop_after_source_patch_guards(
            config=config,
            repair_tools=tools,
            source_patch_guards=1,
            max_source_patch_guards=2,
        )
        is False
    )
    assert (
        scenario.should_stop_after_source_patch_guards(
            config=config,
            repair_tools=tools,
            source_patch_guards=2,
            max_source_patch_guards=2,
        )
        is True
    )
    assert config["vacth_source_patch_guard_exhausted"] is True

    source.write_text("import warnings\nvalue = 1\n", encoding="utf-8")
    config = {"method": "vacth_optimized"}
    assert "only changes import/comment/blank lines" in scenario.source_patch_guard_reason(tools)
    assert (
        scenario.should_stop_after_source_patch_guards(
            config=config,
            repair_tools=tools,
            source_patch_guards=2,
            max_source_patch_guards=2,
        )
        is True
    )


def test_test_status_verifies_same_diff_and_invalidates_changed_diff(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "calculator.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(
        workspace,
        test_command=[sys.executable, "-c", "import sys; sys.exit(0)"],
    )

    run_result = json.loads(tools.run_tests())
    verified_status = json.loads(tools.test_status())
    source.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    stale_status = json.loads(tools.test_status())

    assert run_result["passed"] is True
    assert verified_status["current_diff_verified"] is True
    assert stale_status["current_diff_verified"] is False
    assert stale_status["changed_files"] == ["calculator.py"]


def test_run_tests_caches_failed_same_diff(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(
        workspace,
        test_command=[sys.executable, "-c", "import sys; sys.exit(1)"],
    )

    first = json.loads(tools.run_tests())
    second = json.loads(tools.run_tests())
    status = json.loads(tools.test_status())

    assert first["passed"] is False
    assert first["executed"] is True
    assert second["passed"] is False
    assert second["cached"] is True
    assert second["executed"] is False
    assert status["current_diff_tested"] is True
    assert status["current_diff_failed"] is True


def test_run_tests_installs_missing_test_dependency_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(
        workspace,
        test_command=[sys.executable, "-m", "pytest", "test_calculator.py"],
    )
    calls: list[list[str]] = []
    test_attempts = 0

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal test_attempts
        calls.append(command)
        if command[:3] == [sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(command, 0, "installed pytest", "")
        test_attempts += 1
        if test_attempts == 1:
            return subprocess.CompletedProcess(command, 1, "", "No module named pytest")
        return subprocess.CompletedProcess(command, 0, "test_calculator.py::test_add PASSED\n", "")

    monkeypatch.setattr(scenario.subprocess, "run", fake_run)

    result = json.loads(tools.run_tests())
    status = json.loads(tools.test_status())

    assert result["passed"] is True
    assert result["dependency_install_attempts"][0]["packages"] == ["pytest"]
    assert [sys.executable, "-m", "pip", "install", "pytest"] in calls
    assert test_attempts == 2
    assert status["current_diff_verified"] is True


def test_run_tests_installs_version_dependency_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(
        workspace,
        test_command=[sys.executable, "-m", "pytest", "test_calculator.py"],
    )
    calls: list[list[str]] = []
    test_attempts = 0

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal test_attempts
        calls.append(command)
        if command[:3] == [sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(command, 0, "installed numpy", "")
        test_attempts += 1
        if test_attempts == 1:
            return subprocess.CompletedProcess(
                command,
                1,
                "",
                "ImportError: Numpy version 1.13.0 or later must be installed to use Astropy",
            )
        return subprocess.CompletedProcess(command, 0, "test_calculator.py::test_add PASSED\n", "")

    monkeypatch.setattr(scenario.subprocess, "run", fake_run)

    result = json.loads(tools.run_tests())

    assert result["passed"] is True
    assert result["dependency_install_attempts"][0]["packages"] == ["numpy"]
    assert [sys.executable, "-m", "pip", "install", "numpy"] in calls
    assert test_attempts == 2


def test_run_tests_installs_distutils_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(
        workspace,
        test_command=[sys.executable, "-m", "pytest", "test_calculator.py"],
    )
    calls: list[list[str]] = []
    test_attempts = 0

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal test_attempts
        calls.append(command)
        if command[:3] == [sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(command, 0, "installed setuptools", "")
        test_attempts += 1
        if test_attempts == 1:
            return subprocess.CompletedProcess(command, 1, "", "No module named 'distutils'")
        return subprocess.CompletedProcess(command, 0, "test_calculator.py::test_add PASSED\n", "")

    monkeypatch.setattr(scenario.subprocess, "run", fake_run)

    result = json.loads(tools.run_tests())

    assert result["passed"] is True
    assert result["dependency_install_attempts"][0]["packages"] == ["setuptools"]
    assert [sys.executable, "-m", "pip", "install", "setuptools"] in calls
    assert test_attempts == 2


def test_run_tests_installs_django_runtime_dependency(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "calculator.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(
        workspace,
        test_command=[sys.executable, "-m", "pytest", "test_calculator.py"],
    )
    calls: list[list[str]] = []
    test_attempts = 0

    def fake_run(command: list[str], **_: Any) -> subprocess.CompletedProcess[str]:
        nonlocal test_attempts
        calls.append(command)
        if command[:3] == [sys.executable, "-m", "pip"]:
            return subprocess.CompletedProcess(command, 0, "installed pytz", "")
        test_attempts += 1
        if test_attempts == 1:
            return subprocess.CompletedProcess(command, 1, "", "No module named 'pytz'")
        return subprocess.CompletedProcess(command, 0, "test_calculator.py::test_add PASSED\n", "")

    monkeypatch.setattr(scenario.subprocess, "run", fake_run)

    result = json.loads(tools.run_tests())

    assert result["passed"] is True
    assert result["dependency_install_attempts"][0]["packages"] == ["pytz"]
    assert [sys.executable, "-m", "pip", "install", "pytz"] in calls
    assert test_attempts == 2


def test_changed_source_files_excludes_test_paths(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "calculator.py"
    test_file = workspace / "tests" / "test_calculator.py"
    test_file.parent.mkdir()
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    test_file.write_text("def test_add():\n    assert True\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(workspace)

    test_file.write_text("def test_add():\n    assert False\n", encoding="utf-8")
    assert scenario.changed_source_files(tools) == []

    source.write_text("def add(a, b):\n    return a - b\n", encoding="utf-8")
    assert scenario.changed_source_files(tools) == ["calculator.py"]


def test_source_patch_guard_flags_missing_and_trivial_source_diff(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "calculator.py"
    source.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(workspace)

    assert "no non-test source changes" in scenario.source_patch_guard_reason(tools)

    source.write_text(
        "import math\n\ndef add(a, b):\n    return a + b\n",
        encoding="utf-8",
    )
    assert "only changes import/comment/blank" in scenario.source_patch_guard_reason(tools)

    source.write_text(
        "def add(a, b):\n    return a - b\n",
        encoding="utf-8",
    )
    assert scenario.source_patch_guard_reason(tools) is None


def test_source_patch_guard_flags_unused_added_imports(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = workspace / "merge.py"
    source.write_text("def merge(items):\n    return list(items)\n", encoding="utf-8")
    tools = scenario.SoftwareRepairTools(workspace)

    source.write_text(
        "from helpers import stable_topological_sort\n\ndef merge(items):\n    return sorted(set(items))\n",
        encoding="utf-8",
    )
    reason = scenario.source_patch_guard_reason(tools)
    assert reason is not None
    assert "adds imports/helpers" in reason
    assert "stable_topological_sort" in reason

    source.write_text(
        "from helpers import stable_topological_sort\n\n"
        "def merge(items):\n"
        "    return stable_topological_sort(items, {})\n",
        encoding="utf-8",
    )
    assert scenario.source_patch_guard_reason(tools) is None


def test_test_evidence_extracts_fail_to_pass_snippets(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    tests = workspace / "tests" / "test_edge.py"
    tests.parent.mkdir(parents=True)
    tests.write_text(
        "def test_old_case():\n"
        "    assert True\n\n"
        "class TestEdges:\n"
        "    def test_zero_size_input(self):\n"
        "        result = transform([], [1])\n"
        "        assert result[0] == []\n"
        "        assert result[1] == [1]\n",
        encoding="utf-8",
    )
    task_spec = {
        "category": "swebench_lite",
        "benchmark_metadata": {
            "fail_to_pass": ["tests/test_edge.py::TestEdges::test_zero_size_input"],
        },
        "repo_source": {
            "test_patch": "diff --git a/tests/test_edge.py b/tests/test_edge.py\n"
            "--- a/tests/test_edge.py\n"
            "+++ b/tests/test_edge.py\n",
        },
    }
    tools = scenario.SoftwareRepairTools(workspace, task_spec=task_spec)

    evidence = json.loads(tools.test_evidence())

    assert evidence["available"] is True
    assert evidence["candidate_test_files"] == ["tests/test_edge.py"]
    assert evidence["snippets"][0]["label"] == "tests/test_edge.py::TestEdges::test_zero_size_input"
    assert "test_zero_size_input" in evidence["snippets"][0]["content"]
    assert "assert result[1] == [1]" in evidence["snippets"][0]["content"]


def test_repair_test_env_sets_pytest_version_shim(tmp_path: Path) -> None:
    scenario = _load_scenario_module()
    workspace = tmp_path / "workspace"
    (workspace / "src" / "_pytest").mkdir(parents=True)

    env = scenario.repair_test_env(workspace)

    assert env["SETUPTOOLS_SCM_PRETEND_VERSION"] == "8.0.0"
    assert env["SETUPTOOLS_SCM_PRETEND_VERSION_FOR_PYTEST"] == "8.0.0"


def test_swebench_eval_treats_zero_returncode_missing_labels_as_resolved() -> None:
    scenario = _load_scenario_module()
    evaluation = scenario._evaluate_swebench_style(
        {"returncode": 0, "stdout": "Ran 120 tests in 1.0s\n\nOK\n", "stderr": "", "passed": True},
        {
            "benchmark_metadata": {
                "fail_to_pass": ["test_new_case (app.tests.Case)"],
                "pass_to_pass": ["test_old_case (app.tests.Case)"],
            }
        },
    )

    assert evaluation["resolved"] is True
    assert evaluation["f2p_pass"] == 1
    assert evaluation["p2p_pass"] == 1
    assert evaluation["missing_f2p"] == []
