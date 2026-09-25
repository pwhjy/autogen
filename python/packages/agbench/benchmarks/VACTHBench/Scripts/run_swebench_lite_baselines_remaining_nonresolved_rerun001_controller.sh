#!/usr/bin/env bash
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="$(cd "${BENCH_DIR}/../../../.." && pwd)"
RUN_NAME="swebench_lite_full_baselines_remaining_nonresolved_rerun001"
BENCHMARK_PREFIX="external_${RUN_NAME}"
METHODS=(summary vector_memory autogen_broadcast sliding_window structured_summary vacth_full)
METHODS_CSV="summary,vector_memory,autogen_broadcast,sliding_window,structured_summary,vacth_full"
RUN_PARENT="${RUN_NAME}"
PREDICTION_DIR="swebench_predictions_${RUN_NAME}"
OFFICIAL_DIR="${RUN_NAME}"
MANIFEST="Results/${RUN_NAME}_manifest.json"
COMBINED_TASK_FILE="Tasks/${BENCHMARK_PREFIX}_all.jsonl"
LOG_DIR="Results/${RUN_PARENT}/logs"
OFFICIAL_ROOT="Results/swebench_official_eval/${OFFICIAL_DIR}"
MAX_LOCAL_JOBS="${VACTHBENCH_REMAINING_BASELINE_RERUN_MAX_JOBS:-1}"
TASK_TIMEOUT_SEC="${VACTHBENCH_TASK_TIMEOUT_SEC:-5400}"
MODEL_CALL_TIMEOUT_SEC="${VACTHBENCH_MODEL_CALL_TIMEOUT_SEC:-900}"
MODEL_CALL_RETRIES="${VACTHBENCH_MODEL_CALL_RETRIES:-2}"
ALLOW_OVERLAP="${VACTHBENCH_REMAINING_BASELINE_ALLOW_OVERLAP:-0}"

cd "${BENCH_DIR}"
mkdir -p "${LOG_DIR}" "Results/${RUN_PARENT}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

write_status() {
  uv run --no-project --offline --isolated python Scripts/write_baselines_remaining_nonresolved_rerun001_status.py
}

blocking_official_eval_count() {
  ps ax -o command= | awk '
    /run_swebench_official_patched.py/ && $0 !~ /swebench_lite_full_baselines_remaining_nonresolved_rerun001/ && $0 !~ /awk/ {count++}
    END {print count+0}
  '
}

method_task_file() {
  local method="$1"
  echo "Tasks/${BENCHMARK_PREFIX}_${method}.jsonl"
}

method_results_arg() {
  local method="$1"
  echo "${RUN_PARENT}/${BENCHMARK_PREFIX}_${method}"
}

expected_for_method() {
  local method="$1"
  METHOD="${method}" uv run --no-project --offline --isolated python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_baselines_remaining_nonresolved_rerun001_manifest.json").read_text())
print(manifest["methods"][os.environ["METHOD"]]["expected_tasks"])
PY
}

instances_csv_for_method() {
  local method="$1"
  METHOD="${method}" uv run --no-project --offline --isolated python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_baselines_remaining_nonresolved_rerun001_manifest.json").read_text())
print(",".join(manifest["methods"][os.environ["METHOD"]]["selected_instance_ids"]))
PY
}

count_results_for_method() {
  local method="$1"
  find "Results/${RUN_PARENT}/${BENCHMARK_PREFIX}_${method}" -path "*/0/result.json" -type f 2>/dev/null | wc -l | tr -d " "
}

count_predictions_for_method() {
  local method="$1"
  local path="Results/${PREDICTION_DIR}/${method}.jsonl"
  if [[ -f "${path}" ]]; then
    wc -l < "${path}" | tr -d " "
  else
    echo "0"
  fi
}

agent_uv=(
  uv run --no-project --offline --isolated
  --with "${PYTHON_DIR}/packages/agbench"
  --with "${PYTHON_DIR}/packages/autogen-agentchat"
  --with "${PYTHON_DIR}/packages/autogen-core"
  --with "${PYTHON_DIR}/packages/autogen-ext[openai]"
  python Scripts/run_baseline.py
)

run_method() {
  local method="$1"
  local task_file
  local results_arg
  local instances
  local log_path="${LOG_DIR}/local_${method}.log"
  task_file="$(method_task_file "${method}")"
  results_arg="$(method_results_arg "${method}")"
  instances="$(instances_csv_for_method "${method}")"
  {
    echo "[$(timestamp)] START local ${method}"
    echo "task_file=${task_file}"
    echo "results_arg=${results_arg}"
    echo "instances=${instances}"
    AGBENCH_ALLOW_NATIVE=Yes \
      VACTHBENCH_TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_TIMEOUT_SEC="${MODEL_CALL_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_RETRIES="${MODEL_CALL_RETRIES}" \
      VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE=1 \
      VACTHBENCH_DOCKER_PULL=0 \
      VACTHBENCH_GIT_FETCH_RETRIES=2 \
      VACTHBENCH_DOCKER_PROXY= \
      "${agent_uv[@]}" "${task_file}" --results-dir "${results_arg}" --instances "${instances}"
    exit_status=$?
    echo "[$(timestamp)] END local ${method} status=${exit_status}"
    exit "${exit_status}"
  } > "${log_path}" 2>&1
}

wait_first_job() {
  local pid="$1"
  local name="$2"
  if wait "${pid}"; then
    echo "[$(timestamp)] local ${name} finished ok"
  else
    echo "[$(timestamp)] local ${name} finished nonzero"
  fi
  write_status
}

verify_local_complete() {
  local ok=1
  local method
  for method in "${METHODS[@]}"; do
    local expected
    local results
    expected="$(expected_for_method "${method}")"
    results="$(count_results_for_method "${method}")"
    echo "[$(timestamp)] ${method} local results=${results}/${expected}"
    if [[ "${results}" != "${expected}" ]]; then
      ok=0
    fi
  done
  [[ "${ok}" == "1" ]]
}

verify_predictions_complete() {
  local ok=1
  local method
  for method in "${METHODS[@]}"; do
    local expected
    local predictions
    expected="$(expected_for_method "${method}")"
    predictions="$(count_predictions_for_method "${method}")"
    echo "[$(timestamp)] ${method} prediction rows=${predictions}/${expected}"
    if [[ "${predictions}" != "${expected}" ]]; then
      ok=0
    fi
  done
  [[ "${ok}" == "1" ]]
}

copy_method_official_artifacts() {
  local method="$1"
  local method_official_dir="${OFFICIAL_ROOT}/per_method/${method}"
  mkdir -p "${OFFICIAL_ROOT}/reports" "${OFFICIAL_ROOT}/reports_corrected" "${OFFICIAL_ROOT}/logs" "${OFFICIAL_ROOT}/run_logs"
  find "${method_official_dir}/reports" -maxdepth 1 -type f ! -name "._*" -name "*.json" -exec cp {} "${OFFICIAL_ROOT}/reports/" \; 2>/dev/null || true
  find "${method_official_dir}/reports_corrected" -maxdepth 1 -type f ! -name "._*" -name "*.json" -exec cp {} "${OFFICIAL_ROOT}/reports_corrected/" \; 2>/dev/null || true
  find "${method_official_dir}/logs" -maxdepth 1 -type f ! -name "._*" -exec cp {} "${OFFICIAL_ROOT}/logs/" \; 2>/dev/null || true
  if [[ -d "${method_official_dir}/run_logs" ]]; then
    cp -R "${method_official_dir}/run_logs/." "${OFFICIAL_ROOT}/run_logs/"
  fi
}

run_official_for_method() {
  local method="$1"
  local instance_ids
  local method_official_dir="${OFFICIAL_ROOT}/per_method/${method}"
  instance_ids="$(instances_csv_for_method "${method}")"
  mkdir -p /tmp/vacthbench_docker_config
  printf '{}' > /tmp/vacthbench_docker_config/config.json
  VACTHBENCH_DOCKER_PROXY= DOCKER_CONFIG=/tmp/vacthbench_docker_config \
    uv run --no-project --isolated --with swebench --with docker --with datasets --with pandas python \
    Scripts/run_swebench_official_patched.py \
    --prediction-dir "Results/${PREDICTION_DIR}" \
    --output-dir "${method_official_dir}" \
    --run-id-prefix "${OFFICIAL_DIR}_${method}" \
    --methods "${method}" \
    --instance-ids "${instance_ids}" \
    --timeout 1800 \
    --namespace none \
    --no-offline \
    > "${LOG_DIR}/official_eval_${method}.log" 2>&1
  status=$?
  copy_method_official_artifacts "${method}"
  return "${status}"
}

write_flip_analysis() {
  uv run --no-project --offline --isolated python - <<'PY'
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path

run_name = "swebench_lite_full_baselines_remaining_nonresolved_rerun001"
root = Path("Results")
manifest = json.loads((root / f"{run_name}_manifest.json").read_text())
official_root = root / "swebench_official_eval" / run_name


def corrected_report(method: str) -> Path:
    candidates = [
        path
        for path in (official_root / "reports_corrected").glob(f"{method}.*.corrected.json")
        if not path.name.startswith("._")
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one corrected report for {method}, found {len(candidates)}")
    return candidates[0]


def ids_for(method: str, key: str) -> set[str]:
    path = corrected_report(method)
    data = json.loads(path.read_text())
    values = data.get(key, [])
    if not isinstance(values, list):
        raise TypeError(f"{key} in {path} must be a list")
    return {str(item) for item in values}


status_by_method: dict[str, dict[str, str]] = {}
for method in manifest["methods"]:
    resolved = ids_for(method, "resolved_ids")
    empty = ids_for(method, "empty_patch_ids")
    errors = ids_for(method, "error_ids")
    status_by_method[method] = {}
    for instance_id in manifest["methods"][method]["selected_instance_ids"]:
        if instance_id in resolved:
            status = "resolved"
        elif instance_id in empty:
            status = "empty_patch"
        elif instance_id in errors:
            status = "error"
        else:
            status = "unresolved"
        status_by_method[method][instance_id] = status

rows = []
for row in manifest["rows"]:
    rerun_status = status_by_method[row["method"]][row["instance_id"]]
    rows.append(
        {
            **row,
            "rerun_official_status": rerun_status,
            "flip_to_resolved": rerun_status == "resolved",
            "source_was_empty_patch": row["source_official_status"] == "empty_patch",
        }
    )

by_method = []
for method in manifest["methods"]:
    method_rows = [row for row in rows if row["method"] == method]
    flips = sum(row["flip_to_resolved"] for row in method_rows)
    by_method.append(
        {
            "method": method,
            "rerun_expected": len(method_rows),
            "flip_to_resolved": flips,
            "flip_rate_pct": round(flips / len(method_rows) * 100, 2) if method_rows else 0.0,
            "source_empty_patch": sum(row["source_official_status"] == "empty_patch" for row in method_rows),
            "source_unresolved": sum(row["source_official_status"] == "unresolved" for row in method_rows),
            "rerun_resolved": sum(row["rerun_official_status"] == "resolved" for row in method_rows),
            "rerun_empty_patch": sum(row["rerun_official_status"] == "empty_patch" for row in method_rows),
            "rerun_unresolved": sum(row["rerun_official_status"] == "unresolved" for row in method_rows),
            "rerun_error": sum(row["rerun_official_status"] == "error" for row in method_rows),
        }
    )

output = {
    "run_name": run_name,
    "source_selection": manifest["selection"],
    "methods": by_method,
    "rows": rows,
    "total": {
        "rerun_expected": len(rows),
        "flip_to_resolved": sum(row["flip_to_resolved"] for row in rows),
        "flip_rate_pct": round(sum(row["flip_to_resolved"] for row in rows) / len(rows) * 100, 2) if rows else 0.0,
    },
}
(root / f"{run_name}_flip_analysis.json").write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")

lines = [
    "# SWE-bench Lite Remaining Baseline Non-Resolved Rerun Flip Analysis",
    "",
    f"Updated: {dt.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
    "",
    "| method | rerun N | flip to resolved | flip % | source empty | source unresolved | rerun resolved | rerun empty | rerun unresolved | rerun error |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
]
for row in by_method:
    lines.append(
        f"| {row['method']} | {row['rerun_expected']} | {row['flip_to_resolved']} | {row['flip_rate_pct']:.2f} | "
        f"{row['source_empty_patch']} | {row['source_unresolved']} | {row['rerun_resolved']} | "
        f"{row['rerun_empty_patch']} | {row['rerun_unresolved']} | {row['rerun_error']} |"
    )
lines.append(
    f"| **TOTAL** | {output['total']['rerun_expected']} | {output['total']['flip_to_resolved']} | "
    f"{output['total']['flip_rate_pct']:.2f} |  |  |  |  |  |  |"
)
lines.extend(
    [
        "",
        "## Instance Rows",
        "",
        "| source official run | method | instance | source status | rerun status | flip |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
)
for row in rows:
    lines.append(
        f"| {row['source_official_run']} | {row['method']} | {row['instance_id']} | "
        f"{row['source_official_status']} | {row['rerun_official_status']} | {row['flip_to_resolved']} |"
    )
(root / f"{run_name}_flip_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"wrote Results/{run_name}_flip_analysis.md")
PY
}

echo "[$(timestamp)] controller start"
echo "run_name=${RUN_NAME}"
echo "manifest=${MANIFEST}"
echo "max_local_jobs=${MAX_LOCAL_JOBS}"
write_status

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[$(timestamp)] missing manifest ${MANIFEST}"
  exit 1
fi

if [[ "${ALLOW_OVERLAP}" != "1" ]]; then
  blocking_count="$(blocking_official_eval_count)"
  if [[ "${blocking_count}" != "0" ]]; then
    echo "[$(timestamp)] refusing to start because ${blocking_count} other official eval process(es) are active"
    echo "[$(timestamp)] set VACTHBENCH_REMAINING_BASELINE_ALLOW_OVERLAP=1 to override intentionally"
    exit 2
  fi
fi

echo "[$(timestamp)] prebuilding images from ${COMBINED_TASK_FILE}"
mkdir -p /tmp/vacthbench_docker_config
printf '{}' > /tmp/vacthbench_docker_config/config.json
VACTHBENCH_DOCKER_PROXY= DOCKER_CONFIG=/tmp/vacthbench_docker_config \
  uv run --no-project --isolated --with swebench --with docker --with datasets python \
  Scripts/prebuild_swebench_images.py \
  --task-file "${COMBINED_TASK_FILE}" \
  --output "Results/${RUN_NAME}_image_prebuild_status.json" \
  --namespace none \
  --max-workers 1 \
  --retry-missing-per-instance \
  --retry-build-timeout-sec 1800 \
  > "${LOG_DIR}/prebuild_images.log" 2>&1
prebuild_status=$?
echo "[$(timestamp)] prebuild status=${prebuild_status}"
write_status

echo "[$(timestamp)] starting local method jobs"
pids=()
names=()
for method in "${METHODS[@]}"; do
  while [[ "${#pids[@]}" -ge "${MAX_LOCAL_JOBS}" ]]; do
    wait_first_job "${pids[0]}" "${names[0]}"
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
  done
  run_method "${method}" &
  pids+=("$!")
  names+=("${method}")
  last_index=$((${#pids[@]} - 1))
  echo "[$(timestamp)] launched ${method} pid=${pids[${last_index}]}"
done

for i in "${!pids[@]}"; do
  wait_first_job "${pids[$i]}" "${names[$i]}"
done

if ! verify_local_complete; then
  write_status
  echo "[$(timestamp)] stopping because not all local results are complete"
  exit 1
fi

echo "[$(timestamp)] exporting predictions"
uv run --no-project --offline --isolated python Scripts/export_swebench_predictions_aligned.py \
  --results-dir "Results/${RUN_PARENT}" \
  --output-dir "Results/${PREDICTION_DIR}" \
  --methods "${METHODS_CSV}" \
  --benchmark-prefixes "${BENCHMARK_PREFIX}" \
  --task-prefixes "${BENCHMARK_PREFIX}" \
  > "${LOG_DIR}/export_predictions.log" 2>&1
export_status=$?
echo "[$(timestamp)] export status=${export_status}"
write_status
if [[ "${export_status}" != "0" ]] || ! verify_predictions_complete; then
  echo "[$(timestamp)] stopping because predictions are incomplete"
  exit 1
fi

echo "[$(timestamp)] official eval"
for method in "${METHODS[@]}"; do
  echo "[$(timestamp)] official eval ${method}"
  if ! run_official_for_method "${method}"; then
    write_status
    echo "[$(timestamp)] official eval failed for ${method}"
    exit 1
  fi
  write_status
done

echo "[$(timestamp)] summarizing extended metrics"
uv run --no-project --offline --isolated python Scripts/summarize_swebench_aligned_metrics.py \
  --metadata "Results/${PREDICTION_DIR}/summary.json" \
  --reports-dir "${OFFICIAL_ROOT}/reports_corrected" \
  --run-logs-dir "${OFFICIAL_ROOT}/run_logs" \
  --output-dir "${OFFICIAL_ROOT}" \
  > "${LOG_DIR}/summarize.log" 2>&1
summary_status=$?
echo "[$(timestamp)] summarize status=${summary_status}"
if [[ "${summary_status}" != "0" ]]; then
  write_status
  exit "${summary_status}"
fi

echo "[$(timestamp)] writing flip analysis"
write_flip_analysis > "${LOG_DIR}/flip_analysis.log" 2>&1
flip_status=$?
echo "[$(timestamp)] flip analysis status=${flip_status}"
write_status
if [[ "${flip_status}" != "0" ]]; then
  exit "${flip_status}"
fi

echo "[$(timestamp)] controller complete"
