#!/usr/bin/env bash
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="$(cd "${BENCH_DIR}/../../../.." && pwd)"
RUN_NAME="swebench_lite_full_baselines_nonresolved_rerun001"
BENCHMARK_PREFIX="external_${RUN_NAME}"
METHODS=(summary vector_memory autogen_broadcast sliding_window structured_summary)
METHODS_CSV="summary,vector_memory,autogen_broadcast,sliding_window,structured_summary"
RUN_PARENT="${RUN_NAME}"
PREDICTION_DIR="swebench_predictions_${RUN_NAME}"
OFFICIAL_DIR="${RUN_NAME}"
MANIFEST="Results/${RUN_NAME}_manifest.json"
COMBINED_TASK_FILE="Tasks/${BENCHMARK_PREFIX}_all.jsonl"
LOG_DIR="Results/${RUN_PARENT}/logs"
STATUS_FILE="Results/${RUN_PARENT}_status.md"
CHUNK_DIR="Results/${RUN_PARENT}/chunks"
OFFICIAL_ROOT="Results/swebench_official_eval/${OFFICIAL_DIR}"
MAX_LOCAL_JOBS="${VACTHBENCH_BASELINE_RERUN_MAX_JOBS:-4}"
CHUNKS_PER_METHOD="${VACTHBENCH_BASELINE_RERUN_CHUNKS_PER_METHOD:-4}"
TASK_TIMEOUT_SEC="${VACTHBENCH_TASK_TIMEOUT_SEC:-5400}"
MODEL_CALL_TIMEOUT_SEC="${VACTHBENCH_MODEL_CALL_TIMEOUT_SEC:-900}"
MODEL_CALL_RETRIES="${VACTHBENCH_MODEL_CALL_RETRIES:-2}"

cd "${BENCH_DIR}"
mkdir -p "${LOG_DIR}" "${CHUNK_DIR}" "Results/${RUN_PARENT}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
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
  METHOD="${method}" PYTHONPATH=Scripts uv run --no-project --offline --isolated python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_baselines_nonresolved_rerun001_manifest.json").read_text())
print(manifest["methods"][os.environ["METHOD"]]["expected_tasks"])
PY
}

instances_csv_for_method() {
  local method="$1"
  METHOD="${method}" PYTHONPATH=Scripts uv run --no-project --offline --isolated python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_baselines_nonresolved_rerun001_manifest.json").read_text())
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

count_common_reports_for_method() {
  local report_dir="$1"
  local method="$2"
  if [[ -d "${report_dir}" ]]; then
    find "${report_dir}" -maxdepth 1 -type f ! -name "._*" -name "${method}.*.json" 2>/dev/null | wc -l | tr -d " "
  else
    echo "0"
  fi
}

active_local_jobs() {
  ps ax -o command= | awk '/run_baseline.py/ && /swebench_lite_full_baselines_nonresolved_rerun001/ && !/awk/ {count++} END {print count+0}'
}

write_status() {
  local phase="$1"
  local total_expected=0
  local total_results=0
  local total_predictions=0
  local method

  {
    echo "# SWE-bench Lite Baselines Non-Resolved Rerun001 Status"
    echo
    echo "Updated: $(timestamp)"
    echo
    echo "- phase: ${phase}"
    echo "- selection: official_status != resolved from swebench_lite_full_baselines300, per method"
    echo "- methods: ${METHODS_CSV}"
    echo "- max_local_jobs: ${MAX_LOCAL_JOBS}"
    echo "- chunks_per_method: ${CHUNKS_PER_METHOD}"
    echo "- task_timeout_sec: ${TASK_TIMEOUT_SEC}"
    echo "- model_call_timeout_sec: ${MODEL_CALL_TIMEOUT_SEC}"
    echo "- model_call_retries: ${MODEL_CALL_RETRIES}"
    echo "- active local run_baseline jobs: $(active_local_jobs)"
    echo
    echo "## Counts"
    echo
    echo "| method | expected | result.json | predictions | raw reports | corrected reports | source statuses |"
    echo "| --- | ---: | ---: | ---: | ---: | ---: | --- |"
    for method in "${METHODS[@]}"; do
      local expected=0
      local results=0
      local predictions=0
      local raw_reports=0
      local corrected_reports=0
      local source_statuses=""
      expected="$(expected_for_method "${method}")"
      results="$(count_results_for_method "${method}")"
      predictions="$(count_predictions_for_method "${method}")"
      raw_reports="$(count_common_reports_for_method "${OFFICIAL_ROOT}/reports" "${method}")"
      corrected_reports="$(count_common_reports_for_method "${OFFICIAL_ROOT}/reports_corrected" "${method}")"
      source_statuses="$(
        METHOD="${method}" uv run --no-project --offline --isolated python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_baselines_nonresolved_rerun001_manifest.json").read_text())
counts = manifest["methods"][os.environ["METHOD"]]["source_status_counts"]
print(", ".join(f"{key}: {value}" for key, value in sorted(counts.items())))
PY
      )"
      total_expected=$((total_expected + expected))
      total_results=$((total_results + results))
      total_predictions=$((total_predictions + predictions))
      echo "| ${method} | ${expected} | ${results} | ${predictions} | ${raw_reports} | ${corrected_reports} | ${source_statuses} |"
    done
    echo "| **TOTAL** | ${total_expected} | ${total_results} | ${total_predictions} |  |  |  |"
    echo
    echo "## Paths"
    echo
    echo "- manifest: \`${MANIFEST}\`"
    echo "- combined task file: \`${COMBINED_TASK_FILE}\`"
    echo "- local results: \`Results/${RUN_PARENT}/\`"
    echo "- predictions: \`Results/${PREDICTION_DIR}/\`"
    echo "- official eval: \`${OFFICIAL_ROOT}/\`"
    echo "- logs: \`${LOG_DIR}/\`"
  } > "${STATUS_FILE}"
}

write_chunks() {
  PYTHONPATH=Scripts CHUNKS_PER_METHOD="${CHUNKS_PER_METHOD}" uv run --no-project --offline --isolated python - <<'PY'
import json
import os
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_baselines_nonresolved_rerun001_manifest.json").read_text())
chunk_count = int(os.environ["CHUNKS_PER_METHOD"])
chunk_dir = Path("Results/swebench_lite_full_baselines_nonresolved_rerun001/chunks")
chunk_dir.mkdir(parents=True, exist_ok=True)
for old in chunk_dir.glob("*.instances"):
    old.unlink()
jobs_by_index: list[list[tuple[str, str, Path]]] = [[] for _ in range(chunk_count)]
for method, info in manifest["methods"].items():
    ids = info["selected_instance_ids"]
    for index in range(chunk_count):
        chunk = ids[index::chunk_count]
        path = chunk_dir / f"{method}_chunk{index}.instances"
        path.write_text(",".join(chunk) + "\n", encoding="utf-8")
        jobs_by_index[index].append((method, f"chunk{index}", path))
        print(f"{method} chunk{index}: {len(chunk)}")

with (chunk_dir / "job_order.tsv").open("w", encoding="utf-8") as fh:
    for jobs in jobs_by_index:
        for method, chunk_name, path in jobs:
            fh.write(f"{method}\t{chunk_name}\t{path}\n")
PY
}

agent_uv=(
  uv run --no-project --offline --isolated
  --with "${PYTHON_DIR}/packages/agbench"
  --with "${PYTHON_DIR}/packages/autogen-agentchat"
  --with "${PYTHON_DIR}/packages/autogen-core"
  --with "${PYTHON_DIR}/packages/autogen-ext[openai]"
  python Scripts/run_baseline.py
)

run_chunk() {
  local method="$1"
  local chunk_name="$2"
  local instances="$3"
  local task_file
  local results_arg
  local log_path="${LOG_DIR}/local_${method}_${chunk_name}.log"
  task_file="$(method_task_file "${method}")"
  results_arg="$(method_results_arg "${method}")"
  {
    echo "[$(timestamp)] START local ${method} ${chunk_name}"
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
    echo "[$(timestamp)] END local ${method} ${chunk_name} status=${exit_status}"
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
  write_status "local-running"
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

write_status "starting"
echo "[$(timestamp)] controller start"
echo "run_name=${RUN_NAME}"
echo "manifest=${MANIFEST}"

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[$(timestamp)] missing manifest ${MANIFEST}"
  write_status "missing-manifest"
  exit 1
fi

write_status "prebuild-images"
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

write_chunks > "${LOG_DIR}/chunks.log" 2>&1

write_status "local-running"
echo "[$(timestamp)] starting local chunks"
pids=()
names=()
while IFS=$'\t' read -r method chunk_name chunk_path; do
  [[ -z "${method}" ]] && continue
  instances="$(cat "${chunk_path}")"
  while [[ "${#pids[@]}" -ge "${MAX_LOCAL_JOBS}" ]]; do
    wait_first_job "${pids[0]}" "${names[0]}"
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
  done
  run_chunk "${method}" "${chunk_name}" "${instances}" &
  pids+=("$!")
  names+=("${method}_${chunk_name}")
  last_index=$((${#pids[@]} - 1))
  echo "[$(timestamp)] launched ${method} ${chunk_name} pid=${pids[${last_index}]}"
done < "${CHUNK_DIR}/job_order.tsv"

for i in "${!pids[@]}"; do
  wait_first_job "${pids[$i]}" "${names[$i]}"
done

if ! verify_local_complete; then
  write_status "local-incomplete"
  echo "[$(timestamp)] stopping because not all local results are complete"
  exit 1
fi

write_status "exporting-predictions"
uv run --no-project --offline --isolated python Scripts/export_swebench_predictions_aligned.py \
  --results-dir "Results/${RUN_PARENT}" \
  --output-dir "Results/${PREDICTION_DIR}" \
  --methods "${METHODS_CSV}" \
  --benchmark-prefixes "${BENCHMARK_PREFIX}" \
  --task-prefixes "${BENCHMARK_PREFIX}" \
  > "${LOG_DIR}/export_predictions.log" 2>&1
export_status=$?
echo "[$(timestamp)] export status=${export_status}"
if [[ "${export_status}" != "0" ]] || ! verify_predictions_complete; then
  write_status "export-incomplete"
  exit 1
fi

write_status "official-eval-running"
for method in "${METHODS[@]}"; do
  echo "[$(timestamp)] official eval ${method}"
  if ! run_official_for_method "${method}"; then
    write_status "official-eval-incomplete"
    echo "[$(timestamp)] official eval failed for ${method}"
    exit 1
  fi
  write_status "official-eval-running"
done

write_status "summarizing"
uv run --no-project --offline --isolated python Scripts/summarize_swebench_aligned_metrics.py \
  --metadata "Results/${PREDICTION_DIR}/summary.json" \
  --reports-dir "${OFFICIAL_ROOT}/reports_corrected" \
  --run-logs-dir "${OFFICIAL_ROOT}/run_logs" \
  --output-dir "${OFFICIAL_ROOT}" \
  > "${LOG_DIR}/summarize.log" 2>&1
summary_status=$?
echo "[$(timestamp)] summarize status=${summary_status}"
if [[ "${summary_status}" != "0" ]]; then
  write_status "summary-incomplete"
  exit "${summary_status}"
fi

write_status "complete"
echo "[$(timestamp)] controller complete"
