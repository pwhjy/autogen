#!/usr/bin/env bash
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="$(cd "${BENCH_DIR}/../../../.." && pwd)"
BENCHMARK_NAME="swebench_lite_full_vopt_unresolved131_rerun001"
METHOD="vacth_optimized"
TASK_FILE="Tasks/external_${BENCHMARK_NAME}_${METHOD}.jsonl"
RUN_PARENT="${BENCHMARK_NAME}"
RESULTS_ARG="${RUN_PARENT}/external_${BENCHMARK_NAME}_${METHOD}"
PREDICTION_DIR="swebench_predictions_${BENCHMARK_NAME}"
OFFICIAL_DIR="${BENCHMARK_NAME}"
LOG_DIR="Results/${RUN_PARENT}/logs"
STATUS_FILE="Results/${RUN_PARENT}_status.md"
CHUNK_DIR="Results/${RUN_PARENT}/chunks"
MAX_LOCAL_JOBS="${VACTHBENCH_VOPT_RERUN_MAX_JOBS:-4}"
TASK_TIMEOUT_SEC="${VACTHBENCH_TASK_TIMEOUT_SEC:-5400}"
MODEL_CALL_TIMEOUT_SEC="${VACTHBENCH_MODEL_CALL_TIMEOUT_SEC:-900}"
MODEL_CALL_RETRIES="${VACTHBENCH_MODEL_CALL_RETRIES:-2}"

cd "${BENCH_DIR}"
mkdir -p "${LOG_DIR}" "${CHUNK_DIR}" "Results/${RUN_PARENT}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

count_results() {
  find "Results/${RUN_PARENT}/external_${BENCHMARK_NAME}_${METHOD}" -path "*/0/result.json" -type f 2>/dev/null | wc -l | tr -d " "
}

write_status() {
  local phase="$1"
  local result_count=0
  local task_rows=0
  local pred_rows=0
  local raw_reports=0
  local corrected_reports=0
  local metrics_files=0

  result_count="$(count_results)"
  if [[ -f "${TASK_FILE}" ]]; then
    task_rows="$(wc -l < "${TASK_FILE}" | tr -d " ")"
  fi
  if [[ -f "Results/${PREDICTION_DIR}/${METHOD}.jsonl" ]]; then
    pred_rows="$(wc -l < "Results/${PREDICTION_DIR}/${METHOD}.jsonl" | tr -d " ")"
  fi
  if [[ -d "Results/swebench_official_eval/${OFFICIAL_DIR}/reports" ]]; then
    raw_reports="$(find "Results/swebench_official_eval/${OFFICIAL_DIR}/reports" -maxdepth 1 -type f ! -name "._*" -name "*.json" 2>/dev/null | wc -l | tr -d " ")"
  fi
  if [[ -d "Results/swebench_official_eval/${OFFICIAL_DIR}/reports_corrected" ]]; then
    corrected_reports="$(find "Results/swebench_official_eval/${OFFICIAL_DIR}/reports_corrected" -maxdepth 1 -type f ! -name "._*" -name "*.json" 2>/dev/null | wc -l | tr -d " ")"
  fi
  if [[ -d "Results/swebench_official_eval/${OFFICIAL_DIR}" ]]; then
    metrics_files="$(find "Results/swebench_official_eval/${OFFICIAL_DIR}" -maxdepth 1 -type f ! -name "._*" \( -name "*metrics*.csv" -o -name "*metrics*.json" -o -name "*metrics*.md" \) 2>/dev/null | wc -l | tr -d " ")"
  fi

  {
    echo "# SWE-bench Lite VACTH Optimized Unresolved131 Rerun001 Status"
    echo
    echo "Updated: $(timestamp)"
    echo
    echo "- phase: ${phase}"
    echo "- method: ${METHOD}"
    echo "- selection: official_status != resolved from swebench_lite_full_vacth_optimized_300_dedup"
    echo "- expected rerun tasks: 131"
    echo "- selected source statuses: 127 unresolved + 4 empty_patch"
    echo "- max_local_jobs: ${MAX_LOCAL_JOBS}"
    echo "- task_timeout_sec: ${TASK_TIMEOUT_SEC}"
    echo "- model_call_timeout_sec: ${MODEL_CALL_TIMEOUT_SEC}"
    echo "- model_call_retries: ${MODEL_CALL_RETRIES}"
    echo
    echo "## Counts"
    echo
    echo "| artifact | count |"
    echo "| --- | ---: |"
    echo "| task rows | ${task_rows} |"
    echo "| local result.json | ${result_count} |"
    echo "| prediction rows | ${pred_rows} |"
    echo "| raw official reports | ${raw_reports} |"
    echo "| corrected reports | ${corrected_reports} |"
    echo "| metrics files | ${metrics_files} |"
    echo
    echo "## Paths"
    echo
    echo "- task file: \`${TASK_FILE}\`"
    echo "- local results: \`Results/${RUN_PARENT}/\`"
    echo "- predictions: \`Results/${PREDICTION_DIR}/\`"
    echo "- official eval: \`Results/swebench_official_eval/${OFFICIAL_DIR}/\`"
    echo "- logs: \`${LOG_DIR}/\`"
  } > "${STATUS_FILE}"
}

instance_ids_csv() {
  PYTHONPATH=Scripts uv run --no-project --offline --isolated python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_vopt_unresolved131_rerun001_manifest.json").read_text())
print(",".join(manifest["selected_instance_ids"]))
PY
}

write_chunks() {
  PYTHONPATH=Scripts uv run --no-project --offline --isolated python - <<'PY'
import json
from pathlib import Path

manifest = json.loads(Path("Results/swebench_lite_full_vopt_unresolved131_rerun001_manifest.json").read_text())
ids = manifest["selected_instance_ids"]
chunk_dir = Path("Results/swebench_lite_full_vopt_unresolved131_rerun001/chunks")
chunk_dir.mkdir(parents=True, exist_ok=True)
chunk_count = 4
for index in range(chunk_count):
    chunk = ids[index::chunk_count]
    (chunk_dir / f"chunk{index}.instances").write_text(",".join(chunk) + "\n", encoding="utf-8")
    print(f"chunk{index}: {len(chunk)}")
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
  local chunk_name="$1"
  local instances="$2"
  local log_path="${LOG_DIR}/local_${chunk_name}.log"
  {
    echo "[$(timestamp)] START local ${chunk_name}"
    echo "instances=${instances}"
    AGBENCH_ALLOW_NATIVE=Yes \
      VACTHBENCH_TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_TIMEOUT_SEC="${MODEL_CALL_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_RETRIES="${MODEL_CALL_RETRIES}" \
      VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE=1 \
      VACTHBENCH_DOCKER_PULL=0 \
      VACTHBENCH_GIT_FETCH_RETRIES=2 \
      VACTHBENCH_DOCKER_PROXY= \
      "${agent_uv[@]}" "${TASK_FILE}" --results-dir "${RESULTS_ARG}" --instances "${instances}"
    status=$?
    echo "[$(timestamp)] END local ${chunk_name} status=${status}"
    exit "${status}"
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

write_status "starting"
echo "[$(timestamp)] controller start"
echo "task_file=${TASK_FILE}"
echo "results_arg=${RESULTS_ARG}"

write_status "prebuild-images"
echo "[$(timestamp)] prebuilding images"
mkdir -p /tmp/vacthbench_docker_config
printf '{}' > /tmp/vacthbench_docker_config/config.json
VACTHBENCH_DOCKER_PROXY= DOCKER_CONFIG=/tmp/vacthbench_docker_config \
  uv run --no-project --isolated --with swebench --with docker --with datasets python \
  Scripts/prebuild_swebench_images.py \
  --task-file "${TASK_FILE}" \
  --output "Results/${BENCHMARK_NAME}_image_prebuild_status.json" \
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
for chunk_file in "${CHUNK_DIR}"/chunk*.instances; do
  chunk_name="$(basename "${chunk_file}" .instances)"
  instances="$(cat "${chunk_file}")"
  while [[ "${#pids[@]}" -ge "${MAX_LOCAL_JOBS}" ]]; do
    wait_first_job "${pids[0]}" "${names[0]}"
    pids=("${pids[@]:1}")
    names=("${names[@]:1}")
  done
  run_chunk "${chunk_name}" "${instances}" &
  pids+=("$!")
  names+=("${chunk_name}")
  last_index=$((${#pids[@]} - 1))
  echo "[$(timestamp)] launched ${chunk_name} pid=${pids[${last_index}]}"
done

for i in "${!pids[@]}"; do
  wait_first_job "${pids[$i]}" "${names[$i]}"
done

result_count="$(count_results)"
echo "[$(timestamp)] local result count=${result_count}"
if [[ "${result_count}" != "131" ]]; then
  write_status "local-incomplete"
  echo "[$(timestamp)] stopping because local result count is not 131"
  exit 1
fi

write_status "exporting-predictions"
uv run --no-project --offline --isolated python Scripts/export_swebench_predictions_aligned.py \
  --results-dir "Results/${RUN_PARENT}" \
  --output-dir "Results/${PREDICTION_DIR}" \
  --methods "${METHOD}" \
  --benchmark-prefixes "external_${BENCHMARK_NAME}" \
  --task-prefixes "external_${BENCHMARK_NAME}" \
  > "${LOG_DIR}/export_predictions.log" 2>&1
export_status=$?
pred_rows=0
if [[ -f "Results/${PREDICTION_DIR}/${METHOD}.jsonl" ]]; then
  pred_rows="$(wc -l < "Results/${PREDICTION_DIR}/${METHOD}.jsonl" | tr -d " ")"
fi
echo "[$(timestamp)] export status=${export_status} rows=${pred_rows}"
if [[ "${export_status}" != "0" || "${pred_rows}" != "131" ]]; then
  write_status "export-incomplete"
  exit 1
fi

INSTANCE_IDS="$(instance_ids_csv)"
write_status "official-eval-running"
mkdir -p /tmp/vacthbench_docker_config
printf '{}' > /tmp/vacthbench_docker_config/config.json
VACTHBENCH_DOCKER_PROXY= DOCKER_CONFIG=/tmp/vacthbench_docker_config \
  uv run --no-project --isolated --with swebench --with docker --with datasets --with pandas python \
  Scripts/run_swebench_official_patched.py \
  --prediction-dir "Results/${PREDICTION_DIR}" \
  --output-dir "Results/swebench_official_eval/${OFFICIAL_DIR}" \
  --run-id-prefix "${OFFICIAL_DIR}" \
  --methods "${METHOD}" \
  --instance-ids "${INSTANCE_IDS}" \
  --timeout 1800 \
  --namespace none \
  --no-offline \
  > "${LOG_DIR}/official_eval.log" 2>&1
official_status=$?
echo "[$(timestamp)] official eval status=${official_status}"
if [[ "${official_status}" != "0" ]]; then
  write_status "official-eval-incomplete"
  exit "${official_status}"
fi

write_status "summarizing"
uv run --no-project --offline --isolated python Scripts/summarize_swebench_aligned_metrics.py \
  --metadata "Results/${PREDICTION_DIR}/summary.json" \
  --reports-dir "Results/swebench_official_eval/${OFFICIAL_DIR}/reports_corrected" \
  --run-logs-dir "Results/swebench_official_eval/${OFFICIAL_DIR}/run_logs" \
  --output-dir "Results/swebench_official_eval/${OFFICIAL_DIR}" \
  > "${LOG_DIR}/summarize.log" 2>&1
summary_status=$?
echo "[$(timestamp)] summarize status=${summary_status}"
if [[ "${summary_status}" != "0" ]]; then
  write_status "summary-incomplete"
  exit "${summary_status}"
fi

write_status "complete"
echo "[$(timestamp)] controller complete"
