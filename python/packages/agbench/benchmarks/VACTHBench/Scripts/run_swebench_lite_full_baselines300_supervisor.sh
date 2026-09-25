#!/usr/bin/env bash
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="$(cd "${BENCH_DIR}/../../../.." && pwd)"
BENCHMARK_NAME="swebench_lite_full_baselines300"
RUN_PARENT="swebench_lite_full_baselines300"
METHODS=(summary vector_memory autogen_broadcast sliding_window structured_summary)
METHODS_CSV="summary,vector_memory,autogen_broadcast,sliding_window,structured_summary"
MAX_LOCAL_JOBS="${VACTHBENCH_BASELINES300_MAX_JOBS:-2}"
TASK_TIMEOUT_SEC="${VACTHBENCH_TASK_TIMEOUT_SEC:-5400}"
MODEL_CALL_TIMEOUT_SEC="${VACTHBENCH_MODEL_CALL_TIMEOUT_SEC:-900}"
MODEL_CALL_RETRIES="${VACTHBENCH_MODEL_CALL_RETRIES:-2}"

cd "${BENCH_DIR}"

LOG_DIR="Results/${RUN_PARENT}/logs"
STATUS_FILE="Results/${RUN_PARENT}_status.md"
mkdir -p "${LOG_DIR}" "Results/${RUN_PARENT}"
if [[ "${MAX_LOCAL_JOBS}" -lt 1 ]]; then
  MAX_LOCAL_JOBS=1
fi

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

write_status() {
  local phase="$1"
  {
    echo "# SWE-bench Lite Full Baselines300 Status"
    echo
    echo "Updated: $(timestamp)"
    echo
    echo "- phase: ${phase}"
    echo "- methods: ${METHODS_CSV}"
    echo "- vacth_full: excluded"
    echo "- max_local_jobs: ${MAX_LOCAL_JOBS}"
    echo "- task_timeout_sec: ${TASK_TIMEOUT_SEC}"
    echo "- model_call_timeout_sec: ${MODEL_CALL_TIMEOUT_SEC}"
    echo "- model_call_retries: ${MODEL_CALL_RETRIES}"
    echo
    echo "## Local Result Counts"
    echo
    echo "| method | result.json | task dirs | task file rows |"
    echo "| --- | ---: | ---: | ---: |"
    for method in "${METHODS[@]}"; do
      local root="Results/${RUN_PARENT}/external_${BENCHMARK_NAME}_${method}"
      local task_file="Tasks/external_${BENCHMARK_NAME}_${method}.jsonl"
      local results_count=0
      local task_dirs=0
      local task_rows=0
      if [[ -d "${root}" ]]; then
        results_count="$(find "${root}" -path "*/0/result.json" -type f 2>/dev/null | wc -l | tr -d " ")"
        task_dirs="$(find "${root}" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | wc -l | tr -d " ")"
      fi
      if [[ -f "${task_file}" ]]; then
        task_rows="$(wc -l < "${task_file}" | tr -d " ")"
      fi
      echo "| ${method} | ${results_count} | ${task_dirs} | ${task_rows} |"
    done
    echo
    echo "## Artifacts"
    echo
    echo "- local results: \`Results/${RUN_PARENT}/\`"
    echo "- local logs: \`Results/${RUN_PARENT}/logs/\`"
    echo "- predictions: \`Results/swebench_predictions_${BENCHMARK_NAME}/\`"
    echo "- official eval: \`Results/swebench_official_eval/${BENCHMARK_NAME}/\`"
  } > "${STATUS_FILE}"
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
  local task_file="Tasks/external_${BENCHMARK_NAME}_${method}.jsonl"
  local results_arg="${RUN_PARENT}/external_${BENCHMARK_NAME}_${method}"
  local log_path="${LOG_DIR}/local_${method}.log"

  {
    echo "[$(timestamp)] START local ${method}"
    echo "task_file=${task_file}"
    echo "results_arg=${results_arg}"
    AGBENCH_ALLOW_NATIVE=Yes \
      VACTHBENCH_TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_TIMEOUT_SEC="${MODEL_CALL_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_RETRIES="${MODEL_CALL_RETRIES}" \
      VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE=1 \
      VACTHBENCH_DOCKER_PULL=0 \
      VACTHBENCH_GIT_FETCH_RETRIES=2 \
      "${agent_uv[@]}" "${task_file}" --results-dir "${results_arg}"
    status=$?
    echo "[$(timestamp)] END local ${method} status=${status}"
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
echo "[$(timestamp)] supervisor start"
echo "methods=${METHODS_CSV}"
echo "max_local_jobs=${MAX_LOCAL_JOBS}"
echo "model_call_timeout_sec=${MODEL_CALL_TIMEOUT_SEC}"
echo "model_call_retries=${MODEL_CALL_RETRIES}"

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
  echo "[$(timestamp)] launched local ${method} pid=${pids[${last_index}]}"
  write_status "local-running"
done

for i in "${!pids[@]}"; do
  wait_first_job "${pids[$i]}" "${names[$i]}"
done

write_status "local-complete-exporting"
echo "[$(timestamp)] local phase complete; exporting predictions"

uv run --no-project --offline --isolated python Scripts/export_swebench_predictions_aligned.py \
  --results-dir "Results/${RUN_PARENT}" \
  --output-dir "Results/swebench_predictions_${BENCHMARK_NAME}" \
  --methods "${METHODS_CSV}" \
  --benchmark-prefixes "external_${BENCHMARK_NAME}" \
  --task-prefixes "external_${BENCHMARK_NAME}" \
  > "${LOG_DIR}/export_predictions.log" 2>&1
export_status=$?
echo "[$(timestamp)] export status=${export_status}"

bad_predictions=0
for method in "${METHODS[@]}"; do
  prediction_file="Results/swebench_predictions_${BENCHMARK_NAME}/${method}.jsonl"
  rows=0
  if [[ -f "${prediction_file}" ]]; then
    rows="$(wc -l < "${prediction_file}" | tr -d " ")"
  fi
  echo "[$(timestamp)] predictions ${method} rows=${rows}"
  if [[ "${rows}" != "300" ]]; then
    bad_predictions=1
  fi
done
if [[ "${export_status}" != "0" || "${bad_predictions}" != "0" ]]; then
  write_status "export-incomplete"
  echo "[$(timestamp)] stopping before official eval because prediction export is incomplete"
  exit 1
fi

INSTANCE_IDS="$(
  uv run --no-project --offline --isolated python - <<'PY'
import json
from pathlib import Path
path = Path("data/external/swebench_lite/index_test.jsonl")
print(",".join(json.loads(line)["instance_id"] for line in path.open()))
PY
)"

write_status "official-eval-running"
echo "[$(timestamp)] starting official eval"
mkdir -p /tmp/vacthbench_docker_config
printf '{}' > /tmp/vacthbench_docker_config/config.json
DOCKER_CONFIG=/tmp/vacthbench_docker_config \
  uv run --no-project --isolated --with swebench --with docker --with datasets --with pandas python \
  Scripts/run_swebench_official_patched.py \
  --prediction-dir "Results/swebench_predictions_${BENCHMARK_NAME}" \
  --output-dir "Results/swebench_official_eval/${BENCHMARK_NAME}" \
  --run-id-prefix "${BENCHMARK_NAME}" \
  --methods "${METHODS_CSV}" \
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
  --metadata "Results/swebench_predictions_${BENCHMARK_NAME}/summary.json" \
  --reports-dir "Results/swebench_official_eval/${BENCHMARK_NAME}/reports_corrected" \
  --run-logs-dir "Results/swebench_official_eval/${BENCHMARK_NAME}/run_logs" \
  --output-dir "Results/swebench_official_eval/${BENCHMARK_NAME}" \
  > "${LOG_DIR}/summarize.log" 2>&1
summary_status=$?
echo "[$(timestamp)] summarize status=${summary_status}"
if [[ "${summary_status}" != "0" ]]; then
  write_status "summary-incomplete"
  exit "${summary_status}"
fi

write_status "complete"
echo "[$(timestamp)] supervisor complete"
