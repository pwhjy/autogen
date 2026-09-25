#!/usr/bin/env bash
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="$(cd "${BENCH_DIR}/../../../.." && pwd)"
BENCHMARK_NAME="swebench_lite_full_baselines300"
RUN_PARENT="swebench_lite_full_baselines300"
METHODS=(summary vector_memory autogen_broadcast sliding_window structured_summary)
METHODS_CSV="summary,vector_memory,autogen_broadcast,sliding_window,structured_summary"
TOTAL_LOCAL_JOBS="${VACTHBENCH_BASELINES300_TOTAL_JOBS:-4}"
TASK_TIMEOUT_SEC="${VACTHBENCH_TASK_TIMEOUT_SEC:-5400}"
MODEL_CALL_TIMEOUT_SEC="${VACTHBENCH_MODEL_CALL_TIMEOUT_SEC:-900}"
MODEL_CALL_RETRIES="${VACTHBENCH_MODEL_CALL_RETRIES:-2}"
POLL_SEC="${VACTHBENCH_BASELINES300_POLL_SEC:-120}"

cd "${BENCH_DIR}"

LOG_DIR="Results/${RUN_PARENT}/logs"
mkdir -p "${LOG_DIR}" "Results/${RUN_PARENT}"

if [[ "${TOTAL_LOCAL_JOBS}" -lt 1 ]]; then
  TOTAL_LOCAL_JOBS=1
fi

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

result_count() {
  local method="$1"
  local root="Results/${RUN_PARENT}/external_${BENCHMARK_NAME}_${method}"
  find "${root}" -mindepth 3 -maxdepth 3 -path "*/0/result.json" -type f 2>/dev/null | wc -l | tr -d " "
}

active_method() {
  local method="$1"
  pgrep -f "run_baseline.py Tasks/external_${BENCHMARK_NAME}_${method}.jsonl" >/dev/null 2>&1
}

active_local_jobs() {
  local count=0
  local method
  for method in "${METHODS[@]}"; do
    if active_method "${method}"; then
      count=$((count + 1))
    fi
  done
  echo "${count}"
}

all_local_done() {
  local method
  for method in "${METHODS[@]}"; do
    if [[ "$(result_count "${method}")" != "300" ]]; then
      return 1
    fi
  done
  return 0
}

write_status() {
  uv run --no-project --offline --isolated python Scripts/write_baselines300_status.py \
    >> "${LOG_DIR}/status_monitor.log" 2>&1 || true
}

agent_uv=(
  uv run --no-project --offline --isolated
  --with "${PYTHON_DIR}/packages/agbench"
  --with "${PYTHON_DIR}/packages/autogen-agentchat"
  --with "${PYTHON_DIR}/packages/autogen-core"
  --with "${PYTHON_DIR}/packages/autogen-ext[openai]"
  python Scripts/run_baseline.py
)

launch_method() {
  local method="$1"
  local task_file="Tasks/external_${BENCHMARK_NAME}_${method}.jsonl"
  local results_arg="${RUN_PARENT}/external_${BENCHMARK_NAME}_${method}"
  local log_path="${LOG_DIR}/local_${method}.log"

  {
    echo "[$(timestamp)] CONTROLLER START local ${method}"
    echo "task_file=${task_file}"
    echo "results_arg=${results_arg}"
    echo "existing_result_json=$(result_count "${method}")"
    AGBENCH_ALLOW_NATIVE=Yes \
      VACTHBENCH_TASK_TIMEOUT_SEC="${TASK_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_TIMEOUT_SEC="${MODEL_CALL_TIMEOUT_SEC}" \
      VACTHBENCH_MODEL_CALL_RETRIES="${MODEL_CALL_RETRIES}" \
      VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE=1 \
      VACTHBENCH_DOCKER_PULL=0 \
      VACTHBENCH_GIT_FETCH_RETRIES=2 \
      "${agent_uv[@]}" "${task_file}" --results-dir "${results_arg}"
    status=$?
    echo "[$(timestamp)] CONTROLLER END local ${method} status=${status}"
    exit "${status}"
  } >> "${log_path}" 2>&1 &
  echo "[$(timestamp)] launched ${method} pid=$!"
}

export_predictions() {
  echo "[$(timestamp)] exporting predictions"
  uv run --no-project --offline --isolated python Scripts/export_swebench_predictions_aligned.py \
    --results-dir "Results/${RUN_PARENT}" \
    --output-dir "Results/swebench_predictions_${BENCHMARK_NAME}" \
    --methods "${METHODS_CSV}" \
    --benchmark-prefixes "external_${BENCHMARK_NAME}" \
    --task-prefixes "external_${BENCHMARK_NAME}" \
    > "${LOG_DIR}/export_predictions.log" 2>&1
}

prediction_rows_ok() {
  local method
  for method in "${METHODS[@]}"; do
    local file="Results/swebench_predictions_${BENCHMARK_NAME}/${method}.jsonl"
    if [[ ! -f "${file}" ]]; then
      return 1
    fi
    if [[ "$(wc -l < "${file}" | tr -d " ")" != "300" ]]; then
      return 1
    fi
  done
  return 0
}

instance_ids() {
  uv run --no-project --offline --isolated python - <<'PY'
import json
from pathlib import Path
path = Path("data/external/swebench_lite/index_test.jsonl")
print(",".join(json.loads(line)["instance_id"] for line in path.open()))
PY
}

run_official_eval() {
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
    --instance-ids "$(instance_ids)" \
    --timeout 1800 \
    --namespace none \
    --no-offline \
    > "${LOG_DIR}/official_eval.log" 2>&1
}

run_summary() {
  echo "[$(timestamp)] summarizing official eval"
  uv run --no-project --offline --isolated python Scripts/summarize_swebench_aligned_metrics.py \
    --metadata "Results/swebench_predictions_${BENCHMARK_NAME}/summary.json" \
    --reports-dir "Results/swebench_official_eval/${BENCHMARK_NAME}/reports_corrected" \
    --run-logs-dir "Results/swebench_official_eval/${BENCHMARK_NAME}/run_logs" \
    --output-dir "Results/swebench_official_eval/${BENCHMARK_NAME}" \
    > "${LOG_DIR}/summarize.log" 2>&1
}

echo "[$(timestamp)] controller start"
echo "methods=${METHODS_CSV}"
echo "total_local_jobs=${TOTAL_LOCAL_JOBS}"
echo "task_timeout_sec=${TASK_TIMEOUT_SEC}"
echo "model_call_timeout_sec=${MODEL_CALL_TIMEOUT_SEC}"
echo "model_call_retries=${MODEL_CALL_RETRIES}"

while true; do
  write_status
  if all_local_done; then
    echo "[$(timestamp)] all local result.json counts are 300"
    break
  fi

  active="$(active_local_jobs)"
  for method in "${METHODS[@]}"; do
    if [[ "$(result_count "${method}")" == "300" ]]; then
      continue
    fi
    if active_method "${method}"; then
      continue
    fi
    if [[ "${active}" -ge "${TOTAL_LOCAL_JOBS}" ]]; then
      break
    fi
    launch_method "${method}"
    active=$((active + 1))
  done
  sleep "${POLL_SEC}"
done

write_status
export_predictions || exit $?
if ! prediction_rows_ok; then
  echo "[$(timestamp)] prediction export incomplete; stopping before official eval"
  exit 1
fi
run_official_eval || exit $?
run_summary || exit $?
write_status
echo "[$(timestamp)] controller complete"
