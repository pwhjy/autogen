#!/usr/bin/env bash
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_DIR="$(cd "${BENCH_DIR}/../../../.." && pwd)"
BENCHMARK_NAME="swebench_lite_full_baselines300"
RUN_PARENT="swebench_lite_full_baselines300"
METHODS=(summary vector_memory autogen_broadcast sliding_window structured_summary)
MAX_LOCAL_JOBS="${VACTHBENCH_BASELINES300_OBSERVER_MAX_JOBS:-5}"
TASK_TIMEOUT_SEC="${VACTHBENCH_TASK_TIMEOUT_SEC:-5400}"
MODEL_CALL_TIMEOUT_SEC="${VACTHBENCH_MODEL_CALL_TIMEOUT_SEC:-900}"
MODEL_CALL_RETRIES="${VACTHBENCH_MODEL_CALL_RETRIES:-2}"

cd "${BENCH_DIR}"

LOG_DIR="Results/${RUN_PARENT}/logs"
mkdir -p "${LOG_DIR}" "Results/${RUN_PARENT}"

if [[ "${MAX_LOCAL_JOBS}" -lt 1 ]]; then
  MAX_LOCAL_JOBS=1
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

official_summary_done() {
  [[ -f "Results/swebench_official_eval/${BENCHMARK_NAME}/extended_summary_metrics.csv" ]]
}

tmux_running() {
  local session="$1"
  tmux has-session -t "${session}" >/dev/null 2>&1
}

write_status() {
  uv run --no-project --offline --isolated python Scripts/write_baselines300_status.py \
    >> "${LOG_DIR}/status_monitor.log" 2>&1 || true
}

backfill_timeouts() {
  uv run --no-project --offline --isolated python Scripts/backfill_baselines300_fallback_results.py \
    >> "${LOG_DIR}/timeout_backfill.log" 2>&1 || true
}

start_controller() {
  if tmux_running baselines300-controller; then
    return 0
  fi

  tmux new-session -d -s baselines300-controller \
    "cd '${BENCH_DIR}' && { echo '[observer start controller]' \"\$(date '+%Y-%m-%d %H:%M:%S %Z')\"; VACTHBENCH_BASELINES300_TOTAL_JOBS=5 VACTHBENCH_TASK_TIMEOUT_SEC='${TASK_TIMEOUT_SEC}' VACTHBENCH_MODEL_CALL_TIMEOUT_SEC='${MODEL_CALL_TIMEOUT_SEC}' VACTHBENCH_MODEL_CALL_RETRIES='${MODEL_CALL_RETRIES}' ./Scripts/run_swebench_lite_full_baselines300_controller.sh; status=\$?; echo '[observer controller end]' \"\$(date '+%Y-%m-%d %H:%M:%S %Z')\" status=\$status; exit \$status; } >> Results/${RUN_PARENT}/logs/controller.log 2>&1"
  echo "[$(timestamp)] observer launched controller"
}

launch_method() {
  local method="$1"
  local session="baselines300-${method//_/-}"

  if active_method "${method}"; then
    return 0
  fi
  if tmux_running "${session}"; then
    return 0
  fi

  tmux new-session -d -s "${session}" \
    "cd '${BENCH_DIR}' && { echo '[observer restart ${method}]' \"\$(date '+%Y-%m-%d %H:%M:%S %Z')\"; AGBENCH_ALLOW_NATIVE=Yes VACTHBENCH_TASK_TIMEOUT_SEC='${TASK_TIMEOUT_SEC}' VACTHBENCH_MODEL_CALL_TIMEOUT_SEC='${MODEL_CALL_TIMEOUT_SEC}' VACTHBENCH_MODEL_CALL_RETRIES='${MODEL_CALL_RETRIES}' VACTHBENCH_SWEBENCH_USE_LOCAL_CACHE=1 VACTHBENCH_DOCKER_PULL=0 VACTHBENCH_GIT_FETCH_RETRIES=2 uv run --no-project --offline --isolated --with '${PYTHON_DIR}/packages/agbench' --with '${PYTHON_DIR}/packages/autogen-agentchat' --with '${PYTHON_DIR}/packages/autogen-core' --with '${PYTHON_DIR}/packages/autogen-ext[openai]' python Scripts/run_baseline.py Tasks/external_${BENCHMARK_NAME}_${method}.jsonl --results-dir ${RUN_PARENT}/external_${BENCHMARK_NAME}_${method}; status=\$?; echo '[observer ${method} end]' \"\$(date '+%Y-%m-%d %H:%M:%S %Z')\" status=\$status; exit \$status; } >> Results/${RUN_PARENT}/logs/local_${method}.log 2>&1"
  echo "[$(timestamp)] observer launched ${method}"
}

echo "[$(timestamp)] observer tick max_local_jobs=${MAX_LOCAL_JOBS} task_timeout_sec=${TASK_TIMEOUT_SEC} model_call_timeout_sec=${MODEL_CALL_TIMEOUT_SEC} model_call_retries=${MODEL_CALL_RETRIES}"

write_status
backfill_timeouts
write_status

if official_summary_done; then
  echo "[$(timestamp)] official summary already exists"
  exit 0
fi

if ! tmux_running baselines300-controller; then
  start_controller
fi

if all_local_done; then
  echo "[$(timestamp)] all local results are present; controller will handle export/eval"
  exit 0
fi

active="$(active_local_jobs)"
for method in "${METHODS[@]}"; do
  if [[ "$(result_count "${method}")" == "300" ]]; then
    continue
  fi
  if active_method "${method}"; then
    continue
  fi
  if [[ "${active}" -ge "${MAX_LOCAL_JOBS}" ]]; then
    echo "[$(timestamp)] observer at local job cap active=${active}; not launching more"
    break
  fi
  launch_method "${method}"
  active=$((active + 1))
done

write_status
