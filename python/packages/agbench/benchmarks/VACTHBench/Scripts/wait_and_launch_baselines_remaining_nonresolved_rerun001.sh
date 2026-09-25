#!/usr/bin/env bash
set -uo pipefail

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="swebench_lite_full_baselines_remaining_nonresolved_rerun001"
LOG_DIR="${BENCH_DIR}/Results/${RUN_NAME}/logs"
LOG_PATH="${LOG_DIR}/wait_and_launch.log"
POLL_SEC="${VACTHBENCH_REMAINING_BASELINE_WAIT_POLL_SEC:-300}"
MIN_FREE_GIB="${VACTHBENCH_REMAINING_BASELINE_MIN_FREE_GIB:-50}"

mkdir -p "${LOG_DIR}"
cd "${BENCH_DIR}"

timestamp() {
  date "+%Y-%m-%d %H:%M:%S %Z"
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${LOG_PATH}"
}

big_finalizer_count() {
  ps ax -o command= | awk '
    /finalize_swebench_lite_full_baselines_nonresolved_rerun001.py/ && $0 !~ /awk/ {count++}
    END {print count+0}
  '
}

other_official_eval_count() {
  ps ax -o command= | awk '
    /run_swebench_official_patched.py/ && $0 !~ /swebench_lite_full_baselines_remaining_nonresolved_rerun001/ && $0 !~ /awk/ {count++}
    END {print count+0}
  '
}

free_gib() {
  df -g /Volumes/'ZHITAI SSD' | awk 'NR==2 {print $4}'
}

remaining_controller_count() {
  ps ax -o command= | awk '
    /run_swebench_lite_baselines_remaining_nonresolved_rerun001_controller.sh/ && $0 !~ /awk/ {count++}
    END {print count+0}
  '
}

refresh_status() {
  uv run --no-project --offline --isolated python Scripts/write_baselines_remaining_nonresolved_rerun001_status.py >> "${LOG_PATH}" 2>&1 || true
}

log "wait-and-launch start poll_sec=${POLL_SEC} min_free_gib=${MIN_FREE_GIB}"
refresh_status

while true; do
  if [[ "$(remaining_controller_count)" != "0" ]]; then
    log "remaining rerun controller is already active; exiting waiter"
    exit 0
  fi

  finalizers="$(big_finalizer_count)"
  official="$(other_official_eval_count)"
  free="$(free_gib)"
  log "gate finalizer=${finalizers} other_official_eval=${official} free_gib=${free}"

  if [[ "${free}" =~ ^[0-9]+$ ]] && (( free < MIN_FREE_GIB )); then
    log "disk free ${free}GiB is below ${MIN_FREE_GIB}GiB; not starting remaining rerun"
    exit 3
  fi

  if [[ "${finalizers}" == "0" && "${official}" == "0" ]]; then
    break
  fi

  sleep "${POLL_SEC}"
done

log "resource gate clear; launching remaining rerun controller"
exec Scripts/run_swebench_lite_baselines_remaining_nonresolved_rerun001_controller.sh
