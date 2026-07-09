#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 RUN_ID ARMS_SPEC TASKS_SPEC LIMIT MODEL_PATH BATCH_SIZE GPU_MEM_UTIL MAX_MODEL_LEN LMEVAL_TIMEOUT_SECONDS" >&2
  exit 64
fi

RUN_ID="$1"
ARMS_SPEC="$2"
TASKS_SPEC="$3"
LIMIT="$4"
MODEL_PATH="$5"
BATCH_SIZE="$6"
GPU_MEM_UTIL="$7"
MAX_MODEL_LEN="$8"
LMEVAL_TIMEOUT_SECONDS="$9"

MIN_CUDA_GLOBAL_MEMORY_MB="${MIN_CUDA_GLOBAL_MEMORY_MB:-40000}"
PIP_TIMEOUT_SECONDS="${PIP_TIMEOUT_SECONDS:-900}"
RESULT_DIR="${PWD}/retention_failure_suite_${RUN_ID}"
OUT_BUNDLE="retention_failure_suite_${RUN_ID}_results.tgz"
PYDEPS="${PWD}/pydeps"
ADAPTER_ROOT="${ADAPTER_ROOT:-/staging/s/suresh27/adapters}"

mkdir -p "${RESULT_DIR}" "${PYDEPS}"

log_step() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RESULT_DIR}/progress.log"
}

stop_monitor() {
  if [[ -n "${MONITOR_PID:-}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
}

finish() {
  status=$?
  stop_monitor
  log_step "runner_exit status=${status}"
  echo "${status}" > "${RESULT_DIR}/exit_status.txt"
  tar -czf "${OUT_BUNDLE}" -C "${RESULT_DIR}" . 2>/dev/null || true
  exit "${status}"
}
trap finish EXIT

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "No python interpreter found in container." >&2
  exit 127
fi

ORIGINAL_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-unset}"
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=${PWD}"
  echo "run_id=${RUN_ID}"
  echo "arms_spec=${ARMS_SPEC}"
  echo "tasks_spec=${TASKS_SPEC}"
  echo "limit=${LIMIT}"
  echo "model_path=${MODEL_PATH}"
  echo "adapter_root=${ADAPTER_ROOT}"
  echo "local_adapter_bundles=${LOCAL_ADAPTER_BUNDLES:-}"
  echo "batch_size=${BATCH_SIZE}"
  echo "gpu_mem_util=${GPU_MEM_UTIL}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "min_cuda_global_memory_mb=${MIN_CUDA_GLOBAL_MEMORY_MB}"
  echo "pip_timeout_seconds=${PIP_TIMEOUT_SECONDS}"
  echo "lmeval_timeout_seconds=${LMEVAL_TIMEOUT_SECONDS}"
  echo "python_bin=${PYTHON_BIN}"
  echo "original_cuda_visible_devices=${ORIGINAL_CUDA_VISIBLE_DEVICES}"
  echo "effective_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "CONDOR_SLOT=${_CONDOR_SLOT:-unset}"
} > "${RESULT_DIR}/run_env.txt"

log_step "gpu_probe_start"
(nvidia-smi || true) > "${RESULT_DIR}/nvidia_smi.txt" 2>&1
(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits || true) > "${RESULT_DIR}/gpu_memory_query.txt" 2>&1
GPU_MEM_MB="$(head -n 1 "${RESULT_DIR}/gpu_memory_query.txt" | awk -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2}')"
if [[ -z "${GPU_MEM_MB}" || ! "${GPU_MEM_MB}" =~ ^[0-9]+$ ]]; then
  echo "Could not determine GPU memory from nvidia-smi." >&2
  exit 67
fi
if (( GPU_MEM_MB < MIN_CUDA_GLOBAL_MEMORY_MB )); then
  echo "GPU memory ${GPU_MEM_MB} MB is below required ${MIN_CUDA_GLOBAL_MEMORY_MB} MB." >&2
  exit 67
fi
log_step "gpu_probe_ok memory_mb=${GPU_MEM_MB}"

IFS='+' read -r -a ARMS <<< "${ARMS_SPEC}"
IFS='+' read -r -a TASKS <<< "${TASKS_SPEC}"

if [[ -n "${LOCAL_ADAPTER_BUNDLES:-}" ]]; then
  LOCAL_ADAPTER_ROOT="${PWD}/adapters"
  mkdir -p "${LOCAL_ADAPTER_ROOT}"
  IFS='+' read -r -a BUNDLE_SPECS <<< "${LOCAL_ADAPTER_BUNDLES}"
  for spec in "${BUNDLE_SPECS[@]}"; do
    [[ -z "${spec}" ]] && continue
    if [[ "${spec}" != *=* ]]; then
      echo "Invalid LOCAL_ADAPTER_BUNDLES entry '${spec}', expected arm=tarball.tgz" >&2
      exit 65
    fi
    arm="${spec%%=*}"
    bundle="${spec#*=}"
    target="${LOCAL_ADAPTER_ROOT}/${arm}"
    if [[ ! -r "${bundle}" ]]; then
      echo "Missing local adapter bundle: ${bundle}" >&2
      exit 66
    fi
    mkdir -p "${target}"
    tar -xzf "${bundle}" -C "${target}"
    log_step "local_adapter_bundle_unpacked arm=${arm} bundle=${bundle} target=${target}"
  done
  ADAPTER_ROOT="${LOCAL_ADAPTER_ROOT}"
fi

{
  echo "timestamp_utc,gpu_index,name,utilization_gpu_pct,utilization_memory_pct,memory_used_mb,memory_total_mb,power_draw_w,temperature_c"
  while true; do
    if command -v nvidia-smi >/dev/null 2>&1; then
      nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits \
        | awk -v ts="$(date -u +%Y-%m-%dT%H:%M:%SZ)" -F',' '{gsub(/^[ \t]+|[ \t]+$/, "", $0); print ts "," $0}' || true
    else
      echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),NA,nvidia-smi-missing,NA,NA,NA,NA,NA,NA"
    fi
    sleep 5
  done
} > "${RESULT_DIR}/gpu_metrics.csv" &
MONITOR_PID="$!"

log_step "staged_input_check"
adapter_paths=()
for arm in "${ARMS[@]}"; do
  case "${arm}" in
    base) ;;
    taskvec_a0p25|taskvec_a0p5|lowLR|lowrank) adapter_paths+=("${ADAPTER_ROOT}/${arm}") ;;
    *)
      echo "Unknown arm for input check: ${arm}" >&2
      exit 65
      ;;
  esac
done

for p in "${MODEL_PATH}" "${adapter_paths[@]}"; do
  if [[ ! -e "${p}" ]]; then
    echo "Missing staged path: ${p}" >&2
    exit 66
  fi
done
if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config at ${MODEL_PATH}/config.json" >&2
  exit 66
fi
if [[ "${#adapter_paths[@]}" -gt 0 ]]; then
  (du -sh "${MODEL_PATH}" "${adapter_paths[@]}" 2>&1 || true) > "${RESULT_DIR}/staged_path_sizes.txt"
else
  (du -sh "${MODEL_PATH}" 2>&1 || true) > "${RESULT_DIR}/staged_path_sizes.txt"
fi

export HF_HOME="${HF_HOME:-${PWD}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PWD}/hf_datasets_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PWD}/triton_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PWD}/vllm_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-0}"
export PIP_NO_CACHE_DIR=1

missing_modules="$("${PYTHON_BIN}" - <<'PY'
import importlib.util

missing = []
for module in ("torch", "vllm", "lm_eval"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)
print(" ".join(missing))
PY
)"

hub_needs_pin="$("${PYTHON_BIN}" - <<'PY'
from importlib.metadata import PackageNotFoundError, version

try:
    hub_version = version("huggingface-hub")
except PackageNotFoundError:
    print("1")
else:
    major = int(hub_version.split(".", 1)[0])
    print("1" if major >= 1 else "0")
PY
)"

if [[ -n "${missing_modules}" || "${hub_needs_pin}" == "1" ]]; then
  packages=("huggingface-hub>=0.24.0,<1.0" "datasets>=2.16.0,<5.0.0")
  for module in ${missing_modules}; do
    case "${module}" in
      vllm) packages+=("vllm==${VLLM_VERSION:-0.6.6.post1}") ;;
      lm_eval) packages+=("lm-eval==${LM_EVAL_VERSION:-0.4.12}") ;;
      torch) packages+=("torch") ;;
    esac
  done
  log_step "pip_install_start timeout=${PIP_TIMEOUT_SECONDS}s packages=${packages[*]}"
  set +e
  timeout "${PIP_TIMEOUT_SECONDS}" "${PYTHON_BIN}" -m pip install --upgrade --target "${PYDEPS}" "${packages[@]}" > "${RESULT_DIR}/pip_install.txt" 2>&1
  pip_rc=$?
  set -e
  echo "${pip_rc}" > "${RESULT_DIR}/pip_install_exit_status.txt"
  log_step "pip_install_exit rc=${pip_rc}"
  if [[ "${pip_rc}" -ne 0 ]]; then
    exit "${pip_rc}"
  fi
  export PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}"
else
  echo "all required modules already import" > "${RESULT_DIR}/pip_install.txt"
  echo "0" > "${RESULT_DIR}/pip_install_exit_status.txt"
  log_step "pip_install_skipped"
fi

log_step "import_probe_start"
"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/imports_after_install.txt" 2>&1
import importlib

for name in ("torch", "vllm", "lm_eval"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")

import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch_cuda_device_name={torch.cuda.get_device_name(0)}")
PY
log_step "import_probe_ok"

CMD=(
  "${PYTHON_BIN}" run_retention_failure_suite_gate.py
  --manifest suite_manifest.json
  --out-dir "${RESULT_DIR}/gate"
  --arms "${ARMS[@]}"
  --tasks "${TASKS[@]}"
  --limit "${LIMIT}"
  --model-path "${MODEL_PATH}"
  --adapter-root "${ADAPTER_ROOT}"
  --batch-size "${BATCH_SIZE}"
  --gpu-mem-util "${GPU_MEM_UTIL}"
  --max-model-len "${MAX_MODEL_LEN}"
  --python-bin "${PYTHON_BIN}"
)

printf '%q ' "${CMD[@]}" > "${RESULT_DIR}/command.txt"
printf '\n' >> "${RESULT_DIR}/command.txt"

log_step "gate_start timeout=${LMEVAL_TIMEOUT_SECONDS}s"
set +e
timeout "${LMEVAL_TIMEOUT_SECONDS}" "${CMD[@]}" > "${RESULT_DIR}/gate_stdout.txt" 2> "${RESULT_DIR}/gate_stderr.txt"
rc=$?
set -e
echo "${rc}" > "${RESULT_DIR}/gate_exit_status.txt"
log_step "gate_exit rc=${rc}"

find "${RESULT_DIR}" -maxdepth 6 -type f -printf '%P\t%s\n' > "${RESULT_DIR}/file_inventory.tsv" || true
exit "${rc}"
