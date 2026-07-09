#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 6 ]]; then
  echo "usage: $0 MODE RUN_ID MODEL_PATH GPU_MEM_UTIL MAX_MODEL_LEN MAX_NUM_SEQS" >&2
  exit 64
fi

MODE="$1"
RUN_ID="$2"
MODEL_PATH="$3"
GPU_MEM_UTIL="$4"
MAX_MODEL_LEN="$5"
MAX_NUM_SEQS="$6"

MIN_CUDA_GLOBAL_MEMORY_MB="${MIN_CUDA_GLOBAL_MEMORY_MB:-40000}"
PIP_TIMEOUT_SECONDS="${PIP_TIMEOUT_SECONDS:-1200}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-21600}"
PYDEPS="${PWD}/pydeps"
WORK="${PWD}/work"
SUBMIT_DIR="${PWD}"
RESULT_DIR="${SUBMIT_DIR}/exp3_${MODE}_${RUN_ID}"
OUT_BUNDLE="${SUBMIT_DIR}/exp3_${MODE}_${RUN_ID}_results.tgz"

mkdir -p "${PYDEPS}" "${WORK}" "${RESULT_DIR}"

log_step() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RESULT_DIR}/progress.log"
}

finish() {
  status=$?
  log_step "runner_exit status=${status}"
  echo "${status}" > "${RESULT_DIR}/exit_status.txt"
  if [[ -d "${WORK}/experiments/exp3_directional_confusions" ]]; then
    mkdir -p "${RESULT_DIR}/work_snapshot/experiments"
    cp -a "${WORK}/experiments/exp3_directional_confusions" "${RESULT_DIR}/work_snapshot/experiments/" 2>/dev/null || true
  fi
  tar -czf "${OUT_BUNDLE}" -C "${RESULT_DIR}" . 2>/dev/null || true
  if [[ ! -s "${OUT_BUNDLE}" ]]; then
    echo "failed_to_create_result_bundle" > "${OUT_BUNDLE}.txt"
    tar -czf "${OUT_BUNDLE}" "${OUT_BUNDLE}.txt" 2>/dev/null || true
  fi
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
  echo "mode=${MODE}"
  echo "run_id=${RUN_ID}"
  echo "model_path=${MODEL_PATH}"
  echo "gpu_mem_util=${GPU_MEM_UTIL}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_seqs=${MAX_NUM_SEQS}"
  echo "min_cuda_global_memory_mb=${MIN_CUDA_GLOBAL_MEMORY_MB}"
  echo "pip_timeout_seconds=${PIP_TIMEOUT_SECONDS}"
  echo "run_timeout_seconds=${RUN_TIMEOUT_SECONDS}"
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

if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config at ${MODEL_PATH}/config.json" >&2
  exit 66
fi
(du -sh "${MODEL_PATH}" 2>&1 || true) > "${RESULT_DIR}/staged_path_sizes.txt"

tar -xzf exp3_directional_confusions_bundle.tgz -C "${WORK}"
cd "${WORK}"

export HF_HOME="${HF_HOME:-${PWD}/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${PWD}/hf_datasets_cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${PWD}/triton_cache}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${PWD}/vllm_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PIP_NO_CACHE_DIR=1

missing_modules="$("${PYTHON_BIN}" - <<'PY'
import importlib.util

missing = []
for module in ("torch", "vllm", "pandas", "transformers"):
    if importlib.util.find_spec(module) is None:
        missing.append(module)
print(" ".join(missing))
PY
)"

if [[ -n "${missing_modules}" ]]; then
  packages=()
  for module in ${missing_modules}; do
    case "${module}" in
      vllm) packages+=("vllm==${VLLM_VERSION:-0.6.6.post1}" "transformers==${TRANSFORMERS_VERSION:-4.47.1}" "huggingface-hub>=0.24.0,<1.0") ;;
      transformers) packages+=("transformers==${TRANSFORMERS_VERSION:-4.47.1}" "huggingface-hub>=0.24.0,<1.0") ;;
      pandas) packages+=("pandas") ;;
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

for name in ("torch", "vllm", "pandas"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")

import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch_cuda_device_name={torch.cuda.get_device_name(0)}")
PY
log_step "import_probe_ok"

common_vllm_args=(
  --model-path "${MODEL_PATH}"
  --gpu_mem_util "${GPU_MEM_UTIL}"
  --max_model_len "${MAX_MODEL_LEN}"
  --max_num_seqs "${MAX_NUM_SEQS}"
  --overwrite
)

run_with_timeout() {
  log_step "command_start $*"
  timeout "${RUN_TIMEOUT_SECONDS}" "$@"
  log_step "command_ok"
}

case "${MODE}" in
  geometry)
    {
      printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py init --no-log
      printf '\n'
      printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py run-triplet-suite "${common_vllm_args[@]}"
      printf '\n'
      printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py build-rdm
      printf '\n'
      printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py register-neighbors
      printf '\n'
      printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py generate-items
      printf '\n'
    } > "${RESULT_DIR}/command.txt"
    log_step "pipeline_start mode=${MODE}"
    run_with_timeout "${PYTHON_BIN}" scripts/run_experiment3.py init --no-log
    run_with_timeout "${PYTHON_BIN}" scripts/run_experiment3.py run-triplet-suite "${common_vllm_args[@]}"
    run_with_timeout "${PYTHON_BIN}" scripts/run_experiment3.py build-rdm
    run_with_timeout "${PYTHON_BIN}" scripts/run_experiment3.py register-neighbors
    run_with_timeout "${PYTHON_BIN}" scripts/run_experiment3.py generate-items
    ;;
  items)
    {
      printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py run-items --out-run step1_items_v1 "${common_vllm_args[@]}"
      printf '\n'
      printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py score --run step1_items_v1
      printf '\n'
    } > "${RESULT_DIR}/command.txt"
    log_step "pipeline_start mode=${MODE}"
    run_with_timeout "${PYTHON_BIN}" scripts/run_experiment3.py run-items --out-run step1_items_v1 "${common_vllm_args[@]}"
    run_with_timeout "${PYTHON_BIN}" scripts/run_experiment3.py score --run step1_items_v1
    ;;
  *)
    echo "Unknown MODE=${MODE}; expected geometry or items." >&2
    exit 65
    ;;
esac

log_step "pipeline_ok mode=${MODE}"
