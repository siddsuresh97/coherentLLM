#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 7 ]]; then
  echo "usage: $0 RUN_ID MODEL_PATH OUT_RUN PROMPT_VARIANT GPU_MEM_UTIL MAX_MODEL_LEN MAX_NUM_SEQS" >&2
  exit 64
fi

RUN_ID="$1"
MODEL_PATH="$2"
OUT_RUN="$3"
PROMPT_VARIANT="$4"
GPU_MEM_UTIL="$5"
MAX_MODEL_LEN="$6"
MAX_NUM_SEQS="$7"

MIN_CUDA_GLOBAL_MEMORY_MB="${MIN_CUDA_GLOBAL_MEMORY_MB:-40000}"
PIP_TIMEOUT_SECONDS="${PIP_TIMEOUT_SECONDS:-1200}"
RUN_TIMEOUT_SECONDS="${RUN_TIMEOUT_SECONDS:-21600}"
PYDEPS="${PWD}/pydeps"
WORK="${PWD}/work"
SUBMIT_DIR="${PWD}"
RESULT_DIR="${SUBMIT_DIR}/exp3_triplet_${RUN_ID}"
OUT_BUNDLE="${SUBMIT_DIR}/exp3_triplet_${RUN_ID}_results.tgz"

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
  echo "model_path=${MODEL_PATH}"
  echo "out_run=${OUT_RUN}"
  echo "prompt_variant=${PROMPT_VARIANT}"
  echo "gpu_mem_util=${GPU_MEM_UTIL}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "max_num_seqs=${MAX_NUM_SEQS}"
  echo "min_cuda_global_memory_mb=${MIN_CUDA_GLOBAL_MEMORY_MB}"
  echo "python_bin=${PYTHON_BIN}"
  echo "original_cuda_visible_devices=${ORIGINAL_CUDA_VISIBLE_DEVICES}"
  echo "effective_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
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
  log_step "pip_install_start packages=${packages[*]}"
  timeout "${PIP_TIMEOUT_SECONDS}" "${PYTHON_BIN}" -m pip install --upgrade --target "${PYDEPS}" "${packages[@]}" > "${RESULT_DIR}/pip_install.txt" 2>&1
  export PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}"
else
  echo "all required modules already import" > "${RESULT_DIR}/pip_install.txt"
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

{
  printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py init --no-log
  printf '\n'
  printf '%q ' "${PYTHON_BIN}" scripts/run_experiment3.py run-triplets --out-run "${OUT_RUN}" --prompt-variant "${PROMPT_VARIANT}" "${common_vllm_args[@]}"
  printf '\n'
} > "${RESULT_DIR}/command.txt"

log_step "pipeline_start triplet_one out_run=${OUT_RUN} prompt_variant=${PROMPT_VARIANT}"
timeout "${RUN_TIMEOUT_SECONDS}" "${PYTHON_BIN}" scripts/run_experiment3.py init --no-log
timeout "${RUN_TIMEOUT_SECONDS}" "${PYTHON_BIN}" scripts/run_experiment3.py run-triplets \
  --out-run "${OUT_RUN}" \
  --prompt-variant "${PROMPT_VARIANT}" \
  "${common_vllm_args[@]}"
log_step "pipeline_ok triplet_one"
