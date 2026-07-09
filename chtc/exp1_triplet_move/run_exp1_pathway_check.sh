#!/usr/bin/env bash
# CHTC wrapper for the first real Experiment 1 pathway check.
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 EDIT_ID SUPERVISION MODEL_PATH HF_CACHE TRAIN_STEPS" >&2
  exit 64
fi

EDIT_ID="$1"
SUPERVISION_ARG="$2"
MODEL_PATH="$3"
HF_CACHE="$4"
TRAIN_STEPS_ARG="$5"

RESULT_DIR="${PWD}/exp1_pathway_${EDIT_ID}_${SUPERVISION_ARG}"
PYDEPS="${PWD}/pydeps"
OUT_BUNDLE="exp1_pathway_${EDIT_ID}_${SUPERVISION_ARG}_results.tgz"

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
  find "${RESULT_DIR}" -maxdepth 6 -type f -printf '%P\t%s\n' > "${RESULT_DIR}/file_inventory.tsv" 2>/dev/null || true
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
  echo "edit_id=${EDIT_ID}"
  echo "supervision=${SUPERVISION_ARG}"
  echo "model_path=${MODEL_PATH}"
  echo "hf_cache=${HF_CACHE}"
  echo "train_steps=${TRAIN_STEPS_ARG}"
  echo "python_bin=${PYTHON_BIN}"
  echo "original_cuda_visible_devices=${ORIGINAL_CUDA_VISIBLE_DEVICES}"
  echo "effective_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "CONDOR_SLOT=${_CONDOR_SLOT:-unset}"
} > "${RESULT_DIR}/run_env.txt"

log_step "gpu_probe_start"
(nvidia-smi || true) > "${RESULT_DIR}/nvidia_smi.txt" 2>&1
(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits || true) > "${RESULT_DIR}/gpu_memory_query.txt" 2>&1
log_step "gpu_probe_done"

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

if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
  echo "Missing staged model config: ${MODEL_PATH}/config.json" >&2
  exit 66
fi

log_step "unpack_bundle"
mkdir -p work
tar -xzf exp1_triplet_move_bundle.tgz -C work
cd work

export PYTHONPATH="${PWD}:${PWD}/src:${PYDEPS}:${PYTHONPATH:-}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export HF_DATASETS_CACHE="${PWD}/hf_datasets_cache"
export TRITON_CACHE_DIR="${PWD}/triton_cache"
export VLLM_CACHE_ROOT="${PWD}/vllm_cache"
export MPLCONFIGDIR="${PWD}/out/matplotlib_cache"
export PIP_NO_CACHE_DIR=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

log_step "pip_install_start"
set +e
timeout "${PIP_TIMEOUT_SECONDS:-1800}" "${PYTHON_BIN}" -m pip install --upgrade --target "${PYDEPS}" \
  "huggingface-hub>=0.24.0,<1.0" \
  "numpy<2.3" \
  "pandas>=2.0" \
  "pyarrow>=14.0" \
  "scipy>=1.10" \
  "scikit-learn>=1.3" \
  "matplotlib>=3.7" \
  "pyyaml>=6.0" \
  "transformers>=4.45.0,<5.0.0" \
  "accelerate>=0.33.0" \
  "datasets>=2.16.0,<5.0.0" \
  "peft>=0.12.0" \
  "trl>=0.10.0" \
  "bitsandbytes>=0.43.0" \
  "safetensors>=0.4.5" \
  "tokenizers>=0.22.0,<0.23.0" \
  "vllm==${VLLM_VERSION:-0.6.6.post1}" \
  > "${RESULT_DIR}/pip_install.txt" 2>&1
pip_rc=$?
set -e
echo "${pip_rc}" > "${RESULT_DIR}/pip_install_exit_status.txt"
log_step "pip_install_exit rc=${pip_rc}"
if [[ "${pip_rc}" -ne 0 ]]; then
  exit "${pip_rc}"
fi

log_step "import_probe_start"
"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/imports_after_install.txt" 2>&1
import importlib
for name in ("torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "vllm", "pandas", "pyarrow"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")
import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch_cuda_device_name={torch.cuda.get_device_name(0)}")
PY
log_step "import_probe_done"

log_step "exp1_pathway_start edit=${EDIT_ID} supervision=${SUPERVISION_ARG}"
set +e
SUPERVISION="${SUPERVISION_ARG}" \
BASE_MODEL_PATH="${MODEL_PATH}" \
TRAIN_BASE_MODEL="${MODEL_PATH}" \
HF_HOME="${HF_CACHE}" \
TRAIN_STEPS="${TRAIN_STEPS_ARG}" \
LORA_RANK="${LORA_RANK:-32}" \
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.86}" \
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}" \
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}" \
scripts/run_experiment1_real_gpu.sh "${EDIT_ID}" \
  > "${RESULT_DIR}/exp1_stdout.txt" 2> "${RESULT_DIR}/exp1_stderr.txt"
exp_rc=$?
set -e
echo "${exp_rc}" > "${RESULT_DIR}/exp1_exit_status.txt"
log_step "exp1_pathway_exit rc=${exp_rc}"

mkdir -p "${RESULT_DIR}/experiment"
tar -czf "${RESULT_DIR}/experiment/exp1_triplet_concept_move.tgz" \
  experiments/exp1_triplet_concept_move \
  --exclude='experiments/exp1_triplet_concept_move/lora_control/*/adapter_model.safetensors' \
  --exclude='experiments/exp1_triplet_concept_move/lora_edit/*/adapter_model.safetensors' \
  --exclude='experiments/exp1_triplet_concept_move/lora_control_similarity/*/adapter_model.safetensors' \
  --exclude='experiments/exp1_triplet_concept_move/lora_edit_similarity/*/adapter_model.safetensors' \
  2>/dev/null || true

if [[ "${exp_rc}" -ne 0 ]]; then
  exit "${exp_rc}"
fi
