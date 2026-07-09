#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 13 ]]; then
  echo "usage: $0 RUN_MODE STATE_NAME SHARD_ID TASK_SPEC LIMIT EXPECTED_N NUM_FEWSHOT MODEL_PATH ADAPTER_PATH MAX_LORA_RANK BATCH_SIZE GPU_MEM_UTIL MAX_MODEL_LEN" >&2
  exit 64
fi

RUN_MODE="$1"
STATE_NAME="$2"
SHARD_ID="$3"
TASK_SPEC="$4"
TASKS="${TASK_SPEC//+/,}"
LIMIT="$5"
EXPECTED_N="$6"
NUM_FEWSHOT="$7"
MODEL_PATH="$8"
ADAPTER_PATH="$9"
MAX_LORA_RANK="${10}"
BATCH_SIZE="${11}"
GPU_MEM_UTIL="${12}"
MAX_MODEL_LEN="${13}"
MIN_CUDA_GLOBAL_MEMORY_MB="${MIN_CUDA_GLOBAL_MEMORY_MB:-40000}"

OUT_BUNDLE="mmlu_${RUN_MODE}_${STATE_NAME}_${SHARD_ID}_results.tgz"
RESULT_DIR="${PWD}/mmlu_${RUN_MODE}_${STATE_NAME}_${SHARD_ID}"
EVAL_OUT="${RESULT_DIR}/lm_eval"
PYDEPS="${PWD}/pydeps"

mkdir -p "${RESULT_DIR}" "${EVAL_OUT}" "${PYDEPS}"

finish() {
  status=$?
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
  echo "run_mode=${RUN_MODE}"
  echo "state_name=${STATE_NAME}"
  echo "shard_id=${SHARD_ID}"
  echo "task_spec=${TASK_SPEC}"
  echo "tasks=${TASKS}"
  echo "limit=${LIMIT}"
  echo "expected_n=${EXPECTED_N}"
  echo "num_fewshot=${NUM_FEWSHOT}"
  echo "model_path=${MODEL_PATH}"
  echo "adapter_path=${ADAPTER_PATH}"
  echo "max_lora_rank=${MAX_LORA_RANK}"
  echo "batch_size=${BATCH_SIZE}"
  echo "gpu_mem_util=${GPU_MEM_UTIL}"
  echo "max_model_len=${MAX_MODEL_LEN}"
  echo "min_cuda_global_memory_mb=${MIN_CUDA_GLOBAL_MEMORY_MB}"
  echo "python_bin=${PYTHON_BIN}"
  echo "original_cuda_visible_devices=${ORIGINAL_CUDA_VISIBLE_DEVICES}"
  echo "effective_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "CONDOR_SLOT=${_CONDOR_SLOT:-unset}"
} > "${RESULT_DIR}/run_env.txt"

{
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -m pip --version
} > "${RESULT_DIR}/python.txt" 2>&1 || true

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

(du -sh "${MODEL_PATH}" "${ADAPTER_PATH}" 2>&1 || true) > "${RESULT_DIR}/staged_path_sizes.txt"
(find "${MODEL_PATH}" -maxdepth 2 -type f | sed -n '1,80p' 2>&1 || true) > "${RESULT_DIR}/model_file_sample.txt"
(find "${ADAPTER_PATH}" -maxdepth 2 -type f | sed -n '1,80p' 2>&1 || true) > "${RESULT_DIR}/adapter_file_sample.txt"

if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config at ${MODEL_PATH}/config.json" >&2
  exit 66
fi
if [[ "${ADAPTER_PATH}" != "NONE" && ! -r "${ADAPTER_PATH}/adapter_config.json" ]]; then
  echo "Missing adapter config at ${ADAPTER_PATH}/adapter_config.json" >&2
  exit 66
fi

export HF_HOME="${HF_HOME:-/staging/s/suresh27/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/staging/s/suresh27/hf_home/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/staging/s/suresh27/hf_datasets_cache}"
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
  packages=()
  packages+=("huggingface-hub>=0.24.0,<1.0")
  packages+=("datasets>=2.16.0,<5.0.0")
  for module in ${missing_modules}; do
    case "${module}" in
      vllm) packages+=("vllm==${VLLM_VERSION:-0.6.6.post1}") ;;
      lm_eval) packages+=("lm-eval==${LM_EVAL_VERSION:-0.4.12}") ;;
      torch) packages+=("torch") ;;
    esac
  done
  if [[ ${#packages[@]} -gt 0 ]]; then
    "${PYTHON_BIN}" -m pip install --upgrade --target "${PYDEPS}" "${packages[@]}" > "${RESULT_DIR}/pip_install.txt" 2>&1
    export PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}"
  fi
else
  echo "all required modules already import" > "${RESULT_DIR}/pip_install.txt"
fi

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

MODEL_ARGS="pretrained=${MODEL_PATH},dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=${GPU_MEM_UTIL},max_model_len=${MAX_MODEL_LEN},trust_remote_code=True"
if [[ "${ADAPTER_PATH}" != "NONE" ]]; then
  MODEL_ARGS="${MODEL_ARGS},enable_lora=True,lora_local_path=${ADAPTER_PATH},max_lora_rank=${MAX_LORA_RANK}"
fi

CMD=(
  "${PYTHON_BIN}" -m lm_eval run
  --model vllm
  --model_args "${MODEL_ARGS}"
  --tasks "${TASKS}"
  --limit "${LIMIT}"
  --batch_size "${BATCH_SIZE}"
  --apply_chat_template
  --num_fewshot "${NUM_FEWSHOT}"
  --output_path "${EVAL_OUT}"
)

printf '%q ' "${CMD[@]}" > "${RESULT_DIR}/command.txt"
printf '\n' >> "${RESULT_DIR}/command.txt"

set +e
"${CMD[@]}" > "${RESULT_DIR}/lm_eval_stdout.txt" 2> "${RESULT_DIR}/lm_eval_stderr.txt"
rc=$?
set -e
echo "${rc}" > "${RESULT_DIR}/lm_eval_exit_status.txt"

"${PYTHON_BIN}" - <<'PY' "${EVAL_OUT}" "${TASKS}" "${EXPECTED_N}" "${LIMIT}" > "${RESULT_DIR}/shard_summary.json"
import glob
import json
import sys
from pathlib import Path

out_dir = Path(sys.argv[1])
expected_tasks = [task for task in sys.argv[2].split(",") if task]
expected_n = int(sys.argv[3])
limit = int(sys.argv[4])
matches = sorted(glob.glob(str(out_dir / "**" / "results_*.json"), recursive=True))
summary = {
    "result_json": matches[-1] if matches else None,
    "expected_tasks": expected_tasks,
    "expected_n": expected_n,
    "limit": limit,
    "found_tasks": [],
    "found_n": 0,
    "missing_tasks": expected_tasks,
}
if matches:
    with open(matches[-1]) as f:
        data = json.load(f)
    results = data.get("results", {})
    found = [task for task in expected_tasks if task in results]
    summary["found_tasks"] = found
    summary["missing_tasks"] = [task for task in expected_tasks if task not in results]
    summary["found_n"] = sum(int(results[task].get("sample_len", 0)) for task in found)
print(json.dumps(summary, indent=2, sort_keys=True))
PY

find "${RESULT_DIR}" -maxdepth 4 -type f -printf '%P\t%s\n' > "${RESULT_DIR}/file_inventory.tsv" || true
exit "${rc}"
