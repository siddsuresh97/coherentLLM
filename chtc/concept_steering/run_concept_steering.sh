#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 9 ]]; then
  echo "usage: $0 RUN_MODE LAYERS_SPEC ALPHAS_SPEC MAX_PAIRS MAX_ITEMS EXTRACT_BATCH EVAL_BATCH DTYPE OUT_TAG" >&2
  exit 64
fi

RUN_MODE="$1"
LAYERS_SPEC="$2"
ALPHAS_SPEC="$3"
MAX_PAIRS="$4"
MAX_ITEMS="$5"
EXTRACT_BATCH="$6"
EVAL_BATCH="$7"
DTYPE="$8"
OUT_TAG="$9"

MIN_CUDA_GLOBAL_MEMORY_MB="${MIN_CUDA_GLOBAL_MEMORY_MB:-40000}"
PIP_TIMEOUT_SECONDS="${PIP_TIMEOUT_SECONDS:-900}"
EXTRACT_TIMEOUT_SECONDS="${EXTRACT_TIMEOUT_SECONDS:-3600}"
EVAL_TIMEOUT_SECONDS="${EVAL_TIMEOUT_SECONDS:-3600}"
MODEL_PATH="${MODEL_PATH:-/staging/s/suresh27/models/llama31-8b-instruct}"
HF_CACHE="${HF_CACHE:-/staging/s/suresh27/hf_home}"

RESULT_DIR="${PWD}/concept_steering_${RUN_MODE}_${OUT_TAG}"
VECTOR_DIR="${RESULT_DIR}/vectors"
SWEEP_DIR="${RESULT_DIR}/sweep"
PYDEPS="${PWD}/pydeps"
OUT_BUNDLE="concept_steering_${RUN_MODE}_${OUT_TAG}_results.tgz"

mkdir -p "${RESULT_DIR}" "${VECTOR_DIR}" "${SWEEP_DIR}" "${PYDEPS}" configs

log_step() {
  printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" | tee -a "${RESULT_DIR}/progress.log"
}

finish() {
  status=$?
  log_step "runner_exit status=${status}"
  echo "${status}" > "${RESULT_DIR}/exit_status.txt"
  find "${RESULT_DIR}" -maxdepth 5 -type f -printf '%P\t%s\n' > "${RESULT_DIR}/file_inventory.tsv" 2>/dev/null || true
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

cat > configs/models.yaml <<YAML
hf_cache: ${HF_CACHE}

local:
  llama-3.1-8b-instruct:
    path: ${MODEL_PATH}
    chat: true
YAML

export PYTHONPATH="${PWD}/src:${PYDEPS}:${PYTHONPATH:-}"
export HF_HOME="${HF_CACHE}"
export HF_HUB_CACHE="${HF_CACHE}/hub"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/staging/s/suresh27/hf_datasets_cache}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="${PWD}/triton_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PIP_NO_CACHE_DIR=1

log_step "runner_start mode=${RUN_MODE} layers=${LAYERS_SPEC} alphas=${ALPHAS_SPEC}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=${PWD}"
  echo "run_mode=${RUN_MODE}"
  echo "layers_spec=${LAYERS_SPEC}"
  echo "alphas_spec=${ALPHAS_SPEC}"
  echo "max_pairs=${MAX_PAIRS}"
  echo "max_items=${MAX_ITEMS}"
  echo "extract_batch=${EXTRACT_BATCH}"
  echo "eval_batch=${EVAL_BATCH}"
  echo "dtype=${DTYPE}"
  echo "out_tag=${OUT_TAG}"
  echo "model_path=${MODEL_PATH}"
  echo "hf_cache=${HF_CACHE}"
  echo "min_cuda_global_memory_mb=${MIN_CUDA_GLOBAL_MEMORY_MB}"
  echo "pip_timeout_seconds=${PIP_TIMEOUT_SECONDS}"
  echo "extract_timeout_seconds=${EXTRACT_TIMEOUT_SECONDS}"
  echo "eval_timeout_seconds=${EVAL_TIMEOUT_SECONDS}"
  echo "python_bin=${PYTHON_BIN}"
  echo "original_cuda_visible_devices=${ORIGINAL_CUDA_VISIBLE_DEVICES}"
  echo "effective_cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "CONDOR_SLOT=${_CONDOR_SLOT:-unset}"
} > "${RESULT_DIR}/run_env.txt"

{
  "${PYTHON_BIN}" --version
  "${PYTHON_BIN}" -m pip --version
} > "${RESULT_DIR}/python.txt" 2>&1 || true

log_step "gpu_probe_start"
(nvidia-smi || true) > "${RESULT_DIR}/nvidia_smi.txt" 2>&1
(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits || true) > "${RESULT_DIR}/gpu_memory_query.txt" 2>&1
"${PYTHON_BIN}" - <<'PY' "${RESULT_DIR}/gpu_memory_query.txt" "${MIN_CUDA_GLOBAL_MEMORY_MB}" > "${RESULT_DIR}/gpu_check.txt" 2>&1
import re
import sys
from pathlib import Path

query_path = Path(sys.argv[1])
minimum = int(sys.argv[2])
text = query_path.read_text()
print(text.strip())
memories = []
for line in text.strip().splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) >= 2 and re.fullmatch(r"\d+", parts[1]):
        memories.append(int(parts[1]))
if not memories:
    print("Could not determine GPU memory from nvidia-smi.", file=sys.stderr)
    sys.exit(67)
if max(memories) < minimum:
    print(f"GPU memory {max(memories)} MB is below required {minimum} MB.", file=sys.stderr)
    sys.exit(67)
print(f"max_gpu_memory_mb={max(memories)}")
PY
log_step "gpu_probe_ok"

if [[ ! -r "${MODEL_PATH}/config.json" ]]; then
  echo "Missing model config at ${MODEL_PATH}/config.json" >&2
  exit 66
fi

(du -sh "${MODEL_PATH}" 2>&1 || true) > "${RESULT_DIR}/staged_model_size.txt"
(find "${MODEL_PATH}" -maxdepth 2 -type f | sed -n '1,80p' 2>&1 || true) > "${RESULT_DIR}/model_file_sample.txt"

needs_install="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
from importlib.metadata import PackageNotFoundError, version

mods = ["torch", "transformers", "accelerate", "safetensors", "yaml", "numpy"]
missing = [name for name in mods if importlib.util.find_spec(name) is None]
try:
    hub_version = version("huggingface-hub")
    hub_bad = int(hub_version.split(".", 1)[0]) >= 1
except PackageNotFoundError:
    hub_bad = True
print("1" if missing or hub_bad else "0")
PY
)"

if [[ "${needs_install}" == "1" ]]; then
  log_step "pip_install_start timeout=${PIP_TIMEOUT_SECONDS}s"
  set +e
  timeout "${PIP_TIMEOUT_SECONDS}" "${PYTHON_BIN}" -m pip install --upgrade --no-deps --target "${PYDEPS}" \
    "huggingface-hub>=0.24.0,<1.0" \
    "transformers>=4.45.0,<5.0.0" \
    "accelerate>=0.33.0" \
    "safetensors>=0.4.5" \
    "pyyaml>=6.0" \
    "numpy<2.3" \
    "tokenizers>=0.22.0,<0.23.0" \
    "regex" \
    "filelock" \
    "requests" \
    "tqdm" \
    "packaging" \
    "psutil" \
    "typing-extensions" \
    > "${RESULT_DIR}/pip_install.txt" 2>&1
  pip_rc=$?
  set -e
  echo "${pip_rc}" > "${RESULT_DIR}/pip_install_exit_status.txt"
  log_step "pip_install_exit rc=${pip_rc}"
  if [[ "${pip_rc}" -ne 0 ]]; then
    exit "${pip_rc}"
  fi
else
  echo "all required modules already import and huggingface-hub is <1.0" > "${RESULT_DIR}/pip_install.txt"
  echo "0" > "${RESULT_DIR}/pip_install_exit_status.txt"
  log_step "pip_install_skipped"
fi

log_step "import_probe_start"
"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/imports_after_install.txt" 2>&1
import importlib

for name in ("torch", "transformers", "accelerate", "safetensors", "yaml", "numpy"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")

import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch_cuda_device_name={torch.cuda.get_device_name(0)}")
PY
log_step "import_probe_ok"

IFS='+' read -r -a ALPHAS_ARGS <<< "${ALPHAS_SPEC}"

EXTRACT_CMD=(
  "${PYTHON_BIN}" src/sft/extract_concept_vectors.py
  --concepts coherence human_alignment
  --batch-size "${EXTRACT_BATCH}"
  --out-dir "${VECTOR_DIR}"
  --dtype "${DTYPE}"
  --device cuda
  --overwrite
)
if [[ "${MAX_PAIRS}" != "ALL" ]]; then
  EXTRACT_CMD+=(--max-pairs "${MAX_PAIRS}")
fi

EVAL_CMD=(
  "${PYTHON_BIN}" src/sft/run_concept_steering_eval.py
  --vector-dir "${VECTOR_DIR}"
  --out-dir "${SWEEP_DIR}"
  --steer-concepts coherence human_alignment
  --eval-concepts coherence human_alignment
  --layers "${LAYERS_SPEC}"
  --batch-size "${EVAL_BATCH}"
  --dtype "${DTYPE}"
  --device cuda
  --overwrite
)
if [[ "${MAX_ITEMS}" != "ALL" ]]; then
  EVAL_CMD+=(--max-items "${MAX_ITEMS}")
fi
EVAL_CMD+=(--alphas "${ALPHAS_ARGS[@]}")

printf '%q ' "${EXTRACT_CMD[@]}" > "${RESULT_DIR}/extract_command.txt"
printf '\n' >> "${RESULT_DIR}/extract_command.txt"
printf '%q ' "${EVAL_CMD[@]}" > "${RESULT_DIR}/eval_command.txt"
printf '\n' >> "${RESULT_DIR}/eval_command.txt"

log_step "extract_start timeout=${EXTRACT_TIMEOUT_SECONDS}s"
set +e
timeout "${EXTRACT_TIMEOUT_SECONDS}" "${EXTRACT_CMD[@]}" > "${RESULT_DIR}/extract_stdout.txt" 2> "${RESULT_DIR}/extract_stderr.txt"
extract_rc=$?
set -e
echo "${extract_rc}" > "${RESULT_DIR}/extract_exit_status.txt"
log_step "extract_exit rc=${extract_rc}"
if [[ "${extract_rc}" -ne 0 ]]; then
  exit "${extract_rc}"
fi

log_step "eval_start timeout=${EVAL_TIMEOUT_SECONDS}s"
set +e
timeout "${EVAL_TIMEOUT_SECONDS}" "${EVAL_CMD[@]}" > "${RESULT_DIR}/eval_stdout.txt" 2> "${RESULT_DIR}/eval_stderr.txt"
eval_rc=$?
set -e
echo "${eval_rc}" > "${RESULT_DIR}/eval_exit_status.txt"
log_step "eval_exit rc=${eval_rc}"
if [[ "${eval_rc}" -ne 0 ]]; then
  exit "${eval_rc}"
fi

{
  echo "# Concept Steering ${RUN_MODE} Job Summary"
  echo
  echo "- mode: ${RUN_MODE}"
  echo "- layer spec: ${LAYERS_SPEC}"
  echo "- alpha spec: ${ALPHAS_SPEC}"
  echo "- vector max pairs: ${MAX_PAIRS}"
  echo "- eval max items per set: ${MAX_ITEMS}"
  echo "- vector dir: vectors/"
  echo "- sweep dir: sweep/"
  echo
  echo "## Expected Smoke Shape"
  echo
  echo "For smoke defaults: 2 steering concepts x 1 layer x 3 alphas x 3 eval sets = 18 summary rows."
  echo
  echo "## Sweep Summary"
  echo
  if [[ -r "${SWEEP_DIR}/SUMMARY.md" ]]; then
    sed -n '1,220p' "${SWEEP_DIR}/SUMMARY.md"
  else
    echo "Missing sweep/SUMMARY.md"
  fi
} > "${RESULT_DIR}/job_summary.md"

log_step "complete"
