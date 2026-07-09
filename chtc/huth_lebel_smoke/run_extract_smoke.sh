#!/usr/bin/env bash
set -euo pipefail

RESULT_DIR="${PWD}/huth_extract_smoke"
FEATURE_DIR="${FEATURE_DIR:-${RESULT_DIR}/features}"
PYDEPS="${PWD}/pydeps"
OUT_BUNDLE="huth_extract_smoke_results.tgz"
export FEATURE_DIR

finish() {
  status=$?
  mkdir -p "${RESULT_DIR}" 2>/dev/null || true
  echo "${status}" > "${RESULT_DIR}/exit_status.txt"
  tar -czf "${OUT_BUNDLE}" -C "${RESULT_DIR}" . 2>/dev/null || true
  exit "${status}"
}
trap finish EXIT

mkdir -p "${RESULT_DIR}" "${FEATURE_DIR}" "${PYDEPS}"
if [[ ! -d /staging/s/suresh27 ]]; then
  echo "missing worker staging mount: /staging/s/suresh27" > "${RESULT_DIR}/missing_staging.txt"
  exit 68
fi

PYTHON_BIN="$(command -v python3 || command -v python)"
export PYTHONPATH="${PWD}/src:${PYDEPS}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/staging/s/suresh27/hf_home}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/staging/s/suresh27/hf_home/hub}"
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TRITON_CACHE_DIR="${PWD}/triton_cache"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

STORIES="${STORIES:-sweetaspie,againstthewind,wheretheressmoke}"
ARMS="${ARMS:-base,lowLR,scrambled,taskvec_a0p25}"
LAYERS="${LAYERS:-16,24,32}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MAX_CONTEXT_TOKENS="${MAX_CONTEXT_TOKENS:-512}"
LIMIT_WORDS="${LIMIT_WORDS:-0}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=${PWD}"
  echo "python=${PYTHON_BIN}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "MIN_CUDA_GLOBAL_MEMORY_MB=${MIN_CUDA_GLOBAL_MEMORY_MB:-unset}"
  echo "FEATURE_DIR=${FEATURE_DIR}"
  echo "STORIES=${STORIES}"
  echo "ARMS=${ARMS}"
  echo "LAYERS=${LAYERS}"
  echo "BATCH_SIZE=${BATCH_SIZE}"
  echo "MAX_CONTEXT_TOKENS=${MAX_CONTEXT_TOKENS}"
  echo "LIMIT_WORDS=${LIMIT_WORDS}"
} > "${RESULT_DIR}/run_env.txt"

(nvidia-smi || true) > "${RESULT_DIR}/nvidia_smi.txt" 2>&1

"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/gpu_check.txt" 2>&1
import os
import re
import subprocess
import sys

minimum = int(os.environ.get("MIN_CUDA_GLOBAL_MEMORY_MB", "40000"))
out = subprocess.check_output(
    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
    text=True,
)
print(out.strip())
memories = []
for line in out.strip().splitlines():
    parts = [part.strip() for part in line.split(",")]
    if len(parts) >= 2 and re.match(r"^\d+$", parts[1]):
        memories.append(int(parts[1]))
if not memories or max(memories) < minimum:
    print(f"no GPU with memory >= {minimum} MB", file=sys.stderr)
    sys.exit(67)
PY

"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/imports_before_install.txt" 2>&1 || true
import importlib
for name in ("torch", "transformers", "safetensors", "yaml", "numpy"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{type(exc).__name__}:{exc}")
PY

missing="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = []
for name in ("transformers", "safetensors", "yaml", "numpy"):
    if importlib.util.find_spec(name) is None:
        mods.append(name)
print(" ".join(mods))
PY
)"

if [[ -n "${missing}" ]]; then
  packages=()
  for module in ${missing}; do
    case "${module}" in
      yaml) packages+=("pyyaml") ;;
      *) packages+=("${module}") ;;
    esac
  done
  "${PYTHON_BIN}" -m pip install --target "${PYDEPS}" "${packages[@]}" > "${RESULT_DIR}/pip_install.txt" 2>&1
else
  echo "all required modules already import" > "${RESULT_DIR}/pip_install.txt"
fi

"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/imports_after_install.txt" 2>&1
import importlib
for name in ("torch", "transformers", "safetensors", "yaml", "numpy"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")
PY

for path in \
  /staging/s/suresh27/datasets/ds003020-smoke \
  /staging/s/suresh27/models/llama31-8b-instruct \
  /staging/s/suresh27/adapters/lowLR \
  /staging/s/suresh27/adapters/scrambled \
  /staging/s/suresh27/adapters/taskvec_a0p25
do
  if [[ ! -e "${path}" ]]; then
    echo "missing staged path: ${path}" >&2
    exit 66
  fi
done

CMD=(
  "${PYTHON_BIN}" src/sft/huth_lebel_extract_word_states.py
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke
  --model_path /staging/s/suresh27/models/llama31-8b-instruct
  --hf_cache /staging/s/suresh27/hf_home
  --out_dir "${FEATURE_DIR}"
  --stories "${STORIES}"
  --arms "${ARMS}"
  --adapter lowLR=/staging/s/suresh27/adapters/lowLR
  --adapter scrambled=/staging/s/suresh27/adapters/scrambled
  --adapter taskvec_a0p25=/staging/s/suresh27/adapters/taskvec_a0p25
  --layers "${LAYERS}"
  --batch_size "${BATCH_SIZE}"
  --max_context_tokens "${MAX_CONTEXT_TOKENS}"
  --device cuda
  --dtype bfloat16
  --save_dtype float16
  --skip_existing
)

if [[ "${LIMIT_WORDS}" != "0" ]]; then
  CMD+=(--limit_words "${LIMIT_WORDS}")
fi

printf '%q ' "${CMD[@]}" > "${RESULT_DIR}/command.txt"
printf '\n' >> "${RESULT_DIR}/command.txt"

set +e
"${CMD[@]}" > "${RESULT_DIR}/extract_stdout.txt" 2> "${RESULT_DIR}/extract_stderr.txt"
rc=$?
set -e
echo "${rc}" > "${RESULT_DIR}/extract_exit_status.txt"

find "${FEATURE_DIR}" -maxdepth 3 -type f -printf '%P\t%s\n' \
  > "${RESULT_DIR}/feature_inventory.tsv" || true

"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/npz_shapes.tsv" 2> "${RESULT_DIR}/npz_shapes.err" || true
import os
from pathlib import Path

import numpy as np

feature_dir = Path(os.environ["FEATURE_DIR"])
print("path\tkeys\thidden_shape\thidden_dtype\tlayers\tn_words")
for path in sorted(feature_dir.glob("*/*.npz")):
    data = np.load(path, allow_pickle=False)
    hidden = data["hidden"]
    layers = ",".join(str(int(x)) for x in data["layer_indices"])
    print(
        f"{path.relative_to(feature_dir)}\t{','.join(data.files)}\t"
        f"{'x'.join(map(str, hidden.shape))}\t{hidden.dtype}\t{layers}\t"
        f"{data['words'].shape[0]}"
    )
PY

exit "${rc}"
