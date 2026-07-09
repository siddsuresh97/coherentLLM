#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${PWD}/mmlu_gpu_probe"
PYDEPS="${PWD}/pydeps"
mkdir -p "${OUT_DIR}" "${PYDEPS}"

cleanup() {
  tar -czf mmlu_gpu_probe_results.tgz -C "${OUT_DIR}" . 2>/dev/null || true
}
trap cleanup EXIT

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=${PWD}"
  echo "PATH=${PATH}"
  echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "CONDOR_SLOT=${_CONDOR_SLOT:-unset}"
} > "${OUT_DIR}/env.txt"

{
  which python || true
  which python3 || true
  python --version || true
  python3 --version || true
} > "${OUT_DIR}/python.txt" 2>&1

(nvidia-smi || true) > "${OUT_DIR}/nvidia_smi.txt" 2>&1

python - <<'PY' > "${OUT_DIR}/imports_before_lmeval.txt" 2>&1
import importlib

for name in ("torch", "vllm", "lm_eval"):
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"{name}={version}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{type(exc).__name__}:{exc}")

try:
    import torch
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda_device_count={torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"torch_cuda_device_name={torch.cuda.get_device_name(0)}")
except Exception as exc:
    print(f"torch_cuda_probe_error={type(exc).__name__}:{exc}")
PY

python -m pip install --no-cache-dir --target "${PYDEPS}" "vllm==0.6.6.post1" "lm-eval==0.4.12" \
  > "${OUT_DIR}/pip_install_lmeval.txt" 2>&1

PYTHONPATH="${PYDEPS}:${PYTHONPATH:-}" python - <<'PY' > "${OUT_DIR}/imports_after_lmeval.txt" 2>&1
import importlib

for name in ("torch", "vllm", "lm_eval"):
    mod = importlib.import_module(name)
    version = getattr(mod, "__version__", "unknown")
    print(f"{name}={version}")

import torch
print(f"torch_cuda_available={torch.cuda.is_available()}")
print(f"torch_cuda_device_count={torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"torch_cuda_device_name={torch.cuda.get_device_name(0)}")
PY
