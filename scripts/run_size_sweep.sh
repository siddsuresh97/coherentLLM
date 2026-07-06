#!/usr/bin/env bash
# Sequential size sweep with GPU-drain wait between models (avoids vLLM cleanup-lag OOM).
set -uo pipefail
cd "$(dirname "$0")/.."
wait_gpu_free() {
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0 | head -1)
    [ "$used" -lt 1000 ] && return 0
    sleep 3
  done
}
for m in "$@"; do
  wait_gpu_free
  echo "### launching $m (gpu0 free)"
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_model_full.sh "$m" 2048 0.85
  sleep 5
done
echo "SWEEP DONE"
