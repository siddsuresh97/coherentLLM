#!/usr/bin/env bash
# Run all three methods for a set of local models, one model per GPU in parallel.
# Usage: scripts/run_all_local.sh [feature_sample] model1 model2 ...
set -euo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export VLLM_LOGGING_LEVEL=WARNING

FEAT="${1:-200}"; shift || true
MODELS=("$@")
NGPU=$(nvidia-smi -L | wc -l)

i=0
for m in "${MODELS[@]}"; do
  gpu=$(( i % NGPU ))
  echo "launching $m on GPU $gpu"
  CUDA_VISIBLE_DEVICES=$gpu python src/run_local.py --model "$m" \
    --methods triplet pairwise feature --feature_sample "$FEAT" \
    --gpu_mem_util 0.90 --max_model_len 2048 \
    > "logs/${m}.log" 2>&1 &
  i=$(( i + 1 ))
  # after filling all GPUs, wait for this wave before launching the next
  if (( i % NGPU == 0 )); then wait; fi
done
wait
echo "all local runs done"
