#!/usr/bin/env bash
# Full pipeline for one BIG model (4-bit, tensor-parallel across 2 GPUs) in the
# coherence_big env: triplet + pairwise + feature (listing -> union -> self-verify).
# Usage: scripts/run_big_model.sh gemma-3-27b-it
set -euo pipefail
cd "$(dirname "$0")/.."
M="$1"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence_big
export VLLM_LOGGING_LEVEL=WARNING
export CUDA_VISIBLE_DEVICES=0,1
export HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export HF_HUB_CACHE=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models

echo "=== [$M] triplet + pairwise ==="
python src/run_local.py --model "$M" --methods triplet pairwise \
  --gpu_mem_util 0.92 --max_model_len 4096

echo "=== [$M] feature listing ==="
python src/run_listing.py --model "$M" --repeats 5 --temperature 0.7 \
  --max_tokens 256 --gpu_mem_util 0.92 --max_model_len 4096

echo "=== [$M] build union ==="
python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2

echo "=== [$M] feature self-verification ==="
python src/run_local.py --model "$M" --methods feature \
  --pairs_file verify_pairs.csv --feature_batch 20 --overwrite \
  --gpu_mem_util 0.92 --max_model_len 4096

echo "=== [$M] DONE ==="
