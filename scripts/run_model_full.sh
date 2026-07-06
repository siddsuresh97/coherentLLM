#!/usr/bin/env bash
# Full correct pipeline for one LOCAL model: triplet+pairwise (gen) + listing->union
# ->single-pair feature. Resumable. Usage: CUDA_VISIBLE_DEVICES=0 scripts/run_model_full.sh <model>
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export VLLM_LOGGING_LEVEL=WARNING PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export HF_HUB_CACHE=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
M="$1"; MM="${2:-2048}"; GM="${3:-0.85}"
python src/run_local.py --model "$M" --methods triplet pairwise --gpu_mem_util "$GM" --max_model_len "$MM" || echo "[$M] tp FAIL"
python src/run_listing.py --model "$M" --repeats 5 --temperature 0.7 --max_tokens 256 --gpu_mem_util "$GM" --max_model_len "$MM" || echo "[$M] listing FAIL"
python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2 || true
python src/run_local.py --model "$M" --methods feature --pairs_file verify_pairs.csv --overwrite --gpu_mem_util "$GM" --max_model_len "$MM" || echo "[$M] feat FAIL"
echo "=== $M DONE ==="
