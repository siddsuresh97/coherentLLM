#!/usr/bin/env bash
# Generic training-lineage sweep: full pipeline (generation) + logprob for each
# checkpoint. Open models -> single-pair feature. Usage:
#   scripts/run_lineage.sh tulu3-8b-base tulu3-8b-sft tulu3-8b-dpo tulu3-8b-final
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export VLLM_LOGGING_LEVEL=WARNING PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export HF_HUB_CACHE=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models

for M in "$@"; do
  echo "########## $M ##########"
  # logprob (representation-level, works for ALL stages incl base) -> _lp files
  python src/run_base_logprob.py --model "$M" --methods triplet pairwise --suffix _lp --overwrite --gpu_mem_util 0.80 || echo "[$M] logprob FAILED"
  # generation/instruction (only meaningful for tuned stages; base will be noisy)
  python src/run_local.py --model "$M" --methods triplet pairwise --gpu_mem_util 0.85 --max_model_len 2048 || echo "[$M] gen tp FAILED"
  python src/run_listing.py --model "$M" --repeats 5 --temperature 0.7 --max_tokens 256 --gpu_mem_util 0.85 --max_model_len 2048 || echo "[$M] listing FAILED"
  python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2 || true
  python src/run_local.py --model "$M" --methods feature --pairs_file verify_pairs.csv --overwrite --gpu_mem_util 0.85 --max_model_len 2048 || echo "[$M] feature FAILED"
  echo "=== $M DONE ==="
done
echo "LINEAGE DONE"
