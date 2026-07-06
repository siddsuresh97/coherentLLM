#!/usr/bin/env bash
# OLMo-2-7B training-stage sweep: run the full pipeline on each checkpoint
# (base -> SFT -> DPO -> RLVR/Instruct) to see which stage adds coherence.
# All fit one A5000 in bf16 (7B). Uses the base coherence env (transformers 4.47
# supports OLMo-2). Open models -> SINGLE-PAIR feature verification.
set -uo pipefail
cd "$(dirname "$0")/.."

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export VLLM_LOGGING_LEVEL=WARNING
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export HF_HUB_CACHE=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models

for M in olmo2-7b-base olmo2-7b-sft olmo2-7b-dpo olmo2-7b-instruct; do
  echo "########## $M ##########"
  python src/run_local.py --model "$M" --methods triplet pairwise \
    --gpu_mem_util 0.90 --max_model_len 2048 || { echo "[$M] tp/pw FAILED"; continue; }
  python src/run_listing.py --model "$M" --repeats 5 --temperature 0.7 \
    --max_tokens 256 --gpu_mem_util 0.90 --max_model_len 2048 || { echo "[$M] listing FAILED"; continue; }
  python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2
  python src/run_local.py --model "$M" --methods feature \
    --pairs_file verify_pairs.csv --overwrite \
    --gpu_mem_util 0.90 --max_model_len 2048 || echo "[$M] feature FAILED"
  echo "=== $M DONE ==="
done
echo "OLMO LINEAGE DONE"
