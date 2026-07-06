#!/usr/bin/env bash
# OLMo-2-32B lineage on H100 (bf16). base in muri cache, tuned stages in nsf cache.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
# OLMo-2-0325-32B needs vLLM 0.8.5 (0.6.6 mis-executes it: "hidden_size found 2390")
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence_big
export VLLM_LOGGING_LEVEL=WARNING PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export HF_HUB_CACHE=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export COHERENCE_EXTRA_CACHE=/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/shared_model_weights
export COHERENCE_FORCE_BF16=1
GM=0.85

drain() {
  for i in $(seq 1 40); do
    u=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${u:-99999}" -lt 2000 ] && return 0
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p "$pid" 2>/dev/null)" = "$USER" ] && kill -9 "$pid" 2>/dev/null
    done; sleep 4
  done
}
step() { drain; "$@" || echo "[STEP FAIL] $*"; }
run() {
  local M="$1"
  echo "########## $M ##########"
  step python src/run_local.py --model "$M" --methods triplet pairwise --gpu_mem_util "$GM" --max_model_len 2048
  step python src/run_listing.py --model "$M" --repeats 5 --temperature 0.7 --max_tokens 256 --gpu_mem_util "$GM" --max_model_len 2048
  python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2 || true
  step python src/run_local.py --model "$M" --methods feature --pairs_file verify_pairs.csv --overwrite --gpu_mem_util "$GM" --max_model_len 2048
  echo "=== $M DONE ==="
}
step python src/run_base_logprob.py --model olmo2-32b-base --methods triplet pairwise --suffix _lp --overwrite --gpu_mem_util "$GM"
step python src/run_fewshot.py --model olmo2-32b-base --methods triplet pairwise --overwrite --gpu_mem_util "$GM"
run olmo2-32b-sft
run olmo2-32b-dpo
run olmo2-32b-instruct
echo "OLMO32B LINEAGE DONE"
