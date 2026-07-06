#!/usr/bin/env bash
# Run pending BIG models on the H100 (80GB) in bf16. Robust against vLLM's slow
# GPU-memory release: DRAIN the GPU (kill leftover python, wait for <2GB free-used)
# before every step, so back-to-back model loads never collide/OOM/orphan.
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export VLLM_LOGGING_LEVEL=WARNING
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export HF_HUB_CACHE=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export COHERENCE_FORCE_BF16=1
GM=0.80   # leave headroom so a lagging release doesn't starve the next load

drain() {
  # wait until GPU is actually free; if a leftover python is squatting, reap it.
  for i in $(seq 1 40); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    [ "${used:-99999}" -lt 2000 ] && return 0
    # only our own leftover workers (never touch other users)
    for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader); do
      [ "$(ps -o user= -p "$pid" 2>/dev/null)" = "$USER" ] && kill -9 "$pid" 2>/dev/null
    done
    sleep 4
  done
}

step() { drain; "$@" || echo "[STEP FAIL] $*"; }

run() {  # model [max_len]
  local M="$1"; local ML="${2:-2048}"
  echo "########## $M ##########"
  step python src/run_local.py --model "$M" --methods triplet pairwise --gpu_mem_util "$GM" --max_model_len "$ML"
  step python src/run_listing.py --model "$M" --repeats 5 --temperature 0.7 --max_tokens 256 --gpu_mem_util "$GM" --max_model_len "$ML"
  python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2 || true
  step python src/run_local.py --model "$M" --methods feature --pairs_file verify_pairs.csv --overwrite --gpu_mem_util "$GM" --max_model_len "$ML"
  echo "=== $M DONE ==="
}

# Size ladder top (bf16)
run qwen2.5-14b-instruct-size
run qwen2.5-32b-instruct
# OLMo-2-13B lineage (base measured by logprob + few-shot)
step python src/run_base_logprob.py --model olmo2-13b-base --methods triplet pairwise --suffix _lp --overwrite --gpu_mem_util "$GM"
step python src/run_fewshot.py --model olmo2-13b-base --methods triplet pairwise --overwrite --gpu_mem_util "$GM"
run olmo2-13b-sft
run olmo2-13b-dpo
run olmo2-13b-instruct
# OLMo-2-32B lineage (largest with full training axis) - only if downloaded
if [ -d /mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--allenai--OLMo-2-0325-32B-Instruct ]; then
  step python src/run_base_logprob.py --model olmo2-32b-base --methods triplet pairwise --suffix _lp --overwrite --gpu_mem_util "$GM"
  step python src/run_fewshot.py --model olmo2-32b-base --methods triplet pairwise --overwrite --gpu_mem_util "$GM"
  run olmo2-32b-sft
  run olmo2-32b-dpo
  run olmo2-32b-instruct
fi
echo "H100 BATCH DONE"
