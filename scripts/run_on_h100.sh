#!/usr/bin/env bash
# Run the pending BIG models on an H100 (80GB) in bf16 - no quantization needed,
# so we avoid all the A5000 memory/quant/env pain. Same filesystem, same repo path.
#
# Models registered with quantization: bitsandbytes will still 4-bit unless we
# override. On H100 we want bf16, so we clear quantization via an env flag the
# runners can read (or just run the non-quantized registry entries).
#
# Usage on the H100:
#   cd .../coherence_experiments && bash scripts/run_on_h100.sh
set -uo pipefail
cd "$(dirname "$0")/.."
source "$(conda info --base)/etc/profile.d/conda.sh"
# Prefer an H100-appropriate env; fall back to the base coherence env.
if conda env list | grep -q "coherence_h100"; then
  conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence_h100
else
  conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
fi
export VLLM_LOGGING_LEVEL=WARNING
export HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export HF_HUB_CACHE=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models
export COHERENCE_FORCE_BF16=1   # runners: ignore spec 'quantization' when set

run() {  # model  [max_len]
  local M="$1"; local ML="${2:-2048}"
  echo "########## $M ##########"
  python src/run_local.py --model "$M" --methods triplet pairwise --gpu_mem_util 0.90 --max_model_len "$ML" || echo "[$M] tp FAIL"
  python src/run_listing.py --model "$M" --repeats 5 --temperature 0.7 --max_tokens 256 --gpu_mem_util 0.90 --max_model_len "$ML" || echo "[$M] listing FAIL"
  python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2 || true
  python src/run_local.py --model "$M" --methods feature --pairs_file verify_pairs.csv --overwrite --gpu_mem_util 0.90 --max_model_len "$ML" || echo "[$M] feat FAIL"
  echo "=== $M DONE ==="
}

# Size ladder top end (bf16 on H100)
run qwen2.5-14b-instruct-size
run qwen2.5-32b-instruct
# OLMo-2-13B lineage (size x tuning) - base needs logprob
python src/run_base_logprob.py --model olmo2-13b-base --methods triplet pairwise --suffix _lp --overwrite --gpu_mem_util 0.90 || echo "[olmo13b-base lp FAIL]"
python src/run_fewshot.py --model olmo2-13b-base --methods triplet pairwise --overwrite --gpu_mem_util 0.90 || echo "[olmo13b-base fs FAIL]"
run olmo2-13b-sft
run olmo2-13b-dpo
run olmo2-13b-instruct
echo "H100 BATCH DONE"
