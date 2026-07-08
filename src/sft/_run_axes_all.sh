#!/bin/bash
set -uo pipefail
cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments
source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export COHERENCE_FORCE_BF16=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export COHERENCE_RAW_DIR=results/sft_eval/raw COHERENCE_STIM_DIR=data/scale128/stimuli
run_one(){
  local om="$1" lo="$2"
  local d="results/sft_eval/raw/$om"
  if [ -s "$d/transitivity.csv" ] && [ -s "$d/paraphrase_feature.csv" ] && [ -s "$d/paraphrase_pairwise.csv" ] && [ -s "$d/paraphrase_triplet.csv" ]; then
    echo "[axes] skip $om (complete)"; return 0; fi
  local lf=""; [ "$lo" != "-" ] && lf="--lora $lo"
  echo "[axes] === $om ==="
  python src/sft/run_extra_eval.py --model llama-3.1-8b-instruct --out_model "$om" $lf \
    --gpu_mem_util 0.90 --max_model_len 2048 --max_num_seqs 128 --overwrite || echo "[axes] $om FAILED"
}
run_one llama-3.1-8b-instruct -
run_one llama31-sft-real out/adapters_vllm_fixed/real
run_one llama31-sft-scrambled out/adapters_vllm_fixed/scrambled
echo "[axes] ALL-AXES-DONE"
