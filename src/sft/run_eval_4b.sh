#!/usr/bin/env bash
# Self-contained, resumable task-4B eval driver. Owns the WHOLE remaining pipeline so
# it survives driver/codex death and does not need babysitting.
#
# Design:
#  - Each vLLM step is its OWN python process that EXITS (frees the GPU) before the next
#    loads -> no two vLLM instances at once -> fixes the H100 OOM.
#  - expandable_segments avoids fragmentation OOM.
#  - Idempotent: every step is SKIPPED if its output already exists, so a restart resumes.
#  - retry(): each step retried up to 3x before the script gives up on it.
#  - Runs real-logprob on A5000#1 and retention on the H100 sequentially here (single
#    driver, no cross-GPU contention); both GPUs are cheap for these small jobs.
#  - NEVER GPU 0 locally (CUDA_VISIBLE_DEVICES=1). H100 work goes over ssh.
#
# Usage:  nohup bash src/sft/run_eval_4b.sh </dev/null >logs/eval_4b_driver.log 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
ROOT=$(pwd)
mkdir -p logs

CONDA='source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence'
# SALMON needs the dedicated 'salmon' env (skorch); coherence env lacks it.
SALMON_CONDA='source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/salmon'
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
RAW=results/sft_eval/raw
# Stimulus files (pairs.csv, triplets.csv, concepts.csv) live in the stimuli/ subdir,
# which is what COHERENCE_STIM_DIR must point to (matches the original launcher scripts).
STIM=data/scale128/stimuli
SSH='ssh -i /mnt/ws/home/ssuresh/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=15 ssuresh@opt-a007.discovery.wisc.edu'

log(){ echo "[eval4b $(date +%H:%M:%S)] $*"; }

# retry CMD... : run in the coherence env, up to 3 attempts
retry(){
  local n=0
  while [ $n -lt 3 ]; do
    n=$((n+1))
    log "attempt $n: $*"
    bash -c "$CONDA && $*" && return 0
    log "FAILED (attempt $n): $*"
    sleep 5
  done
  log "GIVING UP after 3 attempts: $*"
  return 1
}

# retry_salmon CMD... : same as retry() but in the salmon env (has skorch)
retry_salmon(){
  local n=0
  while [ $n -lt 3 ]; do
    n=$((n+1))
    log "salmon attempt $n: $*"
    bash -c "$SALMON_CONDA && $*" && return 0
    log "SALMON FAILED (attempt $n): $*"
    sleep 5
  done
  log "GIVING UP (salmon) after 3 attempts: $*"
  return 1
}

# --- H100: real-adapter logprob (missing pairwise_lp, feature_lp) ---
# prompt_logprobs scoring on 8B+LoRA does NOT fit in the 24 GB A5000 (OOMs), so this
# runs on the 80 GB H100 over ssh. Shared FS -> writes straight into the local RAW dir.
# Run SYNCHRONOUSLY (foreground) so SALMON/aggregate wait for the _lp files.
run_real_logprob(){
  for m in pairwise feature; do
    if [ -s "$RAW/llama31-sft-real/${m}_lp.csv" ]; then log "skip real ${m}_lp (exists)"; continue; fi
    log "real ${m}_lp on H100"
    $SSH "cd $ROOT && $CONDA && export COHERENCE_FORCE_BF16=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      COHERENCE_RAW_DIR=$RAW COHERENCE_STIM_DIR=$STIM && \
      python src/run_base_logprob.py --model llama-3.1-8b-instruct --out_model llama31-sft-real \
      --lora out/adapters_vllm_fixed/real --methods $m --suffix _lp --gpu_mem_util 0.90 --max_model_len 2048" \
      || { log "real ${m}_lp FAILED on H100"; return 1; }
  done
}

# --- SALMON d5 fits: real + scrambled, gen and lp triplets ---
run_salmon(){
  for tag in "llama31-sft-real:triplet" "llama31-sft-real:triplet_lp" \
             "llama31-sft-scrambled:triplet" "llama31-sft-scrambled:triplet_lp"; do
    model="${tag%%:*}"; trip="${tag##*:}"
    suffix=""; [ "$trip" = "triplet_lp" ] && suffix="_lp"
    out="results/sft_eval/${model}_${trip}_d5.npy"
    # fit_triplet_salmon writes <model><tag>.npy; we just need the npy to exist. Skip if present.
    if ls results/sft_eval/${model}*d5.npy >/dev/null 2>&1 && [ -s "$out" ]; then log "skip SALMON $tag"; continue; fi
    retry_salmon "COHERENCE_RAW_DIR=$RAW python src/fit_triplet_salmon.py --raw_dir $RAW --stim_dir $STIM \
      --out_dir results/sft_eval --d 5 --suffix '$suffix' $model" || log "SALMON $tag failed (continuing)"
  done
}

# --- Retention on the H100 (needs lm_eval; install if missing) ---
run_retention(){
  if [ -s results/sft_eval/retention.json ] || [ -s results/sft_eval/retention.csv ]; then
    log "skip retention (exists)"; return 0; fi
  $SSH "cd $ROOT && nohup bash -c '$CONDA && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
    python -c \"import lm_eval\" 2>/dev/null || pip install -q lm-eval && \
    COHERENCE_FORCE_BF16=1 python src/sft/eval_retention.py' </dev/null >logs/h100_retention.log 2>&1 & disown; echo H100-RETENTION-LAUNCHED" \
    && log "retention launched on H100" || log "retention ssh launch failed"
}

# --- CPU-light axes: paraphrase + transitivity (safe to run locally) ---
run_axes(){
  [ -f src/sft/run_extra_eval.py ] || { log "no run_extra_eval.py; skipping axes"; return 0; }
  retry "python src/sft/run_extra_eval.py" || log "extra axes failed (continuing)"
}

# --- Aggregate: only when the raw + SALMON inputs exist ---
aggregate(){
  retry "python src/sft/analyze_eval.py" || { log "aggregate failed"; return 1; }
}

log "=== task-4B driver start ==="
run_real_logprob
run_salmon
run_retention          # fires H100 job in background; retention.json lands async
run_axes

# wait for retention to finish (H100 async), up to ~40 min, then aggregate regardless
for i in $(seq 1 80); do
  [ -s results/sft_eval/retention.json ] || [ -s results/sft_eval/retention.csv ] && break
  sleep 30
done
aggregate

if [ -s results/sft_eval/eval_results.csv ]; then
  log "=== DONE: eval_results.csv written ==="
  cat results/sft_eval/eval_results.csv
else
  log "=== INCOMPLETE: eval_results.csv missing after full run — needs attention ==="
fi
