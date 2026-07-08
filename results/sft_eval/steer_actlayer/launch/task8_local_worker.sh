#!/usr/bin/env bash
set -u

GPU="${1:?usage: task8_local_worker.sh GPU}"
ROOT="/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments"
QUEUE="$ROOT/results/sft_eval/steer_actlayer/launch/task8_single_queue.tsv"
LOCK="$ROOT/results/sft_eval/steer_actlayer/launch/task8_single_queue.lock"
WORKER_LOG="$ROOT/results/sft_eval/steer_actlayer/launch/task8-local-gpu${GPU}.log"
CONDA_SH="/mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh"
ENV_PATH="/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence"

cd "$ROOT" || exit 1
source "$CONDA_SH"
conda activate "$ENV_PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$(dirname "$QUEUE")"
touch "$LOCK"

claim_next() {
  flock "$LOCK" bash -c '
    queue="$1"
    tmp="${queue}.tmp.$$"
    [ -s "$queue" ] || exit 1
    IFS= read -r first < "$queue" || exit 1
    tail -n +2 "$queue" > "$tmp"
    mv "$tmp" "$queue"
    printf "%s\n" "$first"
  ' _ "$QUEUE"
}

{
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] worker start gpu=$GPU"
  while true; do
    next="$(claim_next)" || {
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] queue empty gpu=$GPU"
      break
    }
    layer="$(printf '%s' "$next" | awk '{print $1}')"
    alpha="$(printf '%s' "$next" | awk '{print $2}')"
    if [ -z "$layer" ] || [ -z "$alpha" ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] malformed queue row: $next"
      continue
    fi
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=$GPU run layer=$layer alpha=$alpha"
    python -u src/sft/eval_actlayer_steer.py run-config \
      --layer-set "$layer" \
      --alpha "$alpha" \
      --batch-size 8 \
      --device cuda
    rc=$?
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] gpu=$GPU done layer=$layer alpha=$alpha rc=$rc"
    if [ "$rc" -ne 0 ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] stopping gpu=$GPU after failure"
      exit "$rc"
    fi
  done
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] worker exit gpu=$GPU"
} 2>&1 | tee -a "$WORKER_LOG"
