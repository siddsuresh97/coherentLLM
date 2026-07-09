#!/usr/bin/env bash
# End-to-end real GPU run for Experiment 1.
#
# This script assumes it is running on a GPU host with the repo's coherence conda
# env and cached Llama-3.1-8B-Instruct weights. It preserves the hard gate:
# base canonical/paraphrase triplet runs are collected and `run_experiment1.py`
# is rerun before any LoRA data/training step.
set -euo pipefail

cd "$(dirname "$0")/.."

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
fi

export VLLM_LOGGING_LEVEL="${VLLM_LOGGING_LEVEL:-WARNING}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HOME="${HF_HOME:-/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/out/matplotlib_cache}"
if [[ "${CUDA_VISIBLE_DEVICES:-}" == GPU-* ]]; then
  export CUDA_VISIBLE_DEVICES=0
fi

MODEL="${MODEL:-llama-3.1-8b-instruct}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-}"
TRAIN_BASE_MODEL="${TRAIN_BASE_MODEL:-${BASE_MODEL_PATH}}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.88}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-256}"
TRAIN_STEPS="${TRAIN_STEPS:-400}"
LORA_RANK="${LORA_RANK:-32}"
SEED="${SEED:-1729}"
SUPERVISION="${SUPERVISION:-feature}"  # feature or similarity

if [ "$#" -gt 0 ]; then
  EDIT_IDS=("$@")
else
  # Cheap pathway check first, then a coarse boundary sweep.
  EDIT_IDS=(
    concentrated_drop_100
    concentrated_drop_065
    concentrated_drop_035
    concentrated_drop_015
    diffuse_drop_100
    diffuse_drop_065
    diffuse_drop_035
    diffuse_drop_015
  )
fi

drain_gpu() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    return 0
  fi
  for _ in $(seq 1 60); do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 || true)"
    if [ "${used:-0}" -lt 1000 ]; then
      return 0
    fi
    sleep 3
  done
}

run_triplets() {
  local run_name="$1"
  local variant="$2"
  local lora_arg=()
  local model_path_arg=()
  if [ "${3:-}" != "" ]; then
    lora_arg=(--lora "$3")
  fi
  if [ "${BASE_MODEL_PATH}" != "" ]; then
    model_path_arg=(--model-path "${BASE_MODEL_PATH}" --hf-cache "${HF_HOME}")
  fi
  drain_gpu
  python scripts/run_experiment1_triplets.py \
    --model "$MODEL" \
    "${model_path_arg[@]}" \
    --out-run "$run_name" \
    --prompt-variant "$variant" \
    --gpu_mem_util "$GPU_MEM_UTIL" \
    --max_model_len "$MAX_MODEL_LEN" \
    --max_num_seqs "$MAX_NUM_SEQS" \
    --overwrite \
    "${lora_arg[@]}"
}

echo "[exp1] collecting frozen-protocol base floor"
run_triplets base_seed_a_canonical_prompt canonical
run_triplets base_seed_b_canonical_prompt canonical
run_triplets base_seed_a_paraphrase_prompt paraphrase

echo "[exp1] regenerating base artifacts; floor must turn green before training"
python scripts/run_experiment1.py --allow-provisional-edits
python - <<'PY'
import json
from pathlib import Path
floor = json.loads(Path("experiments/exp1_triplet_concept_move/floor_stats.json").read_text())
if not floor.get("gate_passed"):
    raise SystemExit(f"behavioral gate did not pass: {floor.get('why_not_green')}")
print("[exp1] behavioral gate green")
PY

for edit_id in "${EDIT_IDS[@]}"; do
  echo "[exp1] cell ${edit_id}: data"
  if [ "$SUPERVISION" = "feature" ]; then
    python scripts/build_experiment1_sft_data.py \
      --edit-spec "experiments/exp1_triplet_concept_move/candidate_edits/${edit_id}.json" \
      --max-features-per-concept 80 \
      --repeats 8 \
      --train-max-steps "$TRAIN_STEPS" \
      --lora-rank "$LORA_RANK" \
      --seed "$SEED"
    control_data="experiments/exp1_triplet_concept_move/sft_data/${edit_id}/control.jsonl"
    edit_data="experiments/exp1_triplet_concept_move/sft_data/${edit_id}/edit.jsonl"
    control_adapter="experiments/exp1_triplet_concept_move/lora_control/${edit_id}"
    edit_adapter="experiments/exp1_triplet_concept_move/lora_edit/${edit_id}"
  elif [ "$SUPERVISION" = "similarity" ]; then
    python scripts/build_experiment1_similarity_sft_data.py \
      --edit-spec "experiments/exp1_triplet_concept_move/candidate_edits/${edit_id}.json" \
      --repeats 6 \
      --train-max-steps "$TRAIN_STEPS" \
      --lora-rank "$LORA_RANK" \
      --seed "$SEED"
    control_data="experiments/exp1_triplet_concept_move/sft_similarity_data/${edit_id}/control.jsonl"
    edit_data="experiments/exp1_triplet_concept_move/sft_similarity_data/${edit_id}/edit.jsonl"
    control_adapter="experiments/exp1_triplet_concept_move/lora_control_similarity/${edit_id}"
    edit_adapter="experiments/exp1_triplet_concept_move/lora_edit_similarity/${edit_id}"
  else
    echo "Unknown SUPERVISION=$SUPERVISION (expected feature or similarity)" >&2
    exit 2
  fi

  echo "[exp1] cell ${edit_id}: train control"
  drain_gpu
  train_model_arg=()
  if [ "${TRAIN_BASE_MODEL}" != "" ]; then
    train_model_arg=(--base_model "${TRAIN_BASE_MODEL}" --hf_cache "${HF_HOME}")
  fi
  python src/sft/train_lora.py \
    --data "$control_data" \
    --out "$control_adapter" \
    --max_steps "$TRAIN_STEPS" \
    --epochs 1 \
    --lora_rank "$LORA_RANK" \
    --seed "$SEED" \
    --report_to none \
    "${train_model_arg[@]}"

  echo "[exp1] cell ${edit_id}: train edit"
  drain_gpu
  python src/sft/train_lora.py \
    --data "$edit_data" \
    --out "$edit_adapter" \
    --max_steps "$TRAIN_STEPS" \
    --epochs 1 \
    --lora_rank "$LORA_RANK" \
    --seed "$SEED" \
    --report_to none \
    "${train_model_arg[@]}"

  control_run="control_${SUPERVISION}_${edit_id}"
  edit_run="edit_${SUPERVISION}_${edit_id}"
  echo "[exp1] cell ${edit_id}: behavioral triplets"
  run_triplets "$control_run" canonical "$control_adapter"
  run_triplets "$edit_run" canonical "$edit_adapter"

  echo "[exp1] cell ${edit_id}: detection"
  python scripts/score_experiment1_cell.py \
    --edit-id "$edit_id" \
    --control-run "$control_run" \
    --edit-run "$edit_run"
done

python scripts/summarize_experiment1_detection.py
echo "[exp1] done: experiments/exp1_triplet_concept_move/figs/resolution_heatmap.png"
