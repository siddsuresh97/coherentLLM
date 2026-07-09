#!/usr/bin/env bash
# End-to-end real GPU run for Experiment 1.
#
# This script assumes it is running on a GPU host with the repo's coherence conda
# env and cached Llama-3.1-8B-Instruct weights. It preserves the hard gate:
# base canonical/paraphrase triplet runs are collected and `run_experiment1.py`
# is rerun before any LoRA data/training step.
set -euo pipefail

cd "$(dirname "$0")/.."

COHERENCE_ENV="${COHERENCE_ENV:-/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence}"
if [ -x "${COHERENCE_ENV}/bin/python" ]; then
  export PATH="${COHERENCE_ENV}/bin:${PATH}"
  export CONDA_PREFIX="${COHERENCE_ENV}"
elif command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$COHERENCE_ENV"
else
  echo "Cannot find Python env at COHERENCE_ENV=${COHERENCE_ENV}" >&2
  exit 2
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
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}"
TRAIN_GRAD_ACCUM="${TRAIN_GRAD_ACCUM:-1}"
TRAIN_BACKEND="${TRAIN_BACKEND:-auto}"
TRAIN_LEARNING_RATE="${TRAIN_LEARNING_RATE:-2e-4}"
LORA_RANK="${LORA_RANK:-32}"
SEED="${SEED:-1729}"
SUPERVISION="${SUPERVISION:-feature}"  # feature or similarity
TRAIN_REPORT_TO="${TRAIN_REPORT_TO:-wandb}"
WANDB_PROJECT="${WANDB_PROJECT:-coherentLLM-exp1}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DIR="${WANDB_DIR:-$PWD/wandb}"
WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$WANDB_DIR/cache}"
WANDB_NETRC="${WANDB_NETRC:-/mnt/home/ssuresh/.netrc}"
REBUILD_SFT_DATA="${REBUILD_SFT_DATA:-1}"
FEATURE_TARGET_REPEATS="${FEATURE_TARGET_REPEATS:-64}"
TRIPLET_BACKEND="${TRIPLET_BACKEND:-vllm}"  # vllm or transformers
TRIPLET_BATCH_SIZE="${TRIPLET_BATCH_SIZE:-16}"
TRIPLET_LOAD_IN_4BIT="${TRIPLET_LOAD_IN_4BIT:-0}"
export WANDB_MODE WANDB_DIR WANDB_CACHE_DIR WANDB_PROJECT

if [ "$TRAIN_REPORT_TO" = "wandb" ]; then
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"
  if [ -z "${WANDB_API_KEY:-}" ] && [ -r "$WANDB_NETRC" ]; then
    WANDB_API_KEY="$(
      python - "$WANDB_NETRC" <<'PY'
import netrc
import sys

path = sys.argv[1]
try:
    auth = netrc.netrc(path).authenticators("api.wandb.ai")
except Exception:
    auth = None
if auth and auth[2]:
    print(auth[2], end="")
PY
    )"
    export WANDB_API_KEY
  fi
  if [ "$WANDB_MODE" = "online" ] && [ -z "${WANDB_API_KEY:-}" ]; then
    echo "TRAIN_REPORT_TO=wandb and WANDB_MODE=online, but no W&B API key was found." >&2
    echo "Set WANDB_API_KEY or make WANDB_NETRC point at a readable netrc with api.wandb.ai." >&2
    exit 2
  fi
  echo "[exp1] wandb project=${WANDB_PROJECT} mode=${WANDB_MODE} dir=${WANDB_DIR}"
fi

if [ "${EXP1_PREFLIGHT_ONLY:-0}" = "1" ]; then
  echo "[exp1] preflight OK"
  exit 0
fi

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
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || true)"
    if [[ ! "${used:-}" =~ ^[0-9]+$ ]]; then
      return 0
    fi
    if [ "$used" -lt 1000 ]; then
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
  local backend_arg=(--backend "$TRIPLET_BACKEND" --batch-size "$TRIPLET_BATCH_SIZE")
  if [ "$TRIPLET_LOAD_IN_4BIT" = "1" ]; then
    backend_arg+=(--load-in-4bit)
  fi
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
    "${backend_arg[@]}" \
    --overwrite \
    "${lora_arg[@]}"
}

echo "[exp1] collecting frozen-protocol base floor"
run_triplets base_seed_a_canonical_prompt canonical
run_triplets base_seed_b_canonical_prompt canonical
run_triplets base_seed_a_paraphrase_prompt paraphrase

echo "[exp1] refreshing behavioral floor from frozen-protocol runs"
python - <<'PY'
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path.cwd()
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
sys.path.insert(0, str(ROOT / "scripts"))
from run_experiment1 import parse_triplet_raw, pearson_corr, rdm_from_triplet_rows, upper_values  # noqa: E402

items = json.loads((EXP_DIR / "items.json").read_text())
config = json.loads((EXP_DIR / "config.json").read_text())
required_runs = config["triplet_protocol"]["required_baseline_runs"]
rdms = {}
for run_name in required_runs:
    raw_csv = EXP_DIR / "raw" / run_name / "triplet.csv"
    if not raw_csv.exists():
        raise SystemExit(f"missing required frozen-protocol run: {raw_csv}")
    rdms[run_name] = rdm_from_triplet_rows(parse_triplet_raw(raw_csv), items["concepts"])

base_run = "base_seed_a_canonical_prompt"
base_rdm = rdms[base_run]
rdm_dir = EXP_DIR / "artifacts" / "rdms"
rdm_dir.mkdir(parents=True, exist_ok=True)
np.save(rdm_dir / "rdm_base.npy", base_rdm)

comparisons = []
for run_name, rdm in rdms.items():
    if run_name == base_run:
        continue
    delta = upper_values(rdm - base_rdm)
    comparisons.append(
        {
            "run": run_name,
            "upper_triangle_pearson_vs_base": pearson_corr(upper_values(base_rdm), upper_values(rdm)),
            "overall_rms_vs_base": float(np.sqrt(np.mean(delta**2))),
            "pair_abs_median_vs_base": float(np.median(np.abs(delta))),
            "pair_abs_p90_vs_base": float(np.quantile(np.abs(delta), 0.90)),
        }
    )

mean_corr = float(np.nanmean([row["upper_triangle_pearson_vs_base"] for row in comparisons]))
mean_rms = float(np.nanmean([row["overall_rms_vs_base"] for row in comparisons]))
gate_passed = bool(mean_corr >= 0.80 and np.isfinite(mean_rms))
floor = {
    "status": "green" if gate_passed else "red",
    "gate_passed": gate_passed,
    "why_not_green": "" if gate_passed else "Frozen-protocol runs exist, but reliability is below threshold.",
    "source": "frozen_30_item_protocol_runs",
    "base_run": base_run,
    "required_protocol_runs": required_runs,
    "present_required_protocol_runs": required_runs,
    "missing_required_protocol_runs": [],
    "protocol_comparisons": comparisons,
    "protocol_floor": {
        "mean_upper_triangle_pearson": mean_corr,
        "overall_rms_floor_mean": mean_rms,
        "overall_rms_floor_max": float(np.max([row["overall_rms_vs_base"] for row in comparisons])),
        "pair_abs_floor_median_max": float(np.max([row["pair_abs_median_vs_base"] for row in comparisons])),
        "pair_abs_floor_p90_max": float(np.max([row["pair_abs_p90_vs_base"] for row in comparisons])),
    },
    "floor_interpretation": (
        "This is the experiment floor: canonical rerun plus paraphrase variation "
        "under the frozen 30-item protocol."
    ),
}
(EXP_DIR / "floor_stats.json").write_text(json.dumps(floor, indent=2, sort_keys=True) + "\n")
if not gate_passed:
    raise SystemExit(f"behavioral gate did not pass: {floor['why_not_green']}")
print("[exp1] behavioral gate green")
PY

for edit_id in "${EDIT_IDS[@]}"; do
  echo "[exp1] cell ${edit_id}: data"
  if [ "$SUPERVISION" = "feature" ]; then
    python scripts/build_experiment1_sft_data.py \
      --edit-spec "experiments/exp1_triplet_concept_move/candidate_edits/${edit_id}.json" \
      --max-features-per-concept 80 \
      --repeats 8 \
      --target-repeats "$FEATURE_TARGET_REPEATS" \
      --train-max-steps "$TRAIN_STEPS" \
      --lora-rank "$LORA_RANK" \
      --seed "$SEED"
    control_data="experiments/exp1_triplet_concept_move/sft_data/${edit_id}/control.jsonl"
    edit_data="experiments/exp1_triplet_concept_move/sft_data/${edit_id}/edit.jsonl"
    control_adapter="experiments/exp1_triplet_concept_move/lora_control/${edit_id}"
    edit_adapter="experiments/exp1_triplet_concept_move/lora_edit/${edit_id}"
  elif [ "$SUPERVISION" = "similarity" ]; then
    control_data="experiments/exp1_triplet_concept_move/sft_similarity_data/${edit_id}/control.jsonl"
    edit_data="experiments/exp1_triplet_concept_move/sft_similarity_data/${edit_id}/edit.jsonl"
    if [ "$REBUILD_SFT_DATA" = "1" ] || [ ! -s "$control_data" ] || [ ! -s "$edit_data" ]; then
      python scripts/build_experiment1_similarity_sft_data.py \
        --edit-spec "experiments/exp1_triplet_concept_move/candidate_edits/${edit_id}.json" \
        --repeats 6 \
        --train-max-steps "$TRAIN_STEPS" \
        --lora-rank "$LORA_RANK" \
        --seed "$SEED"
    else
      echo "[exp1] cell ${edit_id}: using prebuilt similarity SFT data"
    fi
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
    --backend "$TRAIN_BACKEND" \
    --learning_rate "$TRAIN_LEARNING_RATE" \
    --per_device_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$TRAIN_GRAD_ACCUM" \
    --seed "$SEED" \
    --report_to "$TRAIN_REPORT_TO" \
    --wandb_project "$WANDB_PROJECT" \
    "${train_model_arg[@]}"

  echo "[exp1] cell ${edit_id}: train edit"
  drain_gpu
  python src/sft/train_lora.py \
    --data "$edit_data" \
    --out "$edit_adapter" \
    --max_steps "$TRAIN_STEPS" \
    --epochs 1 \
    --lora_rank "$LORA_RANK" \
    --backend "$TRAIN_BACKEND" \
    --learning_rate "$TRAIN_LEARNING_RATE" \
    --per_device_batch_size "$TRAIN_BATCH_SIZE" \
    --gradient_accumulation_steps "$TRAIN_GRAD_ACCUM" \
    --seed "$SEED" \
    --report_to "$TRAIN_REPORT_TO" \
    --wandb_project "$WANDB_PROJECT" \
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
