#!/usr/bin/env bash
# Build the small source/data bundle needed by the CHTC Exp 1 pathway job.
set -euo pipefail

cd "$(dirname "$0")/../.."

OUT="${1:-chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz}"
mkdir -p "$(dirname "$OUT")"

tar \
  --exclude='__pycache__' \
  --exclude='.pytest_cache' \
  --exclude='experiments/exp1_triplet_concept_move/raw' \
  --exclude='experiments/exp1_triplet_concept_move/rdms' \
  --exclude='experiments/exp1_triplet_concept_move/detection' \
  --exclude='experiments/exp1_triplet_concept_move/lora_control' \
  --exclude='experiments/exp1_triplet_concept_move/lora_edit' \
  --exclude='experiments/exp1_triplet_concept_move/lora_control_similarity' \
  --exclude='experiments/exp1_triplet_concept_move/lora_edit_similarity' \
  --exclude='experiments/exp1_triplet_concept_move/sft_data' \
  --exclude='experiments/exp1_triplet_concept_move/sft_similarity_data' \
  -czf "$OUT" \
  Makefile \
  scripts/run_experiment1.py \
  scripts/run_experiment1_triplets.py \
  scripts/run_experiment1_real_gpu.sh \
  scripts/build_experiment1_sft_data.py \
  scripts/build_experiment1_similarity_sft_data.py \
  scripts/build_experiment1_rdm.py \
  scripts/score_experiment1_detection.py \
  scripts/score_experiment1_cell.py \
  scripts/summarize_experiment1_detection.py \
  scripts/simulate_experiment1_detection.py \
  src/prompts.py \
  src/run_local.py \
  src/stimuli.py \
  src/sft/train_lora.py \
  configs/models.yaml \
  data/nova/verified_matrix_cogsci2025.parquet \
  data/scale128/concepts.csv \
  data/scale128/llama-3.1-8b-instruct_triplet_d5.npy \
  results/raw_128/llama-3.1-8b-instruct/triplet.csv \
  experiments/exp1_triplet_concept_move

du -h "$OUT"
