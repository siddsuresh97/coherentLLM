#!/usr/bin/env bash
# SALMON d=5 fit for qwen2.5-32b at n=60 and n=128. Run in the SALMON env with
# PYTHONNOUSERSITE=1 from repo root, AFTER run_qwen32b_scale.sh produced triplets.
set -e
REPO=/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments
cd "$REPO"; mkdir -p out
M=qwen2.5-32b-instruct
# n=128 already fit (data/scale128/qwen2.5-32b-instruct_triplet_d5.npy) - skip if present
if [ ! -f "data/scale128/${M}_triplet_d5.npy" ]; then
  COHERENCE_RAW_DIR="$REPO/results/raw_128" COHERENCE_SCALE_DIR="$REPO/data/scale128" \
    python src/elbow_d.py $M 5
fi
COHERENCE_RAW_DIR="$REPO/results/raw_60" COHERENCE_SCALE_DIR="$REPO/data/scale60" \
  python src/elbow_d.py $M 5
echo "===== qwen-32b SALMON d=5 done for n=60,128 ====="
