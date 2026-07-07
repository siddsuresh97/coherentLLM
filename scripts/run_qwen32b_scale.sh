#!/usr/bin/env bash
# Full qwen2.5-32b-instruct scale pipeline for n=60 and n=128, on the H100.
# Run FROM the repo root in the coherence env. Emits triplet+listing+verify; SALMON d=5
# is a separate step in the salmon env (see run_qwen32b_salmon.sh).
set -e
REPO=/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments
cd "$REPO"
M=qwen2.5-32b-instruct
export COHERENCE_FORCE_BF16=1

for SCALE in 60 128; do
  STIM="$REPO/data/scale${SCALE}/stimuli"
  RAWD="$REPO/results/raw_${SCALE}"
  echo "===== n=${SCALE}: $M ====="

  # 1. triplet (skip if already present with enough rows)
  if [ ! -f "$RAWD/$M/triplet.csv" ] || [ "$(wc -l < "$RAWD/$M/triplet.csv" 2>/dev/null || echo 0)" -lt 100 ]; then
    COHERENCE_STIM_DIR="$STIM" COHERENCE_RAW_DIR="$RAWD" \
      python src/run_local.py --model $M --methods triplet \
        --max_model_len 128 --gpu_mem_util 0.95 --max_num_seqs 512 --overwrite
  fi

  # 2. feature listing
  if [ ! -f "$RAWD/$M/listing.csv" ]; then
    COHERENCE_STIM_DIR="$STIM" COHERENCE_RAW_DIR="$RAWD" \
      python src/run_listing.py --model $M --repeats 3 --temperature 0.7 \
        --max_model_len 1024 --gpu_mem_util 0.92 --max_num_seqs 256
  fi

  # 3. consolidate union (features shared by >=3 concepts)
  COHERENCE_STIM_DIR="$STIM" COHERENCE_RAW_DIR="$RAWD" \
    python src/build_union.py --model $M --min_concepts 3

  # 4. single-pair feature verification
  COHERENCE_STIM_DIR="$STIM" COHERENCE_RAW_DIR="$RAWD" \
    python src/run_local.py --model $M --methods feature --pairs_file verify_pairs.csv \
      --max_model_len 256 --gpu_mem_util 0.92 --max_num_seqs 512 --overwrite
done
echo "===== qwen-32b triplet+listing+verify done for n=60,128 ====="
