#!/usr/bin/env bash
# Full pipeline for one OpenRouter model: triplet + pairwise + (listing -> union ->
# self-verify feature). Resumable: each stage skips if its output exists.
# Usage: scripts/run_frontier_model.sh <registry-name>
set -euo pipefail
cd "$(dirname "$0")/.."
M="$1"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export OPENROUTER_API_KEY="$(cat .openrouter_key)"

CONC="${CONC:-10}"

echo "=== [$M] triplet + pairwise ==="
python src/run_openrouter.py --model "$M" --methods triplet pairwise \
  --concurrency "$CONC" --reasoning_effort low

echo "=== [$M] feature listing ==="
python src/run_openrouter.py --model "$M" --methods listing \
  --repeats 5 --temperature 0.7 --concurrency "$CONC" --reasoning_effort low

echo "=== [$M] build union ==="
python src/build_union.py --model "$M" --min_listings 1 --min_concepts 2

echo "=== [$M] feature self-verification ==="
python src/run_openrouter.py --model "$M" --methods feature \
  --pairs_file verify_pairs.csv --feature_batch 20 \
  --concurrency "$CONC" --reasoning_effort low

echo "=== [$M] DONE ==="
