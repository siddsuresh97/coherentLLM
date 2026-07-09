#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="${ROOT}/chtc/exp3_directional_confusions/exp3_directional_confusions_bundle.tgz"

cd "${ROOT}"
python scripts/run_experiment3.py init --no-log

tar \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='experiments/exp3_directional_confusions/raw' \
  --exclude='experiments/exp3_directional_confusions/results' \
  -czf "${OUT}" \
  scripts/run_experiment3.py \
  src/prompts.py \
  experiments/exp3_directional_confusions \
  data/human/leuven_groundtruth_matrix.csv \
  data/human/leuven300_feature_map.csv

echo "wrote ${OUT}"
