#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/sft_huth_lebel

ROOTS=("$PWD")
if [[ -n "${STAGING:-}" ]]; then
  ROOTS+=(
    "$STAGING/datasets/ds003020"
    "$STAGING/datasets/lebel/ds003020"
    "$STAGING/ds003020"
  )
fi
ROOTS+=("$PWD/download/ds003020")

python huth_lebel_audit.py \
  --roots "${ROOTS[@]}" \
  --out_dir results/sft_huth_lebel \
  --max_scan_depth 4

tar -czf huth_lebel_audit_results.tgz results/sft_huth_lebel
