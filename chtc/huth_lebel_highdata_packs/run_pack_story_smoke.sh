#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/staging/s/suresh27/datasets/ds003020-smoke}"
MANIFEST="${MANIFEST:-staging_manifest_smoke.csv}"
STORY="${STORY:-againstthewind}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRATCH_ROOT="$PWD"
OUT_DIR="$SCRATCH_ROOT/results/huth_lebel_pack_story_smoke"
PACK_DIR="$OUT_DIR/packs"
EXTRACT_DIR="$OUT_DIR/extracted"
TARBALL="$SCRATCH_ROOT/huth_lebel_pack_story_smoke_results.tgz"
KEEP_PACK_OUTPUT="${KEEP_PACK_OUTPUT:-0}"

mkdir -p "$OUT_DIR" "$PACK_DIR" "$EXTRACT_DIR"
trap 'status=$?; echo "$status" > "$OUT_DIR/exit_status.txt"; tar -czf "$TARBALL" -C "$SCRATCH_ROOT" results || true; exit "$status"' EXIT

{
  echo "started=$(date -Is)"
  echo "hostname=$(hostname -f || hostname)"
  echo "pwd=$PWD"
  echo "dataset_root=$DATASET_ROOT"
  echo "manifest=$MANIFEST"
  echo "story=$STORY"
  echo "python_bin=$PYTHON_BIN"
  echo "keep_pack_output=$KEEP_PACK_OUTPUT"
  printf "zstd_path="
  command -v zstd || true
  tar --version | head -1 || true
} | tee "$OUT_DIR/run_env.txt"

if [[ ! -d "$DATASET_ROOT" ]]; then
  echo "missing DATASET_ROOT=$DATASET_ROOT" >&2
  exit 2
fi
if [[ ! -f "$MANIFEST" ]]; then
  echo "missing MANIFEST=$MANIFEST" >&2
  exit 3
fi

"$PYTHON_BIN" huth_lebel_pack_stories.py plan \
  --manifest "$MANIFEST" \
  --out-dir "$OUT_DIR/plan" \
  --pack-root /staging/s/suresh27/datasets/ds003020-highdata-packs \
  --label smoke

"$PYTHON_BIN" huth_lebel_pack_stories.py pack \
  --manifest "$MANIFEST" \
  --ds-root "$DATASET_ROOT" \
  --story "$STORY" \
  --out-dir "$PACK_DIR" \
  --compression auto \
  --overwrite \
  --sha256 \
  --result-json "$OUT_DIR/pack_result.json"

PACK_PATH="$(find "$PACK_DIR" -maxdepth 1 -type f -name "${STORY}.tar.*" | head -1)"
if [[ -z "$PACK_PATH" ]]; then
  echo "pack output missing for story=$STORY" >&2
  exit 4
fi

"$PYTHON_BIN" huth_lebel_pack_stories.py verify \
  --manifest "$MANIFEST" \
  --story "$STORY" \
  --pack "$PACK_PATH" \
  --extract-dir "$EXTRACT_DIR" \
  --result-json "$OUT_DIR/verify_result.json"

if [[ "$PACK_PATH" == *.tar.zst ]]; then
  tar --use-compress-program=zstd -tf "$PACK_PATH" > "$OUT_DIR/pack_members.txt"
else
  tar -tzf "$PACK_PATH" > "$OUT_DIR/pack_members.txt"
fi
du -sh "$PACK_PATH" "$EXTRACT_DIR" > "$OUT_DIR/pack_du.txt"

if [[ "$KEEP_PACK_OUTPUT" != "1" ]]; then
  rm -rf "$EXTRACT_DIR" "$PACK_PATH"
fi

find "$OUT_DIR" -maxdepth 4 -type f -printf '%s\t%p\n' | sort -n > "$OUT_DIR/file_inventory.tsv"
du -sh "$OUT_DIR" > "$OUT_DIR/output_du.txt"
echo "finished=$(date -Is)" | tee -a "$OUT_DIR/run_env.txt"
