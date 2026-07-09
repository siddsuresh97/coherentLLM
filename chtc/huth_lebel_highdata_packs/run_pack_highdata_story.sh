#!/usr/bin/env bash
set -euo pipefail

STORY="${1:-${STORY:-}}"
if [[ -z "$STORY" ]]; then
  echo "usage: run_pack_highdata_story.sh <story>" >&2
  exit 2
fi

REPO_URL="${REPO_URL:-https://github.com/OpenNeuroDatasets/ds003020.git}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-3.1.1}"
ANNEX_JOBS="${ANNEX_JOBS:-4}"
MANIFEST="${MANIFEST:-staging_manifest_highdata.csv}"
PACK_ROOT="${PACK_ROOT:-/staging/s/suresh27/datasets/ds003020-highdata-packs}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRATCH_ROOT="$PWD"
DS_ROOT="$SCRATCH_ROOT/ds003020-$STORY"
OUT_DIR="$SCRATCH_ROOT/results/huth_lebel_pack_highdata_story_${STORY}"
LOG_DIR="$OUT_DIR/logs"
TARBALL="$SCRATCH_ROOT/huth_lebel_pack_highdata_story_${STORY}_results.tgz"

mkdir -p "$OUT_DIR" "$LOG_DIR" "$PACK_ROOT"
trap 'status=$?; echo "$status" > "$OUT_DIR/exit_status.txt"; tar -czf "$TARBALL" -C "$SCRATCH_ROOT" results || true; exit "$status"' EXIT

{
  echo "started=$(date -Is)"
  echo "hostname=$(hostname -f || hostname)"
  echo "pwd=$PWD"
  echo "story=$STORY"
  echo "repo_url=$REPO_URL"
  echo "snapshot_tag=$SNAPSHOT_TAG"
  echo "annex_jobs=$ANNEX_JOBS"
  echo "manifest=$MANIFEST"
  echo "pack_root=$PACK_ROOT"
  printf "zstd_path="
  command -v zstd || true
  git --version || true
  git annex version || true
  datalad --version || true
} | tee "$OUT_DIR/run_env.txt"

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing MANIFEST=$MANIFEST" >&2
  exit 3
fi

"$PYTHON_BIN" - "$MANIFEST" "$STORY" "$OUT_DIR/story_paths.txt" <<'PY'
import csv
import sys

manifest, story, out_path = sys.argv[1:]
paths = ["dataset_description.json"]
with open(manifest, newline="") as f:
    for row in csv.DictReader(f):
        if row["story"] == story:
            paths.append(row["relative_path"])
if len(paths) == 1:
    raise SystemExit(f"story not present in manifest: {story}")
with open(out_path, "w") as f:
    f.write("\n".join(paths) + "\n")
PY

git clone --no-checkout "$REPO_URL" "$DS_ROOT"
cd "$DS_ROOT"
git fetch --tags origin
git fetch origin git-annex:git-annex || true
git sparse-checkout init --no-cone
git sparse-checkout set --stdin < "$OUT_DIR/story_paths.txt"
git checkout "$SNAPSHOT_TAG"
git config user.email "codex@chtc.local"
git config user.name "Codex CHTC"
git annex init "chtc-ds003020-pack-$STORY"
tail -n +2 "$OUT_DIR/story_paths.txt" > "$OUT_DIR/annex_paths.txt"
git annex get --jobs="$ANNEX_JOBS" -- $(cat "$OUT_DIR/annex_paths.txt")

cd "$SCRATCH_ROOT"
"$PYTHON_BIN" huth_lebel_pack_stories.py pack \
  --manifest "$MANIFEST" \
  --ds-root "$DS_ROOT" \
  --story "$STORY" \
  --out-dir "$PACK_ROOT" \
  --compression auto \
  --overwrite \
  --result-json "$OUT_DIR/pack_result.json"

PACK_PATH="$("$PYTHON_BIN" - "$OUT_DIR/pack_result.json" <<'PY'
import json, sys
print(json.load(open(sys.argv[1]))["archive"])
PY
)"
"$PYTHON_BIN" huth_lebel_pack_stories.py verify \
  --manifest "$MANIFEST" \
  --story "$STORY" \
  --pack "$PACK_PATH" \
  --result-json "$OUT_DIR/verify_result.json"

find "$OUT_DIR" -maxdepth 4 -type f -printf '%s\t%p\n' | sort -n > "$OUT_DIR/file_inventory.tsv"
echo "finished=$(date -Is)" | tee -a "$OUT_DIR/run_env.txt"
