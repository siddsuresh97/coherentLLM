#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT="${DATASET_ROOT:-/staging/s/suresh27/datasets/ds003020-smoke}"
ANNEX_JOBS="${ANNEX_JOBS:-4}"
REPO_URL="${REPO_URL:-https://github.com/OpenNeuroDatasets/ds003020.git}"
SNAPSHOT_TAG="${SNAPSHOT_TAG:-3.1.1}"
MANIFEST="${MANIFEST:-staging_manifest_smoke.csv}"
OUT_DIR="results/sft_huth_lebel_stage_smoke"

mkdir -p "$OUT_DIR"

{
  echo "started=$(date -Is)"
  echo "hostname=$(hostname -f || hostname)"
  echo "pwd=$PWD"
  echo "dataset_root=$DATASET_ROOT"
  echo "repo_url=$REPO_URL"
  echo "snapshot_tag=$SNAPSHOT_TAG"
  echo "annex_jobs=$ANNEX_JOBS"
  echo "manifest=$MANIFEST"
} | tee "$OUT_DIR/stage_env.txt"

echo "Tool versions:"
git --version | tee -a "$OUT_DIR/stage_env.txt"
git annex version | tee -a "$OUT_DIR/stage_env.txt"
datalad --version | tee -a "$OUT_DIR/stage_env.txt" || true

if [[ ! -f "$MANIFEST" ]]; then
  echo "missing manifest: $MANIFEST" >&2
  exit 2
fi

mkdir -p "$(dirname "$DATASET_ROOT")"
if [[ ! -d "$DATASET_ROOT/.git" ]]; then
  git clone "$REPO_URL" "$DATASET_ROOT"
fi

cd "$DATASET_ROOT"
git fetch --tags origin
git fetch origin git-annex:git-annex || true
git checkout "$SNAPSHOT_TAG"
git annex init "chtc-ds003020-smoke"

echo "Annex info before get:"
git annex info | tee "$OLDPWD/$OUT_DIR/git_annex_info_before.txt"

cd "$OLDPWD"
tail -n +2 "$MANIFEST" | cut -d, -f5 > "$OUT_DIR/manifest_paths.txt"
mapfile -t RELPATHS < "$OUT_DIR/manifest_paths.txt"

cd "$DATASET_ROOT"
echo "Manifest paths:"
printf '%s\n' "${RELPATHS[@]}" | tee "$OLDPWD/$OUT_DIR/manifest_paths_resolved.txt"

echo "Whereis sample before get:"
git annex whereis -- "${RELPATHS[@]:0:3}" | tee "$OLDPWD/$OUT_DIR/git_annex_whereis_sample_before.txt" || true

git annex get --jobs="$ANNEX_JOBS" -- "${RELPATHS[@]}"

echo "Annex info after get:"
git annex info | tee "$OLDPWD/$OUT_DIR/git_annex_info_after.txt"

cd "$OLDPWD"
python huth_lebel_audit.py \
  --roots "$DATASET_ROOT" \
  --out_dir "$OUT_DIR/audit" \
  --max_scan_depth 2

python - "$DATASET_ROOT" "$MANIFEST" "$OUT_DIR/stage_summary.json" <<'PY'
import csv
import json
import os
import sys
from pathlib import Path

root = Path(sys.argv[1])
manifest = Path(sys.argv[2])
out = Path(sys.argv[3])
rows = []
total_expected = 0
total_actual = 0
missing = []
with manifest.open() as f:
    for row in csv.DictReader(f):
        path = root / row["relative_path"]
        expected = int(row["size_bytes"])
        actual = path.stat().st_size if path.exists() else 0
        item = {
            "relative_path": row["relative_path"],
            "kind": row["kind"],
            "subject": row["subject"],
            "story": row["story"],
            "expected_bytes": expected,
            "actual_bytes": actual,
            "exists": path.exists(),
            "is_symlink": path.is_symlink(),
        }
        rows.append(item)
        total_expected += expected
        total_actual += actual
        if not path.exists():
            missing.append(row["relative_path"])
summary = {
    "dataset_root": str(root),
    "n_files": len(rows),
    "n_missing": len(missing),
    "missing": missing,
    "expected_bytes": total_expected,
    "actual_bytes": total_actual,
    "expected_gb": total_expected / 1_000_000_000,
    "actual_gb": total_actual / 1_000_000_000,
    "files": rows,
}
out.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps({k: summary[k] for k in ["n_files", "n_missing", "expected_gb", "actual_gb"]}, indent=2))
if missing:
    raise SystemExit(3)
PY

du -sh "$DATASET_ROOT" | tee "$OUT_DIR/dataset_du.txt"
tar -czf huth_lebel_stage_smoke_results.tgz "$OUT_DIR"
echo "finished=$(date -Is)"
