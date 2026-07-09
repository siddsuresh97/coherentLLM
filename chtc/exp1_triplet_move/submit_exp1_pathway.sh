#!/usr/bin/env bash
# Build, push, and submit the Exp 1 pathway-check job via the user's CHTC wrappers.
set -euo pipefail

cd "$(dirname "$0")/../.."

REMOTE_DIR="${REMOTE_DIR:-chtc-runs/exp1-triplet-move-pathway}"
EDIT_ID="${EDIT_ID:-concentrated_drop_100}"
SUPERVISION="${SUPERVISION:-feature}"

for tool in chtc-ssh chtc-push; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Missing $tool. Start/verify the CHTC SSH master first; see chtc/exp1_triplet_move/README.md." >&2
    exit 127
  fi
done

chtc/exp1_triplet_move/prepare_exp1_bundle.sh

chtc-ssh "mkdir -p ~/${REMOTE_DIR}/logs"
chtc-push chtc/exp1_triplet_move/run_exp1_pathway_check.sh "${REMOTE_DIR}/"
chtc-push chtc/exp1_triplet_move/exp1_pathway_check.sub "${REMOTE_DIR}/"
chtc-push chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz "${REMOTE_DIR}/"

append_args=()
if [[ "${EDIT_ID}" != "concentrated_drop_100" ]]; then
  append_args+=("-append 'EDIT_ID = ${EDIT_ID}'")
fi
if [[ "${SUPERVISION}" != "feature" ]]; then
  append_args+=("-append 'SUPERVISION = ${SUPERVISION}'")
fi

if [[ "${#append_args[@]}" -gt 0 ]]; then
  append_text="${append_args[*]}"
  chtc-ssh "cd ~/${REMOTE_DIR} && condor_submit ${append_text} exp1_pathway_check.sub"
else
  chtc-ssh "cd ~/${REMOTE_DIR} && condor_submit exp1_pathway_check.sub"
fi
