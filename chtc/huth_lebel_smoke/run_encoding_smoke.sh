#!/usr/bin/env bash
set -euo pipefail

RESULT_NAME="${RESULT_NAME:-huth_encoding_smoke}"
OUT_BUNDLE="${OUT_BUNDLE:-${RESULT_NAME}_results.tgz}"
FEATURE_BUNDLE="${FEATURE_BUNDLE:-huth_extract_smoke_results.tgz}"
FEATURE_DIR="${FEATURE_DIR:-}"
SUBJECTS="${SUBJECTS:-UTS01}"
TRAIN_STORIES="${TRAIN_STORIES:-sweetaspie,againstthewind}"
TEST_STORY="${TEST_STORY:-wheretheressmoke}"
ARMS="${ARMS:-base,lowLR,scrambled,taskvec_a0p25}"
LAYERS="${LAYERS:-16,24,32}"
MAX_VOXELS="${MAX_VOXELS:-2000}"
RIDGE_SOLVER="${RIDGE_SOLVER:-auto}"
RESULT_DIR="${PWD}/${RESULT_NAME}"
UNPACK_DIR="${PWD}/feature_bundle"
PYDEPS="${PWD}/pydeps"

mkdir -p "${RESULT_DIR}" "${UNPACK_DIR}" "${PYDEPS}"

finish() {
  status=$?
  echo "${status}" > "${RESULT_DIR}/exit_status.txt"
  tar -czf "${OUT_BUNDLE}" -C "${RESULT_DIR}" . 2>/dev/null || true
  exit "${status}"
}
trap finish EXIT

PYTHON_BIN="$(command -v python3 || command -v python)"
export PYTHONPATH="${PWD}/src:${PYDEPS}:${PYTHONPATH:-}"

{
  echo "timestamp_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "hostname=$(hostname)"
  echo "pwd=${PWD}"
  echo "python=${PYTHON_BIN}"
  echo "RESULT_NAME=${RESULT_NAME}"
  echo "FEATURE_BUNDLE=${FEATURE_BUNDLE}"
  echo "FEATURE_DIR=${FEATURE_DIR}"
  echo "SUBJECTS=${SUBJECTS}"
  echo "TRAIN_STORIES=${TRAIN_STORIES}"
  echo "TEST_STORY=${TEST_STORY}"
  echo "ARMS=${ARMS}"
  echo "LAYERS=${LAYERS}"
  echo "MAX_VOXELS=${MAX_VOXELS}"
  echo "RIDGE_SOLVER=${RIDGE_SOLVER}"
} > "${RESULT_DIR}/run_env.txt"

"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/imports_before_install.txt" 2>&1 || true
import importlib
for name in ("h5py", "numpy"):
    try:
        mod = importlib.import_module(name)
        print(f"{name}={getattr(mod, '__version__', 'unknown')}")
    except Exception as exc:
        print(f"{name}=IMPORT_ERROR:{type(exc).__name__}:{exc}")
PY

missing="$("${PYTHON_BIN}" - <<'PY'
import importlib.util
mods = []
for name in ("h5py", "numpy"):
    if importlib.util.find_spec(name) is None:
        mods.append(name)
print(" ".join(mods))
PY
)"

if [[ -n "${missing}" ]]; then
  "${PYTHON_BIN}" -m pip install --target "${PYDEPS}" ${missing} > "${RESULT_DIR}/pip_install.txt" 2>&1
else
  echo "all required modules already import" > "${RESULT_DIR}/pip_install.txt"
fi

"${PYTHON_BIN}" - <<'PY' > "${RESULT_DIR}/imports_after_install.txt" 2>&1
import importlib
for name in ("h5py", "numpy"):
    mod = importlib.import_module(name)
    print(f"{name}={getattr(mod, '__version__', 'unknown')}")
PY

if [[ ! -e /staging/s/suresh27/datasets/ds003020-smoke ]]; then
  echo "missing staged ds003020 smoke dataset" >&2
  exit 67
fi

if [[ -z "${FEATURE_DIR}" ]]; then
  if [[ ! -f "${FEATURE_BUNDLE}" ]]; then
    echo "missing feature bundle: ${FEATURE_BUNDLE}" >&2
    exit 66
  fi
  tar -xzf "${FEATURE_BUNDLE}" -C "${UNPACK_DIR}"
  FEATURE_DIR="${UNPACK_DIR}/features"
fi
if [[ ! -d "${FEATURE_DIR}" ]]; then
  echo "missing feature directory: ${FEATURE_DIR}" >&2
  exit 68
fi

SUBJECTS_FOR_ARGS="${SUBJECTS//,/ }"
read -r -a SUBJECT_ARGS <<< "${SUBJECTS_FOR_ARGS}"
CMD=(
  "${PYTHON_BIN}" src/sft/huth_lebel_smoke_encoding.py
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke
  --features_dir "${FEATURE_DIR}"
  --out_dir "${RESULT_DIR}/encoding"
  --subjects "${SUBJECT_ARGS[@]}"
  --train_stories "${TRAIN_STORIES}"
  --test_story "${TEST_STORY}"
  --arms "${ARMS}"
  --layers "${LAYERS}"
  --ridge_solver "${RIDGE_SOLVER}"
  --overwrite
)

if [[ -n "${MAX_VOXELS}" && "${MAX_VOXELS}" != "0" ]]; then
  CMD+=(--max_voxels "${MAX_VOXELS}")
fi

printf '%q ' "${CMD[@]}" > "${RESULT_DIR}/command.txt"
printf '\n' >> "${RESULT_DIR}/command.txt"

set +e
"${CMD[@]}" > "${RESULT_DIR}/encoding_stdout.txt" 2> "${RESULT_DIR}/encoding_stderr.txt"
rc=$?
set -e
echo "${rc}" > "${RESULT_DIR}/encoding_exit_status.txt"

if [[ -f "${RESULT_DIR}/encoding/summary.csv" ]]; then
  cp "${RESULT_DIR}/encoding/summary.csv" "${RESULT_DIR}/summary.csv"
fi
if [[ -f "${RESULT_DIR}/encoding/alpha_cv.csv" ]]; then
  cp "${RESULT_DIR}/encoding/alpha_cv.csv" "${RESULT_DIR}/alpha_cv.csv"
fi

find "${RESULT_DIR}" -maxdepth 5 -type f -printf '%P\t%s\n' > "${RESULT_DIR}/file_inventory.tsv" || true
exit "${rc}"
