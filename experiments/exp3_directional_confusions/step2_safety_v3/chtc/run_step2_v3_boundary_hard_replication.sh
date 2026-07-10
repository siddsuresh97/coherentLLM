#!/usr/bin/env bash
set -euo pipefail

RUN_NAME="${RUN_NAME:-step2_v3_policy_routing_boundary_hard_replication_v1}"
ITEM_SET="${ITEM_SET:-boundary_hard_replication}"
MODEL_PATH="${MODEL_PATH:-/staging/s/suresh27/models/llama31-8b-instruct}"

echo "started_at=$(date -Is)"
echo "host=$(hostname -f)"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES:-unset}"
echo "model_path=${MODEL_PATH}"

export HF_HOME="$PWD/hf_cache"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCH_HOME="$PWD/torch_cache"
export TRITON_CACHE_DIR="$PWD/triton_cache"
export TMPDIR="$PWD/tmp"
export PYTHONUNBUFFERED=1
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$TORCH_HOME" "$TRITON_CACHE_DIR" "$TMPDIR"
SCRATCH_ROOT="$PWD"

python3 - <<'PY'
import json
import os
try:
    import torch
    cuda = torch.cuda.is_available()
    n_gpu = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(n_gpu)] if cuda else []
except Exception as exc:
    cuda = False
    n_gpu = 0
    names = [f"torch_probe_failed: {exc}"]
print(json.dumps({
    "cuda_available": cuda,
    "device_count": n_gpu,
    "gpu_names": names,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
}, indent=2))
PY

tar -xzf exp3_v3_bundle.tgz
if [ -d coherence_experiments ]; then
  cd coherence_experiments
fi

python3 scripts/run_exp3_safety_v3.py run-items \
  --model llama-3.1-8b-instruct \
  --model-path "$MODEL_PATH" \
  --out-run "$RUN_NAME" \
  --item-set "$ITEM_SET" \
  --overwrite \
  --max_model_len 256 \
  --max_num_seqs 4 \
  --gpu_mem_util 0.82 \
  --max-output-tokens 4

python3 scripts/run_exp3_safety_v3.py score-items --run "$RUN_NAME" --item-set "$ITEM_SET"
python3 scripts/run_experiment3.py update-report

tar -czf "${SCRATCH_ROOT}/${RUN_NAME}_outputs.tgz" \
  experiments/exp3_directional_confusions/REPORT.md \
  experiments/exp3_directional_confusions/RESEARCH_LOG.md \
  experiments/exp3_directional_confusions/step2_safety_v3/items/"$ITEM_SET".csv \
  experiments/exp3_directional_confusions/step2_safety_v3/items/"$ITEM_SET".json \
  experiments/exp3_directional_confusions/step2_safety_v3/raw/"$RUN_NAME" \
  experiments/exp3_directional_confusions/step2_safety_v3/results/step2_v3_policy_routing_"$ITEM_SET".json \
  experiments/exp3_directional_confusions/step2_safety_v3/results/step2_v3_policy_routing_"$ITEM_SET"_scored_items.csv \
  experiments/exp3_directional_confusions/step2_safety_v3/results/step2_v3_policy_routing_"$ITEM_SET"_pair_rates.csv \
  experiments/exp3_directional_confusions/step2_safety_v3/results/step2_v3_policy_routing_"$ITEM_SET"_confusion_matrix.csv

cat > "${SCRATCH_ROOT}/${RUN_NAME}_metadata.json" <<EOF
{
  "run_name": "${RUN_NAME}",
  "item_set": "${ITEM_SET}",
  "model_path": "${MODEL_PATH}",
  "finished_at": "$(date -Is)"
}
EOF

echo "finished_at=$(date -Is)"
