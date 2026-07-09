# Experiment 1 Runbook

This runbook separates the local CPU scaffold from the real GPU path.

## CPU Scaffold And Smoke

Regenerate baseline artifacts, item selection, draft edit specs, and the living
report:

```bash
make experiment1
```

Run the oracle simulation smoke test:

```bash
make experiment1-smoke
```

The simulation validates triplet aggregation, scoring, and heatmap code. It does
not satisfy the model-edit criterion.

## Real GPU Path

On a GPU host with the project conda env and cached Llama weights:

```bash
make experiment1-real-gpu
```

To run only the large pathway check first:

```bash
scripts/run_experiment1_real_gpu.sh concentrated_drop_100
```

To use fallback direct pairwise-similarity supervision instead of feature-list
supervision:

```bash
SUPERVISION=similarity scripts/run_experiment1_real_gpu.sh concentrated_drop_100
```

Useful environment overrides:

```bash
MODEL=llama-3.1-8b-instruct
BASE_MODEL_PATH=/staging/s/suresh27/models/llama31-8b-instruct
TRAIN_BASE_MODEL=/staging/s/suresh27/models/llama31-8b-instruct
HF_HOME=/staging/s/suresh27/hf_home
GPU_MEM_UTIL=0.88
MAX_MODEL_LEN=2048
MAX_NUM_SEQS=256
TRAIN_STEPS=400
LORA_RANK=32
SUPERVISION=feature
```

## Order Enforced By `run_experiment1_real_gpu.sh`

1. Run the frozen base protocol three times:
   `base_seed_a_canonical_prompt`, `base_seed_b_canonical_prompt`,
   `base_seed_a_paraphrase_prompt`.
2. Rerun `scripts/run_experiment1.py --allow-provisional-edits`.
3. Stop if `floor_stats.json` is not green.
4. For each edit cell, build control/edit SFT data, train the two LoRAs, run
   frozen triplet recovery for control/edit, score detection, and summarize the
   heatmap.

## Expected Real Outputs

- `experiments/exp1_triplet_concept_move/raw/<run>/triplet.csv`
- `experiments/exp1_triplet_concept_move/rdms/<run>/rdm.npy`
- `experiments/exp1_triplet_concept_move/detection/<edit_id>.json`
- `experiments/exp1_triplet_concept_move/figs/resolution_heatmap.csv`
- `experiments/exp1_triplet_concept_move/figs/resolution_heatmap.png`

## CHTC Pathway Check

The CHTC-ready first-run package lives in:

```text
chtc/exp1_triplet_move/
```

Prepare the transfer bundle locally:

```bash
chtc/exp1_triplet_move/prepare_exp1_bundle.sh
```

Then follow `chtc/exp1_triplet_move/README.md` to push and submit
`exp1_pathway_check.sub`. The default CHTC job runs only
`concentrated_drop_100` with feature-listing supervision; use the fallback
submit override in that README for `SUPERVISION = similarity`.

After `chtc-master start`, the shortcut is:

```bash
chtc/exp1_triplet_move/submit_exp1_pathway.sh
```
