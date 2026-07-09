# Experiment 3 Report

Last updated: 2026-07-09T13:55:38-05:00

## Current status

- Branch/worktree experiment folder: `experiments/exp3_directional_confusions`
- Step 1 concept set: `experiments/exp3_directional_confusions/concepts/step1_neutral.json`
- Triplet protocol frozen: yes
- Required triplet runs present: 0/3
- RDM reliability gate: `missing`
- Neighbors pre-registered: no
- Human sanity gate: `not_started`
- Directional items generated: no
- Item response runs present: none
- H1 verdict: `not_decided`

## Commands

```bash
python scripts/run_experiment3.py init
python scripts/run_experiment3.py run-triplets --out-run base_seed_a_canonical_prompt --prompt-variant canonical --overwrite
python scripts/run_experiment3.py run-triplets --out-run base_seed_b_canonical_prompt --prompt-variant canonical --overwrite
python scripts/run_experiment3.py run-triplets --out-run base_seed_a_paraphrase_prompt --prompt-variant paraphrase --overwrite
python scripts/run_experiment3.py build-rdm
python scripts/run_experiment3.py register-neighbors
python scripts/run_experiment3.py generate-items
python scripts/run_experiment3.py mark-sanity-gate --status pass --note "nearest-neighbor pairs are human-sane"
python scripts/run_experiment3.py run-items --out-run step1_items_v1 --overwrite
python scripts/run_experiment3.py score --run step1_items_v1
```

## Pre-registered Predictions

No pre-registered neighbors yet.

## Live risks

- If the model is near-perfect on these items, H1 is untestable and the item phrasing needs to move into a harder uncertainty band.
- If RDM reliability is red, do not register or interpret neighbors except as an engineering smoke test.
- Step 2 is intentionally absent until neutral H1 is green and the sanity gate passes.
