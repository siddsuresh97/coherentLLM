# Experiment 1 Report - Triplet Detection of Concept Moves

Current status: the CPU-reproducible baseline scaffold and simulated detection smoke test are in place; the behavioral gate is still red because fresh 30-item canonical/paraphrase model runs have not been collected.

## Pipeline State

- 0a feature-space base RDM: green
- 0b behavioral triplet base RDM: red
- 1 item selection: green
- 2 edit operator pre-check: draft only
- 3 two-LoRA training: not started
- 4 detection/localization: not started
- 5 resolution map: simulated smoke green; real map not started

## Item Choice

Target: `antelope`. Neighbor for concentrated move: `bison`.

Rationale: Chose antelope because it sits in a tight animal cluster while still having far items available for contrast. The concentrated move is defined against bison, its nearest selected neighbor in the NOVA feature-space map.

Artifacts:
- Runbook: `experiments/exp1_triplet_concept_move/RUNBOOK.md`
- CHTC pathway check: `chtc/exp1_triplet_move/README.md`
- CHTC input bundle: `chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz`
- Items: `experiments/exp1_triplet_concept_move/items.json`
- Feature RDM: `experiments/exp1_triplet_concept_move/artifacts/rdms/feature_rdm_base.npy`
- Provisional behavioral RDM: `experiments/exp1_triplet_concept_move/artifacts/rdms/rdm_base.npy`
- Protocol: `experiments/exp1_triplet_concept_move/triplet_protocol.json`
- Triplet runner: `scripts/run_experiment1_triplets.py`
- Real GPU runner: `scripts/run_experiment1_real_gpu.sh`
- Fallback pairwise SFT builder: `scripts/build_experiment1_similarity_sft_data.py`
- Floor stats: `experiments/exp1_triplet_concept_move/floor_stats.json`
- Feature MDS: `experiments/exp1_triplet_concept_move/figs/feature_mds.png`
- Feature dendrogram: `experiments/exp1_triplet_concept_move/figs/feature_dendrogram.png`
- Simulated heatmap: `experiments/exp1_triplet_concept_move/simulation/resolution_heatmap_simulated.png`

## Perturbations Tried

| Edit | Intended move (row L2 RMS) | Locality (Gini) | Did it fire? | SNR |
|---|---:|---:|---|---:|
| `concentrated_drop_015` | 0.0179 | 0.324 | not trained | n/a |
| `concentrated_drop_035` | 0.0466 | 0.369 | not trained | n/a |
| `concentrated_drop_065` | 0.1103 | 0.451 | not trained | n/a |
| `concentrated_drop_100` | 0.2623 | 0.390 | not trained | n/a |
| `diffuse_drop_015` | 0.0558 | 0.241 | not trained | n/a |
| `diffuse_drop_035` | 0.1137 | 0.256 | not trained | n/a |
| `diffuse_drop_065` | 0.2093 | 0.328 | not trained | n/a |
| `diffuse_drop_100` | 0.3791 | 0.281 | not trained | n/a |

## Current Course-Correction Reasoning

The archived 128-item Llama triplet embedding is useful as a provisional behavioral map, but it does not satisfy the hard gate. The next action is to run the frozen 30-item protocol for the base model at least twice with the canonical prompt and once with the paraphrase prompt, then replace the provisional floor with the real split/paraphrase reliability estimate. Training any LoRA before that would contaminate the experiment because detection would have no accepted noise denominator.

Exact next runner pattern:

```bash
python scripts/run_experiment1_triplets.py --model llama-3.1-8b-instruct --out-run base_seed_a_canonical_prompt --prompt-variant canonical --overwrite
python scripts/run_experiment1_triplets.py --model llama-3.1-8b-instruct --out-run base_seed_b_canonical_prompt --prompt-variant canonical --overwrite
python scripts/run_experiment1_triplets.py --model llama-3.1-8b-instruct --out-run base_seed_a_paraphrase_prompt --prompt-variant paraphrase --overwrite
python scripts/run_experiment1.py --allow-provisional-edits
```

Full GPU path once a GPU host is available:

```bash
make experiment1-real-gpu
```

CHTC pathway-check path is prepared for the first real run:

```bash
chtc/exp1_triplet_move/submit_exp1_pathway.sh
```

If feature-listing supervision does not move the behavioral triplet RDM, the prepared fallback path is:

```bash
SUPERVISION=similarity make experiment1-real-gpu
```

## Current Findings

- Feature-space map is established over all 128 held-out concepts and the selected 30-item subset.
- `antelope` has a clear local feature-space neighbor (`bison`) plus a broader animal cluster, so a concentrated push has a concrete target pair.
- The frozen triplet protocol has 12180 judgments per run and hashes the exact stimuli files.
- Behavioral base is provisional: archived_triplet_embedding from `data/scale128/llama-3.1-8b-instruct_triplet_d5.npy`.
- Simulated oracle smoke test recovers injected moves through the triplet pipeline: concentrated edits cross median SNR > 1 at `concentrated_drop_065` (median SNR 1.263; target top-ranked in 4/5 seeds) and are clean at `concentrated_drop_100` (median SNR 2.075; target top-ranked in 5/5 seeds). This validates the scorer/heatmap mechanics only, not the model-edit claim.
- Fallback lever 6.2 is staged: `sft_similarity_data/concentrated_drop_100/manifest.json` dry-runs direct pairwise-similarity supervision with 4,872 control examples and 5,220 edit examples, while keeping detection held out as triplets.

## Live Risks

- Missing true behavioral floor: collect required runs listed in `triplet_protocol.json`.
- No local GPU in this environment: LoRA training and vLLM triplet recovery need a GPU host or CHTC submission.
- CHTC authentication is not active for this process: escalated `chtc-ssh 'hostname -f'` reached SSH but failed keyboard-interactive auth. Start `chtc-master`, then run `chtc/exp1_triplet_move/submit_exp1_pathway.sh`.
- Candidate edits are feature-space drafts only: promote them to `edits/` only after the behavioral gate is green.
