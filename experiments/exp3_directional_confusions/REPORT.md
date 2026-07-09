# Experiment 3 Report

Last updated: 2026-07-09T15:22:17-05:00

## Plain-English Result

Step 1 found a positive result for H1 on neutral concepts.

The model's triplet geometry predicted the direction of its later multiple-choice mistakes. When the model was wrong, it usually chose the distractor that had been preregistered as the target's nearest neighbor in the triplet RDM.

Main result:

- Model made `45` parseable errors on `72` neutral concept items.
- `36/45` errors (`80.0%`) landed on the geometry-predicted near neighbor.
- A shuffled-neighbor null expected about `33.3%`; its 95th percentile was `44.4%`.
- Monte Carlo shuffle p-value: `0.0002`.
- Base-rate control expected `45.4%`; observed was `80.0%`, so lift was `+34.6` percentage points.
- Quantitative H2 slope was negative: `-0.620`, 95% CI `[-0.877, -0.359]`.
- Predicted-vs-actual confusion agreement was `r = 0.634`.

Interpretation: this was not just "the model made errors." The errors had the predicted destination. In this neutral set, representational proximity predicted which wrong answer the model moved toward.

## What I Ran

Model:

- `llama-3.1-8b-instruct`
- Local weights: `meta-llama/Llama-3.1-8B-Instruct`
- Served with `vLLM` from the H100 environment.
- Actual H100 path used: `/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`

Concept set:

- 18 neutral Leuven concrete concepts: reptiles/amphibians plus tools.
- Examples: `alligator`, `crocodile`, `snake`, `cobra`, `tortoise`, `turtle`, `axe`, `saw`.
- Step 2 safety concepts were not started.

Geometry prompt, canonical:

```text
System: You are a helpful assistant who gives responses to questions.

Target concept: {anchor}
Candidate A: {concept1}
Candidate B: {concept2}
Which candidate is more similar in semantic meaning to the target? Answer with exactly A or B.
```

Geometry prompt, paraphrase reliability run:

```text
System: You are a helpful assistant who gives responses to questions.

Compare the target to two candidates.
Target: {anchor}
A: {concept1}
B: {concept2}
Which candidate is closer in meaning to the target? Reply with only A or B.
```

Geometry run details:

- Temperature `0.0`.
- Full anchor/candidate enumeration: `2448` triplets per run.
- Required runs: two canonical runs plus one paraphrase run.
- Initial loose concept-name prompt failed reliability because the model sometimes answered a third unlisted concept. I fixed this by switching to labeled A/B prompts before registering neighbors.

Item prompt format:

```text
Which option is the best match for this description?
- {feature clue}
- {feature clue}
- {feature clue}
Answer with only A, B, C, or D.

Options:
A. {distractor_or_target}
B. {distractor_or_target}
C. {distractor_or_target}
D. {distractor_or_target}
```

Each item included:

- the correct target concept,
- one preregistered near distractor from the RDM,
- two far-control distractors,
- counterbalanced answer positions.

There were `72` items total: `4` per target.

## What Changed In The Repo

- Created branch `exp3-directional-confusions`.
- Added `scripts/run_experiment3.py` with the Step 1 pipeline.
- Added `experiments/exp3_directional_confusions/` with concept set, protocol, raw responses, RDMs, preregistered neighbors, generated items, scored results, report, and research log.
- Added CHTC runner files under `chtc/exp3_directional_confusions/`.
- Added a single-load `run-triplet-suite` command so all geometry runs can reuse one vLLM load.
- Pushed the experiment branch to origin.

## Current status

- Branch/worktree experiment folder: `experiments/exp3_directional_confusions`
- Step 1 concept set: `experiments/exp3_directional_confusions/concepts/step1_neutral.json`
- Triplet protocol frozen: yes
- Triplet response format: `labeled_binary_choice_A_or_B`
- Required triplet runs present: 3/3
- RDM reliability gate: `green`
- Neighbors pre-registered: yes
- Human sanity gate: `passed`
- Directional items generated: yes
- Item response runs present: step1_items_v1
- H1 verdict: `green_directional`

## Commands

```bash
python scripts/run_experiment3.py init
python scripts/run_experiment3.py run-triplet-suite --overwrite
python scripts/run_experiment3.py build-rdm
python scripts/run_experiment3.py register-neighbors
python scripts/run_experiment3.py generate-items
python scripts/run_experiment3.py mark-sanity-gate --status pass --note "nearest-neighbor pairs are human-sane"
python scripts/run_experiment3.py run-items --out-run step1_items_v1 --overwrite
python scripts/run_experiment3.py score --run step1_items_v1
```

## Pre-registered Predictions

| Target | Predicted near | Near d | Far controls |
|---|---|---:|---|
| `alligator` | `crocodile` | 0.000 | `blindworm` (0.583), `salamander` (0.552) |
| `caiman` | `crocodile` | 0.031 | `chisel` (0.521), `tortoise` (0.510) |
| `crocodile` | `alligator` | 0.000 | `axe` (0.542), `saw` (0.562) |
| `boa python` | `snake` | 0.000 | `chisel` (0.667), `toad` (0.635) |
| `cobra` | `snake` | 0.031 | `tortoise` (0.656), `salamander` (0.646) |
| `snake` | `boa python` | 0.000 | `toad` (0.562), `chisel` (0.562) |
| `blindworm` | `salamander` | 0.094 | `chisel` (0.656), `alligator` (0.583) |
| `chameleon` | `gecko` | 0.062 | `chisel` (0.656), `tortoise` (0.594) |
| `gecko` | `lizard` | 0.042 | `chisel` (0.729), `saw` (0.750) |
| `lizard` | `gecko` | 0.042 | `hammer` (0.667), `axe` (0.677) |
| `salamander` | `blindworm` | 0.094 | `saw` (0.771), `chisel` (0.781) |
| `toad` | `salamander` | 0.125 | `saw` (0.792), `chisel` (0.812) |
| `tortoise` | `turtle` | 0.000 | `chisel` (0.802), `hammer` (0.844) |
| `turtle` | `tortoise` | 0.000 | `chisel` (0.729), `hammer` (0.750) |
| `axe` | `saw` | 0.042 | `gecko` (0.844), `turtle` (0.844) |
| `chisel` | `axe` | 0.344 | `gecko` (0.729), `turtle` (0.729) |
| `hammer` | `axe` | 0.083 | `blindworm` (0.802), `salamander` (0.802) |
| `saw` | `axe` | 0.042 | `toad` (0.792), `turtle` (0.792) |

## RDM Reliability

- Source runs: base_seed_a_canonical_prompt, base_seed_b_canonical_prompt, base_seed_a_paraphrase_prompt
- Missing runs: none
- Mean pairwise upper-triangle Pearson: `0.8307935259132994`
- Mean split-half upper-triangle Pearson: `0.8199290023048541`
- Gate: `green`

## Step 1 Directional Score

- Accuracy: `0.3750` (27/72)
- Directional errors: `45`
- Near fraction among directional errors: `0.8000`
- Shuffle null p-value: `0.0002`
- Base-rate lift: `0.3464`
- H2 distance slope: `-0.620137`
- H2 slope 95% CI: `[-0.8769970123889884, -0.3586413218928585]`
- Predicted-vs-actual confusion agreement: `0.6342`

Headline figure: `experiments/exp3_directional_confusions/figs/step1_confusion_matrix.png`

## Live risks

- If the model is near-perfect on these items, H1 is untestable and the item phrasing needs to move into a harder uncertainty band.
- If RDM reliability is red, do not register or interpret neighbors except as an engineering smoke test.
- Step 2 is intentionally absent until neutral H1 is green and the sanity gate passes.
