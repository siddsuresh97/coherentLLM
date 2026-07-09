# Experiment 3 Report

Last updated: 2026-07-09T16:03:40-05:00

## Step 1 Story

### What were we trying to find?

We are testing whether triplet geometry predicts the destination of model errors on neutral concepts. The preregistered prediction for each target is its nearest neighbor in the model's triplet RDM; H1 is green only if later errors land on that near neighbor above shuffled-geometry and base-rate nulls.

### What did we run?

- Model: `llama-3.1-8b-instruct`.
- Serving: local vLLM; triplet and item prompts use temperature `0.0`.
- Concept set: 18 neutral Leuven concrete concepts in [experiments/exp3_directional_confusions/concepts/step1_neutral.json](experiments/exp3_directional_confusions/concepts/step1_neutral.json).
- Stimuli: [experiments/exp3_directional_confusions/stimuli/concepts.csv](experiments/exp3_directional_confusions/stimuli/concepts.csv), [experiments/exp3_directional_confusions/stimuli/triplets.csv](experiments/exp3_directional_confusions/stimuli/triplets.csv), [experiments/exp3_directional_confusions/stimuli/pairs.csv](experiments/exp3_directional_confusions/stimuli/pairs.csv).
- Geometry raw responses: [base_seed_a_canonical_prompt](experiments/exp3_directional_confusions/raw/base_seed_a_canonical_prompt/triplet.csv), [base_seed_b_canonical_prompt](experiments/exp3_directional_confusions/raw/base_seed_b_canonical_prompt/triplet.csv), [base_seed_a_matched_paraphrase_prompt](experiments/exp3_directional_confusions/raw/base_seed_a_matched_paraphrase_prompt/triplet.csv).
- Geometry fitting: `salmon_embedding`; distance metric for `rdm.npy`: `cosine_distance`.
- RDM artifact: [experiments/exp3_directional_confusions/artifacts/rdm.npy](experiments/exp3_directional_confusions/artifacts/rdm.npy); metadata: [experiments/exp3_directional_confusions/artifacts/rdm_meta.json](experiments/exp3_directional_confusions/artifacts/rdm_meta.json).
- SALMON pooled embedding: [experiments/exp3_directional_confusions/artifacts/embeddings/pooled_salmon_d5.npy](experiments/exp3_directional_confusions/artifacts/embeddings/pooled_salmon_d5.npy).
- Pre-registered neighbors: [experiments/exp3_directional_confusions/neighbors.json](experiments/exp3_directional_confusions/neighbors.json).
- Directional items: [experiments/exp3_directional_confusions/items/step1/items.csv](experiments/exp3_directional_confusions/items/step1/items.csv) and [experiments/exp3_directional_confusions/items/step1/items.json](experiments/exp3_directional_confusions/items/step1/items.json).
- Item responses: [experiments/exp3_directional_confusions/raw/step1_items_v1/items.csv](experiments/exp3_directional_confusions/raw/step1_items_v1/items.csv).
- Scored outputs: [experiments/exp3_directional_confusions/results/step1.json](experiments/exp3_directional_confusions/results/step1.json), [experiments/exp3_directional_confusions/results/step1_scored_items.csv](experiments/exp3_directional_confusions/results/step1_scored_items.csv), [experiments/exp3_directional_confusions/results/step1_pair_rates.csv](experiments/exp3_directional_confusions/results/step1_pair_rates.csv), [experiments/exp3_directional_confusions/results/step1_confusion_matrix.csv](experiments/exp3_directional_confusions/results/step1_confusion_matrix.csv).

Canonical geometry prompt:

```text
System: You are a helpful assistant who gives responses to questions.

Target concept: {anchor}
Candidate A: {concept1}
Candidate B: {concept2}
Which candidate is more similar in semantic meaning to the target? Answer with exactly A or B.
```

Paraphrase geometry prompt:

```text
System: You are a helpful assistant who gives responses to questions.

Target concept: {anchor}
Candidate A: {concept1}
Candidate B: {concept2}
Which candidate has the closer semantic meaning to the target concept? Reply with exactly A or B.
```

Prompt-variant correction:

- Old non-matched paraphrase: `Compare the target to two candidates. / Target: {anchor} / A: {concept1} / B: {concept2} / Which candidate is closer in meaning to the target? Reply with only A or B.`
- Canonical vs old non-matched raw choice agreement: `1820/2448 = 0.7435`.
- Canonical vs matched paraphrase raw choice agreement: `2297/2448 = 0.9383`.

Directional item template:

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

Concrete generated item example:

From [experiments/exp3_directional_confusions/items/step1/items.csv](experiments/exp3_directional_confusions/items/step1/items.csv) / `step1_alligator_00`:

```text
Which option is the best match for this description?
- is a freshwater fish
- lives by the sea
- lives in a swamp
Answer with only A, B, C, or D.

Options:
A. cobra
B. turtle
C. chameleon
D. alligator
```

### What did we find?

- RDM source: `salmon_embedding`.
- RDM distance metric: `cosine_distance`.
- SALMON pooled held-out accuracy: `0.8590878248214722`
- SALMON per-run held-out accuracies: `{'base_seed_a_canonical_prompt': 0.8306122422218323, 'base_seed_a_matched_paraphrase_prompt': 0.8183673620223999, 'base_seed_b_canonical_prompt': 0.8571428656578064}`
- Source runs: base_seed_a_canonical_prompt, base_seed_b_canonical_prompt, base_seed_a_matched_paraphrase_prompt.
- Missing runs: none
- Mean pairwise upper-triangle Pearson: `0.870450357132683`
- Mean pairwise SALMON embedding Procrustes R^2: `0.8440765796776389`
- Mean nearest-neighbor top-1 agreement across geometry runs: `0.46296296296296297`
- Mean nearest-neighbor top-2 agreement across geometry runs: `0.6481481481481483`
- Mean split-half upper-triangle Pearson: `0.7120252173491437`
- RDM reliability gate: `green`

- Accuracy: `0.4861` (35/72)
- Directional errors: `37`
- Near fraction among directional errors: `0.7838`
- Shuffle null p-value: `0.0002`
- Base-rate lift: `0.3187`
- H2 distance slope: `-0.233021`
- H2 slope 95% CI: `[-0.3382161163791806, -0.12500199099784218]`
- Predicted-vs-actual confusion agreement: `0.5781`
- H1 verdict: `green_directional`

Headline figure: [experiments/exp3_directional_confusions/figs/step1_confusion_matrix.png](experiments/exp3_directional_confusions/figs/step1_confusion_matrix.png)

### What does this mean?

Step 1 is green: the neutral-model errors are directional under the current geometry. The model did not merely make mistakes; its mistakes preferentially landed on the preregistered nearest-neighbor distractor.

The current report is the SALMON-based version. The earlier direct choice-rate RDM result is superseded for the active Experiment 3 claim and remains only in git history.

## Current Status

- Branch/worktree experiment folder: `experiments/exp3_directional_confusions`
- Step 1 concept set: `experiments/exp3_directional_confusions/concepts/step1_neutral.json`
- Triplet protocol frozen: yes
- Triplet response format: `labeled_binary_choice_A_or_B`
- Required triplet runs present: 3/3
- RDM reliability gate: `green`
- Neighbors pre-registered for current RDM: yes
- Human sanity gate for current RDM: `passed`
- Directional items generated for current RDM: yes
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

## Pre-Registered Predictions

| Target | Predicted near | Near d | Far controls |
|---|---|---:|---|
| `alligator` | `turtle` | 0.136 | `chameleon` (1.160), `cobra` (1.154) |
| `caiman` | `crocodile` | 0.063 | `chisel` (1.561), `cobra` (1.375) |
| `crocodile` | `tortoise` | 0.043 | `chisel` (1.468), `saw` (1.743) |
| `boa python` | `snake` | 0.048 | `chameleon` (1.372), `gecko` (1.297) |
| `cobra` | `snake` | 0.421 | `caiman` (1.375), `chisel` (1.205) |
| `snake` | `boa python` | 0.048 | `chameleon` (1.216), `saw` (1.203) |
| `blindworm` | `toad` | 0.070 | `saw` (1.457), `axe` (1.806) |
| `chameleon` | `gecko` | 0.003 | `hammer` (1.459), `boa python` (1.372) |
| `gecko` | `chameleon` | 0.003 | `hammer` (1.528), `saw` (1.597) |
| `lizard` | `gecko` | 0.046 | `cobra` (1.662), `saw` (1.715) |
| `salamander` | `toad` | 0.035 | `saw` (1.448), `axe` (1.772) |
| `toad` | `salamander` | 0.035 | `saw` (1.612), `chisel` (1.810) |
| `tortoise` | `turtle` | 0.001 | `chisel` (1.451), `saw` (1.612) |
| `turtle` | `tortoise` | 0.001 | `chisel` (1.430), `saw` (1.591) |
| `axe` | `hammer` | 0.041 | `lizard` (1.809), `blindworm` (1.806) |
| `chisel` | `hammer` | 0.278 | `snake` (1.740), `toad` (1.810) |
| `hammer` | `axe` | 0.041 | `crocodile` (1.875), `salamander` (1.867) |
| `saw` | `axe` | 0.116 | `toad` (1.612), `tortoise` (1.612) |

## Live Risks

- If the model is near-perfect on these items, H1 is untestable and the item phrasing needs to move into a harder uncertainty band.
- If RDM reliability is red, do not register or interpret neighbors except as an engineering smoke test.
- Step 2 is intentionally absent until neutral H1 is green and the sanity gate passes.
