# Paper-Adapted Semantic Hub Plan

Created: 2026-07-08

## Goal

Adapt Wu, Yu, Yogatama, Lu, and Kim, "The Semantic Hub Hypothesis"
(`arXiv:2411.04986`) to the coherence-SFT setting.

The current semantic-hub run is useful but incomplete. It shows that aligned
states increase cross-format invariance across triplet, pairwise, and
feature-listing prompts. The paper's stronger pattern is not just invariance:
it uses relative matched-vs-mismatched baselines, logit-lens anchoring to
dominant-language semantic tokens, and causal interventions in the shared
mid-layer space.

## What The Paper Tests

Paper method to mirror:

- Eq. 1 / similarity: semantically equivalent inputs from different data types
  should be closer than unrelated inputs, especially in intermediate layers.
- Eq. 2-3 / logit lens: intermediate hidden states should be anchored by
  semantic tokens in the model's dominant language before final layers project
  back to the surface form required by the input context.
- Intervention: changing the shared mid-layer representation should
  predictably change outputs in another data type or surface format.

The released codebase separates most data types into `similarity.py`,
`logit_lens.py`, and `intervention.py`, which is the right structure to copy
locally.

Sources:

- Paper: https://arxiv.org/abs/2411.04986
- Paper HTML: https://arxiv.org/html/2411.04986
- Code: https://github.com/ZhaofengWu/semantic-hub

## Our Analogue

We do not have true multimodal inputs in this project. Our direct analogue is
different elicitation "spokes" for the same held-out THINGS concept:

- raw concept label;
- pairwise similarity prompt;
- triplet similarity prompt;
- feature-listing prompt;
- optional symbolic S* prompt, e.g. a structured row containing close/far
  neighbors and similarities.

For coherence-SFT, the behavioral surface form is often an answer token such as
`A`, `B`, `yes`, or `no`, while the semantic content is a concept or neighbor
concept. A paper-style hub result would show that middle layers represent the
concept/neighbor semantics before final layers collapse to the task's required
surface answer.

## Test 1: Relative Similarity With Controls

Purpose: replace the current broad RDM/CKA-only read with the paper's relative
matched-vs-baseline test.

Implementation:

- New script: `src/sft/run_semantic_hub_paper_similarity.py`.
- Reuse `results/sft_semantic_hub/hidden_states/{arm}.npz`.
- For every arm, layer, and format pair, compute cosine similarity for:
  - same concept across formats;
  - random nonmatching concepts across formats;
  - S*-close nonmatching concepts across formats;
  - S*-far nonmatching concepts across formats.
- Report layer curves for:
  - `same_minus_random`;
  - `same_minus_close_neighbor`;
  - `same_minus_far_neighbor`;
  - matched-pair retrieval top-1/top-5.
- Bootstrap concepts for 95% CIs.

Decision criterion:

- A real hub should peak in middle layers on `same_minus_random`.
- The stricter result is positive `same_minus_close_neighbor`, because that
  shows concept identity alignment beyond generic semantic relatedness.

Compute:

- CPU-only from existing NPZs; run immediately without using a GPU.

## Test 2: Logit-Lens Semantic Anchoring

Purpose: test whether mid-layer states are closer to concept/semantic tokens
than task-surface tokens, following the paper's dominant-language anchoring
logic.

Implementation:

- New script: `src/sft/run_semantic_hub_logit_lens.py`.
- Load each arm and the model LM head.
- Use stored hidden states where possible; if final-norm handling is ambiguous,
  run a sensitivity analysis with raw unembedding and RMSNorm-before-LM-head.
- Build an anchor table:
  - concept label token;
  - S*-close neighbor token;
  - S*-far neighbor token;
  - triplet answer token (`A`/`B`) when available;
  - pairwise answer token (`yes`/`no`) when available.
- Follow the paper's tokenization caution:
  - test prefix-space and no-prefix variants;
  - keep single-token anchors for the primary analysis;
  - report retained concept count.

Primary metrics:

- `concept_margin = logit(concept) - max(logit(close), logit(far), random)`.
- `close_neighbor_margin = logit(close_neighbor) - logit(far_neighbor)`.
- `semantic_vs_surface_margin = logit(correct_neighbor/concept) - logit(A/B/yes/no)`.

Expected hub pattern:

- Semantic margins rise in middle layers.
- Surface-answer margins dominate only near final layers.
- If coherence-SFT creates a better hub, `taskvec_a0p25` or `lowrank` should
  improve middle-layer semantic margins over base while not making final-layer
  surface calibration worse.

Compute:

- Local GPU after the ARC lane frees. Start with `base`, `lowrank`, and
  `taskvec_a0p25`; add `scrambled` as the control after the script is stable.

## Test 3: Formal/Symbolic Spoke Similarity

Purpose: get closer to the paper's formal-semantics/code cases instead of only
using natural-language prompt variants.

Implementation:

- Add a fourth prompt format to hidden-state extraction:
  `symbolic_sstar`.
- For concept `c`, encode a structured string such as:
  `concept(c); close(c, x); far(c, y); sim_close(s); sim_far(t)`.
- Compare `symbolic_sstar` against raw concept, pairwise, triplet, and
  feature-listing spokes with Test 1.

Controls:

- Swap close/far neighbor positions while preserving lexical overlap.
- Shuffle predicate order in the symbolic string.

Decision criterion:

- Matched symbolic/natural concept states should beat swapped and shuffled
  controls in middle layers.

Compute:

- One hidden-state extraction pass per arm. Debug on `base` and
  `taskvec_a0p25` locally before scaling.

## Test 4: Causal Hub Intervention

Purpose: move beyond correlation and test whether the mid-layer shared
representation is used by the model, matching the spirit of the paper's
intervention section.

Interventions:

1. Same-concept spoke substitution:
   - Run a triplet prompt.
   - Patch the mid-layer last-token hidden state with the feature-listing or
     symbolic spoke hidden state for the same concept.
   - Compare against a random-concept patch.
   - Expected: same-concept patch preserves or strengthens the correct answer;
     random patch degrades it.

2. Concept transplant:
   - Construct paired triplets where option `Y1` is close to concept A and
     option `Y2` is close to concept B.
   - Patch concept A's triplet hidden state with concept B's feature/symbolic
     hidden state at candidate hub layers.
   - Expected: answer logits shift from `Y1` toward `Y2` most strongly in
     middle layers.

3. Activation addition:
   - Estimate a concept direction `h(B) - h(A)` from feature/symbolic spokes.
   - Add it to concept A triplet prompts.
   - Expected: answer logits shift toward B-compatible choices.

Implementation:

- New script: `src/sft/run_semantic_hub_interventions.py`.
- Use forward hooks on HF Llama blocks first; only consider `pyvene` if local
  hooks become brittle.
- Small smoke: 32-64 triplets, layers 8-20, `base` and `taskvec_a0p25`.

Decision criterion:

- A credible causal hub result is a layer-localized intervention effect in
  middle layers that is stronger for coherent/task-vector arms than base and
  absent or degraded for scrambled.

Compute:

- Local GPU only until hooks and output scoring are stable.
- If the sweep becomes large, move independent arm/layer chunks to CHTC after a
  local command is validated.

## Test 5: Link Hub Metrics To Behavior And fMRI

Purpose: decide whether paper-style hub metrics explain the actual scientific
phenomena better than the current RDM-only hub score.

Analysis:

- Join Test 1-4 metrics to:
  - coherence task gains;
  - benchmark drops/gains by skill;
  - THINGS-fMRI RSA;
  - held-out fMRI hub regression.
- Ask whether middle-layer semantic margins/intervention effects predict:
  - gains on semantic similarity tasks;
  - preservation on HellaSwag/WinoGrande;
  - drops on ARC/MMLU/WiC/TruthfulQA calibration;
  - ATL/Language exploratory fMRI scores.

Decision criterion:

- If logit-lens/intervention metrics predict coherence gains without predicting
  benchmark damage, they become the target mitigation objective.
- If they track benchmark damage, the induced hub is probably entangled with
  answer calibration and needs KL/calibration mitigation.

## Priority Order

1. Run Test 1 now from existing hidden states.
2. Run Test 2 on `base`, `lowrank`, `taskvec_a0p25`, and `scrambled`.
3. Add the symbolic S* spoke and rerun Test 1 on `base` and `taskvec_a0p25`.
4. Implement small causal interventions.
5. Join the paper-style hub metrics to benchmark and fMRI tables.

## Reporting

Write outputs under `results/sft_semantic_hub_paper/`:

- `similarity_by_layer.csv`
- `logit_lens_by_layer.csv`
- `symbolic_similarity_by_layer.csv`
- `intervention_by_layer.csv`
- `metric_behavior_bridge.csv`
- `REPORT.md`

Update:

- `README.md`
- `research/EXPERIMENT_LOG.md`
- `research/STATUS.md`

Commit after each major stage.
