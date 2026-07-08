# Literature And Experiment Plan

Created: 2026-07-08

This file synthesizes the sidecar agent memos into one handoff document for
future agents. The full raw memo trail remains in
[`EXPERIMENT_LOG.md`](EXPERIMENT_LOG.md).

## Current Read

The scientific question now has two separable parts:

1. Whether coherence-SFT creates or sharpens a format-invariant semantic hub
   inside the model.
2. Whether that representational shift improves neural predictivity, first in
   THINGS object-fMRI RSA and later in language-fMRI encoding.

The current evidence supports the first part more strongly than the second.
Semantic-hub metrics show large cross-format invariance gains in aligned and
task-vector arms. The first THINGS-fMRI bridge is descriptive and positive in
places, but it does not yet show that the hub explains fMRI RSA.

## Semantic Hub Basis

Anchor paper:

- Wu, Yu, Yogatama, Lu, and Kim, "The Semantic Hub Hypothesis: Language Models
  Share Semantic Representations Across Languages and Modalities,"
  `arXiv:2411.04986`, ICLR 2025:
  https://arxiv.org/abs/2411.04986

What the paper contributes:

- The semantic-hub claim is an intermediate-layer shared representation claim.
- The original paper studies semantically equivalent inputs across languages and
  modalities, then uses representation similarity, logit-lens interpretation,
  and interventions to argue that the shared space is used during processing.
- Our adaptation treats triplet, pairwise, and feature/listing prompts as the
  "spokes"; held-out THINGS concept identity is the shared semantic content.

Operational test:

- Hidden state object: `h[model, layer, concept, format]`.
- Hub signature: same-concept codes across formats are closer than
  different-concept codes, and concept RDMs agree across formats in middle
  layers more than early/late layers.
- Metrics already implemented: cross-format RSA, linear CKA, same-concept
  retrieval top-1/top-5, and concept-vs-format alignment.
- Next decisive model-only test: targeted cross-format activation patching
  under logprob scoring, not free-generation activation steering.

Current result:

- Strong cross-format invariance gain exists inside the model.
- `taskvec_a0p25` is currently the best mid-layer hub candidate.
- It is not yet a clean "concept dominates format" result because
  concept-minus-format alignment remains negative across arms.

## THINGS-fMRI Plan

Local object-fMRI work is the correct first neural test because it is already
implemented and leakage-clean for the held-out overlap.

Primary design:

- Concepts: 90 exact overlaps between held-out SFT eval concepts and
  THINGS-fMRI concepts.
- Subjects: THINGS-fMRI subjects 01/02/03.
- Primary ROI: Ventral Visual, because object-fMRI noise ceilings are reliable
  there.
- Exploratory ROIs: ATL subregions and Language, with explicit caution because
  this is object-fMRI rather than a Fedorenko-style language localizer.
- Model arms: `base`, `scrambled`, `lowLR`, `lowrank`, `taskvec_a0p25`,
  `taskvec_a0p5`, `taskvec_a1p0`.

Completed result:

- The pipeline works end to end.
- Ventral Visual shows expected object-RSA and scrambled-control separation.
- Aligned arms are mostly flat versus base in the primary ROI.
- The next decisive bridge is a held-out regression comparing format-averaged
  hub RDMs against single-format RDMs as predictors of the same fMRI RDMs.

Do not overclaim:

- A positive ATL or Language object-RSA point is exploratory.
- The stronger claim requires language-fMRI with language-localized voxels or
  a clearly labeled approximation.

## Language-fMRI Plan

### Huth / LeBel Route

Best practical candidate:

- LeBel et al. 2023, "A natural language fMRI dataset for voxelwise encoding
  models," Scientific Data:
  https://doi.org/10.1038/s41597-023-02437-z
- Dataset: OpenNeuro `ds003020`:
  https://openneuro.org/datasets/ds003020
- Code: HuthLab `deep-fMRI-dataset`:
  https://github.com/HuthLab/deep-fMRI-dataset
- Related scaling-law code/data:
  https://github.com/HuthLab/encoding-model-scaling-laws

Why this route:

- It has 8 participants listening to natural narrative stories, about 6 hours
  per participant.
- `UTS01`, `UTS02`, and `UTS03` have extended high-data story sets.
- The standard metric is voxelwise held-out prediction correlation between
  predicted and observed BOLD time courses, optionally noise-normalized.

Implementation requirements:

- Feed exact narrative transcript streams, not chat-formatted prompts.
- Extract hidden states per word/final word token.
- Align features to TextGrid word times, downsample to TRs, and concatenate FIR
  delays such as 2/4/6/8 seconds.
- Fit voxelwise ridge or banded-ridge models.
- Select layer, ridge alpha, task-vector alpha, trimming rule, and ROI only
  using training/validation data.
- Report paired deltas against base per subject/ROI/fold.

Safe claim:

- "The aligned representation improves held-out brain predictivity under a
  fixed linear encoding model."
- Do not claim "more human-like language processing" unless controls rule out
  scaling, decodability, lexical content, layer fishing, and prompt-format
  artifacts.

### Fedorenko / EvLab Route

Best-practice target:

- Individually localized language-network masks, ideally from a
  `sentences > nonword lists` localizer.
- If individual localizers are unavailable, atlas/group masks are exploratory.

Metric:

- Fit a linear mapping from LM features to neural responses on training
  stimuli.
- Score held-out predicted-vs-observed responses with Pearson `r`, optionally
  ceiling-normalized.
- Use the same stimuli, splits, preprocessing, feature extraction, mapping
  capacity, context windows, and layer-selection protocol for every model arm.

Main confounds:

- Lexical-semantic content can dominate fMRI predictivity.
- Perplexity/next-word prediction quality can correlate with brain scores.
- Powerful nonlinear mappings can erase representational differences.
- Prompt/chat templates can make base and SFT states incomparable.
- Flexible best-layer selection can inflate small effects.

## Benchmark-Drop Diagnosis

Current mechanism hypothesis:

- Semantic/human gains are expected because training directly supervises one
  concept-similarity geometry across triplet, pairwise, and feature prompts.
- Broad drops are likely a combination of narrow SFT, global LoRA perturbation
  across attention/MLP modules, and multiple-choice logit-rank/calibration
  shifts.
- The drop is not uniform. WinoGrande is stable, WiC is near chance for aligned
  states, and lowrank HellaSwag runs once memory settings are conservative.

Next benchmark work:

- Finish lowrank ARC.
- Summarize `base`, `lowLR`, and `lowrank` with the wide-bench runner.
- Finish the scrambled zero-shot control to decide whether short-task drops are
  generic SFT perturbation or semantic-alignment-specific.
- Continue taskvec `a0p25` long groups on reliable A5000 lanes.

Mitigation candidates, cheapest first:

1. Use `taskvec_a0p25` as the default aligned arm if broad-benchmark retention
   stays acceptable.
2. Run alpha grid `0.10`, `0.20`, `0.25`, `0.35`.
3. Route adapters only for semantic/coherence prompts if the goal is a tool
   rather than a single always-on model.
4. Try KL-to-base or small general-instruction replay if a new training run is
   justified.
5. Ablate LoRA layers/modules to identify gain-vs-drop contributors.

## GPU / Scale-Out Policy

- Local/direct GPU debug first.
- A5000 rank-64 vLLM is viable; `scrambled` zero-shot allocated and runs on
  `rogers-gpu-1`.
- `opt-a007` H100 is currently unsuitable for rank-64 LoRA vLLM evals: both
  shared-path and host-local staged adapters stalled before GPU allocation.
- A rank-64 HF/PEFT smoke on `opt-a007` also stayed pre-GPU for multiple minutes
  and was stopped.
- Use CHTC only after a command works locally with known paths, environment,
  output behavior, and memory settings.

## Concrete Next Experiments

1. Complete current lowrank ARC and scrambled zero-shot jobs.
2. Update `README.md`, `EXPERIMENT_LOG.md`, and wide-bench summaries with those
   metrics; commit and push each checkpoint.
3. Run the held-out fMRI bridge regression:
   format-averaged hub RDMs vs single-format RDMs predicting fMRI RDMs.
4. If language-fMRI is approved for a heavier pass, start with LeBel/Huth
   feasibility on one extended subject and one held-out story, using identical
   extraction for `base`, `scrambled`, `lowLR`, and `taskvec_a0p25`.
5. Only after the local one-subject pipeline is correct, scale independent
   subject/model/layer extraction jobs to CHTC.
