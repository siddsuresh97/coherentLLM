# Coherence-SFT Experiment Log

Last updated: 2026-07-08

## Read this first

This is the running handoff log for follow-up work after Tasks 1-8 on branch
`coherence-sft`. New agents should read, in order:

1. `research/STATUS.md`
2. `results/sft_eval/REPORT.md`
3. this log
4. the active task briefs listed below

Tracking convention:
- Task briefs live in `research/CODEX_TASK_*.md`.
- Major experiment outputs live under `results/`.
- Every meaningful checkpoint should be committed and pushed to `origin/coherence-sft`.
- Append command summaries, artifact paths, and interpretation here after each major run.
- Do not treat a partial/stale artifact as a result unless this log explicitly marks it as valid.

## Active task briefs

- `research/CODEX_TASK_9_FMRI_AND_SEMANTIC_HUB.md`
  - Goal: test whether coherence-SFT improves brain predictivity and whether a
    format-invariant hidden-state semantic hub mediates it.
  - Status: brief created; no fMRI or hidden-state production runs launched yet.

- `research/CODEX_TASK_10_BENCHMARK_DROPS.md`
  - Goal: diagnose why lowLR/lowrank gain semantic/human-alignment tasks but drop
    on standard capability benchmarks, and propose mitigation experiments.
  - Status: brief created; existing results only so far.

## Current branch state

- Branch: `coherence-sft`
- Remote: `origin git@github.com:siddsuresh97/coherentLLM.git`
- Current checked commit before Task 9 docs: `76dbc9f`
- Worktree at log creation: clean.

## Sidecar agents

The following sidecar agents were launched on 2026-07-08. They were instructed not
to edit files; their memos should be integrated into this log and the task briefs.

| Agent | ID | Scope |
|---|---|---|
| Peirce | `019f4341-554b-7b50-b654-2e2dc09ffb2f` | Alex Huth / LeBel / UT Austin natural-language fMRI encoding literature |
| Kierkegaard | `019f4341-6757-78d3-9fb5-94f25969da6e` | Evelina Fedorenko / Ivanova language-network evaluation literature |
| Gibbs | `019f4341-806a-7d51-82f9-49fc95df0c88` | Local benchmark drop/gain diagnosis |
| Nietzsche | `019f4341-bb81-7200-ba91-a7c09b9d4aa2` | Semantic-hub hypothesis and tests |

## 2026-07-08 sidecar memo: Fedorenko / language-network evaluation

Source: Kierkegaard, agent `019f4341-6757-78d3-9fb5-94f25969da6e`.

Key implications for this project:

- Fedorenko/EvLab-style language-brain evaluation should use individually
  localized language-network masks when possible, not broad anatomical
  Broca/Wernicke parcels.
- Standard localizer contrast: `sentences > nonword lists`.
- Current implementations use group-constrained subject-specific masks across
  bilateral fronto-temporal language regions, often selecting the top localizer
  voxels per mask/subject.
- Primary language-fMRI metric should be held-out encoding predictivity:
  fit a linear mapping from LM features to neural responses on training stimuli,
  then score predicted vs observed held-out responses with Pearson `r`, optionally
  normalized by a noise ceiling.
- Fair comparison requirements: same stimuli, same train/test splits, same
  feature extraction/context, same mapping capacity, same layer-selection protocol,
  and paired deltas against the base model.
- Main confounds: lexical-semantic content can dominate brain scores; perplexity
  and next-word prediction quality correlate with brain predictivity; layer
  fishing inflates effects; prompt/chat-template artifacts can make model states
  incomparable; powerful nonlinear mappings can erase representational differences.
- Practical consequence: THINGS-fMRI Glasser Language ROI remains useful as an
  exploratory object-concept control, but it should not be framed as the primary
  Fedorenko-style language-network test.

Primary citations/links from memo:

- Fedorenko et al. 2010 / 2011 language localizer papers.
- Schrimpf et al. 2021, "The neural architecture of language".
- Ivanova et al. 2022, "Beyond linear regression".
- Kauf, Tuckute et al. 2024, lexical-semantic content in fMRI predictivity.
- Hosseini et al. 2024, controlled ANN training-scale/checkpoint analysis.

## 2026-07-08 sidecar memo: Huth / LeBel language-fMRI

Source: Peirce, agent `019f4341-554b-7b50-b654-2e2dc09ffb2f`.

Key implications for this project:

- Primary Huth-style language-fMRI candidate is LeBel et al. 2023,
  OpenNeuro `ds003020`: 8 participants listening to 27 natural narrative
  stories, about 6 hours per participant; `UTS01`, `UTS02`, and `UTS03` also
  have extended high-data story sets.
- Practical code/data entry points:
  - `https://openneuro.org/datasets/ds003020`
  - `https://github.com/HuthLab/deep-fMRI-dataset`
  - `https://github.com/HuthLab/encoding-model-scaling-laws`
- Primary metric: voxelwise Pearson correlation between predicted and held-out
  BOLD time courses, optionally noise-normalized (`CC_norm` in LeBel).
- Standard feature procedure: feed exact narrative text stream, extract hidden
  states per word/final word token, align by TextGrid word times, downsample to
  TRs, concatenate FIR delays such as 2/4/6/8 s, then fit ridge regression.
- Avoid chat templates for passive-listening fMRI; subjects heard narrative text,
  not instruction-formatted dialogue.
- Layer, ridge alpha, task-vector alpha, trimming rules, and ROIs must be chosen
  on training/validation only, never on the final test story.
- Recommended first high-power subjects are the extended LeBel subjects
  (`UTS01`/`UTS02`/`UTS03`), but a smaller first pass can use the repeated
  `wheretheressmoke` test story.
- Important caveat: a better encoding score may mean the LoRA features are more
  linearly decodable or better scaled for ridge, not necessarily more
  human-like. The safe claim is "improves held-out predictivity under a fixed
  linear encoding model" unless controls support a stronger interpretation.

Primary citations/links from memo:

- LeBel et al. 2023, Scientific Data, `https://doi.org/10.1038/s41597-023-02437-z`
- OpenNeuro `ds003020`, `https://doi.org/10.18112/openneuro.ds003020.v2.0.0`
- Antonello et al. 2023, scaling laws, `https://arxiv.org/abs/2305.11863`
- Tang et al. 2023 semantic decoder, Nature Neuroscience.
- Huth et al. 2016 semantic maps, Nature.

## 2026-07-08 sidecar memo: benchmark drops/gains

Source: Gibbs, agent `019f4341-806a-7d51-82f9-49fc95df0c88`.

Key implications for this project:

- The broad standard-capability run on disk is for `lowLR`; there does not appear
  to be a matching wide-benchmark run for `lowrank`.
- `lowLR` gains are strongly concentrated in semantic/human-behavior tasks:
  THINGS odd-one-out `+0.0842`, THINGS triplet similarity `+0.2279`, and gains
  on WordSim-353/MEN/MTurk-771/SimVerb-3500.
- `lowLR` broad capability has 0 gains, 4 flat groups, and 6 drops:
  ARC-Easy, ARC-Challenge, OpenBookQA, HellaSwag, WiC, and MMLU.
- Likely mechanism: SFT is very narrow and geometrically targeted
  (triplet/pairwise/feature prompts from one NOVA similarity target), while the
  LoRA update touches all attention and MLP projections and perturbs
  multiple-choice option ranking. The pattern is not total collapse because
  Winogrande, CommonsenseQA, TruthfulQA, and PIQA are flat.
- The earlier "retention-safe" claim was based on a narrow retention guard; the
  later wide benchmark exposed broader MMLU/ARC/WiC costs.
- The next mitigation evidence should be eval-only:
  1. run wide benchmark on task-vector `alpha=0.25`;
  2. run wide benchmark on `lowrank`;
  3. then consider alpha grid `0.10/0.20/0.25/0.35`, KL-to-base, replay, adapter
     sparsification, or adapter routing.

## 2026-07-08 sidecar memo: semantic hub

Source: Nietzsche, agent `019f4341-bb81-7200-ba91-a7c09b9d4aa2`.

Key implications for this project:

- Method basis is Wu, Yu, Yogatama, Lu, and Kim, "The Semantic Hub Hypothesis:
  Language Models Share Semantic Representations Across Languages and
  Modalities" (`arXiv:2411.04986`, ICLR 2025). The original method studies
  intermediate-layer shared representations across languages/modalities, plus
  logit-lens and cross-domain interventions. Our version adapts the method to
  elicitation formats: triplet, pairwise, and feature/listing prompts.
- Operational hub object: hidden state `h[model, layer, concept, format]`, where
  formats are triplet, pairwise, and feature/listing prompts.
- A semantic hub exists if same-concept codes across formats are closer than
  different-concept codes and if concept RDMs agree across formats in middle
  layers more than early/late layers.
- Primary metrics:
  - cross-format RSA between triplet/pairwise/feature concept RDMs;
  - linear CKA between format matrices with concepts as rows;
  - same-concept cross-format retrieval top-1/top-5 and margin;
  - concept-label vs format-label kernel alignment;
  - mid-layer AUC, e.g. L10-20 for Llama-3.1-8B.
- Expected pattern: `lowLR` and `lowrank` > `base` >> `scrambled`, with
  task-vector alpha monotonicity. If the gain appears only in final layers, it is
  more likely output/report formatting than a representational hub.
- Causal tests should avoid broad activation-addition during free generation
  because Task 8 showed that route collapses generation. Use targeted
  cross-format patching under logprob scoring instead.
- Brain bridge: compare format-specific and format-averaged hub-code RDMs to
  THINGS-fMRI RDMs; the strongest mechanistic link is if hub score predicts fMRI
  RSA across arms/layers.

## 2026-07-08 checkpoint: planning and data audit facts

Facts established locally before implementation:

- The original fMRI proposal is `research/FUTURE_brain_predictivity.md`.
- The semantic-hub proposal is `research/FUTURE_semantic_hub.md`.
- Existing THINGS-fMRI pipeline to reuse is in sibling project:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/`
- Existing Ch4 aggregation script:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/dissertation/scripts/compute_fmri_rsa.py`
- THINGS-fMRI metadata/betas are available locally under:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/`
- The SFT eval concept file has 128 lines in `data/scale128/concepts.csv`.
- Exact overlap with the 720 THINGS-fMRI concepts in `sub-01_StimulusMetadata.csv` is 90 concepts.
- Prior Ch4 noise ceilings warn against overclaiming ATL/language on object fMRI:
  Ventral Visual is high (`lower=0.229`, `upper=0.284`), while Language is low
  (`lower=0.010`, `upper=0.019`) and pooled ATL is near zero in the old object
  setup. Therefore Task 9 uses Ventral Visual as the primary ROI and treats ATL
  subregions / Language as exploratory.

Command used to establish overlap:

```bash
python - <<'PY'
import csv
from pathlib import Path
sft = Path('data/scale128/concepts.csv')
fmri = Path('/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/sub-01_StimulusMetadata.csv')
s = {line.strip() for line in sft.read_text().splitlines() if line.strip()}
with fmri.open() as f:
    r = csv.DictReader(f)
    t = {row['concept'] for row in r}
print(len(s), len(t), len(s & t))
PY
```

Expected output: `128 720 90`.

## 2026-07-08 result: fMRI data audit

Command:

```bash
python src/sft/fmri_audit.py
```

Artifacts:

- `src/sft/fmri_audit.py`
- `results/sft_fmri/concept_overlap.csv`
- `results/sft_fmri/fmri_data_audit.json`
- `results/sft_fmri/REPORT.md`

Result:

- SFT concepts: 128.
- Common THINGS-fMRI concepts across subjects: 720.
- Primary exact held-out overlap: 90 concepts.
- Subjects `01`, `02`, and `03` each have 9,840 trials.
- Response H5 shapes:
  - sub-01: 211,339 voxels x 9,840 trials
  - sub-02: 226,950 voxels x 9,840 trials
  - sub-03: 189,164 voxels x 9,840 trials
- The 90 overlap concepts have no missing trials for any subject.
- Trial count per overlap concept ranges from 12 to 24, mean 13.867.
- Ventral Visual has thousands of voxels in every subject:
  - sub-01: 4,404
  - sub-02: 4,635
  - sub-03: 3,790

Interpretation:

- Track A is unblocked at the data level.
- Primary THINGS-fMRI RSA should use the 90 exact held-out overlap concepts.
- The next implementation step is hidden-state extraction for the 90 overlap
  concepts across `base`, `lowLR`, `lowrank`, `scrambled`, and task-vector arms.

## Open decisions

- Whether to run Phase 1 RSA only on the 90 exact held-out overlaps, or also add
  a secondary full-720 analysis that is not leakage-clean.
- Whether Phase 2 language-fMRI should use LeBel/Huth first or a more turnkey
  OpenNeuro/Narratives route if LeBel preprocessing is costly.
- Whether mitigation work should prioritize alpha-sweep/task-vector strength,
  training recipe changes, or post-hoc adapter composition.
