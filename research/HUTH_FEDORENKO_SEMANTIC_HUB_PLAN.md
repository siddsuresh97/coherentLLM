# Huth/Fedorenko/Semantic-Hub Evaluation Plan

Created: 2026-07-09
Branch: `coherence-sft`
Scope: standalone research plan only. This checkpoint intentionally does not
edit `README.md`, `research/EXPERIMENT_LOG.md`, or `results/sft_huth_lebel/*`.

## Executive Read

The useful claim to test is not "coherence-SFT creates a human-like semantic
hub" in one step. The supported local result is narrower: coherence-aligned
arms already show much stronger cross-format hidden-state invariance over
triplet, pairwise, and feature-listing prompts than the base model. The
unsupported pieces are whether this invariance is exact concept identity rather
than broad semantic smoothing, whether it is causally used by the model, and
whether it improves held-out brain predictivity in Huth/LeBel or Fedorenko-style
language-network analyses.

The plan below makes those distinctions trackable. It combines:

- Huth-style voxelwise encoding: word-level model states predict held-out BOLD
  responses during natural narrative listening.
- Fedorenko-style language-network discipline: language claims should use
  individually localized language fROIs, or be explicitly labeled exploratory.
- Wu et al. semantic-hub methods: relative matched-vs-mismatched similarity,
  logit-lens semantic anchoring, and causal interventions in intermediate
  layers.

## Local Evidence Ledger

### Supported By Existing Local Artifacts

1. Internal semantic-hub invariance is real in the current hidden-state grid.
   `results/sft_semantic_hub/REPORT.md` reports mid-layer cross-format RDM
   Spearman:
   - `base`: `0.3230`
   - `lowLR`: `0.6211`
   - `lowrank`: `0.6637`
   - `taskvec_a0p25`: `0.6770`

2. The stricter paper-style same-vs-mismatch test is positive for random and far
   controls, but much weaker for S*-close controls. `results/sft_semantic_hub_paper/REPORT.md`
   reports mid-layer `same_minus_random`:
   - `base`: `0.0022`
   - `lowLR`: `0.0247`
   - `lowrank`: `0.0311`
   - `taskvec_a0p5`: `0.0617`

   The same report shows the strict `same_minus_close_neighbor` margin remains
   small: best mid-layer value is `taskvec_a1p0 = 0.0099`; `base = -0.0002`.

3. The THINGS-fMRI RSA path works, but it is not yet strong evidence for a brain
   semantic hub. `results/sft_fmri/REPORT.md` reports 90 exact SFT/THINGS-fMRI
   overlap concepts, three subjects, and a clean Ventral Visual object signal:
   - `base`: `0.1990`
   - `lowLR`: `0.2010`
   - `lowrank`: `0.2013`
   - `taskvec_a0p25`: `0.1968`
   - `scrambled`: `0.1293`

   The aligned arms are nearly flat versus base in the primary object ROI.
   Language and ATL values are small exploratory effects, not Fedorenko fROI
   results.

4. The fMRI x hub bridge is suggestive but descriptive. `results/sft_fmri_semantic_bridge/REPORT.md`
   reports positive arm-layer delta correlations between hub RDM gains and some
   fMRI RSA deltas, but arm-summary analyses have only six non-base deltas. The
   report correctly says this does not establish that hub invariance explains
   object-fMRI RSA.

5. The held-out fMRI hub-regression machinery exists. `results/sft_fmri_hub_regression/REPORT.md`
   already compares format-averaged hub predictors against single-format
   predictors with concept-held-out folds. This is the right analysis template,
   but layer/arm selection is still descriptive.

6. Huth/LeBel smoke infrastructure exists, but the active CHTC lane is owned by
   Hubble. `results/sft_huth_lebel/REPORT.md` and `results/sft_huth_lebel/ENCODING_PLAN.md`
   say the smoke data are staged at `/staging/s/suresh27/datasets/ds003020-smoke`;
   GPU extraction debug cluster `5513178` passed; three-story smoke extraction
   cluster `5513245` is running; CPU ridge encoding should follow only after
   those features are inspected.

7. Benchmark retention is a necessary control. `results/sft_eval/wide_bench/skill_diagnostics/REPORT.md`
   shows the intervention improves semantic coherence and human-similarity
   alignment, but hurts ARC/OpenBookQA, WiC, and MMLU for completed aligned
   adapters. Any "hub" target must be checked against answer-ranking calibration
   damage.

### Unsupported Or Not Yet Established

- Unsupported: coherence-SFT creates a true transmodal hub comparable to the
  anterior temporal lobe. Our current spokes are prompt formats, not modalities.
- Unsupported: the local hidden-state effect is exact concept identity. The
  same-vs-close-neighbor margins are still small.
- Unsupported: the hub is causally used. Cross-format patching and concept
  transplant experiments have not yet produced a report.
- Unsupported: coherence-SFT improves Huth/LeBel held-out narrative encoding.
  The smoke encoding run has not yet produced scored BOLD predictions.
- Unsupported: any current result is a Fedorenko-style language-network result.
  The local object-fMRI analysis uses atlas/broad ROIs, not subject-specific
  language fROIs.
- Unsupported: ATL or Language-region THINGS object-RSA deltas are decisive.
  They are exploratory because the stimulus and masks are not a language
  localizer or language encoding design.

## Literature Grounding

### Huth-Style Voxelwise Semantic Encoding

Huth et al. mapped semantic selectivity by fitting voxelwise fMRI encoding
models while subjects listened to narrative stories. The Nature abstract frames
the semantic system as cortex-wide and says the work used voxelwise modeling of
hours of story-listening fMRI to create a semantic atlas:
https://www.nature.com/articles/nature17637

The relevant methodological principle for this project is held-out prediction,
not visual inspection. The local Huth/LeBel smoke script mirrors this:

- extract word-level LLM states from plain story text with
  `src/sft/huth_lebel_extract_word_states.py`;
- align word features to TRs with Lanczos interpolation;
- concatenate FIR delays;
- fit voxelwise ridge models;
- score held-out BOLD prediction with Pearson `r` using
  `src/sft/huth_lebel_smoke_encoding.py`.

The LeBel/Huth Scientific Data descriptor is directly aligned with this lane.
It reports BOLD fMRI for eight participants listening to 27 natural narrative
stories, plus an extended stimulus set for three participants, word and phoneme
timings, preprocessed data, and code for voxelwise encoding models:
https://www.nature.com/articles/s41597-023-02437-z

Its analysis recipe is also the right constraint for this project: word or
phoneme features are downsampled to the fMRI acquisition rate, delayed for the
hemodynamic response, and used in voxelwise regression to predict fMRI data.

Antonello, Vaidya, and Huth provide the modern LLM baseline: transformer
representations predict natural-language fMRI responses, and held-out encoding
performance scaled with model size and fMRI training data in their 2023/2024
language encoding work:
https://arxiv.org/abs/2305.11863

Project implication: our contribution is not "another larger model." It is a
controlled, same-backbone LoRA/task-vector manipulation. The decisive Huth-style
question is whether a coherence-aligned arm improves held-out BOLD prediction
over its own base and over `scrambled`, under identical feature extraction,
alignment, ridge, and layer-selection rules.

### Fedorenko-Style Language-Network Evidence

Fedorenko, Ivanova, and Regev review the core language network as a strongly
interconnected, language-selective set of left frontal and temporal areas that
is distinct from lower-level perceptual/motor mechanisms and from broader
knowledge/reasoning systems:
https://www.nature.com/articles/s41583-024-00802-4

The same review cites the individual-subject localizer tradition, including
Fedorenko et al. 2010, which defines ROIs functionally in individual subjects.
For this project, that means:

- subject-specific language fROIs, ideally from `sentences > nonword lists`, are
  the standard for strong language-network claims;
- atlas parcels, broad "Language" masks, or Huth semantic-map regions are useful
  exploratory summaries but are not Fedorenko/EvLab proof;
- any language-network claim must show selectivity or specificity relative to
  controls such as auditory cortex, multiple-demand/task-control regions,
  visual/object regions, and possibly social-reasoning regions.

Fedorenko, Piantadosi, and Gibson also argue that language and thought can be
dissociated, and that language is not a prerequisite for complex thought:
https://www.nature.com/articles/s41586-024-07522-w

Project implication: if coherence-SFT improves semantic association while
hurting MMLU, ARC, or WiC, that does not by itself contradict a language-network
result. It may mean the intervention sharpened one semantic/communicative
geometry while damaging sharp answer-ranking or reasoning/calibration surfaces.
The plan must keep language, semantic memory, reasoning, and benchmark
calibration separated.

Recent LLM work also mirrors Fedorenko's localization logic in models:
AlKhamissi et al. identify language-selective units in LLMs using a
neuroscience-style localization approach and test causality by ablation:
https://arxiv.org/abs/2411.02280

Project implication: an optional model-internal Fedorenko-style companion test
is to localize units/layers by a sentence-vs-nonword contrast, then test whether
those units overlap with the coherence hub and whether ablation selectively
hurts language/coherence without broadly damaging unrelated tasks.

### Wu et al. Semantic-Hub Methods To Mirror

Wu, Yu, Yogatama, Lu, and Kim define a language-model semantic hub as a shared
intermediate representation space where semantically related inputs from
different data types are close, often anchored by dominant-language tokens, and
causally used for output behavior:
https://arxiv.org/abs/2411.04986

Their paper has three method components we should mirror:

1. Relative similarity: semantically equivalent inputs should be closer than
   unrelated or controlled mismatches, especially in intermediate layers.
2. Logit-lens anchoring: intermediate states should be closer to semantic
   tokens in the dominant language before final layers project to the required
   output surface.
3. Intervention: perturbing or replacing the shared intermediate representation
   should predictably affect output behavior in another data type or surface
   format.

The paper uses true cross-language, code, arithmetic, formal-semantics, visual,
and audio settings. Our current analogue is narrower:

- "spokes" are prompt/elicitation formats: raw concept, triplet, pairwise,
  feature-listing, and a proposed symbolic S* string;
- shared content is held-out THINGS/NOVA concept identity and S* neighborhood;
- output surfaces include `A`/`B`, `yes`/`no`, list text, and natural concept
  labels.

This adaptation is scientifically valid if framed as "format-invariant concept
hub within one text model." It is not valid to claim true cross-modal or
cross-language hub evidence without adding those modalities/languages.

## Main Hypotheses

### H1: Coherence-SFT Tightens A Format-Invariant Concept Hub

Prediction:

- `lowLR`, `lowrank`, and task-vector arms should increase same-concept
  cross-format similarity in middle layers relative to `base`.
- `scrambled` should not show the same structured improvement.
- The effect should survive stricter controls: S*-close neighbor, lexical
  overlap, prompt length, answer-token surface, and category balance.

Current status:

- Partly supported for same-vs-random and cross-format RDM/CKA.
- Not yet established for exact identity over S*-close neighbors.
- Not yet established causally.

### H2: The Hub Is Semantic, Not Just Output Formatting

Prediction:

- In middle layers, semantic anchors such as the concept label or correct
  neighbor should outrank final answer-surface tokens (`A`, `B`, `yes`, `no`) in
  logit-lens probes.
- In final layers, answer-surface tokens may dominate. That late dominance is
  not a hub failure; it is expected autoregressive verbalization.
- If semantic margins only appear in final layers, the result is probably
  answer formatting, not a hub.

Current status:

- Not yet run as a report. It is proposed in
  `research/SEMANTIC_HUB_PAPER_ADAPTED_PLAN.md`, but no local logit-lens report
  is present.

### H3: The Hub Is Functionally Used

Prediction:

- Cross-spoke clean activation patching should recover or preserve
  concept-consistent answer logprobs more than wrong-concept or wrong-layer
  patching.
- Concept transplant should shift triplet/pairwise answer logits in the
  direction predicted by the transplanted concept representation.
- Effects should be layer-localized to the hub band and stronger for aligned
  arms than base, with `scrambled` weaker or nonspecific.

Current status:

- Unsupported. This is the main missing evidence.

### H4: Coherence-SFT Improves Huth/LeBel Held-Out Encoding If The Hub Is Brain-Relevant

Prediction:

- `lowLR`, `lowrank`, or `taskvec_a0p25` should improve held-out voxelwise
  Pearson `r` over `base` in semantic/language-like cortex.
- The improvement should exceed any `scrambled - base` change.
- The improvement should not be concentrated in low-level auditory/acoustic
  regions.

Current status:

- Unsupported. Huth/LeBel smoke extraction/encoding is in progress in the
  active fMRI lane.

### H5: Fedorenko-Style Claims Require fROI Specificity

Prediction:

- If subject-specific language fROIs are available, aligned arms should improve
  held-out encoding in those fROIs more than in auditory, visual, or MD/control
  regions.
- If only atlas/broad masks are available, any "Language" result remains
  exploratory.

Current status:

- Unsupported. Current object-fMRI "Language" ROI results are not a language
  localizer.

## Track A: Mirror The Semantic-Hub Paper Locally

### A1. Relative Similarity With Stronger Controls

Existing script:

```bash
python src/sft/run_semantic_hub_paper_similarity.py
```

Existing outputs:

- `results/sft_semantic_hub_paper/similarity_by_layer.csv`
- `results/sft_semantic_hub_paper/similarity_summary.csv`
- `results/sft_semantic_hub_paper/REPORT.md`

Current method:

- Load hidden states from `results/sft_semantic_hub/hidden_states/{arm}.npz`.
- Compare same-concept cross-format cosine similarity against random
  nonmatching concepts, S*-close nonmatching concepts, and S*-far nonmatching
  concepts.
- Bootstrap concepts and report layer-resolved curves.

Required next controls:

- Lexical overlap: same string frequency and token count controls, especially
  for concepts with multi-token names.
- Prompt length: same-concept effects should not be explained by shorter or
  more stereotyped prompts.
- Option order: triplet answer order should be balanced and explicitly included
  as a nuisance label.
- Category balance: same-vs-close tests should compare within broad semantic
  category when possible.
- Held-out S* relation type: run close/far labels swapped as a negative control.
- Surface-token control: remove or mask final answer tokens when building
  hidden states, or compare last concept-token state versus final prompt-token
  state.

Pass signal:

- Middle-layer `same_minus_close_neighbor` is reliably positive for aligned arms
  and near zero for base/scrambled.
- Layer peak is not purely embedding/early-token or final-answer layers.
- Result survives tokenization and prompt-length controls.

Fail signal:

- Only `same_minus_random` is positive, while same-vs-close stays near zero.
  That means coherence-SFT may increase broad semantic clustering without
  forming exact concept identity.
- `scrambled` improves similarly. That means the metric is sensitive to generic
  adapter perturbation or hidden-state anisotropy.

### A2. Logit-Lens Semantic Anchoring

New output path should be outside protected fMRI lanes, for example:

- `results/sft_semantic_hub_logit_lens/`

Primary probes:

- Concept label anchor: token(s) for the current concept.
- Close-neighbor anchor: S*-close neighbor label.
- Far-neighbor anchor: S*-far neighbor label.
- Surface anchors: `A`, `B`, `yes`, `no`, list punctuation, and common answer
  preambles.
- Distractor anchors: random concepts matched for token count/frequency.

Metrics:

- `concept_margin = logit(concept) - max(logit(close), logit(far), random)`.
- `close_minus_far = logit(close_neighbor) - logit(far_neighbor)`.
- `semantic_vs_surface = max(logit(concept), logit(correct_neighbor)) - max(logit(A/B/yes/no))`.
- Layer of peak semantic margin.
- Final-layer surface dominance as a sanity check, not as a success metric.

Implementation requirements:

- Apply the same final normalization used by the model before the LM head; also
  report a raw-unembedding sensitivity if useful.
- Keep a primary single-token anchor set to mirror Wu et al.'s caution about
  logit-lens tokenization.
- Report retained concept count after tokenization filters.
- For multi-token concepts, use a secondary summed-logit or first-token analysis
  clearly labeled as less clean.

Pass signal:

- Aligned arms show stronger middle-layer semantic margins than base/scrambled.
- Semantic margins peak before final layers; surface tokens dominate near final
  verbalization layers.

Fail signal:

- Semantic margins appear only in final layers or are smaller than answer-token
  margins throughout.
- Improvements are driven by high-frequency concept labels or tokenizer
  artifacts.

### A3. Formal/Symbolic S* Spoke

Purpose: approximate the semantic-hub paper's formal-semantics/code analyses
without claiming a new modality.

Add a symbolic prompt format such as:

```text
concept(c); close(c, x); far(c, y); sim_close(s); sim_far(t)
```

Controls:

- Swap close/far neighbors while preserving lexical overlap.
- Shuffle predicate order.
- Replace concept with category-matched nonmatching concept.
- Keep punctuation and token count matched as much as practical.

Metrics:

- Same relative similarity metrics from A1.
- Symbolic-to-natural retrieval top-1/top-5.
- Symbolic swapped-control margin.
- Logit-lens role anchors: `concept`, `close`, `far`, possibly `similar`.

Pass signal:

- Symbolic/natural same-concept states beat swapped and shuffled controls in
  middle layers.

Fail signal:

- Symbolic string only matches natural prompts because lexical labels overlap;
  swapped controls perform equally well.

### A4. Causal Cross-Spoke Interventions

New output path:

- `results/sft_semantic_hub_interventions/`

Avoid free-generation activation steering as the primary test. Prior local
diagnostics show broad steering can collapse generation. Use logprob-scored,
targeted patching instead.

Interventions:

1. Same-concept clean patch:
   - Run a triplet prompt for concept `c`.
   - Patch the candidate hub-layer last-token state with the feature-listing or
     symbolic-spoke state for the same concept.
   - Score the correct triplet answer logprob.

2. Clean/corrupt recovery:
   - Corrupt concept `c` to category-matched `c_prime`.
   - Patch clean source-format activations for `c`.
   - Measure recovery of `c`-consistent answer logits.

3. Concept transplant:
   - Build paired triplets where option 1 is close to concept A and option 2 is
     close to concept B.
   - Patch A prompt states with B source-spoke states.
   - Measure answer-logit shift from A-compatible to B-compatible option.

4. Directional intervention:
   - Estimate a concept direction `h(B) - h(A)` from source spokes.
   - Add it at candidate layers during target-spoke scoring.
   - Use small alpha grid and logprob-only scoring before any generation.

Controls:

- Wrong concept.
- Wrong layer.
- Random vector with matched norm.
- Same concept but early/final layers.
- Patch after permuting concepts.
- Patch source and target from the same format to check whether cross-format is
  special.

Pass signal:

- Layer-localized causal effects in candidate hub layers.
- Same-concept patches beat wrong-concept patches.
- Aligned arms transfer more strongly than base, and `scrambled` is weaker or
  nonspecific.

Fail signal:

- Patching has no effect.
- Patching works equally in wrong layers or with random vectors.
- Patching improves all arms including `scrambled`, suggesting generic
  activation scale/anisotropy rather than semantic hub use.

## Track B: Huth/LeBel Voxelwise Encoding

### B1. Smoke Encoding

Use the existing active lane rather than duplicating it. The current source of
truth is `results/sft_huth_lebel/ENCODING_PLAN.md`, but this plan records the
interpretive rules.

Core scripts:

- GPU features: `src/sft/huth_lebel_extract_word_states.py`
- CPU ridge: `src/sft/huth_lebel_smoke_encoding.py`

Smoke setup:

- Dataset root on CHTC: `/staging/s/suresh27/datasets/ds003020-smoke`
- Stories: `sweetaspie`, `againstthewind`, `wheretheressmoke`
- Subjects: `UTS01`, `UTS02`, `UTS03`
- Arms: `base`, `lowLR`, `scrambled`, `taskvec_a0p25`
- Layers: currently `16,24,32`
- Held-out story: `wheretheressmoke`
- FIR delays: `1,2,3,4` TRs, corresponding to 2/4/6/8 seconds

Smoke pass criteria:

- Feature NPZs exist for each arm/story with finite states and matching word
  tables.
- CPU ridge writes `summary.csv`, `alpha_cv.csv`, and `run_metadata.json`.
- Held-out Pearson `r` is finite for each arm/layer/subject.
- `scrambled` does not behave identically to aligned arms.

Smoke fail criteria:

- Adapter paths silently fall back to base.
- Word tables and hidden-state word counts disagree.
- Held-out story is used for alpha, layer, or preprocessing decisions.
- Feature scale/norm differences dominate ridge performance.

### B2. Full Huth/LeBel Encoding

Scientific design:

1. Use plain narrative transcripts, not chat templates.
2. Extract per-word hidden states from identical contexts for all arms.
3. Lanczos-align word states to TRs using TextGrid word times.
4. Add identical FIR delays.
5. Fit voxelwise ridge models per subject, arm, layer, and region/voxel.
6. Select alpha and any layer choices using only training/validation stories.
7. Score held-out stories with Pearson `r`, optionally ceiling-normalized when
   reliability estimates are available.
8. Report paired deltas against base.

Arms:

- Required: `base`, `scrambled`, `lowLR`, `taskvec_a0p25`.
- Add `lowrank` after smoke if compute permits, because it is strong in local
  semantic-hub metrics.
- Do not compare to an unrelated model family in the primary claim; the
  scientific strength is same-backbone causal manipulation.

Primary comparisons:

- `lowLR - base`
- `taskvec_a0p25 - base`
- `lowrank - base` if included
- `scrambled - base`
- aligned arm minus `scrambled`

Regions:

- Whole-cortex or available cortical mask: engineering smoke metric.
- Huth semantic-map regions: continuity with semantic mapping, exploratory.
- Auditory cortex: negative/control region for semantic alignment.
- Fedorenko language fROIs: primary language-network result only if
  independently localized.
- Atlas Language/ATL: exploratory if no fROI localizer exists.

Controls:

- Word rate and phoneme/acoustic baselines where available.
- English1000 or other known semantic baseline from the dataset.
- Feature norm/anisotropy normalization before ridge.
- Equal feature dimensionality or PCA controls if comparing layer sets with
  different effective rank.
- Perplexity or next-word log-likelihood control: encoding gains should not be
  reducible to generic LM quality changes.
- Same alpha grid, context length, tokenizer, BOS/reset convention, trim, and
  voxel mask for every arm.
- Blockwise permutation or story-level bootstrap for uncertainty.

Pass signal:

- Aligned arms improve held-out voxelwise prediction over base and scrambled.
- Gains are concentrated in semantic/language-like regions, not auditory-only
  regions.
- The effect survives nested alpha/layer selection.
- The same hub metrics from Track A predict Huth encoding deltas across
  arms/layers.

Fail signal:

- No aligned arm beats base after matched model selection.
- `scrambled` improves as much as aligned arms.
- Gains vanish after feature normalization or PCA.
- Gains appear only in regions dominated by acoustics or word rate.

## Track C: Fedorenko-Style Language-Network Test

### C1. fROI Availability Audit

Before making any Fedorenko claim, audit whether we have:

- subject-specific language localizer contrasts, ideally `sentences > nonword lists`;
- probabilistic language-network masks such as LanA only as a fallback;
- multiple-demand/task-control masks;
- auditory and visual/perceptual control masks;
- story-level held-out encoding data for the same subjects.

If fROIs are unavailable, the report language must say:

```text
Exploratory atlas/broad language-region result, not a Fedorenko-style
subject-specific language-network result.
```

### C2. Encoding In Language fROIs

If fROIs are available:

1. Fit Huth-style voxelwise encoding as in Track B.
2. Aggregate held-out prediction within each subject's language fROIs.
3. Compare aligned arms to base and scrambled using paired subject/fROI deltas.
4. Compare language fROIs against MD, auditory, visual, and semantic-map
   controls.
5. Keep layer selection nested inside training data or use pre-registered layer
   bands from the smoke run.

Pass signal:

- Aligned arm improves held-out `r` in language fROIs.
- Improvement is larger than in MD/auditory/visual controls.
- `scrambled` does not reproduce the effect.

Fail signal:

- Broad improvements appear everywhere, including controls.
- Language fROI gains are smaller than auditory or MD gains.
- Layer selection on held-out data is needed to see the result.

### C3. Optional Model-Internal Language Localizer

This is a companion mechanistic test, not a substitute for fMRI fROIs.

Design:

- Build text stimuli analogous to `sentences > nonword lists` and matched
  nonlinguistic controls.
- Identify units/layers with selective activation differences.
- Test overlap with the semantic-hub band from Track A.
- Ablate or patch localized units and measure:
  - language/coherence tasks;
  - nonlanguage reasoning/control tasks;
  - semantic similarity tasks;
  - benchmark calibration slices.

Pass signal:

- Language-selective units overlap partly with hub layers and causally support
  language/coherence behavior, while controls are less affected.

Fail signal:

- Localized units are just high-variance MLP dimensions or generic answer-format
  units.

## Track D: Brain-Hub Integration

### D1. Pre-Register The Integration Before Full Runs

Primary integration table:

```text
arm x layer x metric
```

Columns:

- semantic-hub metrics:
  - cross-format RDM/CKA;
  - `same_minus_random`;
  - `same_minus_close_neighbor`;
  - logit-lens semantic margins;
  - causal patch effect sizes;
- Huth encoding metrics:
  - held-out voxelwise `r` by subject/region/layer;
  - aligned-minus-base deltas;
  - aligned-minus-scrambled deltas;
- behavioral metrics:
  - generation coherence;
  - human triplet R2;
  - external similarity;
  - ARC/WiC/MMLU/TruthfulQA calibration slices.

Primary questions:

1. Do hub metrics predict semantic behavior gains?
2. Do hub metrics predict Huth/Fedorenko encoding gains?
3. Do hub metrics also predict benchmark damage?
4. Does causal patch effect explain more variance than descriptive RDM
   similarity?

### D2. Mediation Logic

The desired causal story is:

```text
coherence-SFT -> stronger middle-layer format-invariant hub -> better semantic behavior and brain encoding
```

Evidence needed:

- intervention changes hub metrics;
- hub metrics change behavior and/or held-out encoding;
- scrambled/generic perturbation controls do not show the same path;
- benchmark damage is either separable from hub metrics or explicitly part of
  the tradeoff.

If hub metrics correlate with MMLU/WiC/TruthfulQA damage more strongly than with
Huth/Fedorenko gains, the interpretation changes:

```text
coherence-SFT induces semantic smoothing that improves concept similarity but
damages sharp answer-ranking or sense-boundary tasks.
```

That outcome is still scientifically useful, but it is not a clean semantic-hub
success.

## Statistics And Reporting Rules

- Always report layer curves, not only best layers.
- Use fixed mid-layer bands for primary summaries when possible. Existing local
  reports use `10:20`; Huth smoke currently samples `16,24,32`.
- If selecting a best layer, select it on training/validation data or
  leave-one-subject/story out.
- Use paired deltas against base for every subject/region/fold.
- Bootstrap concepts for concept-RDM metrics.
- Use story/block bootstrap or permutation for Huth encoding.
- Correct for many regions/layers or clearly mark exploratory analyses.
- Keep absolute fMRI `r` values and deltas; small deltas can be statistically
  fragile even when directionally consistent.
- Do not claim "language network" unless fROIs are independently localized.
- Do not claim "ATL semantic hub" unless the data and masks justify ATL-specific
  interpretation and controls rule out generic object/visual geometry.

## Trackable Task Table

| ID | Task | Status | Owner Boundary | Output |
|---|---|---|---|---|
| A1 | Existing cross-format hub RDM/CKA/retrieval | Done | Local reports | `results/sft_semantic_hub/REPORT.md` |
| A2 | Existing paper-style same-vs-mismatch similarity | Done | Local reports | `results/sft_semantic_hub_paper/REPORT.md` |
| A3 | Add stricter lexical/token/prompt controls to similarity | Next | New results path, no Huth lane edits | `results/sft_semantic_hub_paper_controls/REPORT.md` |
| A4 | Logit-lens semantic anchoring | Next | New results path | `results/sft_semantic_hub_logit_lens/REPORT.md` |
| A5 | Symbolic S* spoke with swap/shuffle controls | Next | Hidden-state extraction plus new report | `results/sft_semantic_hub_symbolic/REPORT.md` |
| A6 | Causal cross-spoke patching | Next | New results path | `results/sft_semantic_hub_interventions/REPORT.md` |
| B1 | Huth/LeBel GPU extraction smoke | In progress elsewhere | Hubble owns active CHTC lane | `results/sft_huth_lebel/*` |
| B2 | Huth/LeBel CPU ridge smoke | Blocked on B1 completion | Hubble owns active CHTC lane | `smoke_encoding/summary.csv` |
| B3 | Full Huth/LeBel high-data encoding | Pending smoke pass | Coordinate before running | high-data encoding report |
| C1 | fROI availability audit | Next after Huth smoke | Read-only audit first | fROI audit memo/report |
| C2 | Fedorenko fROI encoding comparison | Blocked on fROIs | Only if localizers/masks exist | fROI encoding report |
| D1 | Brain-hub-behavior integration preregistration | Next | New standalone report or results path | integration schema |
| D2 | Hub metrics vs benchmark damage | Next | Use existing wide-bench diagnostics | joined tradeoff report |

## Decision Criteria

### Strong Success

All of the following hold:

- Aligned arms beat base and scrambled on same-vs-close-neighbor hub metrics in
  middle layers.
- Logit-lens semantic anchors peak in middle layers before final surface-token
  dominance.
- Cross-spoke causal patching transfers concept information in aligned arms more
  than in base/scrambled.
- Huth/LeBel held-out encoding improves for aligned arms over base and
  scrambled under nested selection.
- Fedorenko-style fROI gains, if tested, are larger in language fROIs than in
  auditory/visual/MD controls.
- Hub metrics predict semantic behavior and brain-encoding gains without simply
  tracking benchmark damage.

Interpretation:

```text
Coherence-SFT tightens a functional, middle-layer, format-invariant semantic
representation that is useful for behavior and brain encoding.
```

### Partial Success

Likely useful outcomes:

- Same-vs-random and same-vs-far improve, but same-vs-close is weak.
  Interpretation: semantic smoothing or category-level hub, not exact concept
  identity.
- Internal hub metrics improve, but Huth encoding is flat.
  Interpretation: hidden-state coherence does not transfer to natural-language
  brain predictivity, or the Huth features/regions are not sensitive to this
  concept-level intervention.
- Huth encoding improves but scrambled also improves.
  Interpretation: feature geometry or adapter perturbation, not semantic
  alignment, may be responsible.
- Fedorenko fROI unavailable.
  Interpretation: only Huth-style or exploratory atlas language claims are
  possible.

### Failure

Any of the following should stop escalation to larger runs:

- The hub effect disappears under token/prompt/lexical controls.
- `scrambled` matches aligned arms on hub and encoding metrics.
- Causal patching does not work or works equally for wrong concepts/layers.
- Huth smoke cannot produce finite held-out `r`.
- All apparent encoding gains are due to feature norm, dimensionality, or layer
  selection on held-out data.
- Hub metrics correlate mainly with WiC/MMLU/TruthfulQA damage and not with
  brain/semantic gains.

## Recommended Next Checkpoint

Do not launch a larger Huth/LeBel run from this plan until the active smoke lane
finishes and is inspected. The most useful independent next report is:

```text
results/sft_semantic_hub_logit_lens/REPORT.md
```

Minimum contents:

- arms: `base`, `scrambled`, `lowLR`, `lowrank`, `taskvec_a0p25`, `taskvec_a0p5`;
- anchors: concept, close neighbor, far neighbor, random, surface answer tokens;
- tokenization retention table;
- layer curves for semantic margins and surface margins;
- explicit middle-layer pass/fail read;
- join against existing `same_minus_close_neighbor` and benchmark deltas.

Reason: logit-lens anchoring is the cleanest missing piece from Wu et al. that
does not depend on the active Huth/LeBel CHTC lane.

## References

- Wu, Yu, Yogatama, Lu, and Kim. "The Semantic Hub Hypothesis: Language Models
  Share Semantic Representations Across Languages and Modalities." arXiv:2411.04986.
  https://arxiv.org/abs/2411.04986
- Semantic-hub paper HTML. https://ar5iv.org/html/2411.04986v3
- Huth et al. "Natural speech reveals the semantic maps that tile human cerebral
  cortex." Nature 2016. https://www.nature.com/articles/nature17637
- LeBel et al. "A natural language fMRI dataset for voxelwise encoding models."
  Scientific Data 2023. https://www.nature.com/articles/s41597-023-02437-z
- Antonello, Vaidya, and Huth. "Scaling laws for language encoding models in
  fMRI." arXiv:2305.11863. https://arxiv.org/abs/2305.11863
- Fedorenko, Ivanova, and Regev. "The language network as a natural kind within
  the broader landscape of the human brain." Nature Reviews Neuroscience 2024.
  https://www.nature.com/articles/s41583-024-00802-4
- Fedorenko, Piantadosi, and Gibson. "Language is primarily a tool for
  communication rather than thought." Nature 2024.
  https://www.nature.com/articles/s41586-024-07522-w
- AlKhamissi, Tuckute, Bosselut, and Schrimpf. "The LLM Language Network: A
  Neuroscientific Approach for Identifying Causally Task-Relevant Units."
  arXiv:2411.02280. https://arxiv.org/abs/2411.02280
