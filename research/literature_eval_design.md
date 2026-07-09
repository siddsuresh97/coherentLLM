# Literature and Evaluation Design Lane

Created: 2026-07-09
Branch: `coherence-sft`
Owner scope: this file plus `research/lit_notes/`

## Current Read

The clean scientific target is not "coherence-SFT is brain-like" as a single
claim. The tractable target is a ladder:

1. show that the SFT/task-vector arms strengthen a format-invariant concept
   representation inside the model;
2. test whether that representation is specific to exact concept identity, not
   just broad semantic smoothing;
3. test whether the representation is causally used by steering/patching;
4. test whether the same arm/layer choices improve held-out brain encoding or
   cross-modal concept RSA, while controlling for benchmark damage.

I am treating "MEMP-style" as a local shorthand for matched/equivalent-meaning
probes: the same concept is presented through different spokes, and every
matched comparison is paired with random, semantically close, lexical-overlap,
and surface-form controls. I did not find a standard semantic-hub method paper
that uses "MEMP" as the acronym.

Live compute state at this checkpoint:

- Concept-vector CHTC smoke `5513276` and bounded sweep `5513297` passed.
- Huth extraction retry `5513306` passed and returned all 12 arm/story feature
  NPZs; CPU encoding cluster `5513337` has left the queue and should be pulled
  by the Huth lane owner.
- MMLU cache retries still have two GPU jobs running: `5513309.1` and
  `5513313.0`; no held or idle CHTC jobs were visible in the last poll.

## Source Synthesis

### Semantic Hub: Wu et al. arXiv:2411.04986

Wu, Yu, Yogatama, Lu, and Kim define a model semantic hub as a shared
intermediate representation space where semantically equivalent inputs from
different languages, notations, or modalities become close. Their tests have
three parts:

- Relative similarity: matched inputs should be closer than mismatched inputs,
  with the strongest effects in intermediate layers.
- Logit-lens anchoring: intermediate hidden states can be interpreted through
  the model's dominant language tokens before final layers project back to the
  required output form.
- Intervention: changing the shared intermediate representation changes output
  behavior, so the hub is used rather than merely observed.

Primary source: https://arxiv.org/abs/2411.04986

Project implication: our current prompt-format hub is a valid text-only analogue
if we state it narrowly. It does not become a true cross-modal or cross-language
hub until we add language, vision, audio, or fMRI concept spokes.

### Huth / LeBel: Voxelwise Encoding From Narratives

Huth et al. map semantic selectivity by fitting voxelwise encoding models to
fMRI from subjects listening to narrative stories. The decisive metric is
held-out BOLD prediction, not visual map inspection.

Primary sources:

- Huth et al. 2016 semantic maps: https://www.nature.com/articles/nature17637
- LeBel/Huth natural-language fMRI dataset: https://www.nature.com/articles/s41597-023-02437-z
- Scaling laws for language encoding models in fMRI: https://arxiv.org/abs/2305.11863

Project implication: the strongest Huth-style test is same-backbone and
same-dataset. Extract word-level states from `base`, `lowLR`, `lowrank`,
`taskvec_a0p25`, and `scrambled`; align them to TRs; fit identical ridge
encoding models; compare held-out Pearson `r` deltas against `base`.

### Fedorenko / EvLab: Language Network Discipline

Fedorenko-style claims should be tied to subject-specific functional ROIs when
possible, especially a `sentences > nonword lists` language localizer. Atlas
regions and broad masks are useful for exploration, but they should not be
described as decisive language-network evidence.

Two constraints matter for our interpretation:

- Language, semantic/conceptual memory, and general reasoning are dissociable.
- Benchmark drops on MMLU/ARC/TruthfulQA can coexist with better language or
  semantic geometry if the intervention damages calibration or reasoning
  surfaces rather than language-network-like representations.

Primary sources:

- Formal vs functional competence in LLMs:
  https://arxiv.org/abs/2301.06627
- Model/brain concept alignment using semantic consistency:
  https://arxiv.org/abs/2508.11536
- Model-internal language-network localization:
  https://arxiv.org/abs/2411.02280

Project implication: every result should be labeled as one of: formal language,
semantic concept representation, reasoning/calibration, or cross-modal concept
alignment. Do not collapse these into one "coherence" construct.

### Pereira / Ryskina / Fedorenko: Cross-Modal Concepts

Pereira et al.'s 180-concept dataset and the later Ryskina/Tuckute/Fedorenko
analysis are the most direct bridge from semantic-hub thinking to brain data.
The key move is semantic consistency: a voxel/region is concept-relevant if it
responds consistently to the same concept across sentences, word clouds, and
pictures.

Primary sources:

- Pereira et al. universal decoder:
  https://www.nature.com/articles/s41467-018-03068-4
- Ryskina et al. concept consistency / LM-brain alignment:
  https://arxiv.org/abs/2508.11536

Project implication: this gives us a sharper fMRI target than generic object
RSA. We can ask whether coherence-SFT makes LM representations align better
with semantically consistent brain regions, especially when concept identity is
shared across presentation formats.

## Experiment Ladder

### E1: Paper-Style Semantic Hub, Text-Only

Question: did coherence-SFT strengthen a format-invariant concept hub?

Inputs:

- Existing hidden states under `results/sft_semantic_hub/hidden_states/`.
- Formats: raw concept, triplet, pairwise, feature listing.
- Add a symbolic S* spoke later:
  `concept(c); close(c,x); far(c,y); sim_close(s); sim_far(t)`.

Metrics:

- `same_minus_random`
- `same_minus_close_neighbor`
- `same_minus_far_neighbor`
- top-1/top-5 matched retrieval
- cross-format RSA/RDM Spearman

Controls:

- random mismatches;
- S*-close nonmatching concepts;
- lexical overlap and prompt-length controls;
- scrambled adapter;
- surface-answer controls for `A/B` and `yes/no`.

Decision rule:

- first-pass support: middle-layer `same_minus_random` improves over base;
- strong support: positive `same_minus_close_neighbor` and retrieval gains;
- reject or qualify: gains vanish under close-neighbor or lexical controls.

### E2: Logit-Lens Anchoring

Question: do middle layers represent concept semantics before final layers
collapse to task surface tokens?

Primary comparisons:

- `logit(concept label) - logit(random concept)`;
- `logit(close neighbor) - logit(far neighbor)`;
- `logit(semantic token) - logit(surface token A/B/yes/no)`.

Layer prediction:

- semantic anchors peak in middle layers;
- surface answer tokens dominate near final layers.

Decision rule:

- support: aligned arms improve semantic-token margins without worsening final
  answer calibration;
- risk: aligned arms increase semantic margins but also flatten answer-token
  margins, matching benchmark drops.

### E3: Causal Hub Interventions

Question: is the hub used by the model?

Interventions:

- same-concept spoke substitution: patch triplet hidden state with feature-list
  or symbolic hidden state for the same concept;
- concept transplant: patch concept A prompt with concept B hidden state and
  test whether answer logits move toward B-compatible options;
- activation addition: add `h(B) - h(A)` concept directions.

Metrics:

- change in correct option logit margin;
- change in semantic retrieval;
- retention forced-choice accuracy;
- qualitative generation/judge score for coherence and human alignment.

Decision rule:

- support: layer-localized effects in middle layers, stronger for task-vector
  arms than base and absent/degraded for scrambled;
- caution: broad effects at all layers or large retention losses indicate a
  nonspecific activation-energy/calibration artifact.

### E4: Huth/LeBel Narrative Encoding

Question: does coherence-SFT improve held-out natural-language fMRI prediction?

Design:

- Subjects: start with staged `UTS01`-`UTS03`.
- Stories: smoke uses `sweetaspie`, `againstthewind`, and
  `wheretheressmoke`; scale to high-data subset only after smoke scoring.
- Arms: `base`, `lowLR`, `lowrank`, `taskvec_a0p25`, `scrambled`.
- Layers: use the extracted `16,24,32` smoke grid, then expand if needed.
- Model: voxelwise ridge with FIR delays, train-only alpha/layer selection.

Metrics:

- held-out Pearson `r` by subject, voxel/ROI, story, layer, and arm;
- paired deltas vs `base`;
- bootstrap/permutation confidence intervals;
- noise-ceiling-normalized score where available.

Controls:

- `scrambled` adapter;
- word-rate/phoneme/English1000 baseline if cheap;
- no test-set layer selection;
- auditory and visual controls separated from language/semantic ROIs.

Decision rule:

- support: aligned arm beats `base` and `scrambled` on held-out prediction in
  language/semantic regions with consistent layer choice;
- weak result: only visual/auditory regions improve or gains depend on
  selecting the test story/layer.

### E5: Pereira/Fedorenko Cross-Modal Concept Consistency

Question: do our model states better align with brain regions that represent
concepts across sentence, word-cloud, and picture paradigms?

Design:

- Use Pereira Experiment 1 180 concepts.
- For text-only Llama arms, encode sentence and word-cloud stimuli.
- For picture trials, use labels/captions as a text-only proxy first; treat as
  exploratory until a vision-language model or image embedding spoke is added.
- Compute concept-level LM vectors and brain vectors, then RSA by ROI.

Metrics:

- concept-level RSA Spearman against semantically consistent ROIs;
- ridge encoding from LM states to brain activation;
- correlation between ROI semantic consistency and LM predictivity.

Controls:

- shuffled concept labels;
- word frequency/concreteness;
- language selectivity quartiles;
- visual ROI vs language ROI separation.

Decision rule:

- support: aligned arms improve concept RSA or encoding most in
  high-semantic-consistency regions, not merely in high-language-selectivity
  or visual regions.

### E6: Fedorenko-Style Model-Internal Localization

Question: is the induced skill closer to language-selective units, semantic
units, or general task/reasoning units?

Design:

- Build a model localizer: sentences vs nonword strings, coherent paragraphs vs
  word salad, and semantic statements vs syntax-only controls.
- Select units/layers with high language or semantic selectivity.
- Ablate or steer those units and measure coherence, MMLU, ARC, WiC,
  TruthfulQA, HellaSwag, WinoGrande, and retention probes.

Metrics:

- unit/layer selectivity;
- overlap with concept-vector best layers;
- causal ablation effect by benchmark skill family.

Decision rule:

- language-like effect: language/coherence changes with limited reasoning
  damage;
- semantic-memory effect: concept similarity and Huth/Pereira metrics move;
- calibration/reasoning damage: TruthfulQA/MMLU/ARC drops track answer-margin
  flattening rather than hub gains.

## Benchmark Drop Interpretation

The working hypothesis for TruthfulQA and MMLU drops should be split:

- TruthfulQA likely depends on refusal/uncertainty calibration, misconception
  suppression, and answer likelihood ranking. A coherence-tuned semantic hub may
  make plausible narratives more attractive without improving truth selection.
- MMLU and ARC depend on precise option ranking and domain knowledge retrieval.
  A broad coherence direction can improve semantic association while degrading
  fine-grained discriminative margins.
- WiC is a near-neighbor meaning discrimination task, so weak
  `same_minus_close_neighbor` is a plausible mechanistic warning sign.
- HellaSwag/WinoGrande staying stable would suggest narrative/pragmatic
  continuation skills are less damaged than factual/calibrated selection.

Mitigations to test:

- smaller task-vector alpha and layer-local steering instead of global adapter;
- KL or answer-margin regularization on MMLU/TruthfulQA/ARC probes;
- add hard close-neighbor negatives to SFT data;
- two-vector steering: add coherence/human-alignment direction while subtracting
  the empirically measured calibration-damage direction;
- early stop on `same_minus_close_neighbor` plus benchmark retention, not only
  on broad coherence score.

## Compute Plan

Local-first debug:

- Run CPU-only similarity/RSA joins locally.
- Run one-arm, one-layer GPU smoke locally when a visible local GPU is free.
- Only move to CHTC after the command passes locally or the failure is known to
  be a scale-only problem.

CHTC scaling:

- Use CHTC for independent arm/layer/story shards.
- Keep HF/model caches on worker scratch unless staging access is explicitly
  required.
- Bundle output for large intermediate features when staging quota is risky.
- Prefer downloading inside the job for public datasets only when CHTC outbound
  network and repeated-download cost are acceptable; for large reused model
  weights, use existing staging.

GPU allocation:

- Do not queue new low-priority sweeps while MMLU cache retries are occupying
  the remaining GPU slots unless they are short smoke jobs.
- Huth encoding is CPU ridge after GPU extraction, so it should not block GPU
  concept/hub work once its artifacts are pulled.
- Next GPU-worthy jobs: logit-lens anchoring smoke and causal intervention
  smoke, each small enough to run before broad CHTC sweeps.

## Before Asking Human Subjects

Do not ask new subjects to perform tasks until these are done:

1. Huth/LeBel smoke encoding produces a passed held-out score table.
2. Text-only semantic hub has close-neighbor controls and intervention smoke.
3. Pereira/Fedorenko plan is reduced to a concrete stimulus/ROI/metric table.
4. Benchmark retention report identifies which skill families are harmed.
5. The task script, consent/IRB path, data retention plan, and exclusion rules
   are written.

Potential human experiment only after that:

- behavioral semantic discrimination: same concepts as model close-neighbor
  probes, measuring whether model-induced similarity matches human choices;
- reading/story comprehension probes matched to Huth stories;
- optional fMRI only if there is a preregistered ROI/layer hypothesis and a
  clear reason existing public data cannot answer it.

## Immediate Next Actions

1. Huth lane: pull and score `5513337`; if passed, summarize per-arm held-out
   Pearson `r` and decide whether high-data staging is justified.
2. Semantic hub lane: run logit-lens anchoring on `base`, `taskvec_a0p25`,
   `lowrank`, and `scrambled`.
3. Steering lane: run qualitative/judge probes at `coherence` layer 12 alpha 4,
   `human_alignment` layer 24 alpha 4, and `human_alignment` layer 16 alpha 2.
4. Benchmark lane: join benchmark deltas to semantic-hub metrics, especially
   WiC, TruthfulQA, ARC/OpenBookQA, and MMLU.
5. Literature lane: keep this file as the handoff index; append only when new
   cited results or concrete experiment choices change.

## Handoff Map

- `research/literature_eval_design.md`: synthesis, experiment ladder, and next
  actions.
- `research/lit_notes/semantic_hub_2411_04986.md`: paper-method extraction for
  Wu et al.
- `research/lit_notes/huth_fedorenko_sources.md`: source notes for Huth,
  LeBel, Pereira, Ryskina, and Fedorenko-style evaluation.
- `research/HUTH_FEDORENKO_SEMANTIC_HUB_PLAN.md`: prior broader plan; use as
  background, not as the active owner file for this lane.
- `results/sft_eval/concept_steering/REPORT.md`: current concept-vector sweep.
- `results/sft_huth_lebel/REPORT.md`: active Huth/LeBel CHTC status, owned by
  the Huth lane.
