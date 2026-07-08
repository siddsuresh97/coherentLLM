# CODEX TASK 9: fMRI Brain Predictivity + Semantic Hub

Created: 2026-07-08

## Objective

Test whether the coherence-SFT intervention makes Llama-3.1-8B representations
more predictive of human brain data, and whether any gain is mediated by a
format-invariant hidden-state semantic hub.

This task has three connected tracks:

1. THINGS-fMRI object-concept RSA, runnable now with existing local Ch4 machinery.
2. Hidden-state semantic-hub analysis over triplet/pairwise/feature prompts.
3. Language-fMRI feasibility for Huth/LeBel/Fedorenko-style encoding.

## Why this is worth doing

Tasks 1-8 established a controlled semantic intervention:

- Base: `Llama-3.1-8B-Instruct`
- Aligned adapters: `real`, `lowLR`, `lowrank`
- Control: `scrambled`
- Dose response: task-vector alpha sweep `a0p25`, `a0p5`, `a1.0`

The key causal question is whether moving the model along a human-aligned concept
geometry direction also improves neural predictivity. The key mechanistic question
is whether cross-format coherence comes from a shared mid-layer semantic code.

## Primary local assets

Coherence-SFT repo:

- Concepts: `data/scale128/concepts.csv`
- Adapters:
  - `out/adapters_vllm_fixed/real`
  - `out/adapters_vllm_fixed/scrambled`
  - `out/adapters_mitigation/lowLR`
  - `out/adapters_mitigation/lowrank`
  - `out/adapters_taskvec_scaled/a0p25`
  - `out/adapters_taskvec_scaled/a0p5`
- LoRA application helper: `src/sft/lora_apply.py`
- Existing actdiff extractor pattern: `src/sft/extract_coherence_vector.py`

Sibling Ch4 fMRI assets:

- fMRI scripts:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/`
- THINGS betas:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/`
- Aggregation script:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/dissertation/scripts/compute_fmri_rsa.py`

## Track A: THINGS-fMRI RSA

### Concept set

Primary analysis:

- Use only exact overlaps between `data/scale128/concepts.csv` and the 720
  THINGS-fMRI concepts.
- Current exact overlap count: 90.
- This is the clean held-out analysis because these concepts were evaluation
  concepts for SFT.

Secondary/sensitivity analysis:

- Optional full-720 THINGS-fMRI concept run, clearly labeled as not leakage-clean
  unless we retrain with an fMRI-aware holdout.

### Model arms

Minimum:

- `base`
- `scrambled`
- `lowLR`
- `lowrank`
- `taskvec_a0p25`
- `taskvec_a0p5`
- `taskvec_a1p0` if represented by the real adapter or copied from `real`

Do not use activation steering arms for fMRI; Task 8 showed they collapse
generation and are not the viable intervention.

### Representation extraction

For each model arm and concept:

- Prompt: neutral concept prompt first, e.g. `Concept: {concept}` or a minimal
  chat prompt. Keep one primary template fixed.
- Save all-layer last-token hidden states.
- Store as `results/sft_fmri/hidden_states/{arm}.npz` with arrays:
  - `concepts`
  - `layers`
  - `hidden` shaped `(n_concepts, n_layers, hidden_dim)`
  - `prompt_template`
  - adapter metadata / SHA where applicable

Sensitivity:

- Repeat with triplet/pairwise/feature prompts and average across formats only
  after the semantic-hub analysis says that is valid.

### Brain RSA

Reuse the Ch4 logic:

- Average single-trial betas by concept per subject.
- Z-score voxels across concepts.
- Build fMRI RDMs using cosine distance.
- Build model RDMs per layer using cosine distance.
- Compare upper triangles with Spearman RSA.

Regions:

- Primary: Ventral Visual.
- Control: Early Visual.
- Exploratory: Dorsal Visual, Language, Prefrontal, whole brain.
- Exploratory semantic bridge: ATL subregions individually, not only pooled ATL.

Reason for ROI priority:

- Old object-fMRI noise ceiling is high in Ventral Visual and weak/near-zero in
  pooled Language/ATL. Do not frame ATL/Language object-RSA as the primary success
  criterion.

### Statistics and figures

Outputs:

- `results/sft_fmri/concept_overlap.csv`
- `results/sft_fmri/fmri_data_audit.json`
- `results/sft_fmri/rsa_by_layer.csv`
- `results/sft_fmri/rsa_best_layer.csv`
- `results/sft_fmri/figures/layer_curves_by_roi.png`
- `results/sft_fmri/figures/alpha_dose_response.png`
- `results/sft_fmri/figures/atl_subregions.png`

Primary comparisons:

- `lowLR - base` in Ventral Visual.
- `lowrank - base` in Ventral Visual.
- `scrambled - base` should be flat or worse.
- Task-vector alpha monotonic trend in Ventral Visual.
- Early Visual should not show the same semantic-dose effect.

Layer selection:

- Report full layer curves.
- Avoid claiming a best-layer win unless layer selection is held out or clearly
  marked descriptive.
- Preferred: leave-one-subject-out best-layer selection for model/ROI comparisons.
- Also report a fixed mid/late layer band if layer curves are noisy.

Uncertainty:

- Subject-level paired deltas.
- Bootstrap concepts for confidence intervals.
- Permute model concept labels as a null sanity check.

## Track B: Semantic Hub

### Hypothesis

Coherence-SFT should tighten a format-invariant concept code in mid/late layers.
The behavioral outcome is cross-format agreement; the representational mechanism
should be that triplet, pairwise, and feature prompts route through a shared
concept representation.

### Data

Use the 128 held-out concepts and the same model arms as Track A.

For each concept and format:

- triplet prompt
- pairwise prompt
- feature/listing prompt

Capture all-layer last-token hidden states.

### Metrics

Layer-resolved:

- RSA/CKA between concept RDMs from different formats.
- Same-concept cross-format nearest-neighbor accuracy.
- Concept-vs-format decoding: concept identity should become more recoverable
  than format in the putative hub band.
- Hub score: average cross-format RDM correlation minus within-format nuisance
  controls.

Predictions:

- `lowLR` / `lowrank` increase mid-layer cross-format convergence vs `base`.
- `scrambled` does not.
- Task-vector alpha gives a dose-response.
- Peak should be mid/late layers, not embeddings or final logits only.

### Bridge to fMRI

If a model has a hub layer/band:

- Average or align the three format-specific concept codes into a hub code.
- RSA that hub code against THINGS-fMRI RDMs.
- Test whether hub score predicts fMRI RSA across arms/layers.

This is the clean synthesis:

`coherence-SFT -> stronger format-invariant semantic hub -> better object-fMRI RSA`

## Track C: Language-fMRI feasibility

Do not launch a heavy language-fMRI run until a feasibility memo is written.

Questions:

- Which dataset is most practical: LeBel/Huth, Narratives, Pereira, or another
  OpenNeuro language dataset?
- Are transcripts, word timings, TRs, preprocessed BOLD, and masks available
  locally or downloadable?
- What exact feature alignment is needed: token hidden states, word aggregation,
  HRF delays, ridge/banded ridge, held-out story scoring?
- Can we use Fedorenko/Ivanova language-network masks or must we approximate
  with atlas ROIs?

Desired design:

- Extract per-token hidden states per model/layer.
- Align to TRs with delays/HRF.
- Voxelwise ridge encoding.
- Score held-out prediction correlation in language-network voxels.
- Compare `base`, `lowLR`, `scrambled`, and task-vector alpha.

## Initial implementation order

1. Create audit script for exact concept overlap, fMRI file presence, subject
   trial counts, and ROI voxel counts.
2. Implement hidden-state extraction for Track A concept prompts.
3. Implement RSA over saved hidden states and existing fMRI betas.
4. Add semantic-hub prompt extraction and metrics.
5. Integrate sidecar literature memos and decide whether to launch language-fMRI.

## Completion criteria for first solid result

- `results/sft_fmri/concept_overlap.csv` and `fmri_data_audit.json` exist.
- Hidden states are extracted for at least `base`, `lowLR`, and `scrambled`.
- THINGS-fMRI RSA runs end-to-end for those arms.
- `results/sft_fmri/REPORT.md` states whether the first-pass Ventral Visual
  result is positive, null, or inconclusive.
- The result is committed and pushed.
