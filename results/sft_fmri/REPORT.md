# SFT-fMRI Audit Report

Created: 2026-07-08

## Status

The local THINGS-fMRI data audit, hidden-state extraction, and first-pass
object-concept RSA are complete for the 90 exact SFT/THINGS-fMRI overlap
concepts.

## Inputs

- SFT concepts: `data/scale128/concepts.csv`
- THINGS-fMRI betas:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/`
- Audit script: `src/sft/fmri_audit.py`

## Outputs

- `results/sft_fmri/concept_overlap.csv`
- `results/sft_fmri/fmri_data_audit.json`
- `results/sft_fmri/hidden_states/{arm}.npz`
- `results/sft_fmri/rsa_by_layer.csv`
- `results/sft_fmri/rsa_best_layer.csv`
- `results/sft_fmri/rsa_summary.csv`
- `results/sft_fmri/rsa_meta.json`

## Main Counts

| Quantity | Value |
|---|---:|
| SFT held-out concepts | 128 |
| THINGS-fMRI concepts common across subjects | 720 |
| Exact held-out overlap for primary RSA | 90 |
| Subjects | 3 |
| Trials per subject | 9,840 |

## Subject Data

| Subject | Response shape | Overlap trial count range | Mean overlap trials |
|---|---:|---:|---:|
| sub-01 | 211,339 voxels x 9,840 trials | 12-24 | 13.867 |
| sub-02 | 226,950 voxels x 9,840 trials | 12-24 | 13.867 |
| sub-03 | 189,164 voxels x 9,840 trials | 12-24 | 13.867 |

No primary-overlap concept is missing trials in any subject.

## ROI Voxel Counts

| Subject | Early Visual | Ventral Visual | Dorsal Visual | Language | ATL | Prefrontal |
|---|---:|---:|---:|---:|---:|---:|
| sub-01 | 6,507 | 4,404 | 2,173 | 6,784 | 8,204 | 8,381 |
| sub-02 | 7,321 | 4,635 | 1,879 | 6,306 | 8,643 | 8,492 |
| sub-03 | 5,942 | 3,790 | 1,865 | 5,962 | 7,612 | 8,150 |

## Hidden States

Extraction script: `src/sft/extract_fmri_hidden_states.py`.

Prompt template: `concept_colon`, i.e. `Concept: {concept}`.

Each arm has hidden states shaped `90 concepts x 33 layers x 4096 dim`.

Arms extracted:

- `base`
- `scrambled`
- `lowLR`
- `lowrank`
- `taskvec_a0p25`
- `taskvec_a0p5`
- `taskvec_a1p0`

## First-Pass RSA

RSA script: `src/sft/run_fmri_rsa.py`.

Method:

- Average trial betas by concept per subject.
- Z-score voxels across concepts.
- Build cosine-distance RDMs for brain ROIs and model layers.
- Compare upper triangles with Spearman RSA.
- Select descriptive best layers by highest mean RSA across the three subjects.

Core ROI summary:

| ROI | Best arm | Best layer | Mean RSA | Delta vs base | Interpretation |
|---|---|---:|---:|---:|---|
| Ventral Visual | `lowrank` | 6 | 0.2013 | +0.0023 | Primary object ROI; effectively flat vs base, scrambled much lower |
| Early Visual | `lowLR` | 30 | 0.0682 | +0.0003 | Flat vs base |
| Dorsal Visual | `base` | 26 | 0.0632 | 0.0000 | No aligned-arm gain |
| ATL (Semantic) | `taskvec_a0p5` | 30 | 0.0543 | +0.0062 | Small exploratory increase, high subject noise |
| Language | `lowrank` | 32 | 0.0345 | +0.0061 | Small exploratory increase, not a language-localizer result |
| Prefrontal | `taskvec_a1p0` | 31 | 0.0196 | +0.0077 | Very low absolute RSA |

Ventral Visual is the strongest and cleanest signal:

- `base`: 0.1990
- `lowLR`: 0.2010
- `lowrank`: 0.2013
- `taskvec_a0p25`: 0.1968
- `taskvec_a0p5`: 0.1973
- `taskvec_a1p0`: 0.1856
- `scrambled`: 0.1293

## Interpretation

The first-pass THINGS-fMRI RSA is an end-to-end positive pipeline check, but not
yet strong evidence that coherence-SFT improves neural predictivity. The clearest
object-fMRI structure is Ventral Visual: all meaningful model arms beat the
scrambled control, but aligned arms are nearly flat relative to base. ATL and
Language show small descriptive increases for some aligned/task-vector arms, but
these are exploratory because this is object-fMRI, not a language-localizer or
language-encoding dataset.

The current result should be framed as:

- reliable object-RSA signal exists;
- scrambled control behaves as expected;
- coherence-SFT does not produce a large Ventral Visual gain on this first
  prompt/90-concept analysis;
- semantic/language claims require either semantic-hub evidence or a real
  language-fMRI encoding run.

Speed note: for this 90-concept RSA, the bottleneck was HDF5 trial loading rather
than RSA math. `run_fmri_rsa.py` now batches selected HDF5 trial reads per subject
instead of reading per concept. A GPU or ThingsVision-style RSA path may matter
for a full-720 or bootstrap-heavy run, but it is not the limiting factor for this
first pass.

## Next Step

Add the semantic-hub extraction/metrics over triplet, pairwise, and feature
formats, then test whether hub score predicts fMRI RSA across arms/layers. For a
stronger brain claim, run a separate language-fMRI feasibility pass using
held-out encoding in subject-localized language ROIs.
