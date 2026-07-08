# SFT-fMRI Audit Report

Created: 2026-07-08

## Status

The local THINGS-fMRI data audit is complete. No model inference or RSA has been
run yet.

## Inputs

- SFT concepts: `data/scale128/concepts.csv`
- THINGS-fMRI betas:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/`
- Audit script: `src/sft/fmri_audit.py`

## Outputs

- `results/sft_fmri/concept_overlap.csv`
- `results/sft_fmri/fmri_data_audit.json`

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

## Interpretation

The first-pass THINGS-fMRI RSA is data-unblocked. The primary analysis should use
the 90 exact overlaps between the held-out SFT concept set and the 720 THINGS-fMRI
concepts. Ventral Visual should be the primary ROI because prior object-fMRI
noise ceilings were strongest there; Language and ATL analyses should be
exploratory unless later language-fMRI/localizer data support stronger claims.

## Next Step

Extract all-layer Llama hidden states for the 90 primary concepts across the
planned model arms, then compute layer-resolved cosine-RDM Spearman RSA against
the audited fMRI concept betas.
