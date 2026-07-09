# Held-Out fMRI Semantic-Hub Regression

## Scope

This analysis predicts THINGS-fMRI RDM distances from semantic-hub model
RDM predictors. Folds hold out concepts, so train and test distance pairs
do not share concepts. Scores are still descriptive because layers and
arms are selected after looking at test scores.

## Outputs

- `cv_model_scores.csv`: subject x region x arm x layer x model scores.
- `cv_layer_summary.csv`: mean score by region/arm/layer/model.
- `cv_best_layer_summary.csv`: best layer per region/arm/model.
- `cv_midlayer_summary.csv`: mean score over the configured mid-layer band.
- `cv_model_comparison.csv`: mid-layer mean-format vs best single-format comparison.
- `POSITIVE_SIGNAL_REVIEW.md`: post-hoc positive-signal audit with same-layer
  base controls and subject-consistency tables.
- `positive_midlayer_mean_repr.csv`
- `positive_subject_consistency.csv`
- `positive_best_layer_same_base.csv`

## 2026-07-09 Positive-Signal Audit

The strongest current positive brain-alignment lead is the held-out semantic-hub
regression, not the original direct THINGS-fMRI RSA.

In the fixed mid-layer band `10:20`, `mean_repr` held-out Pearson `r` improves
over base for all coherent/non-scrambled arms across all three target regions
(`15/15` arm-region comparisons), while scrambled is below base in all three
regions.

Key task-vector cells:

| Region | Arm | Mean repr r | Delta vs base | Subject consistency |
|---|---|---:|---:|---:|
| Ventral Visual | `taskvec_a0p25` | 0.1679 | +0.0535 | 3/3 |
| ATL (Semantic) | `taskvec_a0p5` | 0.0356 | +0.0324 | 2/3 |
| Language | `taskvec_a0p5` | 0.0511 | +0.0507 | 2/3 |

Read: coherence tuning/task-vector movement appears to make mid-layer semantic
geometry more predictive of held-out object-fMRI RDM distances. This is a
promising representational brain-alignment result, but it remains post-hoc:
direct Ventral Visual RSA is flat versus base, Huth/LeBel smoke is not
base-beating, and `taskvec_a0p5` is not retention-safe. Treat this as the
lead to confirm with fixed metrics and paired subject/concept bootstrap.

## Primary Mid-Layer Comparison

Mid-layer band: `10:20`. Primary metric: held-out Pearson r.

### Ventral Visual

| Arm | Mean repr | Mean RDM | Best single | Best single r | Delta |
|---|---:|---:|---|---:|---:|
| `taskvec_a1p0` | 0.1412 | 0.1264 | `single_triplet` | 0.1395 | 0.0017 |
| `taskvec_a0p5` | 0.1707 | 0.1559 | `single_feature_listing` | 0.1693 | 0.0014 |
| `lowrank` | 0.1751 | 0.1612 | `single_feature_listing` | 0.1742 | 0.0008 |
| `base` | 0.1144 | 0.1035 | `single_feature_listing` | 0.1138 | 0.0006 |
| `taskvec_a0p25` | 0.1679 | 0.1571 | `single_feature_listing` | 0.1702 | -0.0023 |
| `scrambled` | 0.0647 | 0.0748 | `single_pairwise` | 0.0699 | -0.0052 |
| `lowLR` | 0.1623 | 0.1444 | `single_feature_listing` | 0.1748 | -0.0125 |

### ATL (Semantic)

| Arm | Mean repr | Mean RDM | Best single | Best single r | Delta |
|---|---:|---:|---|---:|---:|
| `taskvec_a0p5` | 0.0356 | 0.0357 | `single_triplet` | 0.0310 | 0.0046 |
| `lowLR` | 0.0287 | 0.0252 | `single_feature_listing` | 0.0250 | 0.0037 |
| `lowrank` | 0.0332 | 0.0308 | `single_triplet` | 0.0326 | 0.0006 |
| `taskvec_a0p25` | 0.0276 | 0.0275 | `single_pairwise` | 0.0284 | -0.0007 |
| `taskvec_a1p0` | 0.0273 | 0.0255 | `single_feature_listing` | 0.0337 | -0.0064 |
| `scrambled` | -0.0023 | -0.0023 | `single_triplet` | 0.0091 | -0.0114 |
| `base` | 0.0032 | 0.0096 | `single_triplet` | 0.0171 | -0.0139 |

### Language

| Arm | Mean repr | Mean RDM | Best single | Best single r | Delta |
|---|---:|---:|---|---:|---:|
| `lowLR` | 0.0511 | 0.0511 | `single_pairwise` | 0.0423 | 0.0087 |
| `taskvec_a0p5` | 0.0511 | 0.0477 | `single_pairwise` | 0.0465 | 0.0046 |
| `taskvec_a0p25` | 0.0385 | 0.0369 | `single_feature_listing` | 0.0340 | 0.0045 |
| `taskvec_a1p0` | 0.0422 | 0.0324 | `single_triplet` | 0.0406 | 0.0016 |
| `lowrank` | 0.0503 | 0.0506 | `single_triplet` | 0.0488 | 0.0015 |
| `base` | 0.0004 | 0.0061 | `single_triplet` | 0.0168 | -0.0164 |
| `scrambled` | -0.0095 | -0.0093 | `single_feature_listing` | 0.0083 | -0.0178 |

## Descriptive Best-Layer Winners

### Ventral Visual

| Arm | Model | Layer | Pearson r | R2 vs train mean |
|---|---|---|---:|---:|
| `taskvec_a0p25` | `single_feature_listing` | `layer_18` | 0.2034 | 0.0425 |
| `taskvec_a0p25` | `mean_repr_plus_single_formats` | `layer_18` | 0.2026 | 0.0419 |
| `taskvec_a0p25` | `all_single_formats` | `layer_18` | 0.2021 | 0.0415 |
| `lowrank` | `single_feature_listing` | `layer_18` | 0.2010 | 0.0408 |
| `taskvec_a0p25` | `mean_repr` | `layer_18` | 0.1976 | 0.0400 |
| `lowrank` | `mean_repr_plus_single_formats` | `layer_18` | 0.1959 | 0.0377 |
| `lowrank` | `all_single_formats` | `layer_18` | 0.1958 | 0.0379 |
| `taskvec_a0p5` | `mean_repr_plus_single_formats` | `layer_19` | 0.1946 | 0.0391 |
| `lowLR` | `single_feature_listing` | `layer_18` | 0.1944 | 0.0383 |
| `taskvec_a0p5` | `all_single_formats` | `layer_19` | 0.1938 | 0.0388 |
| `taskvec_a0p5` | `mean_repr` | `layer_18` | 0.1933 | 0.0390 |
| `lowLR` | `mean_repr_plus_single_formats` | `layer_22` | 0.1928 | 0.0370 |

### ATL (Semantic)

| Arm | Model | Layer | Pearson r | R2 vs train mean |
|---|---|---|---:|---:|
| `taskvec_a0p5` | `single_pairwise` | `layer_23` | 0.0565 | 0.0041 |
| `taskvec_a0p5` | `mean_repr_plus_single_formats` | `layer_28` | 0.0547 | 0.0031 |
| `lowrank` | `mean_repr_plus_single_formats` | `layer_22` | 0.0546 | 0.0036 |
| `lowLR` | `mean_repr_plus_single_formats` | `layer_6` | 0.0544 | 0.0010 |
| `taskvec_a0p5` | `mean_repr` | `layer_23` | 0.0498 | 0.0033 |
| `taskvec_a0p5` | `mean_rdm` | `layer_23` | 0.0496 | 0.0031 |
| `taskvec_a1p0` | `mean_repr_plus_single_formats` | `layer_22` | 0.0495 | 0.0014 |
| `taskvec_a0p25` | `single_pairwise` | `layer_23` | 0.0492 | 0.0037 |
| `lowLR` | `all_single_formats` | `layer_6` | 0.0482 | 0.0002 |
| `taskvec_a0p5` | `all_single_formats` | `layer_19` | 0.0471 | 0.0020 |
| `lowrank` | `all_single_formats` | `layer_19` | 0.0469 | 0.0016 |
| `taskvec_a0p25` | `mean_repr_plus_single_formats` | `layer_21` | 0.0464 | 0.0022 |

### Language

| Arm | Model | Layer | Pearson r | R2 vs train mean |
|---|---|---|---:|---:|
| `taskvec_a1p0` | `mean_repr_plus_single_formats` | `layer_18` | 0.0765 | 0.0035 |
| `lowrank` | `mean_repr_plus_single_formats` | `layer_23` | 0.0706 | 0.0042 |
| `lowLR` | `mean_repr_plus_single_formats` | `layer_23` | 0.0688 | 0.0036 |
| `lowrank` | `all_single_formats` | `layer_23` | 0.0684 | 0.0039 |
| `taskvec_a0p5` | `mean_repr_plus_single_formats` | `layer_18` | 0.0679 | 0.0040 |
| `taskvec_a0p5` | `single_pairwise` | `layer_23` | 0.0672 | 0.0038 |
| `taskvec_a0p5` | `all_single_formats` | `layer_19` | 0.0669 | 0.0039 |
| `lowLR` | `all_single_formats` | `layer_27` | 0.0654 | 0.0033 |
| `taskvec_a0p5` | `mean_repr` | `layer_23` | 0.0649 | 0.0035 |
| `taskvec_a0p5` | `mean_rdm` | `layer_23` | 0.0642 | 0.0036 |
| `taskvec_a0p25` | `single_pairwise` | `layer_25` | 0.0635 | 0.0035 |
| `taskvec_a1p0` | `all_single_formats` | `layer_18` | 0.0634 | 0.0024 |

## Initial Read

If `mean_repr_minus_best_single` is positive, the shared/format-averaged
hub representation predicts held-out fMRI geometry better than the best
individual prompt spoke in the same mid-layer band. If it is negative,
the fMRI signal is better explained by a task-specific prompt format
or by visual/object geometry not captured by the hub average.

The 2026-07-09 audit above changes the near-term priority: the mid-layer
`mean_repr` base-delta pattern is consistent enough to justify a confirmatory
fixed-metric run. It does not yet override the negative/flat direct RSA and
Huth/LeBel smoke results.
