# Positive-Signal Audit: fMRI x Semantic Hub

This is a post-hoc audit of existing `results/sft_fmri_hub_regression/`
artifacts. No model features, fMRI responses, or CHTC jobs were rerun.

## Main Read

The most defensible positive lead is not the original direct THINGS-fMRI RSA. It is the concept-held-out semantic-hub regression: mid-layer, format-averaged model geometry predicts held-out fMRI RDM distances better after coherence training/task-vector movement than in base.

Across the three target regions and five coherent/non-scrambled arms, 15/15 mid-layer `mean_repr` comparisons beat base. The scrambled control is below base in 3/3 regions.

This supports a cautious story: coherence tuning makes mid-layer semantic
geometry more brain-predictive under held-out concept regression, while
the raw direct-RSA and Huth/LeBel smoke are not yet base-beating results.

## Mid-Layer Mean-Representation Deltas

Mid-layer band: layers `10` through `20`. Metric: held-out Pearson r.

### Ventral Visual

| Arm | Mean repr r | Delta vs base | Subjects positive | Mean repr - best single |
| --- | --- | --- | --- | --- |
| `lowrank` | 0.1751 | 0.0607 | 2/3 | 0.0008 |
| `taskvec_a0p5` | 0.1707 | 0.0563 | 2/3 | 0.0014 |
| `taskvec_a0p25` | 0.1679 | 0.0535 | 3/3 | -0.0023 |
| `lowLR` | 0.1623 | 0.0479 | 2/3 | -0.0125 |
| `taskvec_a1p0` | 0.1412 | 0.0269 | 2/3 | 0.0017 |
| `base` | 0.1144 | 0.0000 | 0/3 | 0.0006 |
| `scrambled` | 0.0647 | -0.0497 | 0/3 | -0.0052 |

### ATL (Semantic)

| Arm | Mean repr r | Delta vs base | Subjects positive | Mean repr - best single |
| --- | --- | --- | --- | --- |
| `taskvec_a0p5` | 0.0356 | 0.0324 | 2/3 | 0.0046 |
| `lowrank` | 0.0332 | 0.0300 | 2/3 | 0.0006 |
| `lowLR` | 0.0287 | 0.0255 | 2/3 | 0.0037 |
| `taskvec_a0p25` | 0.0276 | 0.0245 | 2/3 | -0.0007 |
| `taskvec_a1p0` | 0.0273 | 0.0241 | 2/3 | -0.0064 |
| `base` | 0.0032 | 0.0000 | 0/3 | -0.0139 |
| `scrambled` | -0.0023 | -0.0054 | 1/3 | -0.0114 |

### Language

| Arm | Mean repr r | Delta vs base | Subjects positive | Mean repr - best single |
| --- | --- | --- | --- | --- |
| `lowLR` | 0.0511 | 0.0507 | 3/3 | 0.0087 |
| `taskvec_a0p5` | 0.0511 | 0.0507 | 2/3 | 0.0046 |
| `lowrank` | 0.0503 | 0.0499 | 3/3 | 0.0015 |
| `taskvec_a1p0` | 0.0422 | 0.0418 | 2/3 | 0.0016 |
| `taskvec_a0p25` | 0.0385 | 0.0381 | 2/3 | 0.0045 |
| `base` | 0.0004 | 0.0000 | 0/3 | -0.0164 |
| `scrambled` | -0.0095 | -0.0099 | 1/3 | -0.0178 |

## Subject Consistency

Subject deltas are mid-layer `mean_repr` means minus base for the same
subject and region.

### Ventral Visual

| Arm | Mean delta | n positive | sub-01 | sub-02 | sub-03 |
| --- | --- | --- | --- | --- | --- |
| `lowrank` | 0.0607 | 2/3 | 0.1061 | 0.0794 | -0.0035 |
| `taskvec_a0p5` | 0.0563 | 2/3 | 0.0919 | 0.0805 | -0.0034 |
| `taskvec_a0p25` | 0.0535 | 3/3 | 0.0820 | 0.0761 | 0.0024 |
| `lowLR` | 0.0479 | 2/3 | 0.0905 | 0.0549 | -0.0016 |
| `taskvec_a1p0` | 0.0269 | 2/3 | 0.0657 | 0.0338 | -0.0190 |
| `scrambled` | -0.0497 | 0/3 | -0.0378 | -0.0917 | -0.0197 |

### ATL (Semantic)

| Arm | Mean delta | n positive | sub-01 | sub-02 | sub-03 |
| --- | --- | --- | --- | --- | --- |
| `taskvec_a0p5` | 0.0324 | 2/3 | 0.0564 | 0.0584 | -0.0176 |
| `lowrank` | 0.0300 | 2/3 | 0.0533 | 0.0471 | -0.0104 |
| `lowLR` | 0.0255 | 2/3 | 0.0497 | 0.0421 | -0.0152 |
| `taskvec_a0p25` | 0.0245 | 2/3 | 0.0421 | 0.0565 | -0.0253 |
| `taskvec_a1p0` | 0.0241 | 2/3 | 0.0530 | 0.0268 | -0.0074 |
| `scrambled` | -0.0054 | 1/3 | 0.0027 | -0.0098 | -0.0093 |

### Language

| Arm | Mean delta | n positive | sub-01 | sub-02 | sub-03 |
| --- | --- | --- | --- | --- | --- |
| `lowLR` | 0.0507 | 3/3 | 0.0899 | 0.0477 | 0.0145 |
| `taskvec_a0p5` | 0.0507 | 2/3 | 0.1048 | 0.0489 | -0.0017 |
| `lowrank` | 0.0499 | 3/3 | 0.0980 | 0.0491 | 0.0025 |
| `taskvec_a1p0` | 0.0418 | 2/3 | 0.1027 | 0.0239 | -0.0012 |
| `taskvec_a0p25` | 0.0381 | 2/3 | 0.0777 | 0.0425 | -0.0058 |
| `scrambled` | -0.0099 | 1/3 | -0.0043 | -0.0320 | 0.0067 |

## Best-Layer Cells With Same-Base Controls

These are descriptive and should not be treated as confirmatory because
layers and models were inspected after scoring. The base comparator is
the same region, predictor model, and layer.

### Ventral Visual

| Rank | Arm | Model | Layer | r | Base same | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `taskvec_a0p25` | `single_feature_listing` | 18 | 0.2034 | 0.1244 | 0.0791 |
| 2 | `taskvec_a0p25` | `mean_repr_plus_single_formats` | 18 | 0.2026 | 0.1325 | 0.0701 |
| 3 | `taskvec_a0p25` | `all_single_formats` | 18 | 0.2021 | 0.1128 | 0.0893 |
| 4 | `lowrank` | `single_feature_listing` | 18 | 0.2010 | 0.1244 | 0.0767 |
| 5 | `taskvec_a0p25` | `mean_repr_plus_single_formats` | 19 | 0.1987 | 0.1330 | 0.0657 |
| 6 | `taskvec_a0p25` | `all_single_formats` | 19 | 0.1979 | 0.1076 | 0.0902 |
| 7 | `lowrank` | `single_feature_listing` | 19 | 0.1978 | 0.1158 | 0.0820 |
| 8 | `taskvec_a0p25` | `mean_repr` | 18 | 0.1976 | 0.1118 | 0.0858 |

### ATL (Semantic)

| Rank | Arm | Model | Layer | r | Base same | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `taskvec_a0p5` | `single_pairwise` | 23 | 0.0565 | 0.0205 | 0.0360 |
| 2 | `taskvec_a0p5` | `single_pairwise` | 25 | 0.0564 | 0.0211 | 0.0352 |
| 3 | `taskvec_a0p5` | `mean_repr_plus_single_formats` | 28 | 0.0547 | 0.0132 | 0.0415 |
| 4 | `lowrank` | `mean_repr_plus_single_formats` | 22 | 0.0546 | -0.0112 | 0.0658 |
| 5 | `lowLR` | `mean_repr_plus_single_formats` | 6 | 0.0544 | -0.0137 | 0.0681 |
| 6 | `taskvec_a0p5` | `mean_repr_plus_single_formats` | 27 | 0.0543 | 0.0159 | 0.0384 |
| 7 | `taskvec_a0p5` | `single_pairwise` | 26 | 0.0537 | 0.0210 | 0.0327 |
| 8 | `taskvec_a0p5` | `mean_repr_plus_single_formats` | 29 | 0.0529 | 0.0040 | 0.0489 |

### Language

| Rank | Arm | Model | Layer | r | Base same | Delta |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | `taskvec_a1p0` | `mean_repr_plus_single_formats` | 18 | 0.0765 | 0.0321 | 0.0444 |
| 2 | `taskvec_a1p0` | `mean_repr_plus_single_formats` | 19 | 0.0716 | 0.0306 | 0.0410 |
| 3 | `lowrank` | `mean_repr_plus_single_formats` | 23 | 0.0706 | 0.0197 | 0.0509 |
| 4 | `lowrank` | `mean_repr_plus_single_formats` | 22 | 0.0695 | 0.0091 | 0.0604 |
| 5 | `lowLR` | `mean_repr_plus_single_formats` | 23 | 0.0688 | 0.0197 | 0.0491 |
| 6 | `lowLR` | `mean_repr_plus_single_formats` | 27 | 0.0687 | 0.0303 | 0.0384 |
| 7 | `taskvec_a1p0` | `mean_repr_plus_single_formats` | 20 | 0.0687 | 0.0257 | 0.0430 |
| 8 | `lowrank` | `all_single_formats` | 23 | 0.0684 | 0.0403 | 0.0281 |

## Interpretation

- Positive claim to test next: coherence-tuned/task-vector mid-layer
  semantic geometry is more predictive of held-out fMRI object-concept
  RDMs than base geometry.
- This is strongest as a representational brain-alignment claim, not as
  a deployment-quality claim. `taskvec_a0p5` is positive here but failed
  the retention gate, so it should not be promoted as a general model.
- The result is still post-hoc. The next run should pre-register the
  mid-layer `mean_repr` metric, use concept-held-out folds, and evaluate
  base, scrambled, lowrank, and the task-vector alphas without selecting
  layers on test scores.
- For a stronger cognitive-neuroscience claim, repeat the same fixed
  analysis in Huth/LeBel high-data language fMRI or a Fedorenko-style
  language-localizer dataset.

## Artifacts

- `positive_midlayer_mean_repr.csv`
- `positive_subject_consistency.csv`
- `positive_best_layer_same_base.csv`
