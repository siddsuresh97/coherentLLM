# Semantic Hub Hubness Control

Created UTC: 2026-07-09T04:05:55.292563+00:00

## Question

Do semantic-hub retrieval gains reflect broad same-concept alignment, or are they inflated
because a few concepts become nearest-neighbor hubs for many unrelated queries?

## Mid-Layer Summary

| Arm | Top1 | Top5 | Unique top1 | Max top1 occ | Gini | Entropy | Read |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 0.0192 | 0.0800 | 0.0346 | 94.1515 | 0.9829 | 0.1416 | weak retrieval; hubness is not the main claim |
| scrambled | 0.1094 | 0.2667 | 0.1838 | 57.3636 | 0.9088 | 0.4033 | retrieval may be inflated by hub concepts |
| lowLR | 0.0729 | 0.3063 | 0.1602 | 48.7879 | 0.9102 | 0.4152 | retrieval may be inflated by hub concepts |
| lowrank | 0.0968 | 0.4079 | 0.1961 | 34.9697 | 0.8888 | 0.4969 | retrieval may be inflated by hub concepts |
| taskvec_a0p25 | 0.2062 | 0.5393 | 0.2996 | 29.1515 | 0.8024 | 0.5815 | strong retrieval with reduced, not eliminated, hubness |
| taskvec_a0p5 | 0.1578 | 0.4744 | 0.2444 | 40.5455 | 0.8477 | 0.5086 | strong retrieval but still hub-skewed |
| taskvec_a1p0 | 0.0637 | 0.1896 | 0.1081 | 75.7273 | 0.9448 | 0.2754 | weak retrieval; hubness is not the main claim |

## CSLS / Mutual-Nearest Correction

| Arm | MNN top1 | CSLS top1 | CSLS top5 | CSLS MNN | CSLS unique | CSLS max occ | Corrected read |
|---|---:|---:|---:|---:|---:|---:|---|
| base | 0.0002 | 0.0348 | 0.1276 | 0.0017 | 0.0733 | 74.0758 | weak after hubness correction |
| scrambled | 0.0355 | 0.1710 | 0.3965 | 0.0767 | 0.3022 | 38.9545 | mostly one-way retrieval after correction |
| lowLR | 0.0069 | 0.1267 | 0.4747 | 0.0230 | 0.2559 | 28.6818 | mostly one-way retrieval after correction |
| lowrank | 0.0140 | 0.1570 | 0.5964 | 0.0400 | 0.3224 | 21.3485 | mostly one-way retrieval after correction |
| taskvec_a0p25 | 0.0777 | 0.2775 | 0.6751 | 0.1319 | 0.4248 | 18.4242 | partly robust; still needs causal/logit-lens validation |
| taskvec_a0p5 | 0.0518 | 0.2206 | 0.5924 | 0.0973 | 0.3420 | 26.5303 | mostly one-way retrieval after correction |
| taskvec_a1p0 | 0.0201 | 0.0897 | 0.2544 | 0.0379 | 0.1644 | 62.4242 | mostly one-way retrieval after correction |

## Interpretation

- Higher `top5` is useful only if `unique top1` and entropy stay reasonably high.
- A high Gini or high max occurrence means a small number of candidate concepts are
  absorbing many nearest-neighbor queries.
- CSLS penalizes concepts that are close to many unrelated queries. Mutual-nearest
  retrieval is stricter still: the query and candidate have to select each other.
- This control should be read alongside `memp_paper_harness/control_summary.csv`;
  it is a hubness artifact check, not a substitute for matched-vs-control deltas.

## Files

- `pair_by_layer.csv`: ordered format-pair hubness rows.
- `layer_summary.csv`: layer means across ordered format pairs.
- `hubness_summary.csv`: mid-layer aggregate used in this report.
- `hubness_concept_counts.csv`: mid-layer attractor counts by arm and concept,
  including raw cosine and CSLS-corrected counts.
