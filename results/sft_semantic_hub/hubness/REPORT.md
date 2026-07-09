# Semantic Hub Hubness Control

Created UTC: 2026-07-09T03:51:52.372780+00:00

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

## Interpretation

- Higher `top5` is useful only if `unique top1` and entropy stay reasonably high.
- A high Gini or high max occurrence means a small number of candidate concepts are
  absorbing many nearest-neighbor queries.
- This control should be read alongside `memp_paper_harness/control_summary.csv`;
  it is a hubness artifact check, not a substitute for matched-vs-control deltas.

## Files

- `pair_by_layer.csv`: ordered format-pair hubness rows.
- `layer_summary.csv`: layer means across ordered format pairs.
- `hubness_summary.csv`: mid-layer aggregate used in this report.
- `hubness_concept_counts.csv`: mid-layer attractor counts by arm and concept.
