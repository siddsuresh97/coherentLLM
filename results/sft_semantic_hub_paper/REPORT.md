# Paper-Style Semantic Hub Similarity Report

Created: 2026-07-08

## Status

Computed matched-vs-mismatched cross-format similarity baselines from
existing semantic-hub hidden states.

## Method

- Same-concept cross-format cosine similarity is compared against random
  nonmatching concepts, S*-close nonmatching concepts, and S*-far
  nonmatching concepts.
- Metrics are layer-resolved and averaged across format pairs.
- Mid-layer band: `10:20`.
- Bootstrap resamples per layer/format pair: `1000`.

## Mid-Layer Summary

| Arm | Same-Random | Same-Close | Same-Far | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|
| `taskvec_a0p5` | 0.0617 | 0.0070 | 0.0805 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | 0.0430 | 0.0099 | 0.0563 | 0.0637 | 0.1896 |
| `taskvec_a0p25` | 0.0380 | 0.0011 | 0.0548 | 0.2062 | 0.5393 |
| `lowrank` | 0.0311 | 0.0003 | 0.0494 | 0.0968 | 0.4079 |
| `lowLR` | 0.0247 | 0.0013 | 0.0371 | 0.0729 | 0.3063 |
| `scrambled` | 0.0095 | 0.0026 | 0.0077 | 0.1094 | 0.2667 |
| `base` | 0.0022 | -0.0002 | 0.0033 | 0.0192 | 0.0800 |

## Initial Read

- Best mid-layer same-minus-random arm: `taskvec_a0p5` (`0.0617`).
- Best mid-layer same-minus-S*-close arm: `taskvec_a1p0` (`0.0099`).
- `same_minus_close_neighbor` is the stricter concept-identity test:
  it asks whether the exact same concept beats a semantically close
  but non-identical concept across prompt formats.

## Files

- Layer metrics: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub_paper/similarity_by_layer.csv`
- Summary: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub_paper/similarity_summary.csv`
- Metadata: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub_paper/similarity_meta.json`
