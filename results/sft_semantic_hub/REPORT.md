# Semantic Hub Report

Created: 2026-07-08

## Status

Layer-resolved semantic-hub metrics computed from extracted hidden states.

## Method

- Concepts: 128 held-out THINGS concepts.
- Formats: triplet, pairwise, feature_listing.
- Triplet prompts use deterministic S*-close and S*-far neighbors.
- Pairwise prompts use the S*-close neighbor.
- Feature/listing prompts use `listing_prompt(concept)`.
- Metrics: cross-format RDM Spearman, linear CKA, same-concept retrieval, and concept-vs-format label alignment.

## Mid-Layer Summary

| Arm | Mid RDM Spearman | Mid CKA | Mid Top-1 | Mid Top-5 | Mid Concept-Format |
|---|---:|---:|---:|---:|---:|
| `taskvec_a0p25` | 0.6770 | 0.7809 | 0.2062 | 0.5393 | -0.8716 |
| `lowrank` | 0.6637 | 0.7467 | 0.0968 | 0.4079 | -0.9311 |
| `taskvec_a0p5` | 0.6372 | 0.7464 | 0.1578 | 0.4744 | -0.8435 |
| `lowLR` | 0.6211 | 0.6899 | 0.0729 | 0.3063 | -0.9304 |
| `taskvec_a1p0` | 0.4867 | 0.5505 | 0.0637 | 0.1896 | -0.7863 |
| `scrambled` | 0.3472 | 0.4601 | 0.1094 | 0.2667 | -0.8652 |
| `base` | 0.3230 | 0.3890 | 0.0192 | 0.0800 | -0.9673 |

## Initial Read

- Strongest mid-layer format-invariance score: `taskvec_a0p25` (RDM Spearman 0.6770, CKA 0.7809, top-1 retrieval 0.2062).
- Base mid-layer score: RDM Spearman 0.3230, CKA 0.3890, top-1 retrieval 0.0192.
- Treat this as evidence for stronger cross-format invariance, not yet a clean concept-dominant semantic hub: concept-minus-format alignment remains negative for every arm.
- Next bridge: correlate arm/layer hub metrics with fMRI RSA and inspect whether format-averaged hub RDMs explain fMRI better than single-format RDMs.

## Best-Layer Summary

| Arm | Best Layer | Best RDM Spearman | Best CKA | Best Top-1 | Best Concept-Format |
|---|---:|---:|---:|---:|---:|
| `lowrank` | layer_8 | 0.8785 | 0.9477 | 0.1029 | -0.9684 |
| `lowLR` | layer_8 | 0.8734 | 0.9530 | 0.0456 | -0.9656 |
| `taskvec_a0p25` | layer_11 | 0.8239 | 0.8783 | 0.0781 | -0.9132 |
| `taskvec_a0p5` | layer_11 | 0.8064 | 0.9014 | 0.2083 | -0.8403 |
| `taskvec_a1p0` | layer_7 | 0.7568 | 0.8899 | 0.2344 | -0.7159 |
| `scrambled` | layer_4 | 0.6797 | 0.4407 | 0.1276 | -0.8758 |
| `base` | layer_3 | 0.6281 | 0.6002 | 0.0182 | -0.8559 |

## Files

- Hidden-state input: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/hidden_states`
- Layer metrics: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/hub_by_layer.csv`
- Summary: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/hub_summary.csv`
