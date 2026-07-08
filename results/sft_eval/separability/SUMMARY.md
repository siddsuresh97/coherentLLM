# Coherence vs THINGS-Human Separability

Primary combined correlation:

- n=6, Pearson r=0.914232, p=0.010719

Correlation table:

| analysis                                  | source        | coherence_col   | human_r2_col                |   n |   pearson_r |   p_value |
|:------------------------------------------|:--------------|:----------------|:----------------------------|----:|------------:|----------:|
| combined_primary                          | combined      | gen_coherence   | things_human_r2             |   6 |    0.914232 |  0.010719 |
| sft_eval_primary                          | sft_eval      | gen_proc_mean   | human_things_triplet_r2     |   3 |    0.931023 |  0.237835 |
| sft_eval_gen_human_things_triplet_r2      | sft_eval      | gen_proc_mean   | gen_human_things_triplet_r2 |   3 |    0.931023 |  0.237835 |
| sft_eval_lp_human_things_triplet_r2       | sft_eval      | gen_proc_mean   | lp_human_things_triplet_r2  |   3 |   -0.955912 |  0.189742 |
| sft_eval_human_things_triplet_r2          | sft_eval      | gen_proc_mean   | human_things_triplet_r2     |   3 |    0.931023 |  0.237835 |
| ch3_model_zoo_primary                     | ch3_model_zoo | cross_mean      | human_procrustes_r2_triplet |   3 |    0.990423 |  0.088179 |
| ch3_model_zoo_human_procrustes_r2_triplet | ch3_model_zoo | cross_mean      | human_procrustes_r2_triplet |   3 |    0.990423 |  0.088179 |

SFT arm deltas vs base:

| state     | model                 |   gen_coherence |   things_human_r2 |   delta_gen_coherence_vs_base |   delta_things_human_r2_vs_base | status       |
|:----------|:----------------------|----------------:|------------------:|------------------------------:|--------------------------------:|:-------------|
| base      | llama-3.1-8b-instruct |        0.322558 |          0.468093 |                      0.000000 |                        0.000000 | baseline     |
| real      | llama31-sft-real      |        0.699402 |          0.604927 |                      0.376844 |                        0.136834 | both_gained  |
| scrambled | llama31-sft-scrambled |        0.031259 |          0.015870 |                     -0.291299 |                       -0.452223 | neither_gain |

No SFT arm gains one metric without the other relative to base.
No pairwise opposite-direction comparisons found in the combined rows.
