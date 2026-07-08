# SFT Step 4 Eval Summary

| state     |   gen_proc_mean |   lp_proc_mean |   para_mean |   sym_viol_rate |   trans_viol_rate |   gen_human_things_triplet_r2 |   lp_human_things_triplet_r2 |   ret_mmlu |   ret_arc_challenge |   ret_hellaswag |   ret_truthfulqa |
|:----------|----------------:|---------------:|------------:|----------------:|------------------:|------------------------------:|-----------------------------:|-----------:|--------------------:|----------------:|-----------------:|
| base      |           0.323 |          0.253 |       0.782 |           0.151 |             0.067 |                         0.468 |                        0.283 |      0.533 |               0.540 |           0.708 |            0.530 |
| real      |           0.699 |          0.315 |       0.840 |           0.000 |             0.006 |                         0.605 |                        0.143 |      0.305 |               0.200 |           0.560 |            0.531 |
| scrambled |           0.031 |          0.047 |       0.842 |           0.000 |             0.045 |                         0.016 |                        0.310 |      0.220 |               0.210 |           0.410 |            0.460 |

1. Axis A: real-base delta is +0.377 for generation and +0.063 for logprob; success requires both to be positive.
2. Scrambled control delta is -0.291 generation and -0.206 logprob; success requires no comparable rise.
3. Axis B paraphrase consistency real-base delta is +0.059; higher is better.
4. Axis C/D shifts for real are symmetry +0.151, transitivity +0.062, human triplet +0.137; lower C violations and higher D alignment are better.
5. Retention is outside or incomplete for the guardrail.
