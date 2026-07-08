# Coherence Steering Sweep

Completed alpha rows:

| route   |    alpha | out_model                |   gen_proc_mean |   human_things_triplet_r2 |
|:--------|---------:|:-------------------------|----------------:|--------------------------:|
| actdiff | 0.000000 | llama31-steer-a0         |        0.322558 |                  0.468093 |
| taskvec | 0.250000 | llama31-steer-task-a0p25 |        0.649454 |                  0.598938 |
| taskvec | 0.500000 | llama31-steer-task-a0p5  |        0.706716 |                  0.584390 |
| taskvec | 1.000000 | llama31-steer-task-a1    |        0.699402 |                  0.604927 |

Reference rows:

| state   | out_model             |   gen_proc_mean |   human_things_triplet_r2 |
|:--------|:----------------------|----------------:|--------------------------:|
| base    | llama-3.1-8b-instruct |        0.322558 |                  0.468093 |
| real    | llama31-sft-real      |        0.699402 |                  0.604927 |

Incomplete rows:

| route   |    alpha | out_model          |   gen_proc_tf |   gen_proc_tp |   gen_proc_pf |   gen_proc_mean |   gen_human_things_triplet_r2 |   human_things_triplet_r2 | complete   | error                                                                                                       |
|:--------|---------:|:-------------------|--------------:|--------------:|--------------:|----------------:|------------------------------:|--------------------------:|:-----------|:------------------------------------------------------------------------------------------------------------|
| actdiff | 0.500000 | llama31-steer-a0p5 |           nan |           nan |           nan |             nan |                           nan |                       nan | False      | missing results/sft_eval/llama31-steer-a0p5_triplet_d5.npy                                                  |
| actdiff | 1.000000 | llama31-steer-a1   |           nan |           nan |           nan |             nan |                           nan |                       nan | False      | missing results/sft_eval/llama31-steer-a1_triplet_d5.npy                                                    |
| actdiff | 2.000000 | llama31-steer-a2   |           nan |           nan |           nan |             nan |                           nan |                       nan | False      | missing results/sft_eval/llama31-steer-a2_triplet_d5.npy                                                    |
| actdiff | 4.000000 | llama31-steer-a4   |           nan |           nan |           nan |             nan |                           nan |                       nan | False      | missing results/sft_eval/raw/llama31-steer-a4/feature.csv; results/sft_eval/llama31-steer-a4_triplet_d5.npy |
| actdiff | 8.000000 | llama31-steer-a8   |           nan |           nan |           nan |             nan |                           nan |                       nan | False      | missing results/sft_eval/llama31-steer-a8_triplet_d5.npy                                                    |
