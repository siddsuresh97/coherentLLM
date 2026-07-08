# Task26: coherence-vs-retention mitigation Pareto

Question: can we teach coherence WITHOUT the catastrophic retention drop that the
default real-SFT causes? Answer: YES — lowering the learning rate (or the LoRA rank)
keeps essentially all the coherence gain while nearly eliminating forgetting.

All coherence = gen_proc_mean over the 128-concept stimuli (SALMON d5 + procrustes).
Retention = mean MC accuracy over arc_challenge / hellaswag / mmlu(abstract_algebra,
anatomy) / truthfulqa_mc2, lm-eval --limit 1000, 0-shot.

| arm                | LR    | rank | steps | coherence | human_r2 | retention | ret vs base |
|--------------------|-------|-----:|------:|----------:|---------:|----------:|------------:|
| base (no SFT)      | -     |   -  |   -   |   0.323   |  0.468   |   0.574   |    0.000    |
| real-SFT (default) | 2e-4  |  64  | 1500  |   0.699   |  0.605   |   0.367   |   -0.207    |
| fewsteps           | 2e-4  |  64  |  400  |   0.703   |  0.591   |   0.407   |   -0.167    |
| **lowLR**          | 5e-5  |  64  | 1500  | **0.757** |  0.651   | **0.538** |  **-0.036** |
| **lowrank**        | 2e-4  |  16  | 1500  | **0.749** |  0.666   | **0.544** |  **-0.030** |

## Headline
- Default real-SFT (LR 2e-4, r64) buys coherence 0.32->0.70 but pays a catastrophic
  retention drop 0.574->0.367 (-0.207).
- **lowLR (LR 5e-5)** and **lowrank (r=16)** BREAK this tradeoff: both reach coherence
  ~0.75 (HIGHER than default real-SFT) while keeping retention ~0.54, only -0.03/-0.04
  below base. Forgetting is almost entirely eliminated.
- Both also improve human-alignment (human_r2 0.65-0.67 vs real-SFT 0.60).
- fewsteps (early stop) keeps coherence but recovers little retention (-0.167): stopping
  early is not the lever; a gentler update (low LR / low rank) is.

## Recommendation
Train the coherence adapter at LR 5e-5 (or rank 16). This delivers coherence with
near-zero capability regression, resolving the retention concern from the main SFT run.
