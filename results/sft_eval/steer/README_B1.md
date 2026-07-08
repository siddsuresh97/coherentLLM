# B1: coherence steering alpha sweep

Two routes tested for adding the coherence direction to the BASE model at inference.

## taskvec route (VALID) — base + alpha * (real_LoRA task vector)
| alpha | GEN coherence (gen_proc_mean) | THINGS-human r2 |
|------:|------------------------------:|----------------:|
| 0.00 (base)        | 0.323 | 0.468 |
| 0.25               | 0.649 | 0.599 |
| 0.50               | 0.707 | 0.584 |
| 1.00 (= real SFT)  | 0.699 | 0.605 |

Headline: adding a FRACTION of the coherence task-vector moves base toward real-SFT coherence.
alpha=0.25 already captures ~87% of the coherence gain (0.32 -> 0.65 vs real 0.70) and matches
real human-alignment (0.60). alpha=0.5 reaches real-level coherence. The coherence direction is a
real, addable weight-space direction; full fine-tuning is not required.

## actdiff route (NEGATIVE RESULT — collapses) — residual-stream activation addition at all layers
Adding alpha * activation-diff vector at the output of ALL 33 layers collapses generation into a
degenerate "features features ..." loop at EVERY tested alpha (0.5, 1, 2, 4, 8). SALMON fits fail
(n_samples=0, no parseable choices). Naive all-layer residual addition of the coherence direction
over-steers and breaks the model; it is not a clean linear steering vector. actdiff rows in
sweep_summary.csv are left blank (incomplete). A norm-preserving / few-layer injection is a possible
follow-up, not pursued here.
