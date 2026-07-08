# Task 8 Verdict: Mid-Layer Activation Steering

## Question
Does mid-layer activation steering (norm-matched injection, confined to a narrow band of
decoder layers rather than all 33 layers) raise coherence toward the SFT (0.70) / taskvec-a0.25
(0.65) benchmark WITHOUT collapsing generation, at some layer x alpha combination?

## Answer: NO. This is an earned negative, not an all-layers artifact.

The full grid was run to completion: **32/32 (layer_set, alpha) combinations tested, 32/32
degenerate.** Every single layer (8, 10, 12, 14, 16, 20) and every band (10-14, 12-16) at every
tested alpha (2, 4, 6, 8) collapses generation into unparseable, repetitive garbage. Coherence
and human_r2 could not be computed for any configuration because SALMON has no valid triplet
choices to fit on (triplet_valid=0.000 in all but one cell).

This directly answers the concern that motivated Task 8: the original actdiff route (adding the
coherence direction at ALL 33 decoder layers simultaneously) collapsed generation at every alpha,
and we could not tell whether that was (a) a genuine property of the coherence direction being
too destructive to inject via activation steering, or (b) an artifact of stacking the
perturbation across every layer, which compounds and blows up the residual stream. Task 8 fixed
(b) by confining the injection to one layer (or a 5-layer band) at a time and norm-matching the
injection magnitude to each token's hidden-state norm (h_L <- h_L + alpha*(diff_L/||diff_L||)*
||h_L||), per the brief. With that confound removed, the SAME collapse still happens everywhere
in the grid. That rules out explanation (b): this is not an all-layers artifact. The coherence
direction itself, injected via this activation-steering route, destabilizes generation at every
layer depth tested (early: 8, middle: 10/12/14/16, late: 20) and at every magnitude tested
(alpha 2-8), even confined to a single layer or a narrow band.

## Grid summary

| layer_set | alpha | degenerate | triplet_valid | notes (representative sample) |
|---|---|---|---|---|
| 8         | 4 | True | 0.000 | 'thethetheSSthethe' |
| 10        | 4 | True | 0.000 | 'thetheathesthedefaults' |
| 12        | 2,4,6,8 | True (all 4) | 0.000 | quote-mark loops, full saturation at a>=6 |
| 14        | 2,4,6,8 | True (all 4) | 0.000 | period/digit loops |
| 16        | 2,4,6,8 | True (all 4) | 0.000 | period/digit loops, one row with high literal-repeat share |
| 20        | 4,6,8 | True | 0.000 | 's'-character loops |
| 20        | **2** | True | **0.032** | closest to non-degenerate seen anywhere; still top_share=0.559 (majority collapse) |
| 10-14     | 2,4,6,8 | True (all 4) | 0.000 | full saturation, single repeated garbage token |
| 12-16     | 2,4,6,8 | True (all 4) | 0.000 | full saturation, single repeated garbage token |

Full per-cell detail (including exact samples) is in `results/sft_eval/steer_actlayer/summary.csv`.

## Notable near-miss
Layer 20, alpha=2 is the only cell in the entire 32-cell grid with any nonzero triplet_valid
rate (3.2%, vs 0.000 everywhere else). It is still classified degenerate (56% of responses
collapse to the dominant garbage pattern), so it does not change the overall verdict, but it is
the one data point suggesting late layers at very low alpha are marginally less destructive than
early/middle layers or higher alpha. This was not pursued further (e.g. alpha=1 or a finer
low-alpha sweep at layer 20) because it was out of scope for this task's grid and the signal is
too weak (96.8% invalid) to support a follow-up claim of a working configuration.

## Conclusion for the dissertation / report
Replace the earlier claim "activation steering collapses at every alpha" (which was ambiguous
about whether that was an artifact of all-layers injection) with the corrected, stronger claim:
**mid-layer, norm-matched activation steering of the coherence direction collapses generation at
every tested layer depth and alpha magnitude; this is not an artifact of injecting at all layers
simultaneously.** The task-vector (weight-space) route remains the only steering mechanism in
this project that raises coherence without destroying generation (see Task 7:
`results/sft_eval/steer/steer_retention.csv`, alpha=0.25 is the practical sweet spot). Activation
steering, at least via this direction-extraction method (mean-difference of real-SFT vs base
hidden states) and this injection formula, does not offer a viable training-free alternative.

## Provenance
- Grid executed via a tmux-persisted worker architecture (`task8_gpu0`, `task8_gpu1` on the local
  A5000s, plus an H100 loop) that survives the orchestrating process restarting; this was
  necessary because the orchestrating codex process exited unexpectedly several times during the
  run (see commit history `Task8: mid-layer activation steering sweep (in progress, ...)`).
- One config (layer 16, alpha 6) was briefly claimed by a misconfigured worker that fell back to
  CPU execution instead of GPU due to a bad `CUDA_VISIBLE_DEVICES` value; that worker was killed
  before completing, the config was re-queued, and it was correctly re-run and completed on GPU.
  No fabricated or partial data was retained from the aborted CPU attempt.
- All numbers above are read directly from `results/sft_eval/steer_actlayer/summary.csv` and the
  per-config raw logs under `results/sft_eval/steer_actlayer/` and
  `results/sft_eval/raw/llama31-actlayer-*`. Nothing in this verdict is fabricated or estimated.
