# CODEX TASK 8: activation steering DONE RIGHT (middle layers, not all layers)

Branch coherence-sft. Env/GPU rules as Task 5-7. Queue behind Task 6 + Task 7; use free GPUs; never idle; commit each step; do NOT push.

## Why (fixing a flawed negative result)
Our earlier `actdiff` route added the coherence direction to the output of EVERY decoder
layer (33 layers) simultaneously (see src/sft/apply_coherence_vector.py docstring: "adds
alpha * actdiff[layer] to each decoder layer output"). Adding a perturbation at all layers
compounds and blows up the residual stream -> generation collapsed at every alpha. That is
an ALL-LAYERS ARTIFACT, not evidence that activation steering fails. The activation-steering
literature (ActAdd/Turner, RepE/Zou, ITI) steers a NARROW BAND OF MIDDLE LAYERS. Redo it
properly before concluding.

We already have the per-layer vector: data/sft/steer/coh_vector_actdiff.npz -> `diff` is
(33, 4096), one direction per layer. Llama-3.1-8B = 32 decoder layers.

## What to run
Modify the actdiff application (a --layers option) so it injects ONLY at a chosen subset of
layers, and NORM-MATCH the injection to avoid blow-up:
  h_L <- h_L + alpha * (diff_L / ||diff_L||) * ||h_L||        (per-token, at layers in the set)
(If norm-matching is awkward, at minimum restrict to the layer subset with raw addition and
smaller alpha.)

**Grid (2D: layers x alpha):**
- LAYER SETS: single-layer sweep at L in {8, 10, 12, 14, 16, 20}, AND a band {10-14}, {12-16}.
- ALPHA: {2, 4, 6, 8} for norm-matched (these are relative-to-hidden-norm, so larger is fine);
  if raw addition, use {0.5, 1, 2}.
- Start coarse: best single layer first (sweep L at one moderate alpha), then refine alpha at
  the best layer(s) and try the 2-layer bands around it.

**Per config**: generate triplet/pairwise/feature on the 128 THINGS concepts (reuse the
run-one path), SALMON d5, coherence (proc mean) + human_things r2. Watch for DEGENERATION
(near-zero/NaN coherence, repeated tokens) and flag it — that's the signal a layer/alpha is
over-steering.

## The question
Does mid-layer activation steering raise coherence toward the SFT (0.70) / taskvec-a0.25
(0.65) WITHOUT collapsing generation, at some layer x alpha? 
- If YES: we have a TRUE activation-steering result (training-free, inference-time), and can
  compare it head-to-head with the weight-space task vector. Report best (layer, alpha) and
  its coherence + human_r2 + a generation-quality sanity check (are the outputs coherent text,
  not repetition?).
- If NO across the whole mid-layer grid: THEN the "activation steering fails for coherence"
  claim is earned (not an all-layers artifact) — report it as a proper negative with the grid.

## Output
- results/sft_eval/steer_actlayer/summary.csv: layer_set, alpha, coherence, human_r2,
  degenerate(bool), notes.
- A short heatmap/plot coherence over (layer x alpha) if easy.
- Commit "Task8: mid-layer activation steering sweep (corrected actdiff)".
- Update the report's actdiff claim: replace "activation steering collapses at every alpha"
  with the correct mid-layer result (win or earned-negative).

## Notes
One vLLM/HF model per process; the forward-hook injection needs the HF path (not vLLM) since
it edits hidden states — reuse whatever apply_coherence_vector used for actdiff. env-activate
every call. Do NOT fabricate; report on-disk numbers + real generation samples.
