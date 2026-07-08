# Future direction: does coherence-SFT induce / sharpen a semantic hub?

Ref: Wu, Yun, Andreas, Kim — "The Semantic Hub Hypothesis: Language Models Share
Semantic Representations Across Languages and Modalities" (arXiv:2411.04986).

## The idea in one sentence
Coherence-SFT is, by construction, cross-format similarity supervision (triplet, pairwise,
feature all derive from one S*). The semantic-hub hypothesis predicts a coherent concept
should occupy the SAME mid-layer location regardless of how it is elicited. So: **does
coherence-SFT induce (or tighten) a format-invariant concept "core" in the middle layers —
a semantic hub — and is that the mechanistic signature of our behavioral coherence gain?**

## Why it connects to what we have
- The semantic hub is a MIDDLE-LAYER phenomenon; our task vector is a mid-layer weight
  change and we are already probing mid-layers (Task 8 activation steering).
- Our behavioral result is "judgments agree across elicitation formats." The hub hypothesis
  gives that a mechanistic prediction: the three formats route through a SHARED mid-layer
  representation. Coherence-SFT may CREATE or SHARPEN that convergence.
- The brain's semantic hub is the ATL — the same region in our Ch4 fMRI work and the
  brain-predictivity proposal (FUTURE_brain_predictivity.md). This ties all three threads:
  coherence-SFT -> LM semantic hub -> brain semantic hub (ATL).

## Assets already on disk (no new data needed)
- models: base, lowLR (best), lowrank, scrambled (control), steer-taskvec alpha sweep.
- the 128 THINGS eval concepts, each with prompts in 3 elicitation formats
  (triplet, pairwise, feature) in results/sft_eval/raw/*/.
This is mostly hidden-state analysis, not new generation.

## Test 1 (core, correlational): format-invariance of the mid-layer concept code
For each concept C and each model {base, lowLR, scrambled}:
1. Run C through each of the 3 elicitation-format prompts; capture the last-token hidden
   state at EVERY layer -> h[C, format, layer].
2. Per layer, build a concept RDM within each format, and measure CROSS-FORMAT convergence:
   - CKA / RSA between the triplet-format concept-RDM and the feature-format concept-RDM
     (and pairwise), per layer.
   - "Same-concept-across-format vs different-concept" separability: is a concept's identity
     (invariant) stronger than the format it was elicited in? (e.g. nearest-neighbour of
     h[C, triplet, L] should be h[C, feature, L], not h[other, triplet, L]).
3. PREDICTION: coherence-SFT (lowLR) raises mid-layer cross-format convergence vs base;
   scrambled does NOT. Effect should PEAK in middle layers (the hub band ~ L10-20), not
   early (token-level) or late (logit-committed) — a layer-resolved curve is the key figure.

## Test 2 (causal, the paper's signature method): cross-format intervention
Extract the similarity/coherence direction from ONE format and patch it during ANOTHER:
- Derive a mid-layer direction from triplet-format activations (e.g. the actdiff vector, or a
  concept-pair contrast), inject it during FEATURE-verification forward passes, and measure
  whether it shifts feature answers in the hub-consistent direction.
- PREDICTION: a triplet-derived mid-layer patch changes feature-verification behavior MORE in
  the coherence-SFT model than base -> the SFT built a shared hub both formats read from.
- This is the strongest evidence: shared representation is functionally USED across formats,
  not just geometrically nearby (mirrors the paper's cross-domain intervention).

## Test 3 (bridge to brain): does the induced hub predict ATL fMRI better?
- Take the FORMAT-INVARIANT mid-layer concept code (averaged/aligned across the 3 formats =
  the induced hub representation) and RSA it against the ATL fMRI RDM (Ch4 pipeline already
  has ATL + ATL subregions for the overlapping concepts).
- PREDICTION: coherence-SFT's format-invariant hub code predicts ATL (the brain's semantic
  hub) better than base's, and better than the format-specific codes. Connects LM hub -> brain hub.

## What each outcome means
- Hub SHARPENS with coherence-SFT (Test 1 up, mid-layer, scrambled flat) + causal (Test 2):
  strong result — coherence supervision induces a functional format-invariant semantic core.
- Hub UNCHANGED despite behavioral coherence gain: also notable — the behavioral agreement is
  NOT routed through a shared mid-layer core (coherence is "reported" at the output, not
  represented centrally). Ties to our OLMo "reporting vs representation" theme + the gen-vs-
  logprob gap we already saw.
- Base ALREADY has a strong hub (the paper shows base models do): then frame as "does targeted
  cross-format supervision TIGHTEN an existing hub, and where," not "create from nothing."

## Scope / caveats
- Keep to the CROSS-ELICITATION-FORMAT version of the hub (triplet/pairwise/feature) — that is
  the part our design actually manipulates. The paper's multimodal (vision/audio) and
  multilingual claims are out of scope for this setup.
- Metrics: CKA is the safe cross-format similarity measure (basis-invariant); pair with RSA
  for interpretability. Report layer-resolved curves, not single numbers.
- Compute: cheap — hidden-state extraction over 128 concepts x 3 formats x few models + CKA/RSA
  (CPU). Test 2 needs forward hooks (HF path, reuse Task 8 machinery).

## Reading to ground it
- Wu, Yun, Andreas, Kim 2024 (2411.04986) — the hub hypothesis + logit-lens + intervention method.
- Ralph/Lambon-Ralph et al. — ATL as the human transmodal semantic hub (for the brain bridge).
- (pair with the logit-lens / cross-format convergence literature; pull exact bibtex when this
  becomes a chapter. Do NOT fabricate.)
