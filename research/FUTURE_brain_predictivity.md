# Future direction: does coherence-SFT improve LLM brain-predictivity?

## The idea in one sentence
We have a controlled pair (and a continuous sweep) of Llama-3.1-8B models that differ
*only* along one interpretable axis — human-like concept geometry (the coherence task
vector). Feed those models into the standard LLM-to-fMRI encoding / RSA machinery
(Ivanova, Huth/LeBel, Schrimpf, Antonello) and ask: **does making a model's semantics
more human-aligned make its features better predictors of the human brain?**

## Why this is a real contribution
Brain-encoding papers show LLM features predict fMRI, and that bigger / better-LM-loss
models tend to predict better. But they lack a *causal, targeted* manipulation of
semantic structure: their comparisons confound size, data, training compute. We have:
- **A clean 2-point contrast**: base vs coherence-SFT (lowLR) — same base weights,
  same size, differ only in a LoRA delta that provably raises human concept-alignment.
- **A scrambled control**: identical training on permuted labels — isolates "aligned
  structure" from "any fine-tuning."
- **A dose-response sweep**: base + alpha*(taskvec), alpha in {0, 0.25, 0.5, 1.0}. If
  brain-predictivity rises monotonically with alpha, that's causal evidence a single
  human-alignment direction carries brain-relevant information — far stronger than a
  two-model comparison.
This reframes "more human-aligned representation" from a behavioral claim (our odd-one-out
+0.084, similarity transfer) into a **neural** one, and gives the brain-encoding literature
a manipulation-based test instead of a correlational model-zoo sweep.

## Models already on disk (no new training needed for Phase 1)
- base Llama-3.1-8B-Instruct
- coherence-SFT: real, lowLR (best, retention-safe), lowrank, scrambled (control)
- steering arms: base + alpha*taskvec for alpha in {0.25, 0.5, 1.0}
All are LoRA adapters (or base+delta); features extractable with one forward pass.

---

## Phase 1 (runnable NOW): RSA on THINGS-fMRI — extend Chapter 4
Reuse the existing Ch4 pipeline verbatim (THINGS-fMRI, 3 subjects, Glasser HCP MMP1.0,
per-region cosine-RDM, Spearman RSA; see memory ch4-fmri-methods). Only the "model" changes.

**Procedure**
1. For each model {base, lowLR, lowrank, scrambled, steer-a0.25/0.5/1.0}: extract
   concept representations for the THINGS concepts (last-token hidden states, ALL layers).
   Build a concept x concept model RDM per layer.
2. Per Glasser macro-ROI (Early Visual, Ventral, Dorsal, **Language**, Prefrontal, **ATL**),
   compute RSA (Spearman) of model-RDM vs brain-RDM, best-layer, mean +/- SE over 3 subjects.
3. **Two analyses on this grid:**
   - **alpha dose-response**: RSA vs alpha per ROI. Monotone rise = the coherence direction
     is brain-relevant. Expect the effect concentrated in semantic ROIs (ATL, Language),
     not Early Visual (control ROI — should be flat; a nice specificity check).
   - **layer x alignment**: for each model, which layer peaks in each ROI, and does
     coherence-SFT *shift the peak layer* or just *raise the ceiling*? (Huth/Toneva-style
     layer-depth analysis — do aligned models push semantic representation to different depths?)

**Cost**: cheap. Forward passes on ~1854 THINGS concepts x 7 models; RSA is CPU. Days, on
the 2xA5000 + H100 we already use. This is the proof-of-concept and a direct Ch4/Ch5 result.

**Predictions / kill criteria**
- SUCCESS: lowLR > base in ATL + Language RSA; scrambled ~ base or worse; alpha monotone in
  semantic ROIs; Early Visual flat across all (specificity).
- NULL that's still interesting: if alpha is flat everywhere, human-alignment != brain-alignment
  at the concept-RDM level — worth reporting given our r=0.914 coherence<->human coupling.

---

## Phase 2 (the stronger, novel claim): voxelwise encoding on language fMRI
The Huth/Ivanova paradigm proper: ridge-regress LLM layer activations onto fMRI voxels
during naturalistic language, score prediction r on held-out timepoints/story.

**Datasets (pick by access)**
- **LeBel/Huth "reading/listening"** (UT Austin, 8 subjects, hours of narrative) — the
  canonical Huth encoding dataset; large per-subject data = high-power within-subject tests.
- **Narratives (Nastase/Hasson)** — many subjects, openly on OpenNeuro; good for group tests.
- **Pereira 2018** (sentences -> concept decoding) — closest to our concept-level framing,
  bridges nicely from Phase 1 RSA.
- **"Little Prince"** multilingual — if we want a cross-language angle later.

**Procedure**
1. Tokenize the stimulus transcript; extract per-token hidden states per layer for each
   model; align to TRs (lag/HRF convolution as in Huth/LeBel).
2. Voxelwise ridge encoding (banded ridge, nested CV) per model x layer; score prediction r
   on held-out story, per voxel, cortical maps.
3. Same two analyses: alpha dose-response on mean encoding r (in a language-network mask),
   and layer x alignment (does coherence-SFT change the best-predicting layer?).
4. Compare against the model-zoo trend they report: where does our aligned 8B land vs its
   own base on the size/quality curve — does targeted alignment "buy" what scaling buys?

**Cost**: heavier (banded ridge over ~50-100k voxels x layers x 7 models), but standard;
himalaya/voxelwise-modeling libs exist. H100 for feature extraction; ridge on CPU/GPU.

**Predictions**
- If Phase 1 shows semantic-ROI RSA gains, Phase 2 should show encoding-r gains concentrated
  in the language network, monotone in alpha, scrambled flat.
- Layer analysis: coherence-SFT may raise mid/late-layer predictivity (where semantics live)
  without touching early layers — a mechanistic signature.

---

## Framing / positioning
- Ties Ch4 (neural alignment of vision models) to Ch3 (LLM coherence) and this SFT work:
  "aligning a model's concept geometry to humans improves its fit to the human brain."
- Novelty vs Ivanova/Huth: a **causal, direction-specific** manipulation (task-vector sweep)
  rather than a correlational model comparison; and a **retention-controlled** aligned model
  (rules out "it just got better at language").
- Risk/caveat: our alignment is over concrete object concepts (NOVA/THINGS); language-fMRI
  stimuli are sentences/narratives — the transfer from concept-geometry to sentence-encoding
  is the scientific bet. Phase 1 (concept-level THINGS-fMRI) de-risks it before Phase 2.

## Open questions to resolve before building
1. Which language-fMRI dataset do we have access to / want (LeBel vs Narratives vs Pereira)?
2. Concept-only alignment vs also fine-tuning on sentence-level human-similarity — does the
   concept task vector even move sentence representations? (test cheaply: does lowLR change
   STS-B, which it did: +0.009 — weak but positive signal.)
3. Do we extend the coherence-SFT to a larger base (the encoding literature favors bigger
   models) or keep 8B for the clean controlled contrast?

## Reading to ground it (find real citations before writing)
- Ivanova et al. — LLM-brain alignment / language network.
- Huth, LeBel, Antonello et al. — voxelwise encoding, scaling laws for brain prediction.
- Schrimpf et al. — "neural architecture of language" brain-score.
- Toneva & Wehbe — interpreting/aligning LM layers to brain.
- Caucheteux & King — brain hierarchy vs LM depth.
(Do NOT fabricate; pull exact bibtex when this becomes a chapter.)
