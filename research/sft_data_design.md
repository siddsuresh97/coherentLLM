# Building SFT data for coherence (autoresearch synthesis)

## Winning recipe: shared-ground-truth multi-view distillation
Derive ALL THREE views (triplet, pairwise, feature) from ONE NOVA-derived concept x concept
similarity matrix S* (cosine over NOVA feature rows) so the views are mutually consistent
BY CONSTRUCTION. Coherence becomes a property of the DATA, not something the model discovers.
Direct template: AligNet (Muttenthaler 2024, arXiv:2409.06509, Nature 2025) - KL similarity-
distillation to a teacher geometry + L2 anchor to pretrained weights. Machinery: Similarity-
Preserving KD (1907.09682), Relational KD (1904.05068).

## Data generation from NOVA
1. Fit S* once from NOVA verified matrix (786 concepts, ~700 feats each). Optionally VICE/SPoSE
   low-dim embedding for cleaner triplet sampling.
2. SPLIT BY CONCEPT (not example): ~600 train concepts; hold out the 128 coherence-test concepts
   (+ 60/30 nested) so they NEVER appear in any training example in any view. Most important
   control - forces generalizing a structuring principle, not memorizing per-item answers.
   Check synonyms/sub-superordinates don't leak.
3. Three mutually-consistent views, matching src/prompts.py formats VERBATIM:
   - triplet: (A,B,C), label "more similar to A" from S*; sample multi-abstraction the AligNet
     way (cluster S*, within- vs across-cluster). ~40-80k.
   - pairwise: map S* -> 1-7 scale (invert 7-rating). ~20-40k.
   - feature: exact "In one word True/False ... property {x} true for {y}?" (NOVA/mixtral_ft.py).
     balance T/F. ~15-30k.
   Mixture ~50/25/25 triplet/pairwise/feature.
4. Format as chat instruction->response with RESPONSE-ONLY loss masking (mask prompt; loss on
   answer token only). Like Centaur.

## Proven on Llama-3.1 specifically
- Centaur (Binz 2024, arXiv:2410.20268, Nature 2025): Llama-3.1-70B QLoRA r=8, 4-bit, all
  non-embedding layers, response-only mask, 1 epoch, lr 5e-5, batch 32, on Psych-101 -> predicts
  held-out behavior AND internal reps became more human-aligned. Text SFT CAN move geometry.
- Sensorimotor norms FT (Wu 2026, arXiv:2603.03313): base RDM gained human block structure
  (Spearman 0.13->0.57-0.72), a SELECTIVE reorganization.

## LoRA vs full FT (target IS the representation, so this matters)
- LoRA introduces "intruder dimensions" (2410.21228), changes reps less deeply; effective rank
  << nominal. rsLoRA higher-rank mirrors full FT better. LoRA learns less/forgets less (2405.09673).
  Exact adaptation needs rank ~ embed/2 (2310.17513) - r=8-64 far below arbitrary change.
- Guidance: start QLoRA but rsLoRA r=64-128 (NOT r=8), all attn+MLP proj. Run FULL FT as ceiling
  (8B affordable on H100). If QLoRA moves generation but not logprob coherence, that gap is itself
  a publishable result. Don't conclude "can't" from LoRA alone.

## Pitfall: reporting vs representation (we already saw this - OLMo Finding 4)
SFT can teach FORMAT not geometry (LIMA/URIAL superficial-alignment; Kung&Peng shortcut 2305.11383).
Detect via linear probes on mid-layer activations (Quirky models 2312.01037: probes recover TRUE
knowledge even when generation reports otherwise).
MANDATORY GUARDS:
1. DUAL eval: coherence from generation AND from logprob scoring (lineage_dual.py). Gap = reporting-only.
2. Held-out-concept transfer: coherence must rise on the 128 unseen concepts.
3. Cross-format generalization: Procrustes r2 across views IS this test; also RSA model-RDM vs S*.
4. DELUSIVE-LABEL control arm: train on SCRAMBLED S* labels; if coherence still rises, it was format.

## Prioritized experiments
1. Stage-1 multi-view SFT from shared S*, QLoRA rsLoRA r=64, response-only mask, held-out 128.
   Dual gen/logprob Procrustes r2. (core hypothesis)
2. Full-FT arm = representational ceiling; compare gen-vs-logprob gap vs QLoRA.
3. Delusive-label (scrambled S*) control - rule out format learning.
4. Stage-2 CoIN/AligNet aux loss (hidden-rep contrastive across the 3 prompts, or KL-on-sim + L2
   anchor) if Stage 1 moves reporting but not representation.
5. Stage-3 self-consistency filtering (GV-consistency 2310.01846) - does coherence self-propagate
   to unsupervised concepts?

Working QLoRA template on disk: /mnt/dv/wid/.../sid/Projects/ft_llms/mixtral_ft.py (adapt).
Framing: Sucholutsky "Getting aligned on representational alignment" (2310.13018);
Platonic Rep Hypothesis (2405.07987) makes distilling a target geometry well-posed.
BibTeX: add real entries from the arXiv IDs; do not fabricate.
