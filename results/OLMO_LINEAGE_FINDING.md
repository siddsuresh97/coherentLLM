# OLMo-2-7B training-stage coherence: two measurements, two stories

We measured cross-method coherence (triplet~pairwise RSA) at four training stages
of the OLMo-2-7B pipeline: base (pretrained) -> SFT -> DPO -> RLVR (Instruct).

The result **depends on how you elicit the judgments**, and that contrast is the finding.

## Measurement A: generation / instruction-following
Model reads the instruction prompt and generates an answer. (Base can barely do this.)
| stage | triplet~pairwise |
|---|---|
| base | (unusable - can't follow instructions) |
| SFT  | 0.26 |
| DPO  | 0.39 |
| RLVR | 0.44 |
-> Coherence *rises* with post-training. But base is not comparable (it can't do the task).

## Measurement B: logprob (uniform, representation-level)
Pick the higher-likelihood completion; same method for every stage, so base is comparable.
| stage | triplet~pairwise | human_rsa_triplet |
|---|---|---|
| base | 0.30 | 0.39 |
| SFT  | 0.16 | 0.34 |
| DPO  | 0.17 | 0.36 |
| RLVR | 0.15 | 0.37 |
-> Coherence does NOT rise; the base representation is already (the most) coherent, and
post-training slightly *reduces* representation-level cross-method coherence while
nudging human-alignment up a little.

## Interpretation
Post-training mainly teaches the model to *express* its similarity structure through
instructions (Measurement A improves because the model can now do the task), but does
NOT increase the coherence of the underlying representations (Measurement B is flat/down).
The pretrained model already encodes a coherent similarity structure; instruction/pref
tuning changes how accessibly it is *reported*, not how coherent it *is*.

Caveat: logprob-pairwise ratings for post-trained models are bimodal (many 1s and 5s)
vs the base's smoother spread - a real behavioral shift from preference tuning, not a
measurement artifact (ratings are non-degenerate). Both curves are saved:
results/coherence/coherence_matrix.csv (generation) and olmo_curve_logprob.csv (logprob).

## OLMo-2 vs Tulu-3 (why run both)
- **OLMo-2-7B**: AI2's from-scratch base (open pretraining data) + Tulu-3 post-training.
- **Tulu-3-8B**: **Llama-3.1-8B** base (Meta) + the SAME Tulu-3 post-training recipe.
So post-training is ~constant across the two lineages; the base/pretraining differs.
- If the "generation rises, representation flat" pattern replicates on Tulu, it's a
  property of post-training, not of OLMo's specific (weaker) base.
- Bonus same-base comparison: Tulu-3-8B-final (Llama-3.1 + AI2 recipe) vs
  llama-3.1-8b-instruct (Llama-3.1 + Meta recipe) - two recipes on one base.
