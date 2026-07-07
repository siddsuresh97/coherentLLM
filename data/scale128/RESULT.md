# Scale experiment: does coherence hold at 128 concepts? NO (for models).

Setup: 128 concepts (category-balanced across 13 cats, all in THINGS SPoSE + NOVA).
Human triplet = THINGS SPoSE 49D; human feature = NOVA verified matrix.
Model triplet = SALMON d=5 (elbow: acc plateaus at d=5) from 10k triplets/model.
Model feature = free-list -> consolidate (features shared by >=3 concepts, ~55-272/model,
Leuven-scale) -> single-pair self-verify (~70k total). Metric: RDM-direct Procrustes r^2.

triplet~feature coherence:            n=30      n=128
  HUMAN                               0.90      0.769
  qwen2.5-32b-instruct                0.94*     0.338
  llama-3.1-8b-instruct               ~0.82*    0.318
  olmo2-7b-instruct                   ~0.80*    0.225
  qwen2.5-7b-instruct                 ~0.69*    0.126
(* n=30 full-3-method coherence for reference)

model->human alignment @ n=128: human_triplet 0.15-0.49, human_feature 0.48-0.71.

MAIN FINDING: the human-vs-model coherence gap WIDENS sharply with concept-set size.
- Humans stay coherent (0.90->0.77): robust conceptual core over a large diverse set.
- Models collapse (near-human@30 -> <=0.34@128): their @30 coherence was mostly the easy
  2-cluster reptile/tool split. On 128 diverse concepts they cannot hold a consistent
  structure across triplet vs feature elicitation.
=> Scale reverses the "modern models are as coherent as humans" impression from n=30.
   The 2023 "humans cohere, LLMs don't" result HOLDS at scale even for strong models;
   the 30-concept task was too easy to reveal it. Concept-set size is a key confound.
