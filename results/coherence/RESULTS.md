# Coherence results (v1)

Bridge data for dissertation chapters 1<->2. Re-run of the three semantic
elicitation methods from arXiv:2510.01030 on new models, over the 30 concepts
(15 reptiles + 15 tools), 3000 triplets + 900 pairs.

## Models
- Local (vLLM, shared_models): llama-3.1-8b-instruct, mistral-7b-instruct-v0.3,
  qwen2.5-7b-instruct
- Frontier (OpenRouter): claude-opus-4.8, gpt-5.5 (triplet + pairwise only so far)

## Headline: cross-method coherence (triplet vs pairwise RSA)
| model | triplet~pairwise | FastText RSA (pairwise) | Procrustes R^2 (triplet) |
|---|---|---|---|
| gpt-5.5                | 0.818 | 0.602 | 0.364 |
| claude-opus-4.8        | 0.772 | 0.641 | 0.437 |
| llama-3.1-8b-instruct  | 0.630 | 0.468 | 0.470 |
| mistral-7b-instruct-v0.3 | 0.415 | 0.487 | 0.484 |
| qwen2.5-7b-instruct    | 0.346 | 0.444 | 0.502 |

Frontier models are much more internally coherent across methods than the 8B
open models. Small-model ordering is sensible (Llama-3.1 > Mistral-7B > Qwen2.5-7B
on triplet~pairwise consistency).

## Figures / tables
- `method_matrices.png` : per-model 3x3 method-agreement (triplet, pairwise, feature).
- `coherence_matrix.csv` : all metrics per model.
- `summary_reliable.csv` : the table above.

## Caveats
1. **Feature verification is interim.** The Leuven-derived 764-feature list barely
   describes tools, so even a 36-feature discriminative subset (12-30% true-rate)
   yields low triplet~feature agreement (0.07-0.24). Feature method was NOT run on
   the frontier models (saving API cost) pending a ground-truth concept x feature
   key or a tool-aware feature list from the user.
2. **Human reference is a FastText proxy**, not human judgments. All `fasttext_*`
   columns are alignment to that proxy. Swap in real human data on these 30
   concepts via `--human_rdm` when available.
3. Procrustes R^2 uses metric MDS on the derived similarity (lightweight stand-in
   for the paper's ordinal embedding); RSA is the more directly comparable number.

## Reproduce
Local:   `scripts/run_all_local.sh 200 <model...>`
Frontier: `OPENROUTER_API_KEY=... python src/run_openrouter.py --model gpt-5.5 --methods triplet pairwise`
Analyze: `python src/compute_coherence.py --human_rdm data/human/fasttext_similarity.npy --human_embedding data/human/fasttext_embedding.npy --ref_name fasttext`
         `python src/method_matrix.py`
