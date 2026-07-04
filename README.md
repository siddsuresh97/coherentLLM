# coherence_experiments

Re-running the semantic-coherence experiments from *"Uncovering the Computational
Ingredients of Human-Like Representations in LLMs"* (Studdiford, Rogers, Mukherjee,
Suresh; arXiv:2510.01030) on newer open-weight models (8B-class, local GPU) and
frontier models (GPT-5, Claude Opus via OpenRouter).

This produces data for the **bridge between dissertation chapters 1 and 2**.

**This folder is intentionally OUTSIDE the dissertation repo and must never be pushed
to the dissertation's Overleaf/GitHub remotes.** It has its own git repo.

## Three response methods (from the EMNLP/pipeline paper)

1. **triplet** — anchored similarity ("which of B/C is more similar to A?")
2. **pairwise** — 1..7 similarity rating for each pair
3. **feature** — True/False feature verification over concept×feature pairs

## Coherence

For each model we build a 30D representation of the 30 concepts from its judgments,
then measure:
- **cross-method coherence**: agreement of the similarity structure a model produces
  across the three methods (triplet vs pairwise vs feature).
- **human alignment**: Procrustes R^2 / RSA between the model embedding and the human
  reference (pending human data on these 30 concepts; falls back to THINGS/SPoSE
  overlap).

## Layout
- `configs/` — model registry + run configs
- `data/stimuli/` — the 30 concepts, triplet set, pairwise set, feature set
- `data/human/` — human reference (triplet judgments or SPoSE embedding)
- `src/` — prompts, runners (vLLM + OpenRouter), embedding, coherence
- `results/raw/` — raw model responses (gitignored)
- `results/embeddings/`, `results/coherence/` — derived (embeddings gitignored)

## Environment
`env/` conda env with vLLM. See `scripts/setup_env.sh`.
