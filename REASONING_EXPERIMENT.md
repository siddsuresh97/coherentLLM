# Reasoning as a variable: does reasoning change coherence?

Decision (2026-07-05):
- **Main leaderboard = reasoning OFF** for all OpenRouter frontier models (cost, and
  more comparable to humans who judge intuitively). Small token budgets.
- The 3 already-run frontier models (gpt-5.5, opus, sonnet) used reasoning=low;
  kept as-is (noted as a caveat, not re-run).
- **Reasoning is studied as its own variable** on ONE good, cheap model: run it
  reasoning-ON and reasoning-OFF, compare coherence. Answers "does reasoning make a
  model more internally coherent?" without re-running everything.

## The A/B model
Candidate: a capable but inexpensive OpenRouter model that supports a reasoning
toggle, e.g. `z-ai/glm-5.2` or `google/gemini-3.5-flash`. Run the full pipeline both
ways:
```bash
# OFF (default)
bash scripts/run_frontier_model.sh <model>           # reasoning_effort none
# ON  (separate output dir via a suffixed registry name, or --reasoning_effort high)
python src/run_openrouter.py --model <model> --methods triplet pairwise \
  --reasoning_effort high  --overwrite   # write to a <model>-reasoning variant
```
Then compare cross-method coherence and human alignment between the two.

## Local open models
Local models are FREE (GPU), so reasoning is fine there where a model supports it
(e.g. Qwen3, R1-distills, gpt-oss). Same caveat: don't mix reasoning-on and -off in
one leaderboard; treat as a separate comparison.
