# Resume frontier OpenRouter runs (blocked on credits, 2026-07-05)

Credits ran out mid-batch (402). Completed frontier models (all 3 methods, saved):
gpt-5.5, claude-opus-4.8, claude-sonnet-5.

NOT yet run (credits died before they produced anything): the 7 new models
gemini-3.5-flash, deepseek-v4-pro, gpt-oss-120b, minimax-m3, kimi-k2.6, glm-5.2,
mistral-medium-3.

The runner is now hardened: a 402 stops the run early and SAVES partial results,
so re-running after top-up is safe and resumable (each stage skips if its CSV exists).

## When credits are added:
```bash
cd .../coherence_experiments
source $(conda info --base)/etc/profile.d/conda.sh
# run the sequential batch (Sonnet already done -> it skips; the 7 new models run)
bash scripts/run_frontier_batch.sh
# then rebuild the full matrix (see below)
```

## Full matrix rebuild (any time):
```bash
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
cd src
MODELS="gpt-5.5 claude-opus-4.8 claude-sonnet-5 gemini-3.5-flash deepseek-v4-pro \
gpt-oss-120b minimax-m3 kimi-k2.6 glm-5.2 mistral-medium-3 \
llama-3.1-8b-instruct mistral-7b-instruct-v0.3 qwen2.5-7b-instruct \
olmo2-7b-base olmo2-7b-sft olmo2-7b-dpo olmo2-7b-instruct"
python compute_coherence.py --models $MODELS \
  --human_rdm ../data/human/leuven_similarity.npy \
  --human_embedding ../data/human/leuven_embedding.npy --ref_name human
python method_matrix.py --models $MODELS
```
Note: frontier feature = batched (validated safe); open models = single-pair.
