# Resume: frontier feature self-verification (blocked on OpenRouter credits)

Frontier runs stopped at feature self-verification because the OpenRouter key hit
**402 Insufficient credits** (2026-07-05). Everything upstream is done:
- gpt-5.5 / claude-opus-4.8: triplet + pairwise + feature LISTING complete.
- unions built: results/raw/{gpt-5.5,claude-opus-4.8}/verify_pairs.csv
  (gpt-5.5 7500 pairs, opus 7200 pairs).

## When credits are topped up, run exactly:

```bash
cd .../coherence_experiments
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
export OPENROUTER_API_KEY=$(cat .openrouter_key)   # or a fresh key

python src/run_openrouter.py --model gpt-5.5 --methods feature \
  --pairs_file verify_pairs.csv --feature_batch 20 --reasoning_effort low --overwrite
python src/run_openrouter.py --model claude-opus-4.8 --methods feature \
  --pairs_file verify_pairs.csv --feature_batch 20 --reasoning_effort low --overwrite

# then rebuild the full 5-model 3-method matrix:
cd src
MODELS="gpt-5.5 claude-opus-4.8 llama-3.1-8b-instruct mistral-7b-instruct-v0.3 qwen2.5-7b-instruct"
python method_matrix.py --models $MODELS
python compute_coherence.py --models $MODELS \
  --human_rdm ../data/human/leuven_similarity.npy \
  --human_embedding ../data/human/leuven_embedding.npy --ref_name human
```

Estimated cost of the two feature runs (batched, tight): well under $10 total.
```
gpt-5.5:  ~375 batched calls
opus-4.8: ~360 batched calls
```
