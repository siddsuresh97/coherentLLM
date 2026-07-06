# Feature-verification: batched vs single-pair (validation)

Question: does asking a model to mark True/False for 20 features in one call ("batched")
give the same answers as one (feature, concept) per call ("single-pair")?

A/B test on 600 random pairs from each model's own union, comparing that model's
batched answers to fresh single-pair answers.

| model | class | agreement | Cohen's kappa | true-rate single | true-rate batched |
|---|---|---|---|---|---|
| gpt-5.5              | frontier | 0.922 | 0.82 | 0.36 | 0.30 |
| llama-3.1-8b-instruct | open 8B | 0.555 | 0.10 | 0.08 | 0.49 |

## Conclusion
- **Open/small models: batching is INVALID.** Llama rubber-stamps "True" down a
  20-item list (true-rate jumps 0.08 -> 0.49; kappa ~ chance). Use SINGLE-PAIR.
- **Frontier models: batching is VALID.** GPT-5.5 tracks the list faithfully
  (92% agreement, kappa 0.82 "almost perfect", true-rates close). Batched feature
  data for gpt-5.5 / opus / sonnet stands; no re-run needed.

## Policy
- OpenRouter frontier models -> batched (cheap, validated).
- Local/open models -> single-pair (`--pairs_file` without `--feature_batch`).
