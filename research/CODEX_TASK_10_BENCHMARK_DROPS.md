# CODEX TASK 10: Benchmark Drops, Semantic Gains, and Mitigations

Created: 2026-07-08

## Objective

Explain why coherence-SFT yields large semantic/human-alignment gains but broad
standard benchmark drops, then identify mitigation experiments that preserve the
semantic gains with less capability loss.

## Known facts from Tasks 1-8

Strong gains:

- Generation coherence: base `0.32` to lowLR `0.76`.
- THINGS-human alignment: base around `0.47` to lowLR around `0.65`.
- THINGS odd-one-out agreement: base `0.686` to lowLR `0.770`.
- THINGS triplet similarity Spearman: base `0.483` to lowLR `0.711`.
- Similarity tasks with gains: WordSim-353, MEN, MTurk-771, SimVerb-3500.

Broad drops:

- Full capability battery for lowLR has no >0.02 gains, 4 flat groups, and 6 drops.
- Drops include HellaSwag, OpenBookQA, MMLU mean, ARC-Easy, ARC-Challenge, WiC.
- 57/59 MMLU subjects drop for lowLR in the full evaluation.

Useful existing result files:

- `results/sft_eval/REPORT.md`
- `results/sft_eval/TASK6_SUMMARY.md`
- `results/sft_eval/human_tasks/summary.csv`
- `results/sft_eval/mitigation/summary.csv`
- `results/sft_eval/steer/steer_retention.csv`
- `results/sft_eval/separability/SUMMARY.md`

## Questions to answer

1. Are drops mostly caused by answer-format drift, lost factual knowledge,
   calibration/logit distribution shift, instruction-following changes, or
   benchmark-specific prompt mismatch?
2. Why do semantic/human tasks gain?
3. Does alpha `0.25` preserve enough semantic gain while avoiding most capability
   loss, and should it become the default fMRI arm?
4. Which mitigations are cheapest and scientifically cleanest?

## Analysis plan

### A. Summarize deltas

Build one table across arms:

- `base`
- `real`
- `lowLR`
- `lowrank`
- `taskvec_a0p25`
- `taskvec_a0p5`

Columns:

- coherence
- THINGS-human r2
- human behavior metrics
- similarity benchmarks
- capability benchmarks
- retention delta vs base

Output:

- `results/sft_eval/benchmark_diagnosis/delta_table.csv`
- `results/sft_eval/benchmark_diagnosis/pareto_semantic_vs_capability.png`

### B. Inspect benchmark mechanics

For dropped tasks, check:

- Is evaluation log-likelihood multiple-choice or generation?
- Did the model's likelihood distribution flatten or become overconfident?
- Are drops concentrated in commonsense, factual recall, lexical sense, or
  multi-hop reasoning?
- Does WiC drop suggest lexical sense discrimination was harmed despite semantic
  similarity gains?

### C. Explain gains

Hypothesis:

- SFT teaches a consistent concept similarity manifold derived from NOVA feature
  norms, so gains should appear on tasks whose target is similarity/category/
  feature structure.
- Gains should be largest when the benchmark asks for graded semantic relatedness
  or human object-concept judgments.
- Gains should not transfer to arbitrary factual/reasoning tasks unless the
  needed answer is controlled by the same concept manifold.

Evidence to collect:

- Correlate benchmark delta with task semantic-similarity-ness, if possible.
- Inspect examples from THINGS odd-one-out and similarity tasks where lowLR fixes
  base.
- Inspect examples from WiC/ARC/MMLU where lowLR loses base.

### D. Mitigation candidates

Cheapest first:

1. Use task-vector `alpha=0.25` as the default aligned model.
   - Already recovers most coherence gain for much less retention loss.
2. Try alpha grid between `0.10` and `0.35`.
   - Goal: find an elbow with THINGS-human gain but minimal capability loss.
3. Adapter composition / DARE-style sparsification.
   - Keep only high-signal LoRA deltas or attenuate layers/modules most associated
     with capability loss.
4. Add KL-to-base during SFT.
   - More training but directly targets retention.
5. Mix small general-instruction replay during SFT.
   - Could reduce forgetting, but risks muddying the clean causal story.
6. Layer/module ablation of LoRA adapter.
   - Identify whether MLP or attention deltas drive gains vs drops.

## Completion criteria

- `results/sft_eval/benchmark_diagnosis/REPORT.md` exists.
- It states a grounded mechanism for semantic gains and capability drops.
- It proposes the next two mitigation runs with exact commands and expected cost.
- It is committed and pushed.
