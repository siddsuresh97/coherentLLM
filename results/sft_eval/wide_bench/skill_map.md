# Wide-Benchmark Skill Map

Last updated: 2026-07-08

This map groups the current wide-benchmark deltas by the skill each benchmark
mostly probes. Deltas are relative to the base Llama-3.1-8B-Instruct run in
`results/sft_eval/wide_bench/raw_task_summary.csv`. The `taskvec_a0p25` ARC
and HellaSwag results and the scrambled ARC results are from completed split
runs under `results/sft_eval/wide_bench/runs/`.

| Task | Skill proxy | Base | Lowrank delta | Taskvec 0.25 delta | Scrambled delta | Read |
|---|---|---:|---:|---:|---:|---|
| PIQA | Physical affordance commonsense | 0.799 | -0.023 | -0.010 | -0.259 | Mostly preserved by task-vector mitigation |
| OpenBookQA | Science facts plus commonsense facts | 0.490 | -0.082 | -0.054 | -0.208 | Hurt, but task-vector helps |
| CommonsenseQA | Associative commonsense | 0.651 | -0.029 | -0.070 | -0.454 | Mixed; semantic SFT beats scrambled but task-vector underperforms lowLR |
| WiC | Word-sense disambiguation | 0.652 | -0.154 | -0.150 | -0.171 | Strongly hurt across aligned and scrambled states |
| TruthfulQA-MC2 | Truthfulness and misconception calibration | 0.550 | -0.010 | -0.028 | -0.068 | Mildly hurt; sensitive to calibration and plausible-false lures |
| WinoGrande | Coreference and discourse commonsense | 0.762 | -0.018 | -0.015 |  | Preserved |
| ARC-Easy | Grade-school science retrieval | 0.850 | -0.145 | -0.042 | -0.544 | Improved by task-vector but still hurt; catastrophic under scrambled SFT |
| ARC-Challenge | Hard science reasoning | 0.649 | -0.141 | -0.090 | -0.424 | Improved by task-vector but still hurt; catastrophic under scrambled SFT |
| HellaSwag | Script and event plausibility | 0.685 | -0.002 | -0.005 | running | Preserved by lowrank and task-vector |
| MMLU | Broad exam knowledge and reasoning | 0.693 | -0.101 |  |  | Hurt; task-vector pending |

## Interpretation

The induced skill looks closest to format-invariant semantic association over
object concepts: it helps hidden-state hub metrics and the THINGS-style semantic
tasks, and it survives best on benchmarks that can be solved with broad event,
affordance, or discourse plausibility. That explains why HellaSwag, WinoGrande,
and task-vector PIQA are flat or near-flat.

The hurt skills are those that need sharper decision boundaries than associative
similarity: lexical sense discrimination in WiC, science/exam retrieval in ARC
and MMLU, and option calibration in OpenBookQA. The ARC result is now more
nuanced: task-vector `alpha=0.25` recovers a large part of the lowrank ARC loss
but still remains below base, so the mitigation helps science retrieval without
fully restoring the base model's multiple-choice ranking surface. The scrambled
ARC control is much worse (`ARC-Easy -0.544`, `ARC-Challenge -0.424`), so ARC is
not just detecting the induced semantic skill; it is also highly sensitive to
generic rank-64 adapter/SFT perturbation.

TruthfulQA is not one of the largest lowrank drops in the current numbers:
lowrank is only `-0.010` and lowLR is `-0.017`, both flat by the current
`0.02` verdict threshold. It becomes more negative under task-vector
`alpha=0.25` (`-0.028`) and especially under scrambled SFT (`-0.068`). The
likely mechanism is calibration rather than loss of truth knowledge: TruthfulQA
MC2 rewards probability mass on truthful answers and penalizes attractive
misconception lures. Training on semantic relatedness can make plausible but
false answers look more competitive, while scrambled training shows that generic
adapter perturbation also damages this scoring surface.

The 200-item log-sample diagnostic refines this. Lowrank improves over base on
the bounded slice (`0.5528` vs `0.5224`). The paired decomposition in
`results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/REPORT.md` shows
that lowrank lowers both true and false answer log-likelihood mass, but lowers
false-answer pressure more (`delta_false_logsumexp=-3.9391` vs
`delta_true_logsumexp=-3.5444`), so mean truth log-odds rises by `+0.3946`.
Task-vector `alpha=0.25` is almost flat on aggregate (`0.5267`) but has the
opposite pressure pattern: both true and false masses rise, and false pressure
rises more (`+4.9265` vs `+4.3026`), so mean truth log-odds falls by
`-0.6239`. Scrambled drops to `0.4701`; its mean truth log-odds rises only
because it suppresses both true and false likelihoods extremely hard, producing
catastrophic item-level flips. Overall this supports a calibration/relative
ranking story rather than a simple "truth skill got worse" story.

## Next Tests

- Finish `taskvec_a0p25` MMLU 5-shot to see whether the ARC mitigation extends
  to broad exam knowledge.
- Finish scrambled HellaSwag 10-shot to test whether script/event plausibility
  remains uniquely preserved under a random-label perturbation.
- Use the TruthfulQA item-level deltas to group failure classes:
  medical/safety myths, conspiracy lures, stereotype/generalization lures, and
  ordinary factual confusions.
- Run WiC error slices by same-lemma similarity and part-of-speech to test
  whether the semantic objective collapses word-sense boundaries.
