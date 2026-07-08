# Wide-Benchmark Skill Map

Last updated: 2026-07-08

This map groups the current wide-benchmark deltas by the skill each benchmark
mostly probes. Deltas are relative to the base Llama-3.1-8B-Instruct run in
`results/sft_eval/wide_bench/raw_task_summary.csv`. The `taskvec_a0p25`
HellaSwag result is from the completed split run under
`results/sft_eval/wide_bench/runs/taskvec_a0p25_hellaswag_10shot/`.

| Task | Skill proxy | Base | Lowrank delta | Taskvec 0.25 delta | Scrambled delta | Read |
|---|---|---:|---:|---:|---:|---|
| PIQA | Physical affordance commonsense | 0.799 | -0.023 | -0.010 | -0.259 | Mostly preserved by task-vector mitigation |
| OpenBookQA | Science facts plus commonsense facts | 0.490 | -0.082 | -0.054 | -0.208 | Hurt, but task-vector helps |
| CommonsenseQA | Associative commonsense | 0.651 | -0.029 | -0.070 | -0.454 | Mixed; semantic SFT beats scrambled but task-vector underperforms lowLR |
| WiC | Word-sense disambiguation | 0.652 | -0.154 | -0.150 | -0.171 | Strongly hurt across aligned and scrambled states |
| TruthfulQA-MC2 | Truthfulness and misconception calibration | 0.550 | -0.010 | -0.028 | -0.068 | Mildly hurt; sensitive to calibration and plausible-false lures |
| WinoGrande | Coreference and discourse commonsense | 0.762 | -0.018 | -0.015 |  | Preserved |
| ARC-Easy | Grade-school science retrieval | 0.850 | -0.145 |  |  | Hurt; task-vector pending |
| ARC-Challenge | Hard science reasoning | 0.649 | -0.141 |  |  | Hurt; task-vector pending |
| HellaSwag | Script and event plausibility | 0.685 | -0.002 | -0.005 |  | Preserved by lowrank and task-vector |
| MMLU | Broad exam knowledge and reasoning | 0.693 | -0.101 |  |  | Hurt; task-vector pending |

## Interpretation

The induced skill looks closest to format-invariant semantic association over
object concepts: it helps hidden-state hub metrics and the THINGS-style semantic
tasks, and it survives best on benchmarks that can be solved with broad event,
affordance, or discourse plausibility. That explains why HellaSwag, WinoGrande,
and task-vector PIQA are flat or near-flat.

The hurt skills are those that need sharper decision boundaries than associative
similarity: lexical sense discrimination in WiC, science/exam retrieval in ARC
and MMLU, and option calibration in OpenBookQA. This is consistent with a LoRA
update that improves one semantic geometry while perturbing the base model's
multiple-choice ranking surface.

TruthfulQA is not one of the largest lowrank drops in the current numbers:
lowrank is only `-0.010` and lowLR is `-0.017`, both flat by the current
`0.02` verdict threshold. It becomes more negative under task-vector
`alpha=0.25` (`-0.028`) and especially under scrambled SFT (`-0.068`). The
likely mechanism is calibration rather than loss of truth knowledge: TruthfulQA
MC2 rewards probability mass on truthful answers and penalizes attractive
misconception lures. Training on semantic relatedness can make plausible but
false answers look more competitive, while scrambled training shows that generic
adapter perturbation also damages this scoring surface.

## Next Tests

- Finish `taskvec_a0p25` ARC 25-shot and MMLU 5-shot to see whether the
  task-vector mitigation also protects science/exam retrieval.
- Run a TruthfulQA error slice comparing correct-answer logprob, best
  false-answer logprob, and margin for base vs lowrank vs task-vector vs
  scrambled. If margins shrink mainly on misconception lures, the issue is
  semantic attraction/calibration rather than missing factual knowledge.
- Run WiC error slices by same-lemma similarity and part-of-speech to test
  whether the semantic objective collapses word-sense boundaries.
