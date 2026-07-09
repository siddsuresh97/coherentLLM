# Retention Failure-Suite

This directory contains the cheap gate for future alpha/replay/steering
experiments. It is derived from existing wide-bench summaries only; creating the
manifest does not launch lm-eval, submit jobs, or modify the live MMLU report.

## Files

- `task_slices.csv`: bounded benchmark slices, current base/lowrank/task-vector
  scores, and required diagnostics.
- `suite_manifest.json`: machine-readable gates, metric definitions, current
  `taskvec_a0p25` anchor, TruthfulQA mechanism summary, and mitigation
  candidates.
- `README.md`: this operational note.

Refresh or validate the generated artifacts with:

```bash
python src/sft/build_retention_failure_suite_manifest.py
python src/sft/build_retention_failure_suite_manifest.py --check
```

## First Smoke

The lowest-cost smoke test is a manifest consistency check:

```bash
python src/sft/build_retention_failure_suite_manifest.py --check
```

It verifies that `task_slices.csv` and `suite_manifest.json` still match the
current source CSVs. It does not run model inference.

## 2026-07-09 CHTC TruthfulQA Gate 5513424

Run:

- CHTC cluster: `5513424.0`
- Remote run:
  `~/chtc-runs/coherence-retention-failure-suite-20260709-031120`
- Local artifacts:
  `results/sft_eval/wide_bench/failure_suite/chtc_5513424/`
- Host/GPU: `gpu4006.chtc.wisc.edu`, NVIDIA L40S, 45GB advertised memory
- Exit evidence: Condor `ExitCode=0`, `RemoteWallClockTime=520.0`,
  `exit_status.txt == 0`, `gate_exit_status.txt == 0`
- GPU metrics: max sampled GPU utilization `100%`, max sampled memory
  `39163` MiB, mean sampled memory `14076.9` MiB

Bounded gate:

- Arms: `base`, `taskvec_a0p25`, `lowLR`
- Task: `truthfulqa_mc2`
- Limit: `40`

Aggregate slice:

| Arm | n | MC2 acc | Truth log-odds | Best true - false |
| --- | ---: | ---: | ---: | ---: |
| `base` | 40 | 0.5682 | 1.7411 | 1.8170 |
| `lowLR` | 40 | 0.5661 | 2.5413 | 2.6345 |
| `taskvec_a0p25` | 40 | 0.6037 | 2.3077 | 2.3883 |

Paired deltas vs base:

| Arm | Delta MC2 | Delta truth log-odds | Delta true mass | Delta false pressure | False pressure up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lowLR` | -0.0021 | +0.8002 | -1.9314 | -2.7315 | 0.275 |
| `taskvec_a0p25` | +0.0355 | +0.5666 | +4.1623 | +3.5958 | 0.825 |

Read:

- `lowLR` is effectively flat on MC2 in this slice, but its mechanism is safer:
  it suppresses false-answer pressure more than true-answer mass.
- `taskvec_a0p25` improves bounded MC2 and truth log-odds here, but it also
  increases false-answer pressure on most items. This matches the broader
  concern that task-vector mitigation can raise plausible-lure pressure even
  when aggregate accuracy improves.
- This is a useful gate, not a final TruthfulQA estimate. Next mitigation work
  should keep the false-pressure-up fraction below the gate threshold while
  preserving the task-vector's MC2 gain.

## 2026-07-09 Active CHTC Scale-Up 5513434

Run:

- CHTC cluster: `5513434.0`
- Remote run:
  `~/chtc-runs/coherence-retention-scaleup-20260709-033234`
- Local submission note:
  `results/sft_eval/wide_bench/failure_suite/chtc_5513434/SUBMISSION.md`
- Host/GPU at submission check: `gpu4006.chtc.wisc.edu`, 46GB advertised GPU
  memory.
- Early status: running, GPU probe passed, staged-input check passed, package
  setup started.

Gate:

- Arms: `base`, `taskvec_a0p25`, `lowLR`
- Tasks: `truthfulqa_mc2`, `wic`, `openbookqa`
- Limit: `200`
- Promotion read: do not treat MC2 alone as sufficient. Keep
  `frac_false_pressure_up <= 0.60` as the hard TruthfulQA mechanism gate while
  checking whether WiC/OpenBookQA retention is flat, hurt, or boosted.

## Suite Lanes

P0 slices:

- MMLU failure subjects: `mmlu_moral_scenarios`, `mmlu_formal_logic`,
  `mmlu_medical_genetics`, `mmlu_nutrition`,
  `mmlu_professional_psychology`, `mmlu_high_school_statistics`.
- Science ranking: `arc_easy`, `arc_challenge`, `openbookqa`.
- Sense boundary: `wic`.
- Truthfulness lure pressure: `truthfulqa_mc2` with log samples.

P1 guards:

- `mmlu_business_ethics` as a task-vector regression sentinel.
- `hellaswag`, `winogrande`, and `piqa` as retained commonsense guards.

## Required Metrics

Do not promote a mitigation based on accuracy alone. Each candidate should log
answer-choice scores for:

- Accuracy on the bounded slice.
- Correct-choice margin.
- Best-distractor margin.
- TruthfulQA false-lure pressure and `frac_false_pressure_up`.
- KL to base answer-choice logits.
- Semantic target retention from generation coherence and THINGS human R2.

## Promotion Rule

Run this suite before any full MMLU sweep. Promote a candidate only if it
improves the MMLU/science/WiC failures, keeps TruthfulQA false-lure pressure
controlled, preserves HellaSwag/WinoGrande/PIQA, and keeps the semantic target
above the thresholds in `suite_manifest.json`.
