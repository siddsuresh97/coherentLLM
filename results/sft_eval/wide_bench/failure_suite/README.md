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

## 2026-07-09 CHTC Scale-Up 5513434

Run:

- CHTC cluster: `5513434.0`
- Remote run:
  `~/chtc-runs/coherence-retention-scaleup-20260709-033234`
- Local artifacts:
  `results/sft_eval/wide_bench/failure_suite/chtc_5513434/SUBMISSION.md`
- Host/GPU: `gpu4006.chtc.wisc.edu`, NVIDIA L40S, 45GB Condor-advertised /
  46GB `nvidia-smi` memory
- Exit evidence: Condor `ExitCode=0`, `TimeExecute=848s`,
  `exit_status.txt == 0`, `gate_exit_status.txt == 0`
- GPU metrics: max sampled GPU utilization `100%`, max sampled memory
  `39183` MiB

Gate:

- Arms: `base`, `taskvec_a0p25`, `lowLR`
- Tasks: `truthfulqa_mc2`, `wic`, `openbookqa`
- Limit: `200`

TruthfulQA paired deltas vs base:

| Arm | Delta MC2 | Delta truth log-odds | Delta true mass | Delta false pressure | False pressure up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `lowLR` | +0.0229 | -0.0188 | -2.2445 | -2.2257 | 0.355 |
| `taskvec_a0p25` | +0.0038 | -0.5586 | +4.1479 | +4.7066 | 0.820 |

WiC/OpenBookQA bounded retention:

| Arm | WiC acc | WiC delta | OpenBookQA acc | OpenBookQA acc_norm | OpenBookQA acc_norm delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.660 | 0.000 | 0.405 | 0.510 | 0.000 |
| `lowLR` | 0.490 | -0.170 | 0.355 | 0.445 | -0.065 |
| `taskvec_a0p25` | 0.500 | -0.160 | 0.390 | 0.480 | -0.030 |

Read:

- `lowLR` is the safer TruthfulQA arm at this size: it improves MC2 more than
  `taskvec_a0p25` and keeps `frac_false_pressure_up` below the `0.60` gate.
- `taskvec_a0p25` still carries the plausible-false-lure pressure problem:
  false pressure rises more than true mass and truth log-odds drops hard.
- Both arms strongly hurt WiC, so lexical sense disambiguation remains a
  primary damaged skill.
- `taskvec_a0p25` partly mitigates OpenBookQA relative to `lowLR`, but neither
  arm preserves elementary-science option ranking versus base.

## 2026-07-09 Active CHTC A0.5 Gate 5513444

Run:

- CHTC cluster: `5513444.0`
- Remote run:
  `~/chtc-runs/coherence-retention-a0p5-20260709-041400`
- Local provenance:
  `results/sft_eval/wide_bench/failure_suite/chtc_5513444/SUBMISSION.md`
- Submit file:
  `chtc/retention_failure_suite/retention_failure_suite_a0p5_homeadapter.sub`
- Status: initially idle at submit check; running by 2026-07-08 23:19 CDT.

Gate:

- Arms: `base`, `taskvec_a0p5`
- Tasks: `truthfulqa_mc2`, `wic`, `openbookqa`
- Limit: `200`
- Adapter transfer: dereferenced `taskvec_a0p5_adapter.tgz` copied to CHTC
  `/home` and unpacked inside job scratch; no new `/staging` adapter directory
  is required.

Purpose:

- Test whether the higher-alpha semantic-hub signal survives the same
  TruthfulQA false-pressure, WiC, and OpenBookQA gate that `taskvec_a0p25`
  just failed/partly mitigated.
- Avoid a duplicate MMLU run and avoid worsening the current CHTC staging
  file-count quota.

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
