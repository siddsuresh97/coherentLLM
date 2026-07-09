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
