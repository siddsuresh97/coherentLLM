# Retention Failure-Suite CHTC Gate

This bundle runs the first GPU-backed smoke from the committed retention
failure-suite manifest. It is intentionally narrow: `truthfulqa_mc2`, limit 40,
for `base`, `taskvec_a0p25`, and the staged `lowLR` adapter.

The smoke tests whether the drop mechanism is visible in fresh CHTC logs:
paired TruthfulQA MC2, truth-logodds, true-answer mass, false-answer pressure,
and `frac_false_pressure_up` versus base.

## Local Checks

```bash
python -m py_compile src/sft/run_retention_failure_suite_gate.py src/sft/analyze_truthfulqa_logsamples.py
bash -n chtc/retention_failure_suite/run_retention_failure_suite.sh
python src/sft/run_retention_failure_suite_gate.py --dry-run --manifest results/sft_eval/wide_bench/failure_suite/suite_manifest.json --out-dir /tmp/retention_failure_suite_dryrun --arms base taskvec_a0p25 lowLR --tasks truthfulqa_mc2 --limit 16 --model-path /staging/s/suresh27/models/llama31-8b-instruct
```

## Submit

Copy the bundle files to a CHTC run directory with:

```bash
chtc-push chtc/retention_failure_suite/run_retention_failure_suite.sh "chtc-runs/<run-id>/"
chtc-push chtc/retention_failure_suite/retention_failure_suite_smoke.sub "chtc-runs/<run-id>/"
chtc-push src/sft/run_retention_failure_suite_gate.py "chtc-runs/<run-id>/"
chtc-push src/sft/analyze_truthfulqa_logsamples.py "chtc-runs/<run-id>/"
chtc-push results/sft_eval/wide_bench/failure_suite/suite_manifest.json "chtc-runs/<run-id>/"
chtc-ssh "cd ~/chtc-runs/<run-id> && mkdir -p logs && condor_submit retention_failure_suite_smoke.sub"
```

Required staged inputs:

- `/staging/s/suresh27/models/llama31-8b-instruct`
- `/staging/s/suresh27/adapters/taskvec_a0p25`
- `/staging/s/suresh27/adapters/lowLR`

`/staging/s/suresh27/adapters/lowrank` was not present during the initial
smoke setup, so the first CHTC gate uses `lowLR` as the staged comparison arm.
