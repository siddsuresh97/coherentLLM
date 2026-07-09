# Retention Failure-Suite Scale-Up Submission

Submitted: 2026-07-08 22:34 CDT

Status: completed successfully. See `RESULT.md` for metrics and runtime
evidence.

## Run

- CHTC cluster: `5513434.0`
- Remote directory:
  `~/chtc-runs/coherence-retention-scaleup-20260709-033234`
- Submit file:
  `chtc/retention_failure_suite/retention_failure_suite_scaleup.sub`
- Runner:
  `chtc/retention_failure_suite/run_retention_failure_suite.sh`

## Gate

- Arms: `base`, `taskvec_a0p25`, `lowLR`
- Tasks: `truthfulqa_mc2`, `wic`, `openbookqa`
- Limit: `200`
- Model path: `/staging/s/suresh27/models/llama31-8b-instruct`
- Adapter paths:
  `/staging/s/suresh27/adapters/taskvec_a0p25` and
  `/staging/s/suresh27/adapters/lowLR`
- GPU requirement:
  `TARGET.HasCHTCStaging == true && TARGET.CUDAGlobalMemoryMb >= 40000`

## Validation

Local checks before submission:

```bash
bash -n chtc/retention_failure_suite/run_retention_failure_suite.sh
git diff --check
python src/sft/run_retention_failure_suite_gate.py --dry-run --manifest results/sft_eval/wide_bench/failure_suite/suite_manifest.json --out-dir /tmp/retention_failure_suite_scaleup_dryrun --arms base taskvec_a0p25 lowLR --tasks truthfulqa_mc2 wic openbookqa --limit 200 --model-path /staging/s/suresh27/models/llama31-8b-instruct
```

CHTC matching check after submission:

- `condor_q -batch suresh27`: `1` running, `0` idle, `0` held.
- `condor_q 5513434 -better-analyze`: job is running; 36 slots match the full
  requirement expression, 3 were immediately willing, and the successful match
  occurred at 2026-07-08 22:34:16 CDT.
- Assigned host at first check:
  `slot2_2@gpu4006.chtc.wisc.edu`.
- Early stdout:
  `gpu_probe_ok memory_mb=46068`, `staged_input_check`, and
  `pip_install_start`.

## Intended Read

This job scales the first TruthfulQA mechanism result from limit 40 to limit
200 and adds WiC/OpenBookQA retention slices. The primary question is whether
`taskvec_a0p25` still improves aggregate TruthfulQA while raising
false-answer pressure, and whether the same arm is preserving, hurting, or
boosting word-sense and elementary-science skills.
