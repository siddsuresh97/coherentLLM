# CHTC Retention A0.5 Gate Submission

Status: running on CHTC as of 2026-07-08 23:19 CDT.

## Run

- Cluster: `5513444.0`
- Submitted: 2026-07-08 23:15 CDT
- Remote run directory:
  `~/chtc-runs/coherence-retention-a0p5-20260709-041400`
- Submit file:
  `chtc/retention_failure_suite/retention_failure_suite_a0p5_homeadapter.sub`
- Wrapper:
  `chtc/retention_failure_suite/run_retention_failure_suite.sh`
- Gate runner:
  `src/sft/run_retention_failure_suite_gate.py`

## Gate

- Arms: `base`, `taskvec_a0p5`
- Tasks: `truthfulqa_mc2`, `wic`, `openbookqa`
- Limit: `200`
- Model path: `/staging/s/suresh27/models/llama31-8b-instruct`
- GPU floor: `TARGET.CUDAGlobalMemoryMb >= 40000`
- Staging requirement: `TARGET.HasCHTCStaging == true`

## Adapter Transfer

Current CHTC staging quota is over file count (`1120/1000` files), so this job
does not create `/staging/s/suresh27/adapters/taskvec_a0p5`.

Instead, the local symlinked adapter directory was bundled with symlinks
dereferenced:

```bash
tar -czhf /tmp/taskvec_a0p5_adapter.tgz -C out/adapters_taskvec_scaled/a0p5 .
```

The bundle was pushed to CHTC `/home`:

```bash
chtc-push /tmp/taskvec_a0p5_adapter.tgz 'chtc-runs/coherence-retention-a0p5-20260709-041400/taskvec_a0p5_adapter.tgz'
```

Remote verification before submit showed `taskvec_a0p5_adapter.tgz` at `596M`.

## Local Validation

```bash
python -m py_compile src/sft/run_retention_failure_suite_gate.py src/sft/analyze_truthfulqa_logsamples.py
bash -n chtc/retention_failure_suite/run_retention_failure_suite.sh
python src/sft/run_retention_failure_suite_gate.py --dry-run --manifest results/sft_eval/wide_bench/failure_suite/suite_manifest.json --out-dir /tmp/retention_failure_suite_a0p5_dryrun --arms base taskvec_a0p5 --tasks truthfulqa_mc2 wic openbookqa --limit 200 --model-path /staging/s/suresh27/models/llama31-8b-instruct --adapter-root adapters
git diff --check
```

## Initial Queue State

Immediately after submit:

```text
5513444.0 suresh27 7/8 23:15 0+00:00:00 I run_retention_failure_suite.sh a0p5_truth_wic_obqa_200 base+taskvec_a0p5 truthfulqa_mc2+wic+openbookqa 200 ...
```

Overall queue:

```text
1 jobs; 0 completed, 0 removed, 1 idle, 0 running, 0 held, 0 suspended
```

Follow-up queue check at 2026-07-08 23:19 CDT:

```text
1 jobs; 0 completed, 0 removed, 0 idle, 1 running, 0 held, 0 suspended
```
