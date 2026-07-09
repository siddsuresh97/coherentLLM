# Retention Failure-Suite Gate Run

This run is a bounded GPU-backed gate from the committed failure-suite
manifest. It is not a replacement for the full benchmark sweep.

## Plan

- Arms: `base, taskvec_a0p5`
- Tasks: `truthfulqa_mc2, wic, openbookqa`
- Limit override: `200`
- Adapter root: `/var/lib/condor/execute/slot2/dir_2809285/scratch/adapters`
- Command count: `6`

## TruthfulQA Paired Delta vs Base

| Arm | n | Delta MC2 | Delta truth log-odds | Delta true mass | Delta false pressure | False pressure up |
|---|---:|---:|---:|---:|---:|---:|
| taskvec_a0p5 | 200 | -0.016896982101882282 | -0.9308304878771394 | -3.8411239319549226 | -2.9102934440777832 | 0.305 |

Gate note: the current threshold is `frac_false_pressure_up <= 0.60`
with non-positive false-pressure delta preferred unless truth
log-odds improves.

## Artifacts

- `command_manifest.csv`: exact lm-eval commands.
- `truthfulqa_logsamples/`: raw lm-eval results and samples when TruthfulQA is run.
- `truthfulqa_analysis/`: false-lure pressure diagnostics when analysis is enabled.
