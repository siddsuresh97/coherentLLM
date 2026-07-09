# Retention Failure-Suite Scale-Up Result

Completed: 2026-07-08 22:48 CDT

## Run

- CHTC cluster: `5513434.0`
- Remote directory:
  `~/chtc-runs/coherence-retention-scaleup-20260709-033234`
- Host/GPU: `slot2_2@gpu4006.chtc.wisc.edu`, NVIDIA L40S
- Exit evidence: Condor return value `0`, `exit_status.txt == 0`,
  `gate_exit_status.txt == 0`
- Runtime evidence: `TimeExecute=848s`, `TimeSlotBusy=849s`
- GPU evidence: max sampled utilization `100%`, max sampled memory `39183` MiB

## TruthfulQA MC2

| Arm | MC2 | Delta MC2 | Truth log-odds | Delta truth log-odds | False pressure up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.5230 | 0.0000 | 0.6227 | 0.0000 |  |
| `lowLR` | 0.5459 | +0.0229 | 0.6039 | -0.0188 | 0.355 |
| `taskvec_a0p25` | 0.5268 | +0.0038 | 0.0641 | -0.5586 | 0.820 |

## WiC / OpenBookQA

| Arm | WiC acc | WiC delta | OpenBookQA acc | OpenBookQA acc_norm | OpenBookQA acc_norm delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.660 | 0.000 | 0.405 | 0.510 | 0.000 |
| `lowLR` | 0.490 | -0.170 | 0.355 | 0.445 | -0.065 |
| `taskvec_a0p25` | 0.500 | -0.160 | 0.390 | 0.480 | -0.030 |

## Read

`lowLR` is the safer TruthfulQA arm on this bounded gate: it improves MC2 more
than `taskvec_a0p25` and passes the `frac_false_pressure_up <= 0.60` mechanism
gate. `taskvec_a0p25` still fails the false-lure gate because false-answer
pressure rises more than true-answer mass.

Both aligned arms damage WiC, so lexical sense disambiguation remains a primary
retention failure. `taskvec_a0p25` partly mitigates OpenBookQA relative to
`lowLR`, but neither arm preserves elementary-science option ranking against
base.

## Key Files

- `extracted/gate/RUN_SUMMARY.md`
- `extracted/gate/truthfulqa_analysis/summary.csv`
- `extracted/gate/truthfulqa_analysis/paired_delta_summary.csv`
- `extracted/gate/lm_eval/*_wic_limit200/**/results_*.json`
- `extracted/gate/lm_eval/*_openbookqa_limit200/**/results_*.json`
- `extracted/gpu_metrics.csv`
- `logs/retention_failure_suite_5513434_0.log`
