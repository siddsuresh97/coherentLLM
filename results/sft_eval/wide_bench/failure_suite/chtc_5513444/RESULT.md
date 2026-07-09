# CHTC Retention A0.5 Gate Result

## Run

- CHTC cluster: `5513444.0`
- Remote run:
  `~/chtc-runs/coherence-retention-a0p5-20260709-041400`
- Local artifacts:
  `results/sft_eval/wide_bench/failure_suite/chtc_5513444/`
- Host/GPU: `gpu2010.chtc.wisc.edu`, NVIDIA A100-SXM4-80GB
- Exit evidence: Condor return `0`, `TimeExecute=1216s`,
  `exit_status.txt == 0`, `gate_exit_status.txt == 0`
- GPU metrics: 240 samples, max utilization `100%`, max memory `64923` MiB

## Gate

- Arms: `base`, `taskvec_a0p5`
- Tasks: `truthfulqa_mc2`, `wic`, `openbookqa`
- Limit: `200`
- Adapter transfer: dereferenced `taskvec_a0p5_adapter.tgz` copied to CHTC
  `/home` and unpacked in job scratch.

## TruthfulQA Limit-200

| Arm | MC2 | Delta MC2 | Truth log-odds | Delta truth log-odds | False pressure up |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.5237 | 0.0000 | 0.6636 | 0.0000 |  |
| `taskvec_a0p5` | 0.5068 | -0.0169 | -0.2673 | -0.9308 | 0.305 |

Paired mechanism deltas vs base:

| Arm | Delta true mass | Delta false pressure | Delta best margin | Accuracy down fraction |
| --- | ---: | ---: | ---: | ---: |
| `taskvec_a0p5` | -3.8411 | -2.9103 | -0.9385 | 0.565 |

## WiC / OpenBookQA Limit-200

| Arm | WiC acc | WiC delta | OpenBookQA acc | OpenBookQA acc_norm | OpenBookQA acc_norm delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| `base` | 0.650 | 0.000 | 0.405 | 0.510 | 0.000 |
| `taskvec_a0p5` | 0.490 | -0.160 | 0.255 | 0.425 | -0.085 |

## Interpretation

- `taskvec_a0p5` passes the false-pressure-up threshold (`0.305 <= 0.60`), but
  this is not a useful mitigation: it lowers both true and false answer mass,
  drops truth log-odds by `-0.9308`, and loses MC2 by `-0.0169`.
- WiC remains badly hurt (`-0.160`), so the lexical-sense-disambiguation damage
  is not solved by increasing alpha to `0.5`.
- OpenBookQA acc_norm drops by `-0.085`, worse than the prior `taskvec_a0p25`
  gate (`-0.030`) and worse than `lowLR` (`-0.065`) in `5513434`.
- Conclusion: do not promote `taskvec_a0p5` or spend full MMLU on it. Treat it
  as evidence that stronger semantic-hub-like retrieval can suppress false
  lure pressure only by degrading useful answer ranking.
