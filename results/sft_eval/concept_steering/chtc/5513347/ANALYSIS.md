# CHTC 5513347 Qualitative Analysis

Remote run:
`~/chtc-runs/coherence-concept-qualitative-20260709-015214`

Local artifact root:
`results/sft_eval/concept_steering/chtc/5513347/`

## Status

- Condor job: normal termination with return value `1`.
- GPU host: `xhuanggpu4000.chtc.wisc.edu`, NVIDIA A40 (`46068 MB`).
- `pip_install_exit_status.txt`: `0`.
- `qual_exit_status.txt`: `1`.
- `exit_status.txt`: `1`.

The nonzero exit happened after all generations and judge CSVs were written.
Failure cause:
`src/sft/run_concept_steering_qualitative.py` tried to write `SUMMARY.md` from
compact score rows that did not include the raw `response` field. The script is
patched in the source tree, and the existing generations were rescored without
re-running the GPU job.

## Artifacts

- Original bundle:
  `concept_steering_qualitative_best_thresholds_results.tgz`
- Original generated rows:
  `extracted/qualitative/generations.jsonl` (`112` rows)
- Original score summary:
  `extracted/qualitative/judge_summary.csv`
- Rescored summary:
  `rescored_v2/SUMMARY.md`
- Rescored machine-readable summary:
  `rescored_v2/judge_summary.csv`

## Rescored Readout

| Setting | Coherence | Alignment | Retention |
| --- | ---: | ---: | ---: |
| `baseline` | 0.75 | 1.00 | 1.00 |
| `coherence_l12_a2` | 0.75 | 1.00 | 1.00 |
| `coherence_l12_a4` | 0.75 | 1.00 | 1.00 |
| `coherence_l16_a4` | 1.00 | 1.00 | 1.00 |
| `human_alignment_l24_a4` | 0.75 | 1.00 | 1.00 |
| `human_alignment_l16_a2` | 0.75 | 1.00 | 1.00 |
| `human_alignment_l12_a4` | 0.75 | 0.50 | 1.00 |
| `human_alignment_l12_a-4` | 1.00 | 0.25 | 1.00 |

## Interpretation

- No targeted retention collapse appears in generated text.
- `coherence_l16_a4` is the best current generation candidate.
- `coherence_l12_a4` has the strongest forced-choice margin but did not improve
  this qualitative check and remains retention-risky by margin.
- `human_alignment_l24_a4` and `human_alignment_l16_a2` are retention-safe here,
  but do not improve over the saturated alignment baseline.
- Layer-12 human-alignment steering is a failure mode for alignment behavior.
