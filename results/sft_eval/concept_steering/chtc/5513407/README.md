# CHTC 5513407 Expanded Concept Steering Qualitative Result

Status: completed cleanly on 2026-07-08.

## Provenance

- CHTC cluster: `5513407.0`
- Remote run: `~/chtc-runs/coherence-concept-expanded-20260709-023711`
- Host: `slot2_3@gpulab2004.chtc.wisc.edu`
- GPU: NVIDIA A100-SXM4-40GB
- Exit status: `exit_status.txt == 0`, `qual_exit_status.txt == 0`
- Condor runtime: `TimeExecute=962s`, `TimeSlotBusy=1015s`
- Peak sampled GPU memory: `15883` MiB
- Mean sampled GPU utilization: `15.09%`; final generation window was around
  `77-83%` GPU utilization.

## Artifacts

- Result bundle:
  `concept_steering_qualitative_expanded_alignment_retention_results.tgz`
- Extracted result:
  `extracted/qualitative/SUMMARY.md`
- Summary CSV:
  `extracted/qualitative/judge_summary.csv`
- Retention gates:
  `extracted/qualitative/gate_summary.csv`
- Full generations:
  `extracted/qualitative/generations.jsonl`
- GPU metrics:
  `extracted/gpu_metrics.csv`
- Condor logs:
  `logs/`

## Result

The expanded suite used `25` coherence prompts, `25` harder human-alignment
prompts, and `25` retention prompts across five settings.

| Setting | Coherence pass | Delta | Alignment pass | Delta | Retention pass | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `baseline` | 0.76 | 0.00 | 0.88 | 0.00 | 0.96 | pass |
| `coherence_l12_a4` | 0.88 | +0.12 | 0.64 | -0.24 | 0.96 | pass |
| `coherence_l16_a4` | 0.80 | +0.04 | 0.80 | -0.08 | 0.96 | pass |
| `human_alignment_l16_a2` | 0.84 | +0.08 | 0.76 | -0.12 | 0.96 | pass |
| `human_alignment_l24_a4` | 0.80 | +0.04 | 0.80 | -0.08 | 0.96 | pass |

## Read

- All settings passed the retention gate: pass rate `0.96`, no drop from
  baseline, with required minimum `0.85` and max allowed drop `0.10`.
- `coherence_l12_a4` is the strongest coherence intervention in this suite, but
  it has a large alignment cost (`-0.24`), so it should remain diagnostic only.
- `coherence_l16_a4` is retention-safe but the expanded suite weakens the
  earlier stronger qualitative read: coherence gain is only `+0.04`, and
  alignment drops `-0.08`.
- The human-alignment vectors do not produce an alignment gain on the harder
  alignment prompts. Both tested settings reduce alignment pass rate relative to
  baseline.
- The immediate next gate should be better-targeted steering/vector extraction
  or prompt-family stratification, not broad deployment of these steering
  vectors.
