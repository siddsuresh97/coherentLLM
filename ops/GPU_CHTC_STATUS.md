# GPU / CHTC Live Queue Status

Last updated: 2026-07-08 21:11 CDT
Repo checkpoint when sampled: `43fa525` on `coherence-sft`

## Live CHTC Queue

Source: `chtc-master check`, `chtc-ssh "condor_q -batch suresh27"`, and detailed `condor_q -af` resource attributes.

- Total: 1 running, 1 idle, 0 held.
- Running GPU jobs:
  - `5513347.0`, batch `coherence_concept_steering_qualitative_5513347`: concept qualitative sweep; requests 1 GPU, 8 CPUs, 64 GB RAM, and 80 GB disk. Still running.
- Idle jobs:
  - `5513373.0`, batch `coherence_huth_encoding_smoke_bundle_uncapped_all_5513373`: Huth uncapped-all encoding; requests 0 GPUs, 4 CPUs, 16 GB RAM, and 20 GB disk. `better-analyze` reports 3 immediately willing matches and 143 slots that would match if drained.
- No held jobs in queue at this poll.

## Utilization Notes

- No allocated CHTC GPU appears wasted from this queue view: the only visible GPU allocation is attached to a running job.
- CHTC wrapper access is working: `chtc-master check` reports a live master. Use `chtc-ssh`; direct `ssh -F /dev/null` bypasses the wrapper config/control path and is not a valid access probe for this lane.
- Local direct GPU telemetry is unavailable from this sandbox because `nvidia-smi` cannot communicate with the driver here; this file therefore tracks CHTC queue utilization only.
- `5513350.0`, batch `coherence_huth_encoding_smoke_bundle_uts02_uts03_5513350`, was running at the previous poll and no longer appears in the queue. It requested 0 GPUs, so it did not affect GPU utilization.
- `5513313.0`, batch `coherence_mmlu_full_retry_cache2_5513313`, no longer appears in the queue.
- The idle Huth job is CPU-only and currently satisfiable, so it does not indicate wasted GPU allocation or a submit-file blocker.
- Next material update: revise this file when `5513347.0` leaves the queue, `5513373.0` starts or holds, a job enters hold, or new idle GPU jobs accumulate.
