# GPU / CHTC Live Queue Status

Last updated: 2026-07-08 20:59 CDT
Repo checkpoint when sampled: `805fc12` on `coherence-sft`

## Live CHTC Queue

Source: `chtc-ssh "condor_q -batch suresh27"` and detailed `condor_q -af` resource attributes.

- Total: 2 running, 0 idle, 0 held.
- Running GPU jobs:
  - `5513313.0`, batch `coherence_mmlu_full_retry_cache2_5513313`: MMLU retry shard `00`; requests 1 GPU, 8 CPUs, 40 GB RAM, and 80 GB disk. Still running.
  - `5513347.0`, batch `coherence_concept_steering_qualitative_5513347`: concept qualitative sweep; requests 1 GPU, 8 CPUs, 64 GB RAM, and 80 GB disk. It has moved from idle to running.
- No idle jobs in queue at this poll.
- No held jobs in queue at this poll.

## Utilization Notes

- No allocated CHTC GPU appears wasted from this queue view: both visible GPU allocations are attached to running jobs.
- Local direct GPU telemetry is unavailable from this sandbox because `nvidia-smi` cannot communicate with the driver here; this file therefore tracks CHTC queue utilization only.
- `5513350.0`, batch `coherence_huth_encoding_smoke_bundle_uts02_uts03_5513350`, was running at the previous poll and no longer appears in the queue. It requested 0 GPUs, so it did not affect GPU utilization.
- Next material update: revise this file when `5513313.0` or `5513347.0` leaves the queue, a job enters hold, or new idle GPU jobs accumulate.
