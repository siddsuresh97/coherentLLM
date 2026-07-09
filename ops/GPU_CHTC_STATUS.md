# GPU / CHTC Live Queue Status

Last updated: 2026-07-08 21:45 CDT
Repo checkpoint when sampled: `c3c9624` on `coherence-sft`

## Live CHTC Queue

Source: `chtc-master check`, `chtc-ssh "condor_q -batch suresh27"`,
detailed `condor_q -af` resource attributes, `condor_status -total`, and the
small Condor log/stdout/stderr files for active job `5513407.0`.

- Total: 1 running, 0 idle, 0 held.
- Running GPU jobs:
  - `5513407.0`, batch `coherence_concept_steering_qualitative_5513407`:
    expanded concept qualitative suite owned by Boole's lane. It requests
    1 GPU, 8 CPUs, 64 GB RAM, and 80 GB disk.
- Idle jobs: none.
- No held jobs in queue at this poll.

## Utilization Notes

- CHTC wrapper access is working: `chtc-master check` reports a live master and
  the Access Point is `ap2002.chtc.wisc.edu`. Use `chtc-ssh`; direct
  `ssh -F /dev/null` bypasses the wrapper config/control path.
- Local direct GPU telemetry is unavailable from this sandbox because
  `nvidia-smi` cannot communicate with the driver here. This file therefore
  tracks CHTC queue/utilization evidence only.
- Active job `5513407.0` matched and started quickly on
  `slot2_3@gpulab2004.chtc.wisc.edu`, using an `NVIDIA A100-SXM4-40GB`
  (`Capability=8.0`, `GlobalMemoryMb=40442`). Its stdout shows
  `gpu_probe_ok`, `pip_install_exit rc=0`, `import_probe_ok`, and the planned
  75 cases x 5 settings = 375 generations. Stderr only shows a dtype
  deprecation warning and checkpoint-load progress at this poll.
- Exact available CHTC capacity remains high: 40 unclaimed X86_64 slots satisfy
  `State=="Unclaimed"`, `Gpus>=1`, `Cpus>=8`, `Memory>=65536`,
  `CUDAGlobalMemoryMb>=40000`, and `HasChtcStaging==true`. The same count
  satisfies the recent MMLU shape with 40 GB RAM.
- `condor_q -better-analyze 5513407.0` reports the job is running, with 7
  willing slots for its exact constraints and 42 more that would match if
  drained.

## Submission Decision

No additional job was submitted by this monitor poll.

Rationale:

- Do not duplicate Boole's concept write set while `5513407.0` is already
  running the expanded concept suite.
- The `taskvec_a0p25` MMLU CHTC merge is complete, so rerunning the same shards
  would spend GPUs without adding signal.
- The checked-in Huth uncapped encoding submit is CPU-only and already
  completed. The next useful Huth/Fedorenko or semantic-hub GPU job needs the
  Faraday/design lane to publish a locally smoke-tested bundle first.

Next material update: revise this file when `5513407.0` leaves the queue, a job
enters hold, a new debugged non-concept GPU lane appears, or Faraday publishes a
ready Huth/Fedorenko or semantic-hub CHTC submit bundle.
