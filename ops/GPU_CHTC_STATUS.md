# GPU / CHTC Operational Status

Last updated: 2026-07-08 20:57 CDT
Repo checkpoint when sampled: `712464e` on `coherence-sft`

## Current Allocation

- Local GPU telemetry: unavailable from the current sandbox. `nvidia-smi` cannot communicate with the NVIDIA driver here, and local `condor_q` is not connected to the CHTC scheduler. Use CHTC-side logs plus known result directories as the reliable operational view from this session.
- CHTC queue at 2026-07-08 20:57 CDT: 1 job running, 2 idle, 0 held.
- Running GPU jobs:
  - `5513313.0`, batch `coherence_mmlu_full_retry_cache2_5513313`, MMLU shard `00`, 1 GPU on `mkhodakgpu4000.chtc.wisc.edu`, NVIDIA L40S, 45 GB class. Passed GPU probe and pip/import setup; in `lm_eval_start` since 2026-07-09 01:37:58 UTC.
- Idle jobs:
  - `5513347.0`, batch `coherence_concept_steering_qualitative_5513347`, requests 1 GPU, 8 CPUs, 64 GB RAM, staging, and a 40 GB class GPU. `better-analyze` reports 4 willing matches and 44 would match if drained, so this is a scarce-resource wait rather than a malformed submit.
  - `5513350.0`, batch `coherence_huth_encoding_smoke_bundle_uts02_uts03_5513350`, requests 0 GPUs, 4 CPUs, 24 GB RAM, staging, and docker-volume staging. `better-analyze` reports 4 willing matches and 142 would match if drained. It should not consume GPU capacity when it starts.
- Recently completed non-GPU job:
  - `5513337.0`, batch `coherence_huth_encoding_smoke_bundle_5513337`, completed normally with wrapper and inner `encoding_exit_status.txt == 0`. It used 0 GPUs and transferred `huth_encoding_smoke_bundle_results.tgz`.
- Recently completed GPU job:
  - `5513309.1`, batch `coherence_mmlu_full_retry_cache_5513309`, MMLU shard `06`, completed and produced `mmlu_full_cache_taskvec_a0p25_shard_06_results.tgz`.

## Artifact Readiness

- Huth extraction retry `5513306`: local artifact is present under `results/sft_huth_lebel/chtc_huth_extract_smoke_retry_5513306/`; wrapper and extract status are both `0`.
- Huth encoding bundle `5513337`: local artifact is present under `results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_5513337/`; wrapper and encoding status are both `0`. The tarball contains `summary.csv`, `alpha_cv.csv`, `run_metadata.json`, and status files. Sample summary values are near zero mean Pearson `r` on the 2k-voxel smoke cap, so the smoke validates plumbing more than scientific effect size.
- MMLU CHTC run `coherence-mmlu-shards-20260709-004133`: several result tarballs are already present remotely. Known completed recovery artifacts include shards `02`, `05`, `06`, and `07`; shard `00` is still running. Wait for shard `00` before final merge to avoid another partial aggregate.
- Local MMLU fallback: both `results/sft_eval/wide_bench/runs/taskvec_a0p25_mmlu_5shot_shard00of02/` and `results/sft_eval/wide_bench/runs/taskvec_a0p25_mmlu_5shot_shard01of02/` now contain completed `results_*.json` files. Treat merge/report ownership as the MMLU lane's responsibility unless main explicitly takes it over.

## Bottlenecks And Waste Check

- No allocated CHTC GPU is idle right now: the only allocated GPU is inside MMLU `lm_eval`, not waiting on setup.
- The MMLU bottleneck remains cache/staging robustness and slow shard completion, not job availability. The cache-quota retries using per-job scratch caches are the correct mitigation; avoid submitting duplicate MMLU jobs while shard `00` is still running.
- Huth encoding was CPU-only and fast. It does not justify occupying GPUs; the GPU-heavy part is feature extraction, which now has a successful bundled-output path. The new UTS02/UTS03 encoding job also requests no GPU.
- The concept qualitative job may wait because it requests both 64 GB RAM and a 40 GB staging GPU. If queue delay becomes the bottleneck, the first mitigation to consider is whether qualitative generation can run with a 40 GB memory request instead of 64 GB, after local/short CHTC smoke validation.
- Transfer strategy: for small code/manifests, `chtc-push` is fine. For large reusable assets such as model/adapters/datasets, prefer staging or job-side direct access when available. For job outputs, bundled tarball return is better than writing new feature trees under `/staging/s/suresh27`, because the earlier Huth staged-feature attempt hit `Disk quota exceeded` despite apparent quota headroom.

## Next Operational Checks

1. Poll `condor_q -batch suresh27` until MMLU `5513313.0` leaves the queue.
2. Pull or have the MMLU lane pull the final MMLU tarball `mmlu_full_cache2_taskvec_a0p25_shard_00_results.tgz`; shard `06` is already ready remotely.
3. Have the Huth lane update the Huth report with the already-pulled encoding smoke summary.
4. Do not submit additional CHTC jobs from this monitor lane without coordinating with the main experiment owner or the relevant lane agent.
