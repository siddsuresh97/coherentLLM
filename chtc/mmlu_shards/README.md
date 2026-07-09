# CHTC GPU MMLU Shards

This directory prepares CHTC GPU execution for taskvec `a0p25` MMLU without waiting on the local two-GPU run.

The jobs expect these staged paths on CHTC:

- Model: `/staging/s/suresh27/models/llama31-8b-instruct`
- Adapter: `/staging/s/suresh27/adapters/taskvec_a0p25`

Both smoke and full submit files require `TARGET.HasCHTCStaging == true` and `TARGET.CUDAGlobalMemoryMb >= 40000`. The staging requirement is necessary because the model and adapter are read directly from `/staging/s/suresh27`; without it, backfill hosts can run the wrapper but fail before model load because `/staging` is absent. The GPU memory requirement targets useful vLLM GPUs such as A100-40GB, L40S, A100-80GB, and H200 while excluding 11GB-class cards. The runner also checks `nvidia-smi` memory before loading vLLM and exits without running eval if the GPU is below `MIN_CUDA_GLOBAL_MEMORY_MB`.

The full run uses the 57 MMLU subject tasks only. Aggregate tasks such as `mmlu`, `mmlu_stem`, `mmlu_other`, `mmlu_social_sciences`, and `mmlu_humanities` are not in the shard manifests.

## Files

- `mmlu_subjects.csv`: the explicit 57 subject tasks and sample counts from the base local MMLU run.
- `mmlu_smoke_manifest.tsv`: one short smoke shard, `mmlu_abstract_algebra` with `--limit 20`.
- `mmlu_full_manifest.tsv`: eight balanced full shards, roughly 1.65k-1.72k questions each. The task list uses `+` as a queue-safe delimiter; the runner converts it back to comma-separated `lm_eval` tasks.
- `mmlu_smoke.sub`: one-GPU smoke submit file.
- `mmlu_full.sub`: eight-GPU full submit file.
- `mmlu_full_retry_missing_staging_manifest.tsv`: retry manifest for shards that failed in cluster `5513268` after landing on hosts without `/staging`.
- `mmlu_full_retry_missing_staging.sub`: retry submit file using the staging requirement for that manifest.
- `run_mmlu_shard.sh`: container runner. It checks staged paths, records GPU/env diagnostics, installs missing `lm-eval`/`vllm` packages if needed, runs `lm_eval`, and always returns a tarball.

The submit files use `docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, matching the successful CHTC GPU probe path while also providing a C compiler for Triton/vLLM runtime kernels. The runner installs a per-job Python overlay when needed. The overlay pins `vllm==0.6.6.post1`, `lm-eval==0.4.12`, `huggingface-hub>=0.24,<1.0`, and `datasets>=2.16,<5` so dependency resolution stays compatible with `transformers`.

CHTC may expose the assigned GPU through a UUID-valued `CUDA_VISIBLE_DEVICES` such as `GPU-...`. The runner records that original value and then normalizes the effective value to `0`, because this container exposes only the assigned GPU and vLLM `0.6.6` expects numeric CUDA device IDs.

The runner writes `progress.log` and streams step markers to HTCondor stdout. Smoke jobs use a 15-minute pip timeout and a 1-hour lm-eval timeout; full shards use the same pip timeout and a 4-hour lm-eval timeout.

## Submit

From the local repo, stage this directory to a fresh CHTC run directory:

```bash
RUN_ID=coherence-mmlu-shards-YYYYMMDD-HHMM
chtc-ssh "mkdir -p ~/chtc-runs/${RUN_ID}/logs"
chtc-push chtc/mmlu_shards/ "chtc-runs/${RUN_ID}/"
```

Run the smoke job first:

```bash
chtc-ssh "cd ~/chtc-runs/${RUN_ID} && condor_submit mmlu_smoke.sub"
```

If the smoke tarball contains a successful `lm_eval_exit_status.txt`, submit the full eight-shard run:

```bash
chtc-ssh "cd ~/chtc-runs/${RUN_ID} && condor_submit mmlu_full.sub"
```

For the 2026-07-09 recovery run only, retry the shards that failed on non-staging hosts:

```bash
chtc-push chtc/mmlu_shards/mmlu_full_retry_missing_staging_manifest.tsv "chtc-runs/${RUN_ID}/"
chtc-push chtc/mmlu_shards/mmlu_full_retry_missing_staging.sub "chtc-runs/${RUN_ID}/"
chtc-ssh "cd ~/chtc-runs/${RUN_ID} && condor_submit mmlu_full_retry_missing_staging.sub"
```

Useful status commands:

```bash
chtc-ssh "condor_q -batch suresh27"
chtc-ssh "condor_q <cluster_id> -af:jh ClusterId ProcId JobStatus HoldReason LastRemoteHost RemoteWallClockTime DiskUsage_RAW"
chtc-ssh "condor_tail <cluster_id>.<proc_id>"
```

## Pull

Pull the completed run directory back under the owned local results scope:

```bash
mkdir -p results/sft_eval/wide_bench/chtc_mmlu_shards/${RUN_ID}
chtc-pull "chtc-runs/${RUN_ID}/" "results/sft_eval/wide_bench/chtc_mmlu_shards/${RUN_ID}/"
```

Expected returned tarballs:

- Smoke: `mmlu_smoke_taskvec_a0p25_smoke_abstract_results.tgz`
- Full: `mmlu_full_taskvec_a0p25_shard_00_results.tgz` through `mmlu_full_taskvec_a0p25_shard_07_results.tgz`

## Merge

Merge pulled tarballs or extracted shard directories:

```bash
python src/sft/merge_mmlu_shards.py \
  --inputs results/sft_eval/wide_bench/chtc_mmlu_shards/${RUN_ID} \
  --state-name taskvec_a0p25 \
  --base-csv results/sft_eval/wide_bench/raw_task_summary.csv \
  --out-dir results/sft_eval/wide_bench/chtc_mmlu_shards/${RUN_ID}/merged \
  --require-complete
```

The merge writes:

- `results_merged.json`: merged lm-eval-style `results` JSON.
- `raw_task_summary.csv`: per-subject values plus weighted `mmlu` aggregate.
- `summary.csv`: deltas versus the local base CSV where available.
- `REPORT.md`: human-readable shard coverage and aggregate summary.

## Notes

The submit files use `docker_override_entrypoint = true` because entrypoint-based images can otherwise treat the job script as an argument to their default command. The smoke job is the guardrail for confirming that CHTC's container runtime honors this before launching eight full shards.
