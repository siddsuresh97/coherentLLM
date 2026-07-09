# CHTC GPU MMLU Shards

This directory prepares CHTC GPU execution for taskvec `a0p25` MMLU without waiting on the local two-GPU run.

The jobs expect these staged paths on CHTC:

- Model: `/staging/s/suresh27/models/llama31-8b-instruct`
- Adapter: `/staging/s/suresh27/adapters/taskvec_a0p25`

Both smoke and full submit files require `TARGET.CUDAGlobalMemoryMb >= 40000`, which targets useful vLLM GPUs such as A100-40GB, L40S, A100-80GB, and H200 while excluding 11GB-class cards. The runner also checks `nvidia-smi` memory before loading vLLM and exits without running eval if the GPU is below `MIN_CUDA_GLOBAL_MEMORY_MB`.

The full run uses the 57 MMLU subject tasks only. Aggregate tasks such as `mmlu`, `mmlu_stem`, `mmlu_other`, `mmlu_social_sciences`, and `mmlu_humanities` are not in the shard manifests.

## Files

- `mmlu_subjects.csv`: the explicit 57 subject tasks and sample counts from the base local MMLU run.
- `mmlu_smoke_manifest.tsv`: one short smoke shard, `mmlu_abstract_algebra` with `--limit 20`.
- `mmlu_full_manifest.tsv`: eight balanced full shards, roughly 1.65k-1.72k questions each. The task list uses `+` as a queue-safe delimiter; the runner converts it back to comma-separated `lm_eval` tasks.
- `mmlu_smoke.sub`: one-GPU smoke submit file.
- `mmlu_full.sub`: eight-GPU full submit file.
- `run_mmlu_shard.sh`: container runner. It checks staged paths, records GPU/env diagnostics, installs missing `lm-eval`/`vllm` packages if needed, runs `lm_eval`, and always returns a tarball.

The runner installs a per-job Python overlay when needed. The overlay pins `huggingface-hub>=0.24,<1.0` and `datasets>=2.16,<5` because `vllm/vllm-openai:v0.6.6.post1` can otherwise resolve a newer Hub package that is rejected by the image's `transformers`.

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
