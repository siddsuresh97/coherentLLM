# CHTC MMLU Shard Execution Report

## 2026-07-09 Template Prep

- Created CHTC smoke/full MMLU shard templates under `chtc/mmlu_shards/`.
- Container: `docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`, with per-job overlay installation for `vllm==0.6.6.post1` and `lm-eval==0.4.12`.
- Staged model path expected by jobs: `/staging/s/suresh27/models/llama31-8b-instruct`.
- Staged adapter path expected by jobs: `/staging/s/suresh27/adapters/taskvec_a0p25`.
- Full run shards: 8 GPU jobs over 57 explicit MMLU subject tasks, excluding aggregate `mmlu_*` categories.
- Local merge validation: `src/sft/merge_mmlu_shards.py` reconstructed all 57 subjects from the existing base MMLU JSON and reproduced weighted MMLU `acc=0.6929227124666865` over `n=13508`.
- Submit dry-run validation on CHTC succeeded for one smoke job and eight full jobs.

## GPU Constraint Update

Main-lane probe cluster `5513077` exited `0` and confirmed the PyTorch container can install/import `vllm==0.6.6.post1` and `lm-eval==0.4.12`, but it landed on an 11GB GTX 1080 Ti. The shard submit files now require:

```condor
requirements = (TARGET.CUDAGlobalMemoryMb >= 40000)
rank = TARGET.CUDAGlobalMemoryMb
```

The runner also checks `nvidia-smi --query-gpu=name,memory.total` before invoking `lm_eval`; if the selected GPU reports less than `40000` MB, the job exits with status `67` and still returns a diagnostic tarball.

An unconstrained smoke job submitted before this constraint was canceled while idle:

- Cluster: `5513114`
- Status before removal: idle, no remote host, no GPU memory attribute assigned
- Action: `condor_rm 5513114`

Next allowed CHTC step is a constrained smoke submission from the updated `chtc/mmlu_shards/` files.

## Constrained Smoke 5513123

- Run directory: `coherence-mmlu-shards-20260709-002254`
- Cluster: `5513123`
- Scheduler requirement: `TARGET.CUDAGlobalMemoryMb >= 40000`
- Matched slot: `slot2_1@gpu2003.chtc.wisc.edu`
- Matched GPU: `NVIDIA A100-SXM4-80GB`
- Advertised GPU memory: `81154` MB
- Runtime `nvidia-smi` memory: `81920` MB
- Staged model size: `15G`
- Staged adapter size: `657M`
- Result: failed before `lm_eval` due dependency overlay resolving `huggingface-hub==1.22.0`, which the image's `transformers` rejects because it requires `huggingface-hub<1.0`.
- Fix: `run_mmlu_shard.sh` now installs a per-job overlay with `huggingface-hub>=0.24,<1.0` and `datasets>=2.16,<5` whenever dependencies are missing or the base image has an incompatible Hub package.

## Constrained Smoke 5513138

- Run directory: `coherence-mmlu-shards-20260709-002716`
- Cluster: `5513138`
- Scheduler requirement: `TARGET.CUDAGlobalMemoryMb >= 40000`
- Matched slot: `slot2_4@gpu4001.chtc.wisc.edu`
- Matched GPU: `NVIDIA L40`
- Advertised GPU memory: `45468` MB
- Result: imports succeeded (`torch=2.5.1+cu124`, `vllm=0.6.6.post1`, `lm_eval=0.4.12`) and `lm_eval` reached vLLM initialization, but vLLM failed on UUID-valued `CUDA_VISIBLE_DEVICES=GPU-c41e89c5-a828-686d-745f-cbc2b3c573d1`.
- Fix: `run_mmlu_shard.sh` now records the original `CUDA_VISIBLE_DEVICES` and normalizes UUID-style values to numeric `0` before importing or invoking vLLM. CHTC's container runtime exposes only the assigned GPU, so numeric `0` points to the allocated device.

## Constrained Smoke 5513172

- Run directory: `coherence-mmlu-shards-20260709-003116`
- Cluster: `5513172`
- Matched slot: `slot2_4@gpu4001.chtc.wisc.edu`
- Matched GPU: `NVIDIA L40`
- Advertised GPU memory: `45468` MB
- Action: canceled after remaining opaque for multiple polls with no user-log updates or runtime counters. This avoided continuing to burn an L40 while the runner provided no live step-level visibility.
- Fix: `run_mmlu_shard.sh` now writes `progress.log`, streams step markers to HTCondor stdout, records `pip_install_exit_status.txt`, and wraps pip and lm-eval in timeouts. Smoke jobs use `PIP_TIMEOUT_SECONDS=900` and `LMEVAL_TIMEOUT_SECONDS=3600`; full shards use `PIP_TIMEOUT_SECONDS=900` and `LMEVAL_TIMEOUT_SECONDS=14400`.

## Main-Lane Smoke 5513177

- Run directory: `coherence-mmlu-shards-20260708-1945`
- Cluster: `5513177`
- Matched slot: `slot2_5@gpu5001.chtc.wisc.edu`
- Matched GPU: `NVIDIA H200`
- Advertised GPU memory: `143158` MB
- Effective CUDA device handling: original UUID `GPU-2c782401-f520-1473-d039-74da02991585`, normalized to `CUDA_VISIBLE_DEVICES=0`
- Imports succeeded: `torch=2.5.1+cu124`, `vllm=0.6.6.post1`, `lm_eval=0.4.12`
- Model load succeeded: vLLM loaded all 4 safetensor shards from `/staging/s/suresh27/models/llama31-8b-instruct`.
- Result: failed during vLLM LoRA/Triton profile execution because the PyTorch runtime image lacked a C compiler: `RuntimeError: Failed to find C compiler. Please specify via CC environment variable.`
- Fix: submit files now use `docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` instead of the runtime image, and the runner records `compiler.txt` plus exits early with status `68` if neither `cc` nor `gcc` is present.

## Devel-Image Retry 5513195

- Run directory: `coherence-mmlu-shards-20260709-004133`
- Cluster: `5513195`
- Final status: completed successfully at 2026-07-09T01:07:14Z.
- Matched slot: `slot2_3@gpu4003.chtc.wisc.edu`
- Matched GPU: `NVIDIA H100 80GB HBM3`
- Advertised GPU memory: `81089` MB
- Runtime `nvidia-smi` memory: `81559` MB
- Runtime milestones: GPU probe passed, dependency overlay install completed with `rc=0`, imports succeeded, `lm_eval` started at 2026-07-09T00:48:15Z, and `lm_eval` exited with `rc=0` at 2026-07-09T01:07:12Z.
- Pulled artifact: `results/sft_eval/wide_bench/chtc_mmlu_shards/coherence-mmlu-shards-20260709-004133/mmlu_smoke_taskvec_a0p25_smoke_abstract_results.tgz`
- Extracted diagnostics: `results/sft_eval/wide_bench/chtc_mmlu_shards/coherence-mmlu-shards-20260709-004133/extracted/`
- Smoke coverage: `mmlu_abstract_algebra`, `found_n=20`, `missing_tasks=[]`.
- Smoke metric: `acc,none=0.3`, `acc_stderr,none=0.10513149660756933`.
- Duplicate cleanup: an earlier retry, `5513191`, was still idle after the main-lane smoke was active; it was removed with `condor_rm 5513191` to avoid duplicate GPU use.

## Full Shards 5513268

- Run directory: `coherence-mmlu-shards-20260709-004133`
- Cluster: `5513268`
- Jobs: 8 full MMLU shard procs, `0` through `7`.
- Status at 2026-07-09T01:12Z: all 8 procs idle, no hold reasons, no remote hosts assigned.
- Submit evidence: full-shard log stubs appeared in the run directory at 2026-07-08 20:10 local time after the successful smoke.
- Requirements: `TARGET.CUDAGlobalMemoryMb >= 40000`, `request_gpus=1`, `request_cpus=8`, `request_memory=40GB`, `request_disk=80GB`.
- `condor_q -better-analyze 5513268.0`: requirements are satisfiable; 37 slots match the full shard requirements, 1 slot is currently willing to run the job, and 51 more would match if drained.
- Next action: monitor cluster `5513268`; pull full tarballs when procs complete and merge with `src/sft/merge_mmlu_shards.py`.
