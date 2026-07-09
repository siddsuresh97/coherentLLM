# Concept Steering Smoke Status

Last updated: 2026-07-09.

## Current State

Prepared, locally dry-run validated, and passed on CHTC as cluster `5513276`.
The bounded layer/alpha sweep also passed as cluster `5513297`.

Owned files added in this handoff:

- `chtc/concept_steering/run_concept_steering.sh`
- `chtc/concept_steering/concept_steering_smoke.sub`
- `chtc/concept_steering/concept_steering_sweep.sub`
- `chtc/concept_steering/README.md`
- `results/sft_eval/concept_steering/CHTC_PLAN.md`
- `results/sft_eval/concept_steering/SMOKE_STATUS.md`

## Validation Performed

Local validation intentionally avoids GPU use so it does not interfere with
local MMLU jobs.

Checks run:

```bash
bash -n chtc/concept_steering/run_concept_steering.sh
python -m py_compile src/sft/extract_concept_vectors.py src/sft/run_concept_steering_eval.py
python src/sft/extract_concept_vectors.py --dry-run --concepts coherence human_alignment --max-pairs 4
python src/sft/run_concept_steering_eval.py \
  --dry-run \
  --out-dir /tmp/concept_steering_dryrun \
  --vector-dir results/sft_eval/concept_steering/vectors_smoke \
  --steer-concepts coherence human_alignment \
  --eval-concepts coherence human_alignment \
  --layers 24 \
  --alphas -2 0 2 \
  --max-items 2
```

Results:

- `bash -n chtc/concept_steering/run_concept_steering.sh`: passed.
- `python -m py_compile src/sft/extract_concept_vectors.py src/sft/run_concept_steering_eval.py`: passed.
- extractor dry-run: `coherence` n=4, `human_alignment` n=4.
- eval dry-run: produced the expected layer/alpha/eval-set plan below.
- local `condor_submit -dry-run`: not usable in this container; it fails before
  submit-file parsing with `Failed to determine my IP address using
  NETWORK_INTERFACE=*`.

Observed dry-run eval plan:

```json
{
  "alphas": [-2.0, 0.0, 2.0],
  "eval_sets": {
    "coherence": 2,
    "human_alignment": 2,
    "retention": 2
  },
  "layers": [24],
  "steer_concepts": ["coherence", "human_alignment"]
}
```

## CHTC Attempts

Initial submission:

Submission command:

```bash
chtc-ssh 'cd ~/chtc-runs/coherence-concept-steering-20260709-005726 && condor_submit concept_steering_smoke.sub'
```

Queue state immediately after submission:

- cluster: `5513235`
- submit file: `chtc/concept_steering/concept_steering_smoke.sub`
- run id: `coherence-concept-steering-20260709-005726`
- CHTC requirement: `TARGET.CUDAGlobalMemoryMb >= 40000`
- initial status: idle but satisfiable; `condor_q -better-analyze` found one
  willing high-memory GPU slot and 51 additional possible matches if drained.
- local GPU path: not usable from this session because `nvidia-smi` could not
  communicate with the NVIDIA driver.

Failure diagnosis:

- cluster `5513235` ran on `jcaicedogpu0002.chtc.wisc.edu` with an NVIDIA L40S
  and passed the GPU-memory probe.
- it exited with status `66` before pip/model load because
  `/staging/s/suresh27/models/llama31-8b-instruct/config.json` was not visible
  inside the worker container.
- the CHTC login node can read that path, so the staged model is present.
- the failed worker advertises `HasChtcStaging` as false; both smoke and sweep
  submit files now require `TARGET.HasChtcStaging =?= true`.
- retry cluster `5513261` ran on `gpu5000.chtc.wisc.edu` with an NVIDIA H200
  and confirmed staged model visibility.
- `5513261` failed in extraction after the pip overlay installed
  `torch==2.13.0`, shadowing the container's PyTorch 2.5.1 and causing a
  `torchvision::nms` registration error while importing `LlamaForCausalLM`.
- the runner now uses `pip install --no-deps` for the overlay and explicitly
  lists non-torch dependencies so the container's CUDA-matched torch stack is
  preserved.
- fixed no-deps retry cluster `5513276` ran from
  `~/chtc-runs/coherence-concept-steering-20260709-011336` and passed.

Passing smoke:

- cluster: `5513276`
- local artifacts: `results/sft_eval/concept_steering/chtc/5513276/`
- host: `dbrundagegpu5000.chtc.wisc.edu`
- GPU: NVIDIA L40S, `max_gpu_memory_mb=45460`
- `exit_status.txt`, `extract_exit_status.txt`, `eval_exit_status.txt`, and
  `pip_install_exit_status.txt` are all `0`.
- both vector files were written.
- `sweep/sweep_results.csv` has the expected 18 rows.
- retention positive preference stayed at `1.00` for both steering vectors and
  all alpha settings.
- target margins moved upward for positive alpha:
  `coherence` on coherence `+0.0625`, and `human_alignment` on
  human-alignment `+0.0419`, relative to alpha `0`.

## Bounded Sweep Result

- cluster: `5513297`
- local artifacts: `results/sft_eval/concept_steering/chtc/5513297/`
- host: `gpu4005.chtc.wisc.edu`
- GPU: NVIDIA H100 80GB HBM3
- `exit_status.txt`, `extract_exit_status.txt`, `eval_exit_status.txt`, and
  `pip_install_exit_status.txt` are all `0`.
- `sweep/sweep_results.csv` has 120 rows over 2 steering concepts, 4 layers,
  5 alphas, and 3 eval sets.
- strongest target margin movement: `coherence` layer 12 alpha `4`
  (`+1.0222` vs alpha `0`) and `human_alignment` layer 24 alpha `4`
  (`+0.1197` vs alpha `0`).
- retention preference stayed in `0.8-1.0`, but aggressive layer-12 settings
  reduced retention margins.
- specificity is imperfect: `human_alignment -> coherence` at layer 12 alpha
  `4` produced the largest cross-effect (`+2.5565` margin, `+0.125`
  preference).

## Next Action

Run qualitative generation/judge probes for the strongest low-risk settings and
pair them with wider retention checks before using steering in broad evals.
