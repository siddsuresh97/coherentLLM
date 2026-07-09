# Concept Steering Smoke Status

Last updated: 2026-07-09.

## Current State

Prepared, locally dry-run validated, and submitted to CHTC for the first GPU
smoke. The first smoke failed before model load because the worker did not
expose CHTC staging; the retry submit files now require staged-storage access.

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

## Next Action

Monitor CHTC cluster `5513235`, run directory
`~/chtc-runs/coherence-concept-steering-20260709-005726`.

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
- fixed no-deps retry cluster `5513281` was submitted from
  `~/chtc-runs/coherence-concept-steering-20260709-011445`; first poll is idle,
  no hold reason, and satisfiable.

Smoke success requires all three status files in the returned tarball to be
`0`:

- `exit_status.txt`
- `extract_exit_status.txt`
- `eval_exit_status.txt`

Then inspect `sweep/SUMMARY.md` and `sweep/sweep_results.csv` before launching
`concept_steering_sweep.sub`.
