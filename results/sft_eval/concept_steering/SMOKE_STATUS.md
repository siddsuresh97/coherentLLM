# Concept Steering Smoke Status

Last updated: 2026-07-09.

## Current State

Prepared, locally dry-run validated, and submitted to CHTC for the first GPU
smoke.

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

Smoke success requires all three status files in the returned tarball to be
`0`:

- `exit_status.txt`
- `extract_exit_status.txt`
- `eval_exit_status.txt`

Then inspect `sweep/SUMMARY.md` and `sweep/sweep_results.csv` before launching
`concept_steering_sweep.sub`.
