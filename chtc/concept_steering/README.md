# Concept Steering CHTC Jobs

This directory runs the concept-vector steering scaffold from commit `99c0c60`
on CHTC without competing with local MMLU jobs.

The jobs expect the already staged base model:

- `/staging/s/suresh27/models/llama31-8b-instruct`

The current steering scripts load the base model only. The staged adapter
convention is still respected by not transferring model weights through Condor;
adapter-conditioned steering would need a source-level extension before these
jobs should consume `/staging/s/suresh27/adapters/*`.

## Smoke Design

- concepts extracted: `coherence`, `human_alignment`
- extraction split: train
- vector pairs per concept: first 4 contrast pairs
- steering layer: 24
- alpha sweep: `-2, 0, 2`
- eval items: 2 coherence, 2 human-alignment, 2 retention items
- model dtype: `bfloat16`
- GPU requirement: `TARGET.CUDAGlobalMemoryMb >= 40000`
- storage requirement: `TARGET.HasChtcStaging =?= true`

The extractor always stores all layer rows in each `.npz`; the smoke consumes
only layer 24 during steering. Expected `sweep_results.csv` shape is:

```text
2 steering concepts x 1 layer x 3 alphas x 3 eval sets = 18 rows
```

## Files

- `run_concept_steering.sh`: CHTC container runner. It writes a runtime
  `configs/models.yaml` pointing at staged storage, verifies a 40GB+ GPU,
  installs a small Python overlay if needed, extracts vectors, runs the
  forced-choice sweep, and returns one tarball.
- `concept_steering_smoke.sub`: one-GPU smoke at layer 24 with alphas `-2,0,2`.
- `concept_steering_sweep.sub`: one-GPU scale-up sweep over layers
  `12,16,20,24` and alphas `-4,-2,0,2,4`.

## Submit

Run the smoke first:

```bash
RUN_ID=coherence-concept-steering-YYYYMMDD-HHMM
chtc-ssh "mkdir -p ~/chtc-runs/${RUN_ID}/logs ~/chtc-runs/${RUN_ID}/src ~/chtc-runs/${RUN_ID}/data/sft/steering_contrasts"
chtc-push chtc/concept_steering/ "chtc-runs/${RUN_ID}/"
chtc-push src/ "chtc-runs/${RUN_ID}/src/"
chtc-push data/sft/steering_contrasts/ "chtc-runs/${RUN_ID}/data/sft/steering_contrasts/"
chtc-ssh "cd ~/chtc-runs/${RUN_ID} && condor_submit concept_steering_smoke.sub"
```

Monitor:

```bash
chtc-ssh "condor_q -batch suresh27"
chtc-ssh "condor_q <cluster_id> -af:jh ClusterId ProcId JobStatus HoldReason LastRemoteHost RemoteWallClockTime DiskUsage_RAW"
chtc-ssh "condor_tail <cluster_id>.0"
```

Pull:

```bash
mkdir -p results/sft_eval/concept_steering/chtc/${RUN_ID}
chtc-pull "chtc-runs/${RUN_ID}/concept_steering_smoke_layer24_alpha_neg2_0_pos2_results.tgz" \
  "results/sft_eval/concept_steering/chtc/${RUN_ID}/"
chtc-pull "chtc-runs/${RUN_ID}/logs/" \
  "results/sft_eval/concept_steering/chtc/${RUN_ID}/logs/"
```

If the smoke tarball has `exit_status.txt`, `extract_exit_status.txt`, and
`eval_exit_status.txt` all equal to `0`, submit the scale-up sweep from the same
run directory:

```bash
chtc-ssh "cd ~/chtc-runs/${RUN_ID} && condor_submit concept_steering_sweep.sub"
```

Expected full-sweep tarball:

```text
concept_steering_sweep_layers12_16_20_24_alpha_neg4_neg2_0_pos2_pos4_results.tgz
```

## Success Criteria

Smoke succeeds if:

- both vector files exist:
  `vectors/coherence_caa_vectors.npz` and
  `vectors/human_alignment_caa_vectors.npz`
- `sweep/sweep_results.csv` has 18 rows plus header
- `sweep/SUMMARY.md` has finite `positive_preference` and
  `mean_delta_logprob` values for target and retention rows
- retention rows at `alpha=-2` and `alpha=2` do not collapse relative to
  `alpha=0`

The scale-up sweep should be launched only after the smoke shows finite scores
and no import/model-loading failures.

## Failure Note

The first CHTC smoke, cluster `5513235`, landed on
`jcaicedogpu0002.chtc.wisc.edu` and failed before model loading because the
worker container could not read
`/staging/s/suresh27/models/llama31-8b-instruct/config.json`. The login node
could read that path, and the failed worker advertised `HasChtcStaging` as
false. The submit files now require `TARGET.HasChtcStaging =?= true` so retries
land only on workers that expose CHTC staging.

The second smoke, cluster `5513261`, landed correctly on `gpu5000.chtc.wisc.edu`
with an NVIDIA H200 and saw the staged model. It failed during extraction after
the Python overlay installed `torch==2.13.0` as an `accelerate` dependency,
shadowing the container's CUDA 12.4 PyTorch 2.5.1 and breaking the matching
`torchvision` import path. The runner now installs the lightweight Python
overlay with `pip --no-deps` and explicitly lists non-torch dependencies, so
the base image keeps its tested `torch`/`torchvision` pair.
