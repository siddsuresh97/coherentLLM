# Huth/Fedorenko Next Experiment Plan

Updated: 2026-07-08 21:44 CDT

Scope: fMRI/Huth/Fedorenko lane only. This plan does not touch concept
steering.

## Headline

The Huth/LeBel path is now technically validated through a full-voxel
three-subject smoke, but the smoke does not show a coherence-aligned arm beating
base. The next useful experiment is not another smoke rerun. It is a
pre-registered high-data language-fMRI run that uses more stories, fixed or
nested layer selection, and a Fedorenko-safe interpretation boundary.

No new fMRI CHTC job was submitted in this checkpoint. The current blocker is
CHTC staging file quota, not code correctness:

```text
/staging/s/suresh27  24.3209 GB used / 100 GB limit
Files_Used           1120 / 1000 file limit
```

The high-data manifest is already present:

- `results/sft_huth_lebel/staging_manifest_highdata.csv`
- 84 shared stories for `UTS01,UTS02,UTS03`
- 420 planned files: 252 HF5, 84 WAV, 84 TextGrid
- 76.858043164 GB planned

Because the file limit is already exceeded, do not submit high-data staging
until the staging design uses packed artifacts or the file count is freed.

## Completed Smoke Synthesis

Completed CHTC chain:

| Step | Cluster | Result |
|---|---:|---|
| Smoke data staging audit | `5513059` | passed, 15/15 files, 7.88 GB |
| GPU extraction debug | `5513178` | passed, all four arms, 64-word check |
| Bundle-output extraction | `5513306` | passed, 12 arm/story NPZs |
| Capped CPU encoding | `5513337` | passed, `UTS01`, 2000 voxels |
| Capped CPU scale check | `5513350` | passed, `UTS02,UTS03`, 2000 voxels |
| Uncapped CPU encoding | `5513373` | passed, all three subjects, full voxels |

The uncapped smoke used:

- Train stories: `sweetaspie,againstthewind`
- Test story: `wheretheressmoke`
- Subjects: `UTS01,UTS02,UTS03`
- Arms: `base,lowLR,scrambled,taskvec_a0p25`
- Layers: `16,24,32`
- Full voxel counts: `81126`, `94251`, `95556`
- Output: `results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_uncapped_all_5513373/`

Mean held-out Pearson `r` across subjects:

| Arm | L16 | L24 | L32 |
|---|---:|---:|---:|
| `base` | 0.009785 | 0.007987 | 0.007525 |
| `taskvec_a0p25` | 0.009137 | 0.006598 | 0.004197 |
| `lowLR` | 0.008287 | 0.006338 | 0.003442 |
| `scrambled` | 0.004988 | 0.003622 | -0.002132 |

Read: full-voxel execution works and `taskvec_a0p25` is closer to base than
`lowLR`, but base is still the best subject-level row for all three subjects.
Treat this as a pipeline validation and weak negative/neutral scientific smoke,
not as the final Huth/Fedorenko result.

## Alex Huth / LeBel Evaluation

Primary claim to test:

> Do coherence-aligned model states improve held-out voxelwise prediction of
> passive natural-language fMRI responses under a fixed linear encoding model?

Keep the claim Huth-style:

- Use exact narrative text streams, not chat templates.
- Extract word-aligned hidden states from the same Llama backbone.
- Align word features to TRs using TextGrid word times.
- Concatenate fixed FIR delays.
- Fit voxelwise ridge on training stories.
- Select ridge alpha and any layer rule using only training/validation stories.
- Score held-out stories by voxelwise Pearson `r`.
- Report paired deltas against `base`.

Model-arm priority:

1. `base`: required anchor.
2. `taskvec_a0p25`: primary induced-skill arm because it has the best current
   semantic-hub/benchmark tradeoff.
3. `lowLR`: original aligned LoRA arm.
4. `scrambled`: required negative control.
5. `taskvec_a0p5` and `taskvec_a1p0`: optional diagnostics after the primary
   four-arm run is stable.

Layer rule:

- Primary fixed layers: `16,24,32`, matching the smoke.
- Secondary nested selection: choose from layers `10:20` on validation stories
  only, because the semantic-hub hypothesis predicts mid-layer effects.
- Never choose layers from `wheretheressmoke` or any final held-out story.

## Fedorenko / EvLab Evaluation

Primary caution: the current `ds003020` smoke path is a Huth-style natural
listening encoding experiment. It is not automatically a Fedorenko language
network experiment.

Strong Fedorenko-style claims require:

- subject-specific functional language localizers;
- a language over control contrast, typically sentences over nonword lists;
- individual functional ROIs rather than broad anatomical or group parcels.

Until such masks are available, report only:

- whole-cortex or available cortical mask: primary Huth metric;
- Huth-style semantic-map or language-like atlas regions: exploratory;
- Fedorenko/LanA/probabilistic atlas labels: exploratory language-like ROIs;
- auditory cortex/control regions: negative controls.

If later using Pereira/Ryskina-style concept-consistency data, keep it as a
separate cross-modal concept experiment. Do not treat that as a substitute for
subject-specific ds003020 language localizers.

## Semantic Hub / MEMP Bridge

Hypothesis:

> If coherence SFT induces a reusable semantic memory/hub, Huth/LeBel gains
> should concentrate in semantic/language-like regions and correlate with
> middle-layer hub metrics across arms/layers.

Bridge variables to pre-register:

- Internal hub metrics: same-minus-random, same-minus-S*-close,
  cross-format RDM Spearman, CKA, retrieval top-1/top-5.
- Brain metrics: held-out mean voxelwise `r`, median `r`, fraction positive,
  ROI-level mean delta vs base.
- Paired deltas: `taskvec_a0p25 - base`, `lowLR - base`,
  `scrambled - base`, and `taskvec_a0p25 - scrambled`.

Interpretation gates:

- Good support: `taskvec_a0p25` improves semantic/language-like ROI prediction
  over base and scrambled, and the improvement tracks mid-layer hub metrics.
- Weak support: task-vector is close to base while scrambled is lower, but no
  base-beating gain appears.
- Failure: scrambled improves similarly, gains appear only in auditory control
  regions, or hub metrics predict benchmark damage better than fMRI gains.

## Next Scalable Experiment

### Stage 1: Unblock High-Data Staging

Do not submit raw high-data staging in the current quota state. First use one of
these two paths:

1. Preferred: create packed story artifacts in staging.
   - CPU job downloads one story at a time into scratch.
   - Pack each story's WAV, TextGrid, and three subject HF5 files into one
     `story.tar.zst`.
   - Store packs under `/staging/s/suresh27/datasets/ds003020-highdata-packs/`.
   - Later jobs unpack only needed stories into scratch.

2. Fallback: free staging file count and disk, then raw-stage high-data.
   - Current high-data adds 420 files and 76.86 GB.
   - Because current staging is already at 1120/1000 files, raw staging requires
     reducing file count below roughly 580 before launch.
   - Because current disk is 24.32 GB, raw staging also needs freeing at least
     about 2 GB, or replacing the smoke root after high-data is verified.

Required checks before staging:

```bash
chtc-ssh 'get_quotas'
chtc-ssh 'condor_q -batch suresh27'
chtc-ssh 'du -sh /staging/s/suresh27/datasets/ds003020-smoke /staging/s/suresh27/models /staging/s/suresh27/adapters'
```

### Stage 2: High-Data Extraction

After staging is unblocked, split feature extraction into many one-GPU jobs.

Recommended job unit:

- one story per process;
- all primary arms for that story;
- layers `16,24,32` first;
- `request_gpus=1`, `request_cpus=8`, `request_memory=64GB`;
- require `TARGET.CUDAGlobalMemoryMb >= 40000`;
- write feature packs, not thousands of loose files.

Extraction command shape:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /scratch/ds003020-story-root \
  --model_path /staging/s/suresh27/models/llama31-8b-instruct \
  --hf_cache /staging/s/suresh27/hf_home \
  --out_dir "$PWD/features" \
  --stories "$STORY" \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --adapter lowLR=/staging/s/suresh27/adapters/lowLR \
  --adapter scrambled=/staging/s/suresh27/adapters/scrambled \
  --adapter taskvec_a0p25=/staging/s/suresh27/adapters/taskvec_a0p25 \
  --layers 16,24,32 \
  --max_context_tokens 512 \
  --batch_size 4 \
  --device cuda \
  --dtype bfloat16 \
  --save_dtype float16 \
  --skip_existing
```

### Stage 3: High-Data Encoding

First interpretable run:

- subjects: `UTS01,UTS02,UTS03`;
- train stories: all staged high-data stories except held-out folds;
- held-out folds: pre-register 5 stories spanning size/content variation, or
  use leave-one-story-out if runtime is acceptable;
- arms: `base,taskvec_a0p25,lowLR,scrambled`;
- layers: fixed `16,24,32`;
- save voxel correlations for ROI summaries;
- no `max_voxels` cap.

Command shape after feature packs and data packs are unpacked in scratch:

```bash
python src/sft/huth_lebel_smoke_encoding.py \
  --ds_root "$PWD/ds003020-highdata" \
  --features_dir "$PWD/features" \
  --out_dir "$PWD/huth_encoding_highdata" \
  --subjects UTS01 UTS02 UTS03 \
  --train_stories "$TRAIN_STORIES_CSV" \
  --test_story "$TEST_STORY" \
  --arms base,taskvec_a0p25,lowLR,scrambled \
  --layers 16,24,32 \
  --ridge_solver auto \
  --save_voxel_corrs \
  --overwrite
```

### Stage 4: Reporting

Every high-data job must return:

- exact submit file and wrapper;
- `run_metadata.json`;
- `summary.csv`;
- `alpha_cv.csv`;
- optional `voxel_corrs/*.npy`;
- resource usage and queue/wall time;
- a README with cluster id, local validation commands, and pass/fail read.

Primary tables to produce:

- subject x fold x arm x layer mean/median/p95/frac-positive `r`;
- paired delta table vs base;
- story-fold stability table;
- ROI/control summaries if masks are available;
- hub-metric-to-brain-delta bridge table.

## Local Validation Completed

These checks passed before writing this plan:

```bash
python -m py_compile \
  src/sft/huth_lebel_smoke_encoding.py \
  src/sft/huth_lebel_extract_word_states.py \
  src/sft/plan_huth_lebel_staging.py \
  src/sft/huth_lebel_audit.py
```

```bash
bash -n chtc/huth_lebel_stage_smoke/run_stage_smoke.sh
bash -n chtc/huth_lebel_smoke/run_extract_smoke.sh
```

CHTC state checked:

```bash
chtc-master check
chtc-ssh 'condor_q -batch suresh27'
chtc-ssh 'get_quotas'
chtc-ssh 'condor_status -constraint "State == \"Unclaimed\" && Gpus >= 1 && Cpus >= 8 && Memory >= 64000 && CUDAGlobalMemoryMb >= 40000" -total'
```

At `2026-07-08 21:43 CDT`, the only queued job was concept-steering cluster
`5513407.0` running. GPU availability for the 8-CPU/64GB/40GB-VRAM shape was
broad, with 44 unclaimed matching slots reported, but no fMRI GPU job should be
launched until staging is unblocked.

## Unresolved Blockers

1. CHTC staging file quota is exceeded: `1120/1000` files.
2. Raw high-data staging would also push disk close to or above the 100 GB
   staging limit unless duplicate smoke data or other staging artifacts are
   removed after verification.
3. No subject-specific Fedorenko language localizer masks are staged or
   validated for this dataset lane.
4. High-data feature storage needs packed artifacts or per-story bundles; loose
   NPZ files in staging are not acceptable under the current file quota.
5. A high-data fold manifest still needs final pre-registration before launch.
