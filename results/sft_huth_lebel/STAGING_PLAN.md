# Huth/LeBel ds003020 Staging Plan

Generated: 2026-07-08T23:13:09.747553+00:00

Manual update: 2026-07-08 21:44 CDT after uncapped encoding cluster `5513373`
completed and high-data staging was rechecked.

## Dataset

- Root inspected: `/tmp/ds003020-git`
- Name: An fMRI dataset during a passive natural language listening task
- Dataset DOI: `doi:10.18112/openneuro.ds003020.v3.1.1`
- Subjects planned first: `UTS01, UTS02, UTS03`
- Held-out test story: `wheretheressmoke`
- Shared preprocessed stories across subjects: 84

## Recommended Staging

- Use explicit CHTC staging path `/staging/s/suresh27`; `$STAGING` was unset on `ap2002`.
- Stage the smoke subset first under `/staging/s/suresh27/datasets/ds003020-smoke`.
- If the smoke encoding path works, stage the high-data `UTS01`-`UTS03` subset under `/staging/s/suresh27/datasets/ds003020-highdata`.
- Do staging with a CPU/download job or local download plus rsync; do not consume a GPU for data transfer.
- The AP currently lacks `datalad`, `git-annex`, `openneuro`, `aws`, and `aria2c`, so a containerized downloader is the safer CHTC route.
- Current quota means raw high-data staging should not be submitted yet:
  `/staging/s/suresh27` is using `1120/1000` files and `24.3209/100` GB. The
  high-data manifest adds 420 files and 76.86 GB, so use packed story artifacts
  or free file count/disk first.

## Staging Status

- Smoke staging cluster `5513059` completed with exit code 0.
- Staged root: `/staging/s/suresh27/datasets/ds003020-smoke`.
- Planned files: 15.
- Missing files: 0.
- Expected bytes and actual bytes: `7,877,643,435`.
- Pulled proof bundle:
  `results/sft_huth_lebel/chtc_5513059/`.
- Extraction debug status: cluster `5513178` completed with return value `0`.
  Pulled artifacts are in
  `results/sft_huth_lebel/chtc_huth_extract_debug_5513178/`; all four arms
  produced layer-24 `sweetaspie` features with hidden shape `64 x 1 x 4096`.
- Full smoke extraction status: cluster `5513245` ran from
  `~/chtc-runs/coherence-huth-extract-smoke-20260709-005926`. It wrote the
  three `base` feature NPZs under
  `/staging/s/suresh27/features/huth_lebel_smoke_llama31`, then failed with
  `OSError: [Errno 122] Disk quota exceeded` while creating the `lowLR`
  feature directory. New directories under `/staging/s/suresh27` currently
  fail with the same quota error, so the recovery path avoids writing features
  to staging.
- Active recovery extraction: cluster `5513306`, remote directory
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`. It seeds
  the existing staged `base` features into job scratch and should return all
  features in `huth_extract_smoke_results.tgz`. It passed with Condor return
  value `0`, wrapper status `0`, extractor status `0`, and all 12 arm/story
  NPZs present in `npz_shapes.tsv`.
- CPU encoding smoke cluster `5513337` passed from the same run directory with
  `huth_encoding_smoke_bundle.sub`; the returned bundle includes
  `encoding_exit_status.txt == 0`, `summary.csv`, and `alpha_cv.csv`.
- CPU-only scale check cluster `5513350` passed from the same bundle for capped
  `UTS02,UTS03` encoding.
- Uncapped all-subject CPU encoding cluster `5513373` passed from the same
  bundle with `MAX_VOXELS=0`, all three subjects, and all 36 full-voxel rows.
  This validates the technical path; it does not show a base-beating
  task-vector result.
- High-data staging status at `2026-07-08 21:43 CDT`: not submitted. CHTC
  quota check showed `/staging/s/suresh27` at `1120/1000` files and
  `24.3209/100` GB. The next safe staging design is one packed artifact per
  story, e.g. `ds003020-highdata-packs/<story>.tar.zst`, unpacked in scratch by
  feature-extraction and encoding jobs.
- High-data packed-story plan status at `2026-07-08 21:58 CDT`: implemented
  locally. `src/sft/huth_lebel_pack_stories.py plan` groups
  `staging_manifest_highdata.csv` into 84 story archives instead of 420 loose
  raw files, a 336-entry reduction. The pack manifest is
  `highdata_story_pack_manifest.csv`; the quota/cleanup report is
  `STAGING_UNBLOCK_REPORT.md`.
- Duplicate recovery cluster `5513310` was removed while idle.
- Duplicate cluster `5513244` held before model work because its submit
  expected a missing output tarball; it was removed with `condor_rm`.

## Smoke Subset

- Stories: `sweetaspie, againstthewind, wheretheressmoke`
- Files: 15 total; missing metadata entries: 0
- Planned size: 7.88 GB (7.34 GiB)

| Kind | GB | GiB |
|---|---:|---:|
| hf5 | 7.65 | 7.12 |
| textgrid | 0.00 | 0.00 |
| wav | 0.23 | 0.21 |

## High-Data UTS01-UTS03 Subset

- Stories: all 84 shared preprocessed stories across `UTS01, UTS02, UTS03`.
- Files: 420 total; missing metadata entries: 0
- Planned size: 76.86 GB (71.58 GiB)
- This fits the nominal 100 GB CHTC staging quota only after freeing duplicate
  smoke/other staging data or using packed artifacts. It does not fit the
  current file quota state.

| Kind | GB | GiB |
|---|---:|---:|
| hf5 | 67.19 | 62.57 |
| textgrid | 0.08 | 0.08 |
| wav | 9.59 | 8.93 |

## Artifacts

- Smoke manifest: `results/sft_huth_lebel/staging_manifest_smoke.csv`
- High-data manifest: `results/sft_huth_lebel/staging_manifest_highdata.csv`
- Summary JSON: `results/sft_huth_lebel/staging_summary.json`
- High-data pack manifest:
  `results/sft_huth_lebel/highdata_story_pack_manifest.csv`
- High-data pack summary:
  `results/sft_huth_lebel/highdata_story_pack_summary.json`
- Staging unblock report:
  `results/sft_huth_lebel/STAGING_UNBLOCK_REPORT.md`

## Next

1. Do not uncap or rerun the three-subject smoke; `5513373` already did this.
2. Unblock high-data staging by packed story artifacts or by freeing staging
   file count below the raw-staging requirement.
3. Stage the high-data `UTS01`-`UTS03` subset only after quota checks pass, then
   run the high-data plan in `NEXT_EXPERIMENT_PLAN.md`.
