# Huth/LeBel ds003020 Staging Plan

Generated: 2026-07-08T23:13:09.747553+00:00

Manual update: 2026-07-09 after CHTC extraction debug cluster `5513178`
completed and full smoke extraction cluster `5513245` was submitted.

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
- Full smoke extraction status: cluster `5513245` is running from
  `~/chtc-runs/coherence-huth-extract-smoke-20260709-005926`. This active run
  uses the staged-output wrapper, so features should appear under
  `/staging/s/suresh27/features/huth_lebel_smoke_llama31`; its returned
  `huth_extract_smoke_results.tgz` is status/metadata. Before submitting
  encoding, bundle those staged features into
  `huth_extract_smoke_results_with_features.tgz`.
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
- This fits the observed 100 GB CHTC staging quota if only the needed WAV, TextGrid, and author-preprocessed HF5 files are staged.

| Kind | GB | GiB |
|---|---:|---:|
| hf5 | 67.19 | 62.57 |
| textgrid | 0.08 | 0.08 |
| wav | 9.59 | 8.93 |

## Artifacts

- Smoke manifest: `results/sft_huth_lebel/staging_manifest_smoke.csv`
- High-data manifest: `results/sft_huth_lebel/staging_manifest_highdata.csv`
- Summary JSON: `results/sft_huth_lebel/staging_summary.json`

## Next

1. Poll and pull cluster `5513245` after completion. Pass criteria:
   `extract_exit_status.txt == 0`, `npz_shapes.tsv` contains all 12 arm/story
   NPZs or the staged feature directory contains all 12 arm/story NPZs, each
   with layers `16,24,32`, and no adapter load failures.
2. Package staged features into
   `huth_extract_smoke_results_with_features.tgz`, then submit the CPU ridge
   smoke from `~/chtc-runs/coherence-huth-extract-smoke-20260709-005926` with
   `huth_encoding_smoke_with_features.sub`.
3. Verify artifact writing, feature shapes, TR alignment, and voxelwise Pearson
   scoring in `summary.csv` and `alpha_cv.csv` before using more GPUs.
4. If the smoke passes, stage the high-data `UTS01`-`UTS03` subset under
   `/staging/s/suresh27/datasets/ds003020-highdata`.
5. Run the full high-data encoding jobs only after the smoke report is committed.
