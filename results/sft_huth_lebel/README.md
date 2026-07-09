# Huth/LeBel fMRI Lane

This directory tracks the ds003020 natural-language fMRI path for testing whether coherence-SFT/LoRA model states improve held-out brain predictivity.

## Current Status

- Metadata audit and CHTC staging audit are complete.
- Smoke data are staged on CHTC at `/staging/s/suresh27/datasets/ds003020-smoke`.
- Successful staging artifact: `chtc_5513059/extracted/sft_huth_lebel_stage_smoke/`.
- Smoke subset: 15 files, 7.877643435 GB.
- Stories: `sweetaspie`, `againstthewind`, `wheretheressmoke`.
- Subjects: `UTS01`, `UTS02`, `UTS03`.
- GPU extraction debug is complete: CHTC cluster `5513178`, artifact
  `chtc_huth_extract_debug_5513178/`, four arms, `sweetaspie`, first 64 words,
  layer 24. Wrapper and extractor exit codes are both `0`; each arm produced
  `hidden` arrays with shape `(64, 1, 4096)`.
- Three-story staged-output smoke extraction `5513245` wrote only the base
  features before failing on `/staging/s/suresh27` quota while creating the
  `lowLR` feature directory.
- Bundle-output recovery extraction `5513306` passed from
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`, returning
  `huth_extract_smoke_results.tgz` with all 12 arm/story NPZs for
  `base,lowLR,scrambled,taskvec_a0p25` across layers `16,24,32`.
- Capped CPU encoding `5513337` passed for `UTS01`, and capped CPU scale check
  `5513350` passed for `UTS02,UTS03`. These runs validate the path only because
  they use two training stories and a 2000-voxel cap.
- Uncapped all-subject CPU encoding is running as cluster `5513373` from the
  same `5513306` run directory. It uses
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle_uncapped_all.sub`,
  `MAX_VOXELS=0`, subjects `UTS01,UTS02,UTS03`, no GPU request, and current
  resources `request_cpus=4`, `request_memory=16GB`, `request_disk=20GB`.
  The event log shows input transfer completed and execution started on
  `oconnor2007.chtc.wisc.edu` at `2026-07-08 21:13:46 CDT` with `GPUs=0`.
- Duplicate cluster `5513244` held before work because its submit expected a
  missing output tarball; it was removed with `condor_rm`.

## Executable Smoke Pipeline

- GPU feature extraction: `src/sft/huth_lebel_extract_word_states.py`
- CPU ridge encoding: `src/sft/huth_lebel_smoke_encoding.py`
- CHTC wrappers: `chtc/huth_lebel_smoke/huth_extract_smoke.sub`,
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle.sub`, and
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle_uncapped_all.sub`
- Full command plan: `ENCODING_PLAN.md`

Expected first outputs:

- `word_states_smoke/<arm>/<story>.npz`
- `word_states_smoke/word_tables/<story>.csv`
- `smoke_encoding/summary.csv`
- `smoke_encoding/alpha_cv.csv`
- `smoke_encoding/run_metadata.json`

## Interpretation Boundary

The smoke metric is Huth-style held-out voxelwise encoding: Pearson `r` between predicted and observed BOLD for `wheretheressmoke`. It can compare `base`, `lowLR`, `scrambled`, and `taskvec_a0p25` fairly when all extraction/alignment/ridge settings are fixed.

Fedorenko/EvLab language-network claims require independent subject-specific language localizers. Until those are available, atlas or broad language-region summaries should be labeled exploratory.

## Key Files

- `STAGING_PLAN.md`: smoke/high-data staging manifest plan.
- `staging_manifest_smoke.csv`: exact staged smoke files.
- `chtc_5513059/extracted/sft_huth_lebel_stage_smoke/stage_summary.json`: successful staging byte audit.
- `chtc_huth_extract_debug_5513178/`: successful first GPU feature extraction
  debug bundle.
- `chtc_huth_extract_smoke_retry_5513306/`: successful bundle-output
  three-story feature extraction metadata.
- `chtc_huth_encoding_smoke_bundle_5513337/`: capped `UTS01` encoding smoke.
- `chtc_huth_encoding_smoke_bundle_uts02_uts03_5513350/`: capped
  `UTS02,UTS03` encoding smoke.
- `ENCODING_PLAN.md`: executable smoke experiment and CHTC scaling plan.
