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
- Uncapped all-subject CPU encoding `5513373` passed from the same `5513306`
  run directory. It used
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle_uncapped_all.sub`,
  `MAX_VOXELS=0`, subjects `UTS01,UTS02,UTS03`, no GPU request, and resources
  `request_cpus=4`, `request_memory=16GB`, `request_disk=20GB`. The event log
  shows execution on `oconnor2007.chtc.wisc.edu` from
  `2026-07-08 21:13:46` to `2026-07-08 21:30:09 CDT` with `GPUs=0`, and
  Condor history reports `ExitCode=0`.
- Uncapped `5513373` validation passed: `exit_status.txt == 0`,
  `encoding_exit_status.txt == 0`, `summary.csv` has all 36 full-voxel rows,
  and `alpha_cv.csv` has all 360 alpha-CV rows. Full voxel counts were used:
  `UTS01=81126`, `UTS02=94251`, `UTS03=95556`.
- Uncapped smoke read: `base` is the best subject-level row for all three
  subjects (`UTS01` base L16 mean `r=0.007640`, `UTS02` base L32 mean
  `r=0.011303`, `UTS03` base L16 mean `r=0.011305`). `taskvec_a0p25` beats
  same-layer base only for `UTS02` L16 (`0.011037` vs `0.010409`);
  `lowLR` trails base in all nine subject/layer cells; `scrambled` is usually
  lower and is negative for `UTS02` L24/L32.
- Next-experiment planning is updated in `NEXT_EXPERIMENT_PLAN.md`. The next
  scientific run should be high-data story scaling, not another smoke rerun.
  Current CHTC staging state blocks immediate raw high-data submission:
  `/staging/s/suresh27` is at `1120/1000` files despite only `24.3209/100` GB
  used. Use packed story artifacts or free file count before staging the
  84-story, 420-file, 76.86 GB high-data manifest.
- High-data staging unblock work is tracked in `STAGING_UNBLOCK_REPORT.md`.
  A packed-story plan is now generated: 420 raw high-data files become 84
  planned story archives under
  `/staging/s/suresh27/datasets/ds003020-highdata-packs`, reducing the planned
  file count by 336 entries. The preferred cleanup candidate is the rebuildable
  MMLU HF dataset cache at `/staging/s/suresh27/hf_datasets_cache`; no cleanup
  has been performed yet. Pack-smoke retry `5513422` passed on CHTC with
  `ExitCode=0`, proving the tar dereference path handles ds003020 git-annex
  symlinks.
- Duplicate cluster `5513244` held before work because its submit expected a
  missing output tarball; it was removed with `condor_rm`.

## Current Scientific Conclusion

The Huth/LeBel lane is technically validated but not scientifically positive
yet. The uncapped all-subject smoke proves the feature extraction, alignment,
ridge fitting, alpha CV, and held-out scoring path works. It does not show a
base-beating coherence/task-vector effect:

- Subject-mean layer-16 Pearson `r`: `base=0.009785`,
  `taskvec_a0p25=0.009137`, `lowLR=0.008287`, `scrambled=0.004988`.
- Base is the best subject-level row for all three staged subjects in the
  uncapped smoke.
- `taskvec_a0p25` beats same-layer base only in one cell:
  `UTS02` layer 16 (`0.011037` vs `0.010409`).

The right next experiment is the high-data Huth/LeBel run with packed story
artifacts and fixed or nested layer selection. Do not claim an Alex Huth /
LeBel positive effect from the smoke results.

Fedorenko/EvLab status is separate: there is no conclusive Fedorenko-style
language-network result yet. The current `ds003020` path is natural-listening
encoding, not an individual language-localizer experiment. Atlas or broad
language-like summaries must stay exploratory until subject-specific language
fROIs or localizer contrasts are added.

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
- `chtc_huth_encoding_smoke_bundle_uncapped_all_5513373/`: uncapped
  `UTS01,UTS02,UTS03` full-voxel smoke, result bundle, Condor logs, exact
  summary table, and extracted CSVs.
- `ENCODING_PLAN.md`: executable smoke experiment and CHTC scaling plan.
- `NEXT_EXPERIMENT_PLAN.md`: current Huth/Fedorenko high-data plan, semantic
  hub/MEMP bridge design, exact runnable command shapes, and blockers.
- `STAGING_UNBLOCK_REPORT.md`: current CHTC file-quota audit, cleanup
  candidates, packed-story manifests, and pack-smoke status.
- `chtc_huth_pack_smoke_retry_5513422/`: successful CHTC pack-smoke proof
  files for `againstthewind`.
- `highdata_story_pack_manifest.csv`: 84 planned story pack rows for the
  `UTS01`-`UTS03` high-data subset.
- `staging_cleanup_candidates.csv`: exact CHTC staging paths proposed for
  cleanup or retention.

## Uncapped 5513373 Mean-r Table

These rows use `sweetaspie,againstthewind` for training and held-out
`wheretheressmoke` for testing. Values are mean voxelwise Pearson `r` over all
finite voxels.

| Subject | Arm | L16 | L24 | L32 |
|---|---|---:|---:|---:|
| `UTS01` | `base` | 0.007640 | 0.005942 | 0.003637 |
| `UTS01` | `lowLR` | 0.005228 | 0.004545 | -0.000024 |
| `UTS01` | `scrambled` | 0.003554 | 0.005323 | 0.000069 |
| `UTS01` | `taskvec_a0p25` | 0.005663 | 0.004924 | 0.001469 |
| `UTS02` | `base` | 0.010409 | 0.008560 | 0.011303 |
| `UTS02` | `lowLR` | 0.009891 | 0.006204 | 0.004589 |
| `UTS02` | `scrambled` | 0.001885 | -0.000144 | -0.009166 |
| `UTS02` | `taskvec_a0p25` | 0.011037 | 0.005836 | 0.005713 |
| `UTS03` | `base` | 0.011305 | 0.009458 | 0.007635 |
| `UTS03` | `lowLR` | 0.009744 | 0.008265 | 0.005761 |
| `UTS03` | `scrambled` | 0.009525 | 0.005687 | 0.002701 |
| `UTS03` | `taskvec_a0p25` | 0.010710 | 0.009034 | 0.005410 |
