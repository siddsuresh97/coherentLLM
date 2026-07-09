# Huth/LeBel fMRI Lane

This directory tracks the ds003020 natural-language fMRI path for testing whether coherence-SFT/LoRA model states improve held-out brain predictivity.

## Current Status

- Metadata audit and CHTC staging audit are complete.
- Smoke data are staged on CHTC at `/staging/s/suresh27/datasets/ds003020-smoke`.
- Successful staging artifact: `chtc_5513059/extracted/sft_huth_lebel_stage_smoke/`.
- Smoke subset: 15 files, 7.877643435 GB.
- Stories: `sweetaspie`, `againstthewind`, `wheretheressmoke`.
- Subjects: `UTS01`, `UTS02`, `UTS03`.

## Executable Smoke Pipeline

- GPU feature extraction: `src/sft/huth_lebel_extract_word_states.py`
- CPU ridge encoding: `src/sft/huth_lebel_smoke_encoding.py`
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
- `ENCODING_PLAN.md`: executable smoke experiment and CHTC scaling plan.
