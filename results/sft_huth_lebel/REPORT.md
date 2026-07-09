# Huth/LeBel Language-fMRI Audit

Generated: 2026-07-08T22:59:00.688548+00:00

Manual update: 2026-07-08 after CHTC staging cluster `5513059` and extraction
debug cluster `5513178`.

## Current Read

- Status: the Huth/LeBel narrative-encoding experiment now has enough staged
  data for a first CHTC smoke test, but not yet for the full high-data run.
- Best staged smoke root: `/staging/s/suresh27/datasets/ds003020-smoke`.
- Staged data pieces: 3 WAVs, 3 TextGrids, and 9 author-preprocessed HF5 files
  for `sweetaspie`, `againstthewind`, and `wheretheressmoke` across
  `UTS01`-`UTS03`.
- Blocking piece: validate the tiny CHTC extraction/encoding smoke before
  staging the 76.86 GB high-data subset.
- Reusable local study hook found: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/tribev2/studies/lebel2023bold.py`
  (`exists=True`).
- CHTC submission is now validated. Audit smoke cluster `5513006` completed
  with exit code 0 and returned logs/artifacts under
  `results/sft_huth_lebel/chtc_5513006/`.
- Metadata-only OpenNeuro mirror inspection is validated separately under
  `results/sft_huth_lebel/metadata_audit/`. That check confirms the current
  `derivatives/` layout, `wheretheressmoke` metadata, and DataLad annex
  symlink evidence, but it does not mean the large files are downloaded.
- Staging plan is ready in `results/sft_huth_lebel/STAGING_PLAN.md`: the first
  smoke subset is 7.88 GB (`sweetaspie`, `againstthewind`,
  `wheretheressmoke` for `UTS01`-`UTS03`), and the full `UTS01`-`UTS03`
  author-preprocessed subset is 76.86 GB across 420 manifest files.
- Smoke staging cluster `5513059` completed with exit code 0 and returned
  artifacts under `results/sft_huth_lebel/chtc_5513059/`. Its stage summary
  reports all 15 planned files present, `7,877,643,435` expected bytes and
  actual bytes, and zero missing paths.
- GPU-side extraction debug cluster `5513178` is running from
  `~/chtc-runs/coherence-huth-extract-debug-20260708-1945` on an NVIDIA L40.
  It extracts layer 24 for the first 64 words of `sweetaspie` across
  `base,lowLR,scrambled,taskvec_a0p25`.

## Checked Roots

| Path | Exists | Plausible | TextGrids | WAVs | HF5 | BOLD | Test Story |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/data` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/data/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/data/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/data/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/data/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/data` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/data/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/data/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/data/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/data/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Experiment Design To Run Once Data Is Present

- Subjects: start with `UTS01`, `UTS02`, and `UTS03`; they have the high-data
  extended story set in LeBel/Huth.
- Smoke test: use `wheretheressmoke` as held-out test story and one or two
  short training stories for a tiny local/CHTC validation.
- Feature extraction: feed exact narrative text streams, not chat templates;
  extract final-token word states by layer for `base`, `lowrank`,
  `taskvec_a0p25`, and `scrambled` first.
- Alignment: map word features to TRs using TextGrid word times, TR=2s, and
  concatenate FIR delays such as 2/4/6/8s.
- Encoding model: voxelwise ridge with layer/ridge selected on train/validation
  only; report paired held-out Pearson `r` deltas vs base by subject/ROI/fold.
- Fedorenko/EvLab constraint: individual language localizer masks are preferred;
  atlas or broad regions must be labeled exploratory.

## Artifacts

- JSON audit: `results/sft_huth_lebel/audit.json`
- Candidate CSV: `results/sft_huth_lebel/candidate_roots.csv`
- Metadata audit: `results/sft_huth_lebel/metadata_audit/REPORT.md`
- Staging plan: `results/sft_huth_lebel/STAGING_PLAN.md`
- Smoke manifest: `results/sft_huth_lebel/staging_manifest_smoke.csv`
- High-data manifest: `results/sft_huth_lebel/staging_manifest_highdata.csv`
- CHTC smoke outputs: `results/sft_huth_lebel/chtc_5513006/`
- CHTC staging smoke outputs: `results/sft_huth_lebel/chtc_5513059/`
- CHTC extraction debug outputs, once pulled:
  `results/sft_huth_lebel/chtc_huth_extract_debug_5513178/`
