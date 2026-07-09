# Huth/LeBel Language-fMRI Audit

Generated: 2026-07-08T22:59:00.688548+00:00

Manual update: 2026-07-09 after pulling CHTC extraction debug cluster `5513178`
and submitting full smoke extraction cluster `5513245`.

## Current Read

- Status: the Huth/LeBel narrative-encoding experiment now has enough staged
  data for a first CHTC smoke test, but not yet for the full high-data run.
- Best staged smoke root: `/staging/s/suresh27/datasets/ds003020-smoke`.
- Staged data pieces: 3 WAVs, 3 TextGrids, and 9 author-preprocessed HF5 files
  for `sweetaspie`, `againstthewind`, and `wheretheressmoke` across
  `UTS01`-`UTS03`.
- Blocking piece: full three-story smoke features and CPU ridge encoding must
  complete before staging the 76.86 GB high-data subset.
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
- GPU-side extraction debug cluster `5513178` completed with normal return
  value `0` at `2026-07-08 19:55:28` from
  `~/chtc-runs/coherence-huth-extract-debug-20260708-1945` on
  `gpu4000.chtc.wisc.edu`, an NVIDIA L40.
- Pulled debug artifacts are under
  `results/sft_huth_lebel/chtc_huth_extract_debug_5513178/`. Both
  `exit_status.txt` and `extract_exit_status.txt` are `0`.
- Debug extraction evidence: `base`, `lowLR`, `scrambled`, and
  `taskvec_a0p25` each produced `features/<arm>/sweetaspie.npz` with hidden
  shape `64 x 1 x 4096`, layer index `24`, and float16 activations. The shared
  word table has 64 rows.
- Full smoke extraction cluster `5513245` ran from
  `~/chtc-runs/coherence-huth-extract-smoke-20260709-005926` on an L40S-class
  GPU. It loaded the model, extracted all three `base` story features for
  layers `16,24,32`, then failed with `OSError: [Errno 122] Disk quota
  exceeded` while creating
  `/staging/s/suresh27/features/huth_lebel_smoke_llama31/lowLR`.
- Staged feature evidence from `5513245`: `base/sweetaspie.npz`
  (`697 x 3 x 4096`), `base/againstthewind.npz` (`842 x 3 x 4096`), and
  `base/wheretheressmoke.npz` (`1859 x 3 x 4096`). Adapter-arm features were
  not written.
- Recovery path: rerun the three-story extraction using the bundle-output
  submit path, which writes features in job scratch and returns them inside
  `huth_extract_smoke_results.tgz` instead of creating new staging
  directories.
- Active recovery: cluster `5513306`, remote directory
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`.
- Duplicate recovery cluster `5513310` had identical settings and was removed
  while idle.
- Duplicate cluster `5513244` held before model work because its submit
  expected a missing output tarball; it was removed with `condor_rm`.
- The prepared CPU ridge follow-ups are
  `chtc/huth_lebel_smoke/huth_encoding_smoke.sub` for staged features and
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle.sub` for bundle-returned
  features.

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
- CHTC extraction debug outputs:
  `results/sft_huth_lebel/chtc_huth_extract_debug_5513178/`
- Failed staged-output full smoke extraction:
  `~/chtc-runs/coherence-huth-extract-smoke-20260709-005926`, cluster
  `5513245`
- Active bundle-output recovery extraction:
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`, cluster
  `5513306`
