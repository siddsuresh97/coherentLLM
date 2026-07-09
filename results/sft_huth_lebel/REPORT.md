# Huth/LeBel Language-fMRI Audit

Generated: 2026-07-08T22:59:00.688548+00:00

Manual update: 2026-07-09 after completing uncapped CPU encoding cluster
`5513373`.

## Current Read

- Status: the Huth/LeBel narrative-encoding experiment has passed the staged
  smoke-data audit, GPU feature extraction, bundle-output recovery extraction,
  capped CPU encoding path checks, and the first uncapped all-subject CPU
  encoding smoke on CHTC as cluster `5513373`.
- Best staged smoke root: `/staging/s/suresh27/datasets/ds003020-smoke`.
- Staged data pieces: 3 WAVs, 3 TextGrids, and 9 author-preprocessed HF5 files
  for `sweetaspie`, `againstthewind`, and `wheretheressmoke` across
  `UTS01`-`UTS03`.
- Current blocking piece: decide whether to scale beyond the smoke subset. The
  uncapped smoke encoding `5513373` returned full-voxel
  `summary.csv`/`alpha_cv.csv`; the remaining scientific limitation is data
  volume, not pipeline correctness.
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
- Recovery outcome: `5513306` started on
  `mkhodakgpu4000.chtc.wisc.edu` at `2026-07-08 20:36:09` with an NVIDIA L40S,
  transferred a 254 MB feature/status bundle, and terminated normally with
  return value `0`. Pulled artifacts are under
  `results/sft_huth_lebel/chtc_huth_extract_smoke_retry_5513306/`.
- Bundle extraction evidence: `exit_status.txt == 0`,
  `extract_exit_status.txt == 0`, and `npz_shapes.tsv` lists all 12 expected
  arm/story NPZ files with `hidden` dtype `float16`, layers `16,24,32`, and
  width `4096`.
- CPU encoding smoke cluster `5513337` was submitted from the same run
  directory using `huth_encoding_smoke_bundle.sub`.
- Encoding outcome: `5513337` terminated normally with return value `0`, and
  the pulled bundle under
  `results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_5513337/`
  contains `encoding_exit_status.txt == 0`, `summary.csv`, `alpha_cv.csv`, and
  `run_metadata.json`.
- Encoding smoke read: the path is validated for feature-bundle ingestion,
  TextGrid/word-state alignment, FIR-delayed ridge fitting, alpha CV, and
  held-out Pearson scoring. The metrics are near zero because this is only
  `UTS01`, two short training stories, and `--max_voxels 2000`; do not treat it
  as a scientific arm comparison.
- CPU-only scale check `5513350` passed for the same capped bundle encoding on
  `UTS02,UTS03` from the validated `5513306` feature bundle. `UTS02` remains
  near zero; `UTS03` shows small positive smoke predictivity (`base` layer 16
  mean `r=0.015795`, `taskvec_a0p25` layer 16 mean `r=0.013765`).
- Uncapped all-subject CPU encoding cluster `5513373` was submitted at
  `2026-07-08 21:08:30 CDT` from
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204` using
  `huth_encoding_smoke_bundle_uncapped_all.sub`. It consumed the validated
  `5513306` feature bundle, set `MAX_VOXELS=0` so the wrapper omitted
  `--max_voxels`, evaluated `UTS01,UTS02,UTS03`, and requested CPU only
  (`4` CPUs, `16GB` memory, `20GB` disk). It was initially idle with no hold
  and 5 willing matches at `2026-07-08 21:11:49 CDT`, then executed on
  `oconnor2007.chtc.wisc.edu` from `2026-07-08 21:13:46` to
  `2026-07-08 21:30:09 CDT` with `GPUs=0`. It completed normally with
  `ExitCode=0`, peak logged memory `1722 MB`, and about `986` seconds of slot
  busy time.
- Uncapped encoding outcome: pulled artifacts are under
  `results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_uncapped_all_5513373/`.
  `exit_status.txt == 0`, `encoding_exit_status.txt == 0`, and
  `summary.csv` has all 36 expected rows (`3` subjects x `4` arms x `3`
  layers) with full voxel counts: `UTS01=81126`, `UTS02=94251`,
  `UTS03=95556`. `alpha_cv.csv` has all 360 expected alpha-CV rows.
- Uncapped smoke read: mean held-out Pearson `r` values are small but positive
  for most arms. At layer 16, averaged across `UTS01`-`UTS03`, base is highest
  (`0.009785`), followed by `taskvec_a0p25` (`0.009137`), `lowLR`
  (`0.008287`), and `scrambled` (`0.004988`). `base` is the best subject-level
  row for all three subjects (`UTS01` base L16 mean `r=0.007640`, `UTS02`
  base L32 mean `r=0.011303`, `UTS03` base L16 mean `r=0.011305`).
  `taskvec_a0p25` beats same-layer base only for `UTS02` L16 (`0.011037` vs
  `0.010409`); `lowLR` trails base in all nine subject/layer cells;
  `scrambled` is usually lower and is negative for `UTS02` L24/L32. Treat this
  as evidence that the full-voxel path works; do not treat it as a final
  Huth/Fedorenko arm comparison.
- Duplicate recovery cluster `5513310` had identical settings and was removed
  while idle.
- Duplicate cluster `5513244` held before model work because its submit
  expected a missing output tarball; it was removed with `condor_rm`.
- The prepared CPU ridge follow-ups are
  `chtc/huth_lebel_smoke/huth_encoding_smoke.sub` for staged features,
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle.sub` for capped
  bundle-returned features, and
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle_uncapped_all.sub` for the
  completed uncapped all-subject smoke.

## Uncapped 5513373 Exact Mean-r Results

All rows use `sweetaspie,againstthewind` for training and held-out
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

Subject-mean rows by arm/layer:

| Arm | L16 | L24 | L32 |
|---|---:|---:|---:|
| `base` | 0.009785 | 0.007987 | 0.007525 |
| `lowLR` | 0.008287 | 0.006338 | 0.003442 |
| `scrambled` | 0.004988 | 0.003622 | -0.002132 |
| `taskvec_a0p25` | 0.009137 | 0.006598 | 0.004197 |

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
- Passed bundle-output recovery extraction:
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`, cluster
  `5513306`
- CHTC extraction retry outputs:
  `results/sft_huth_lebel/chtc_huth_extract_smoke_retry_5513306/`
- Passed CPU encoding smoke:
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`, cluster
  `5513337`
- CHTC encoding smoke outputs:
  `results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_5513337/`
- Passed UTS02/UTS03 encoding scale check:
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`, cluster
  `5513350`
- CHTC UTS02/UTS03 encoding outputs:
  `results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_uts02_uts03_5513350/`
- Passed uncapped all-subject CPU encoding:
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`, cluster
  `5513373`
- Local destination/results for `5513373`:
  `results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_uncapped_all_5513373/`
