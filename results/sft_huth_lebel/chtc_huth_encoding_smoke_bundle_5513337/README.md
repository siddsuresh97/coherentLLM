# CHTC Huth Encoding Smoke 5513337

Status: passed as a path-validation smoke.

Cluster `5513337` ran from
`~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204` using
`huth_encoding_smoke_bundle.sub`. It unpacked the feature bundle produced by
extraction cluster `5513306`, used the staged smoke dataset at
`/staging/s/suresh27/datasets/ds003020-smoke`, trained on
`sweetaspie,againstthewind`, and evaluated held-out `wheretheressmoke` for
subject `UTS01`.

Pass evidence:

- Condor normal termination, return value `0`.
- `extracted/exit_status.txt == 0`.
- `extracted/encoding_exit_status.txt == 0`.
- `extracted/summary.csv` has all 12 arm/layer rows.
- `extracted/alpha_cv.csv` records leave-one-training-story-out ridge CV.

Read: this validates feature-bundle ingestion, TextGrid/word-state alignment to
TRs, FIR-delayed ridge fitting, alpha selection, and held-out Pearson scoring.
It is not a scientific result yet: the smoke uses one subject, only two short
training stories, and a 2000-voxel cap.

Best-by-mean smoke rows:

| Arm | Best Layer | Mean r | Median r |
|---|---:|---:|---:|
| `base` | 16 | -0.000117 | -0.001928 |
| `lowLR` | 16 | -0.001191 | -0.002872 |
| `scrambled` | 16 | 0.000169 | -0.000881 |
| `taskvec_a0p25` | 16 | 0.000589 | 0.000862 |
