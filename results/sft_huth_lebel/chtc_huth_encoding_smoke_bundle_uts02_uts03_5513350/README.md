# CHTC Huth Encoding Scale Check 5513350

Status: passed as a capped CPU path-validation scale check.

Cluster `5513350` ran from
`~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204` using
`huth_encoding_smoke_bundle_uts02_uts03.sub`. It reused the feature bundle
produced by extraction cluster `5513306`, trained on
`sweetaspie,againstthewind`, and evaluated held-out `wheretheressmoke` for
subjects `UTS02` and `UTS03`.

Pass evidence:

- Condor normal termination, return value `0`.
- `extracted/exit_status.txt == 0`.
- `extracted/encoding_exit_status.txt == 0`.
- `extracted/summary.csv` has 24 subject/arm/layer rows.
- `extracted/alpha_cv.csv` records leave-one-training-story-out ridge CV.

Best-by-mean smoke rows:

| Subject | Best Arm | Best Layer | Mean r | Median r | Fraction positive |
|---|---|---:|---:|---:|---:|
| `UTS02` | `scrambled` | 16 | 0.001870 | 0.001318 | 0.5100 |
| `UTS03` | `base` | 16 | 0.015795 | 0.010420 | 0.5700 |

Best arm/layer by subject:

| Subject | Arm | Best Layer | Mean r | Median r |
|---|---|---:|---:|---:|
| `UTS02` | `base` | 32 | 0.000295 | -0.001307 |
| `UTS02` | `lowLR` | 24 | 0.001216 | 0.001533 |
| `UTS02` | `scrambled` | 16 | 0.001870 | 0.001318 |
| `UTS02` | `taskvec_a0p25` | 32 | -0.002408 | -0.002646 |
| `UTS03` | `base` | 16 | 0.015795 | 0.010420 |
| `UTS03` | `lowLR` | 16 | 0.014256 | 0.010064 |
| `UTS03` | `scrambled` | 16 | 0.004420 | 0.003518 |
| `UTS03` | `taskvec_a0p25` | 16 | 0.013765 | 0.009104 |

Read: this validates that the bundle encoding path works across all three smoke
subjects (`UTS01` in `5513337`, `UTS02`/`UTS03` here). It is still a smoke
test: two short training stories, one held-out story, and a 2000-voxel cap.
Do not treat these rows as a final scientific arm comparison.
