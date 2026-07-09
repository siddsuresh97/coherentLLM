# CHTC Huth Extraction Retry 5513306

Status: passed.

Cluster `5513306` ran from
`~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204` on
`mkhodakgpu4000.chtc.wisc.edu` with an NVIDIA L40S. The job avoided new feature
writes under `/staging/s/suresh27` by copying the already-written `base`
features from `/staging/s/suresh27/features/huth_lebel_smoke_llama31` into job
scratch and returning all features inside `huth_extract_smoke_results.tgz`.

Pass evidence:

- Condor normal termination, return value `0`.
- `extracted/exit_status.txt == 0`.
- `extracted/extract_exit_status.txt == 0`.
- `extracted/npz_shapes.tsv` lists all 12 arm/story feature files:
  `base`, `lowLR`, `scrambled`, and `taskvec_a0p25` by
  `sweetaspie`, `againstthewind`, and `wheretheressmoke`.
- Each NPZ has `hidden` dtype `float16`, layers `16,24,32`, and hidden width
  `4096`.

The large returned bundle and unpacked NPZ files are present in this local
workspace directory for handoff, but they are intentionally not committed
because this repo has no LFS/filter configured for these 573 MB of binary
artifacts. The committed files retain command, environment, logs, seed
inventory, and NPZ shape metadata.
