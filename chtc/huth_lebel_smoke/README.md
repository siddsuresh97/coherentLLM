# Huth/LeBel CHTC Smoke Jobs

This directory runs the first GPU-side ds003020 smoke check on CHTC after the
smoke subset has been staged under `/staging/s/suresh27/datasets/ds003020-smoke`.

The initial debug job is intentionally small:

- story: `sweetaspie`
- words: first 64 TextGrid words
- layer: 24
- arms: `base`, `lowLR`, `scrambled`, `taskvec_a0p25`
- model: `/staging/s/suresh27/models/llama31-8b-instruct`
- adapters: `/staging/s/suresh27/adapters/{lowLR,scrambled,taskvec_a0p25}`

It returns a tarball containing NPZ feature files and logs. Run this before any
full three-story extraction job.

After that passes, use `huth_extract_smoke.sub` for the held-out encoding
prerequisite. It extracts all three smoke stories for
`base,lowLR,scrambled,taskvec_a0p25` and layers `16,24,32`, then returns
`huth_extract_smoke_results.tgz`.

Submit from a CHTC run directory that contains this folder's scripts plus
`src`:

```bash
condor_submit huth_extract_smoke.sub
```

When that extraction job exits with `extract_exit_status.txt == 0` and
`npz_shapes.tsv` lists all 12 arm/story NPZs, submit the CPU ridge smoke:

```bash
condor_submit huth_encoding_smoke.sub
```

If `/staging/s/suresh27` cannot accept new feature directories, use the
bundle-output path instead. The checked-in `huth_extract_smoke.sub` leaves
`FEATURE_DIR` unset, so `run_extract_smoke.sh` writes features under
`huth_extract_smoke/features` and returns them inside
`huth_extract_smoke_results.tgz`. Then submit:

```bash
condor_submit huth_encoding_smoke_bundle.sub
```

The encoding smoke uses `sweetaspie,againstthewind` for ridge/CV training,
holds out `wheretheressmoke`, and caps responses at `--max_voxels 2000` for the
first pass. Remove the cap only after this CPU smoke returns valid
`summary.csv` and `alpha_cv.csv`.

After capped `UTS01`/`UTS02`/`UTS03` path checks pass, use the uncapped
all-subject bundle smoke:

```bash
condor_submit huth_encoding_smoke_bundle_uncapped_all.sub
```

Current submitted instance: cluster `5513373` from
`~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`, submitted at
`2026-07-08 21:08:30 CDT`. It consumes the `5513306`
`huth_extract_smoke_results.tgz` bundle, sets `MAX_VOXELS=0` so
`run_encoding_smoke.sh` omits the voxel cap, evaluates
`UTS01,UTS02,UTS03`, and requests CPU only (`4` CPUs, `16GB` memory, `20GB`
disk). It began executing on `oconnor2007.chtc.wisc.edu` at
`2026-07-08 21:13:46 CDT` with `GPUs=0`.
