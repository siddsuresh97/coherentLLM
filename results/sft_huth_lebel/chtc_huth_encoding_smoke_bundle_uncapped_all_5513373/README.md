# CHTC Huth Uncapped Encoding Smoke 5513373

Status: completed normally.

- Cluster: `5513373`
- Condor status: `JobStatus=4`, `ExitCode=0`
- Submit event: `2026-07-08 21:08:30 CDT`
- Execution: `2026-07-08 21:13:46` to `2026-07-08 21:30:09 CDT`
- Remote host: `oconnor2007.chtc.wisc.edu`
- Resources: `4` CPUs, `16GB` memory, `20GB` disk, `GPUs=0`
- Peak logged memory: `1722 MB`
- Remote run directory:
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`
- Submit file:
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle_uncapped_all.sub`
- Result bundle:
  `huth_encoding_smoke_bundle_uncapped_all_results.tgz`
- Feature bundle input: `huth_extract_smoke_results.tgz` from extraction
  cluster `5513306`

## Validation

- `exit_status.txt == 0`
- `encoding_exit_status.txt == 0`
- `encoding/summary.csv` has 36 data rows
  (`3 subjects x 4 arms x 3 layers`)
- `encoding/alpha_cv.csv` has 360 data rows
  (`36 summaries x 10 ridge alphas`)
- Metadata reports full voxel counts, not the prior 2000-voxel cap:
  `UTS01=81126`, `UTS02=94251`, `UTS03=95556`.
- The command omitted `--max_voxels`, as intended for `MAX_VOXELS=0`.

Shared settings for every row:

- Train stories: `sweetaspie,againstthewind`
- Held-out test story: `wheretheressmoke`
- Train TRs: `307`
- Test TRs: `281`
- Features: `16384`
- Layers: `16,24,32`
- Arms: `base,lowLR,scrambled,taskvec_a0p25`
- Metric: voxelwise Pearson `r` on the held-out story, summarized over finite
  voxels.

## Exact Summary

| Subject | Arm | Layer | Mean r | Median r | P95 r | Frac positive | Alpha |
|---|---|---:|---:|---:|---:|---:|---:|
| `UTS01` | `base` | 16 | 0.007640 | 0.007541 | 0.114916 | 0.5461 | 1000 |
| `UTS01` | `base` | 24 | 0.005942 | 0.005631 | 0.111321 | 0.5354 | 215.443 |
| `UTS01` | `base` | 32 | 0.003637 | 0.003351 | 0.108621 | 0.5211 | 359.381 |
| `UTS01` | `lowLR` | 16 | 0.005228 | 0.004984 | 0.109747 | 0.5312 | 1000 |
| `UTS01` | `lowLR` | 24 | 0.004545 | 0.004315 | 0.109054 | 0.5278 | 215.443 |
| `UTS01` | `lowLR` | 32 | -0.000024 | -0.000001 | 0.104215 | 0.5000 | 129.155 |
| `UTS01` | `scrambled` | 16 | 0.003554 | 0.003136 | 0.107619 | 0.5207 | 129.155 |
| `UTS01` | `scrambled` | 24 | 0.005323 | 0.005321 | 0.107072 | 0.5345 | 129.155 |
| `UTS01` | `scrambled` | 32 | 0.000069 | 0.000007 | 0.102142 | 0.5001 | 215.443 |
| `UTS01` | `taskvec_a0p25` | 16 | 0.005663 | 0.005485 | 0.113174 | 0.5338 | 1000 |
| `UTS01` | `taskvec_a0p25` | 24 | 0.004924 | 0.004581 | 0.110571 | 0.5288 | 215.443 |
| `UTS01` | `taskvec_a0p25` | 32 | 0.001469 | 0.001125 | 0.104825 | 0.5071 | 599.484 |
| `UTS02` | `base` | 16 | 0.010409 | 0.009237 | 0.127498 | 0.5536 | 1000 |
| `UTS02` | `base` | 24 | 0.008560 | 0.007929 | 0.123834 | 0.5459 | 1000 |
| `UTS02` | `base` | 32 | 0.011303 | 0.009648 | 0.135253 | 0.5548 | 1000 |
| `UTS02` | `lowLR` | 16 | 0.009891 | 0.009228 | 0.123651 | 0.5544 | 1000 |
| `UTS02` | `lowLR` | 24 | 0.006204 | 0.005845 | 0.119934 | 0.5340 | 1000 |
| `UTS02` | `lowLR` | 32 | 0.004589 | 0.004108 | 0.123157 | 0.5232 | 1000 |
| `UTS02` | `scrambled` | 16 | 0.001885 | 0.002019 | 0.105044 | 0.5125 | 10 |
| `UTS02` | `scrambled` | 24 | -0.000144 | 0.000079 | 0.102666 | 0.5005 | 1000 |
| `UTS02` | `scrambled` | 32 | -0.009166 | -0.008743 | 0.096552 | 0.4460 | 1000 |
| `UTS02` | `taskvec_a0p25` | 16 | 0.011037 | 0.010019 | 0.128078 | 0.5576 | 1000 |
| `UTS02` | `taskvec_a0p25` | 24 | 0.005836 | 0.005290 | 0.121373 | 0.5310 | 1000 |
| `UTS02` | `taskvec_a0p25` | 32 | 0.005713 | 0.004914 | 0.124744 | 0.5283 | 1000 |
| `UTS03` | `base` | 16 | 0.011305 | 0.010145 | 0.127264 | 0.5602 | 1000 |
| `UTS03` | `base` | 24 | 0.009458 | 0.008054 | 0.123309 | 0.5497 | 1000 |
| `UTS03` | `base` | 32 | 0.007635 | 0.006748 | 0.121869 | 0.5403 | 1000 |
| `UTS03` | `lowLR` | 16 | 0.009744 | 0.008934 | 0.122560 | 0.5542 | 1000 |
| `UTS03` | `lowLR` | 24 | 0.008265 | 0.006941 | 0.120195 | 0.5420 | 1000 |
| `UTS03` | `lowLR` | 32 | 0.005761 | 0.005182 | 0.117327 | 0.5308 | 1000 |
| `UTS03` | `scrambled` | 16 | 0.009525 | 0.008977 | 0.116684 | 0.5553 | 1000 |
| `UTS03` | `scrambled` | 24 | 0.005687 | 0.005640 | 0.107708 | 0.5355 | 1000 |
| `UTS03` | `scrambled` | 32 | 0.002701 | 0.002475 | 0.107899 | 0.5155 | 1000 |
| `UTS03` | `taskvec_a0p25` | 16 | 0.010710 | 0.009682 | 0.125555 | 0.5579 | 1000 |
| `UTS03` | `taskvec_a0p25` | 24 | 0.009034 | 0.007852 | 0.121499 | 0.5477 | 1000 |
| `UTS03` | `taskvec_a0p25` | 32 | 0.005410 | 0.004875 | 0.115508 | 0.5293 | 1000 |

Mean across subjects by arm/layer:

| Arm | Layer | Mean r | Median r | P95 r | Frac positive |
|---|---:|---:|---:|---:|---:|
| `base` | 16 | 0.009785 | 0.008974 | 0.123226 | 0.5533 |
| `base` | 24 | 0.007987 | 0.007205 | 0.119488 | 0.5436 |
| `base` | 32 | 0.007525 | 0.006583 | 0.121914 | 0.5387 |
| `lowLR` | 16 | 0.008287 | 0.007715 | 0.118653 | 0.5466 |
| `lowLR` | 24 | 0.006338 | 0.005701 | 0.116394 | 0.5346 |
| `lowLR` | 32 | 0.003442 | 0.003097 | 0.114900 | 0.5180 |
| `scrambled` | 16 | 0.004988 | 0.004711 | 0.109782 | 0.5295 |
| `scrambled` | 24 | 0.003622 | 0.003680 | 0.105816 | 0.5235 |
| `scrambled` | 32 | -0.002132 | -0.002087 | 0.102198 | 0.4872 |
| `taskvec_a0p25` | 16 | 0.009137 | 0.008395 | 0.122269 | 0.5497 |
| `taskvec_a0p25` | 24 | 0.006598 | 0.005907 | 0.117815 | 0.5359 |
| `taskvec_a0p25` | 32 | 0.004197 | 0.003638 | 0.115026 | 0.5216 |

Best rows by subject:

- `UTS01`: `base` layer 16, mean `r=0.007640`.
- `UTS02`: `base` layer 32, mean `r=0.011303`.
- `UTS03`: `base` layer 16, mean `r=0.011305`.

## Readout

The uncapped smoke confirms the CPU encoding path on all full subject voxel
sets. Mean held-out correlations are positive but small, as expected from only
two short training stories and one fixed held-out story.

The smoke does not show a convincing coherence-aligned improvement over base.
`base` is the best subject-level row for all three subjects. `taskvec_a0p25`
beats same-layer base in only one cell, `UTS02` layer 16 (`+0.000628` mean
`r`), while it trails base at the other eight subject/layer cells. `lowLR`
trails base in all nine cells. `scrambled` is usually lower and is strongly
negative for `UTS02` layer 32.

Limitations:

- This is a smoke result, not the high-data result.
- Training uses only `sweetaspie` and `againstthewind`.
- Evaluation uses only held-out `wheretheressmoke`.
- There is no blockwise permutation, FDR, ROI/localizer analysis, or
  subject-specific Fedorenko language localizer.
- Layer, alpha, ROI, and task-vector-scale choices should not be tuned on this
  held-out story before a high-data run.
