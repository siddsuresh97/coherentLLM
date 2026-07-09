# CHTC Huth Uncapped Encoding Smoke 5513373

Status at checkpoint: submitted and queued, no hold.

- Cluster: `5513373`
- Submit event: `2026-07-08 21:08:30 CDT`
- Remote run directory:
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`
- Submit file:
  `chtc/huth_lebel_smoke/huth_encoding_smoke_bundle_uncapped_all.sub`
- Feature bundle input: `huth_extract_smoke_results.tgz` from extraction
  cluster `5513306`
- Subjects: `UTS01,UTS02,UTS03`
- Train stories: `sweetaspie,againstthewind`
- Test story: `wheretheressmoke`
- Arms: `base,lowLR,scrambled,taskvec_a0p25`
- Layers: `16,24,32`
- Voxel cap: none (`MAX_VOXELS=0`; the wrapper omits `--max_voxels`)
- Resources after in-place tuning: `request_cpus=4`, `request_memory=16GB`,
  `request_disk=20GB`

Monitor:

```bash
chtc-ssh 'condor_q 5513373 -nobatch'
```

When the job completes, pull:

```bash
chtc-pull 'chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204/huth_encoding_smoke_bundle_uncapped_all_results.tgz' \
  results/sft_huth_lebel/chtc_huth_encoding_smoke_bundle_uncapped_all_5513373/
```

Then extract and verify:

- `exit_status.txt == 0`
- `encoding_exit_status.txt == 0`
- `encoding/summary.csv` has 36 rows
  (`3 subjects x 4 arms x 3 layers`)
- `encoding/alpha_cv.csv` exists
- `encoding/run_metadata.json` reports full voxel counts rather than the prior
  2000-voxel cap

Interpretation limit: this is still a smoke read, not the high-data result. It
uses two short training stories and one fixed held-out story, so successful
metrics should decide whether to stage/run high-data, not support a final
Huth/Fedorenko claim.
