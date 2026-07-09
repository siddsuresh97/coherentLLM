# Huth Pack Smoke Retry 5513422

Status: passed, `ExitCode=0`.

Run directory on CHTC:
`~/chtc-runs/coherence-huth-pack-smoke-retry-20260709-030344`

Purpose: validate the fixed story-pack workflow on the already staged ds003020
smoke dataset without writing new staging files.

Job shape:

- CPU/container only.
- `2` CPUs.
- `4GB` memory.
- `10GB` disk.
- `0` GPUs.
- Staging visibility required.
- Host: `slot1_59@e4049.chtc.wisc.edu`.
- Condor history: `JobStatus=4`, `ExitCode=0`, `RemoteWallClockTime=91.0`.

Validation result:

- Story: `againstthewind`.
- Source files represented: `5`.
- Expected source bytes: `432463521`.
- Archive: `againstthewind.tar.gz`.
- Archive bytes: `399848250`.
- Archive SHA256:
  `728b01346aed14f512d66bdb8699dbbde37455e1287e50a7392ddc9ce6f0cc7a`.
- Archive members: `7`.
- `verified_members=true`.
- `verified_extracted_sizes=true`.
- `exit_status.txt == 0`.

Notes:

- This retry proves the `--dereference` fix handles ds003020 git-annex
  symlinks correctly.
- This retry used the pre-trim wrapper and returned a large
  `huth_lebel_pack_story_smoke_results.tgz` bundle. That tarball is intentionally
  left untracked.
- Future wrapper output is trimmed by default: it keeps metadata, member lists,
  and size summaries rather than returning the pack plus extracted files.

Tracked proof files:

- `logs/5513422_0.log`
- `logs/5513422_0.out`
- `logs/5513422_0.err`
- `extracted_meta/results/huth_lebel_pack_story_smoke/pack_result.json`
- `extracted_meta/results/huth_lebel_pack_story_smoke/verify_result.json`
- `extracted_meta/results/huth_lebel_pack_story_smoke/run_env.txt`
- `extracted_meta/results/huth_lebel_pack_story_smoke/file_inventory.tsv`
- `extracted_meta/results/huth_lebel_pack_story_smoke/output_du.txt`
- `extracted_meta/results/huth_lebel_pack_story_smoke/exit_status.txt`
