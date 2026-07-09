# Huth/Fedorenko High-Data Staging Unblock Report

Updated: 2026-07-08 22:11 CDT

Scope: Huth/Fedorenko language-fMRI lane only. Concept-steering files were not
edited.

## Quota Finding

Read-only CHTC audit:

```text
get_quotas
/home/suresh27       22.1824 / 80 GB, 117195 files, no file limit
/staging/s/suresh27  24.3209 / 100 GB, 1120 / 1000 files
```

The blocker is file count, not disk. The planned high-data raw manifest would
add 420 loose files and 76.858043164 GB, so raw staging remains unsafe.

Per-directory staging audit:

| Path | Files | Dirs | Symlinks | Entries | GB | Recommendation |
|---|---:|---:|---:|---:|---:|---|
| `/staging/s/suresh27/hf_datasets_cache` | 258 | 115 | 0 | 373 | 0.004 | preferred cleanup |
| `/staging/s/suresh27/hf_home` | 244 | 54 | 114 | 412 | 0.004 | defer until active jobs finish |
| `/staging/s/suresh27/datasets/ds003020-smoke` | 127 | 134 | 15 | 276 | 7.887 | cleanup after pack smoke if needed |
| `/staging/s/suresh27/features/huth_lebel_smoke_llama31` | 9 | 3 | 0 | 12 | 0.084 | cleanup after confirm |
| `/staging/s/suresh27/models` | 10 | 1 | 0 | 11 | 16.070 | keep |
| `/staging/s/suresh27/adapters` | 23 | 6 | 0 | 29 | 2.065 | keep |

Exact cleanup candidates are in `staging_cleanup_candidates.csv`. No cleanup was
performed in this checkpoint.

## Packed Artifact Plan

Implemented one story archive per ds003020 story:

- Source manifest: `staging_manifest_highdata.csv`.
- Source files: 420 loose files across 84 stories.
- Pack manifest: `highdata_story_pack_manifest.csv`.
- Story queue: `highdata_pack_stories.txt`.
- Pack root: `/staging/s/suresh27/datasets/ds003020-highdata-packs`.
- Planned archives: 84 `.tar.zst` files.
- File-count reduction: 336 entries before extraction outputs.
- Source bytes represented: 76.858043164 GB.

Each archive contains:

- one WAV;
- one TextGrid;
- three subject HF5 files for `UTS01`, `UTS02`, `UTS03`;
- `HUTH_LEBEL_PACK_MANIFEST.json` with source rows and byte expectations.

Examples:

| Story | Source files | Source GB | Planned pack |
|---|---:|---:|---|
| `againstthewind` | 5 | 0.432464 | `againstthewind.tar.zst` |
| `adollshouse` | 5 | 0.612065 | `adollshouse.tar.zst` |
| `wheretheressmoke` | 5 | 7.045239 | `wheretheressmoke.tar.zst` |

## CHTC Workflow

New helper:

```bash
python src/sft/huth_lebel_pack_stories.py plan \
  --manifest results/sft_huth_lebel/staging_manifest_highdata.csv \
  --out-dir results/sft_huth_lebel \
  --label highdata
```

Small CHTC smoke attempts:

- Failed run id: `coherence-huth-pack-smoke-20260709-025632`.
- Failed cluster: `5513418`, `ExitCode=1`.
- Failure signature: verifier reported `actual_bytes=-1` for all five expected
  `againstthewind` files after extraction.
- Diagnosis: staged ds003020 paths are git-annex symlinks; the first packer
  archived symlinks instead of dereferenced file contents.
- Fix: `src/sft/huth_lebel_pack_stories.py` now calls tar with
  `--dereference`; the self-test includes a symlink case.
- Retry run id: `coherence-huth-pack-smoke-retry-20260709-030344`.
- Retry cluster: `5513422`, passed with `ExitCode=0`.
- Retry host: `slot1_59@e4049.chtc.wisc.edu`.
- Retry Condor history: `JobStatus=4`, `RemoteWallClockTime=91.0`.
- Retry validation: `againstthewind.tar.gz` contains 7 members, represents 5
  source files, has `archive_bytes=399848250` for
  `expected_source_bytes=432463521`, and passes both
  `verified_members=true` and `verified_extracted_sizes=true`.
- Job shape for both attempts: CPU/container only, `2` CPUs, `4GB` memory,
  `10GB` disk, `0` GPUs.
- Staging writes: none.
- Reads existing `/staging/s/suresh27/datasets/ds003020-smoke`.
- Packs and verifies `againstthewind`.
- Current queue after retry: empty.

High-data CHTC array:

- Submit file: `chtc/huth_lebel_highdata_packs/huth_lebel_pack_highdata.sub`.
- Default queue file: `highdata_pack_stories_smoke.txt` with only
  `againstthewind`.
- Scale queue file: `highdata_pack_stories.txt` with all 84 stories.
- Do not submit until `/staging/s/suresh27` is below the file-count limit.

## Cleanup Recommendation

Preferred unblock:

```text
Remove /staging/s/suresh27/hf_datasets_cache after confirming no MMLU jobs are active.
```

Why:

- frees 373 file entries;
- uses only 0.004 GB;
- is rebuildable;
- MMLU is complete;
- would move staging from about 1120 entries to about 747 entries, enough for
  one-story pack smoke and then the 84-pack high-data array.

Do not remove `models` or `adapters`. Defer `hf_home` cleanup while unrelated
HF jobs may still be active. Remove `ds003020-smoke` only after the pack smoke
is pulled or if file quota remains blocked.

## Validation

Passed locally:

```bash
python -m py_compile src/sft/huth_lebel_pack_stories.py src/sft/huth_lebel_smoke_encoding.py src/sft/huth_lebel_extract_word_states.py src/sft/plan_huth_lebel_staging.py src/sft/huth_lebel_audit.py
bash -n chtc/huth_lebel_highdata_packs/run_pack_story_smoke.sh
bash -n chtc/huth_lebel_highdata_packs/run_pack_highdata_story.sh
python src/sft/huth_lebel_pack_stories.py self-test
python src/sft/huth_lebel_pack_stories.py plan --manifest results/sft_huth_lebel/staging_manifest_highdata.csv --out-dir results/sft_huth_lebel --label highdata
```

CHTC checks:

```text
condor_q -better-analyze 5513422
34 slots match and are willing to run the job.

condor_history 5513422
ExitCode=0, RemoteWallClockTime=91.0

condor_q -batch suresh27
0 jobs
```

## Remaining Blockers

1. Staging file count is still above quota until a cleanup is performed.
2. The high-data pack array should start with one story after cleanup, then
   scale to all 84 stories only after the one-story staged pack validates.
3. Fedorenko-style claims still require subject-specific language localizer
   masks; this staging work only unblocks the Huth-style natural-listening run.
