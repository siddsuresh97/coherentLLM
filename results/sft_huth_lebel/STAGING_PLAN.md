# Huth/LeBel ds003020 Staging Plan

Generated: 2026-07-08T23:13:09.747553+00:00

## Dataset

- Root inspected: `/tmp/ds003020-git`
- Name: An fMRI dataset during a passive natural language listening task
- Dataset DOI: `doi:10.18112/openneuro.ds003020.v3.1.1`
- Subjects planned first: `UTS01, UTS02, UTS03`
- Held-out test story: `wheretheressmoke`
- Shared preprocessed stories across subjects: 84

## Recommended Staging

- Use explicit CHTC staging path `/staging/s/suresh27`; `$STAGING` was unset on `ap2002`.
- Stage the smoke subset first under `/staging/s/suresh27/datasets/ds003020-smoke`.
- If the smoke encoding path works, stage the high-data `UTS01`-`UTS03` subset under `/staging/s/suresh27/datasets/ds003020-highdata`.
- Do staging with a CPU/download job or local download plus rsync; do not consume a GPU for data transfer.
- The AP currently lacks `datalad`, `git-annex`, `openneuro`, `aws`, and `aria2c`, so a containerized downloader is the safer CHTC route.

## Smoke Subset

- Stories: `sweetaspie, againstthewind, wheretheressmoke`
- Files: 15 total; missing metadata entries: 0
- Planned size: 7.88 GB (7.34 GiB)

| Kind | GB | GiB |
|---|---:|---:|
| hf5 | 7.65 | 7.12 |
| textgrid | 0.00 | 0.00 |
| wav | 0.23 | 0.21 |

## High-Data UTS01-UTS03 Subset

- Stories: all 84 shared preprocessed stories across `UTS01, UTS02, UTS03`.
- Files: 420 total; missing metadata entries: 0
- Planned size: 76.86 GB (71.58 GiB)
- This fits the observed 100 GB CHTC staging quota if only the needed WAV, TextGrid, and author-preprocessed HF5 files are staged.

| Kind | GB | GiB |
|---|---:|---:|
| hf5 | 67.19 | 62.57 |
| textgrid | 0.08 | 0.08 |
| wav | 9.59 | 8.93 |

## Artifacts

- Smoke manifest: `results/sft_huth_lebel/staging_manifest_smoke.csv`
- High-data manifest: `results/sft_huth_lebel/staging_manifest_highdata.csv`
- Summary JSON: `results/sft_huth_lebel/staging_summary.json`

## Next

1. Submit a CPU-only CHTC downloader that uses a container with `git-annex`/DataLad or the OpenNeuro downloader.
2. Pull only manifest-listed files for the smoke subset first.
3. Run the local/CHTC Huth audit against the staged smoke root.
4. Start GPU feature extraction only after the smoke root passes audit.
