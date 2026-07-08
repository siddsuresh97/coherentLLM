# Huth/LeBel ds003020 Smoke Staging on CHTC

CPU-only staging job for the first Huth/LeBel language-fMRI smoke test.

Inputs:

- `run_stage_smoke.sh`
- `huth_lebel_audit.py`
- `staging_manifest_smoke.csv`

The job stages only the manifest-listed `ds003020` files under:

```text
/staging/s/suresh27/datasets/ds003020-smoke
```

It uses a git-annex container and the public OpenNeuro GitHub mirror. The clone
uses sparse checkout so `/staging/s/suresh27` does not exceed its file-count
quota by materializing the full dataset tree.
The smoke subset is 15 files, 7.88 GB total:

- stories: `sweetaspie`, `againstthewind`, `wheretheressmoke`
- subjects: `UTS01`, `UTS02`, `UTS03`
- assets: WAV, TextGrid, and author-preprocessed HF5 files

Submit from the remote CHTC run directory:

```bash
condor_submit huth_lebel_stage_smoke.sub
```

After completion, pull `huth_lebel_stage_smoke_results.tgz` and the `logs/`
directory. The returned archive contains a stage summary and the Huth audit
against the staged dataset root.

Current retry context, 2026-07-08:

- `5513024` failed because a full checkout exceeded the `/staging/s/suresh27`
  1000-file quota.
- `5513028` failed after sparse checkout because `git annex init` needed a git
  author identity and the failure tarball path was relative after `cd`.
- `5513036` is the fixed sparse-checkout retry with local git identity and
  scratch-anchored output paths.
