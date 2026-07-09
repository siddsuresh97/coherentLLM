# Huth Pack Smoke 5513418

Status: failed, `ExitCode=1`.

Run directory on CHTC:
`~/chtc-runs/coherence-huth-pack-smoke-20260709-025632`

Purpose: validate the story-pack workflow on the already staged ds003020 smoke
dataset without writing new staging files.

Job shape:

- CPU/container only.
- `2` CPUs.
- `4GB` memory.
- `10GB` disk.
- `0` GPUs.
- Staging visibility required.

Failure:

The job packed `againstthewind`, but the archive contained git-annex symlinks
rather than dereferenced file contents. After extraction, those symlinks pointed
outside the extracted tree, so verifier size checks saw `actual_bytes=-1` for
the five expected files:

- `derivatives/preprocessed_data/UTS01/againstthewind.hf5`
- `derivatives/preprocessed_data/UTS02/againstthewind.hf5`
- `derivatives/preprocessed_data/UTS03/againstthewind.hf5`
- `derivatives/TextGrids/againstthewind.TextGrid`
- `stimuli/againstthewind.wav`

Fix applied after this run:

- `src/sft/huth_lebel_pack_stories.py` now calls tar with `--dereference`.
- The local self-test now includes a git-annex-style symlink case.
- Retry cluster: `5513422`.

Pulled files:

- `logs/5513418_0.log`
- `logs/5513418_0.out`
- `logs/5513418_0.err`
- `huth_lebel_pack_story_smoke_results.tgz`
- the submitted wrapper, submit file, helper, and manifest.
