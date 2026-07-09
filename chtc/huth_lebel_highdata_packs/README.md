# Huth/LeBel High-Data Story Packs

This lane converts ds003020 high-data staging from many loose files to one
archive per story. It is scoped to the Huth/Fedorenko experiment and should not
be mixed with concept-steering jobs.

## Why

The current CHTC staging quota is file-count blocked:

- `/staging/s/suresh27`: `1120/1000` files used.
- Disk is only `24.3209/100` GB used.
- Raw high-data staging would add 420 loose files.
- Story packs reduce those 420 files to 84 archives before feature extraction.

## Files

- `src/sft/huth_lebel_pack_stories.py`: local and CHTC pack planner/builder.
- `huth_lebel_pack_story_smoke.sub`: validates pack/verify on the already
  staged three-story smoke dataset and returns the archive as a job output.
- `huth_lebel_pack_highdata.sub`: CHTC-ready per-story packer array. Do not
  submit it until staging file count is below quota, because it writes one pack
  per story to `/staging/s/suresh27/datasets/ds003020-highdata-packs`.
- `highdata_pack_stories_smoke.txt`: one-story high-data smoke queue.
- `highdata_pack_stories.txt`: generated full 84-story queue.

## Local Checks

```bash
python -m py_compile src/sft/huth_lebel_pack_stories.py
python src/sft/huth_lebel_pack_stories.py self-test
python src/sft/huth_lebel_pack_stories.py plan \
  --manifest results/sft_huth_lebel/staging_manifest_highdata.csv \
  --out-dir results/sft_huth_lebel \
  --label highdata
bash -n chtc/huth_lebel_highdata_packs/run_pack_story_smoke.sh
bash -n chtc/huth_lebel_highdata_packs/run_pack_highdata_story.sh
```

## CHTC Smoke

The smallest safe smoke does not add a staging file. It reads
`/staging/s/suresh27/datasets/ds003020-smoke`, packs `againstthewind` in job
scratch, verifies extraction, and transfers the pack back in the job result
tarball.

```bash
mkdir -p ~/chtc-runs/<run-id>/logs
cd ~/chtc-runs/<run-id>
condor_submit huth_lebel_pack_story_smoke.sub
```

## High-Data Pack Submission

Before submitting the high-data array, run:

```bash
get_quotas
```

Only submit once `/staging/s/suresh27` has enough free file slots for at least
the smoke pack. The preferred cleanup candidate is the rebuildable MMLU HF
dataset cache, not model/adapters:

```text
/staging/s/suresh27/hf_datasets_cache
```

After quota is below the file limit, use:

```bash
condor_submit huth_lebel_pack_highdata.sub
```
