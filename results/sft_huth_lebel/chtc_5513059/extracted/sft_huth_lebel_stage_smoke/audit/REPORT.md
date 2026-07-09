# Huth/LeBel Language-fMRI Audit

Generated: 2026-07-09T00:01:44.487539+00:00

## Current Read

- Status: local/staged data are sufficient for a first Huth-style smoke test.
- Best candidate root: `/staging/s/suresh27/datasets/ds003020-smoke`.
- Next: extract narrative hidden states without chat templates, align word
  features to TRs with FIR delays, and fit a small held-out ridge model.
- Reusable local study hook found: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/tribev2/studies/lebel2023bold.py`
  (`exists=False`).
- CHTC submission is currently gated on a reusable SSH ControlMaster/2FA
  session, not on experiment code.

## Checked Roots

| Path | Exists | Plausible | TextGrids | WAVs | HF5 | BOLD | Test Story |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/staging/s/suresh27/datasets/ds003020-smoke` | 1 | 1 | 3 | 3 | 9 | 0 | 1 |
| `/staging/s/suresh27/datasets/ds003020-smoke/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/staging/s/suresh27/datasets/ds003020-smoke/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/staging/s/suresh27/datasets/ds003020-smoke/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/staging/s/suresh27/datasets/ds003020-smoke/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/staging/s/suresh27/datasets/ds003020-smoke/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## Experiment Design To Run Once Data Is Present

- Subjects: start with `UTS01`, `UTS02`, and `UTS03`; they have the high-data
  extended story set in LeBel/Huth.
- Smoke test: use `wheretheressmoke` as held-out test story and one or two
  short training stories for a tiny local/CHTC validation.
- Feature extraction: feed exact narrative text streams, not chat templates;
  extract final-token word states by layer for `base`, `lowrank`,
  `taskvec_a0p25`, and `scrambled` first.
- Alignment: map word features to TRs using TextGrid word times, TR=2s, and
  concatenate FIR delays such as 2/4/6/8s.
- Encoding model: voxelwise ridge with layer/ridge selected on train/validation
  only; report paired held-out Pearson `r` deltas vs base by subject/ROI/fold.
- Fedorenko/EvLab constraint: individual language localizer masks are preferred;
  atlas or broad regions must be labeled exploratory.

## Artifacts

- JSON audit: `/var/lib/condor/execute/slot1/dir_1004558/scratch/results/sft_huth_lebel_stage_smoke/audit/audit.json`
- Candidate CSV: `/var/lib/condor/execute/slot1/dir_1004558/scratch/results/sft_huth_lebel_stage_smoke/audit/candidate_roots.csv`
