# Huth/LeBel Language-fMRI Audit

Generated: 2026-07-08T23:03:23.466578+00:00

## Current Read

- Status: the Huth/LeBel narrative-encoding experiment is not runnable yet
  from the paths visible to this session.
- Best candidate root checked: `none`.
- Blocking data pieces: need `ds003020` with `stimuli/*.wav`,
  `derivative/TextGrids/*.TextGrid`, and either author preprocessed
  `derivative/preprocessed_data/UTS*/<story>.hf5` or fMRIPrep/BIDS BOLD
  files staged locally/CHTC.
- Reusable local study hook found: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/tribev2/studies/lebel2023bold.py`
  (`exists=False`).
- CHTC submission is currently gated on a reusable SSH ControlMaster/2FA
  session, not on experiment code.

## Checked Roots

| Path | Exists | Plausible | TextGrids | WAVs | HF5 | BOLD | Test Story |
|---|---:|---:|---:|---:|---:|---:|---:|
| `/var/lib/condor/execute/slot1/dir_457053/scratch` | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/download/ds003020/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/download/ds003020/download/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/download/ds003020/data/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/download/ds003020/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `/var/lib/condor/execute/slot1/dir_457053/scratch/download/ds003020/download/openneuro/ds003020` | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

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

- JSON audit: `results/sft_huth_lebel/audit.json`
- Candidate CSV: `results/sft_huth_lebel/candidate_roots.csv`
