# CHTC Huth/LeBel Audit Bundle

This bundle is for the first non-heavy CHTC validation step for the LeBel/Huth
language-fMRI lane. It checks whether `ds003020` is present in the job sandbox
or `$STAGING` and transfers a small report back.

Run only after a CHTC SSH ControlMaster is active. Operational rule for this
project: Codex should run `chtc-master start` itself and ask the user only to
approve the Duo push, then continue submission and monitoring once `check`
passes.

```bash
chtc-master start
# User approves Duo push.
chtc-master check
chtc-ssh 'hostname -f; condor_q -totals; echo STAGING=$STAGING'
```

Suggested remote flow:

```bash
RUN_ID=coherence-huth-audit-20260708
chtc-ssh "mkdir -p ~/chtc-runs/$RUN_ID"
chtc-push src/sft/huth_lebel_audit.py "chtc-runs/$RUN_ID/"
chtc-push chtc/huth_lebel_audit/ "chtc-runs/$RUN_ID/"
chtc-ssh "cd ~/chtc-runs/$RUN_ID && condor_submit huth_lebel_audit.sub"
```

Expected staged dataset locations checked by the wrapper:

- `$STAGING/datasets/ds003020`
- `$STAGING/datasets/lebel/ds003020`
- `$STAGING/ds003020`
- `./download/ds003020`

The audit is CPU-only. The GPU lane starts after this finds/stages TextGrids,
stimuli, and BOLD/preprocessed responses; then submit independent one-GPU
feature-extraction jobs by `(arm, subject, story split)`.
