# RMU Triplet Attribution Report

Current status: branch created and Step 0 access check completed. The public WMDP
QA benchmark is accessible, but the exact WMDP-bio forget corpus needed for the
reference RMU recipe is gated in this environment.

## Hypothesis

H1: if RMU unlearns WMDP-bio, a black-box triplet probe should assign larger
pre-to-post behavioral distance shifts to edited bio concepts than to untouched
neutral, cyber, and chem concepts.

H3b: the edit should over-separate targeted hazardous bio concepts from benign
bio neighbors more than neutral-control pairs drift.

## What Has Been Tried

Step 0 only:

- Checked Hugging Face access for public QA dataset `cais/wmdp`.
- Loaded small slices from `wmdp-bio`, `wmdp-chem`, and `wmdp-cyber`.
- Checked access to `cais/wmdp-bio-forget-corpus` and
  `cais/wmdp-cyber-forget-corpus`.

Access-check file:
[results/access_check_2026-07-09.json](results/access_check_2026-07-09.json)

Session handoff with operating preferences:
[CODEX_SESSION_BRIEF.md](CODEX_SESSION_BRIEF.md)

Experiment spec:
[EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)

Append-only decision log:
[RESEARCH_LOG.md](RESEARCH_LOG.md)

## Did It Work?

The access check partly worked.

- Worked: `cais/wmdp` is public and downloaded successfully. Available configs:
  `wmdp-bio`, `wmdp-chem`, `wmdp-cyber`. Columns are `answer`, `question`, and
  `choices`.
- Did not work: the exact WMDP forget corpora are gated. The local environment
  does not currently have access to `cais/wmdp-bio-forget-corpus`.

## Why This Matters

The reference RMU edit depends on the gated WMDP-bio forget corpus. Without that
access, running the exact WMDP-authors' recipe would silently become a different
experiment. The clean choice is either to wait for gated access or explicitly use
a public proxy forget corpus and name the edit accordingly.

## Model, Prompt, And Training State

- Planned base model: `HuggingFaceH4/zephyr-7b-beta`.
- Planned edit method: WMDP/RMU bio-only unlearning.
- RMU training has not been run yet.
- Triplet and pairwise probe prompts have not been frozen yet.
- No W&B runs exist yet for this branch because no training has started.

## Next Decision

Choose one:

- Exact WMDP path: accept/access `cais/wmdp-bio-forget-corpus` on Hugging Face,
  then run the reference RMU recipe.
- Fallback path: use a public bio proxy forget corpus, and describe the edit as a
  representative bio-unlearning edit rather than exact WMDP-bio RMU.

## Live Risks

- Gated forget corpus blocks exact replication of WMDP/RMU.
- RMU may require H100/A100-class memory; if local GPUs are insufficient, use
  CHTC with logged online W&B runs.
- Triplet probe must use identical pre/post prompts and decoding; otherwise the
  attribution signal is contaminated by protocol drift.
