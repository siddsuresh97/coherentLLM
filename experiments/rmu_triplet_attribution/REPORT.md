# RMU Triplet Attribution Report

Current status: Step 0 is green for the exact WMDP-bio path. Public WMDP QA is
accessible, and `cais/wmdp-bio-forget-corpus` now loads through the ignored local
token file at `experiments/rmu_triplet_attribution/.env`. Exact WMDP/RMU can
proceed.

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

Authenticated access-check rerun:
[results/access_check_2026-07-09_authenticated.json](results/access_check_2026-07-09_authenticated.json)

Local `.env` token access-check rerun:
[results/access_check_2026-07-09_local_env_token.json](results/access_check_2026-07-09_local_env_token.json)

Session handoff with operating preferences:
[CODEX_SESSION_BRIEF.md](CODEX_SESSION_BRIEF.md)

Experiment spec:
[EXPERIMENT_SPEC.md](EXPERIMENT_SPEC.md)

Append-only decision log:
[RESEARCH_LOG.md](RESEARCH_LOG.md)

## Did It Work?

The access checks partly worked.

- Worked: `cais/wmdp` is public and downloaded successfully. Available configs:
  `wmdp-bio`, `wmdp-chem`, `wmdp-cyber`. Columns are `answer`, `question`, and
  `choices`.
- Worked after installing the local ignored `.env` token: the exact
  `cais/wmdp-bio-forget-corpus` loads. The first three training rows have
  columns `title`, `abstract`, `text`, and `doi`.
- Still unavailable but not required for this bio-only run:
  `cais/wmdp-cyber-forget-corpus`.

## Why This Matters

The reference RMU edit depends on the gated WMDP-bio forget corpus. That gate is
now open for the bio-only edit, so the next work should use the exact WMDP/RMU
path rather than the public-proxy fallback.

## Model, Prompt, And Training State

- Planned base model: `HuggingFaceH4/zephyr-7b-beta`.
- Planned edit method: WMDP/RMU bio-only unlearning.
- RMU training has not been run yet.
- Triplet and pairwise probe prompts have not been frozen yet.
- No W&B runs exist yet for this branch because no training has started.

## Next Decision

Next step:

- Clone/inspect the WMDP repo, identify the Zephyr RMU default config, baseline
  `HuggingFaceH4/zephyr-7b-beta` on WMDP bio/chem/cyber plus MMLU controls, then
  run bio-only RMU with W&B online.

## Live Risks

- RMU may require H100/A100-class memory; if local GPUs are insufficient, use
  CHTC with logged online W&B runs.
- Triplet probe must use identical pre/post prompts and decoding; otherwise the
  attribution signal is contaminated by protocol drift.
