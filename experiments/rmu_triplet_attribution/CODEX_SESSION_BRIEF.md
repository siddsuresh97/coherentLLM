# Codex Session Brief: RMU Triplet Attribution

Start here for a fresh Codex session.

## Branch And Working Directory

- Branch: `exp-rmu-triplet-attribution`
- Worktree: `/tmp/exp-rmu-triplet-attribution-worktree`
- Remote: `origin` -> `git@github.com:siddsuresh97/coherentLLM.git`

## User Preferences To Preserve

- Always create a branch, commit, and push meaningful progress.
- Do not delete files or outputs.
- Use W&B online for every training run. Do not use offline W&B unless the user
  explicitly overrides it.
- Put W&B run links in `REPORT.md` and `RESEARCH_LOG.md`.
- Log the model, dataset, prompt/template, hyperparameters, training examples,
  hardware, backend, runtime, and reason for each run.
- Show representative training examples before or alongside training.
- Prefer fast, correct training settings: use known-good hyperparameters, small
  smoke tests, then scale. Do not assume 100k examples are needed; justify data
  size from the hypothesis and observed failure mode.
- Question assumptions and metrics when a run fails. Log why the next run was
  chosen over plausible alternatives.
- Use local GPUs when free. Local A5000s are good for small probes/training;
  local H100 is preferred for heavier 7B RMU work when available.
- Use CHTC for parallel or larger GPU jobs. Follow CHTC GPU-job practice:
  short/resumable jobs, broad GPU constraints, GPU utilization logs, W&B online,
  caches in scratch/staging, and no runtime package installs unless unavoidable.
- Prefer vLLM for batched probing/inference when it supports the needed
  probabilities/adapters. If vLLM is incompatible, use Transformers and log the
  reason.
- Use efficient KV caching / prefix caching for repeated probe prompts when the
  backend supports it.
- Keep pre/post probe conditions byte-identical except model weights.
- For empirical writeups, lead with AUC, effect sizes, QA evidence that the edit
  landed, and patching robustness. Do not drift into broad values framing.

## Current State

Step 0 has been run three times: once without an active token, once with the
shared token at `/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/token`,
and once with the ignored local token in
`experiments/rmu_triplet_attribution/.env`.

- `cais/wmdp` public QA is accessible and downloaded for configs `wmdp-bio`,
  `wmdp-chem`, and `wmdp-cyber`.
- The ignored local `.env` token authenticates as the user account and can access
  `cais/wmdp-bio-forget-corpus`.
- `cais/wmdp-cyber-forget-corpus` is still gated, but it is not needed for the
  planned bio-only RMU edit.

Machine-readable result:
[results/access_check_2026-07-09.json](results/access_check_2026-07-09.json)

Authenticated rerun:
[results/access_check_2026-07-09_authenticated.json](results/access_check_2026-07-09_authenticated.json)

Successful local `.env` token rerun:
[results/access_check_2026-07-09_local_env_token.json](results/access_check_2026-07-09_local_env_token.json)

## Immediate Next Decision

The exact WMDP-bio forget corpus is accessible now. Proceed with the exact WMDP
path unless the user explicitly redirects.

Do not commit `experiments/rmu_triplet_attribution/.env`; it is intentionally
ignored and contains the Hugging Face token.

## Next Technical Steps After Access/Fallback Choice

1. Clone or vendor `github.com/centerforaisafety/wmdp` into
   `experiments/rmu_triplet_attribution/external/` or document the external path.
2. Install or containerize dependencies for RMU and `lm-eval`.
3. Baseline `HuggingFaceH4/zephyr-7b-beta` on WMDP-bio, WMDP-cyber, WMDP-chem,
   and selected MMLU controls.
4. Apply bio-only RMU.
5. Re-score QA to confirm the edit landed cleanly.
6. Build the concept inventory and freeze Probe A/Probe B prompts.
7. Run pre/post triplet and pairwise probes with identical protocols.
8. Score H1 AUC, H3b over-separation, and the output-patching variant.

## Reporting Standard

`REPORT.md` should be readable by the user without opening the log. It must say:

- What hypothesis is being tested.
- Which model and edit were used.
- Which prompt/probe was used.
- What was tried.
- Whether it worked.
- Why it worked or why it failed.
- What the next run changes and why.

`RESEARCH_LOG.md` is append-only and should preserve the full decision trail.
