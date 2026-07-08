# Coherence-SFT Experiment Log

Last updated: 2026-07-08

## Read this first

This is the running handoff log for follow-up work after Tasks 1-8 on branch
`coherence-sft`. New agents should read, in order:

1. `research/STATUS.md`
2. `results/sft_eval/REPORT.md`
3. this log
4. the active task briefs listed below

Tracking convention:
- Task briefs live in `research/CODEX_TASK_*.md`.
- Major experiment outputs live under `results/`.
- Every meaningful checkpoint should be committed and pushed to `origin/coherence-sft`.
- Append command summaries, artifact paths, and interpretation here after each major run.
- Do not treat a partial/stale artifact as a result unless this log explicitly marks it as valid.

## Active task briefs

- `research/CODEX_TASK_9_FMRI_AND_SEMANTIC_HUB.md`
  - Goal: test whether coherence-SFT improves brain predictivity and whether a
    format-invariant hidden-state semantic hub mediates it.
  - Status: brief created; no fMRI or hidden-state production runs launched yet.

- `research/CODEX_TASK_10_BENCHMARK_DROPS.md`
  - Goal: diagnose why lowLR/lowrank gain semantic/human-alignment tasks but drop
    on standard capability benchmarks, and propose mitigation experiments.
  - Status: brief created; existing results only so far.

## Current branch state

- Branch: `coherence-sft`
- Remote: `origin git@github.com:siddsuresh97/coherentLLM.git`
- Current checked commit before Task 9 docs: `76dbc9f`
- Worktree at log creation: clean.

## Sidecar agents

The following sidecar agents were launched on 2026-07-08. They were instructed not
to edit files; their memos should be integrated into this log and the task briefs.

| Agent | ID | Scope |
|---|---|---|
| Peirce | `019f4341-554b-7b50-b654-2e2dc09ffb2f` | Alex Huth / LeBel / UT Austin natural-language fMRI encoding literature |
| Kierkegaard | `019f4341-6757-78d3-9fb5-94f25969da6e` | Evelina Fedorenko / Ivanova language-network evaluation literature |
| Gibbs | `019f4341-806a-7d51-82f9-49fc95df0c88` | Local benchmark drop/gain diagnosis |
| Nietzsche | `019f4341-bb81-7200-ba91-a7c09b9d4aa2` | Semantic-hub hypothesis and tests |

## 2026-07-08 checkpoint: planning and data audit facts

Facts established locally before implementation:

- The original fMRI proposal is `research/FUTURE_brain_predictivity.md`.
- The semantic-hub proposal is `research/FUTURE_semantic_hub.md`.
- Existing THINGS-fMRI pipeline to reuse is in sibling project:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/`
- Existing Ch4 aggregation script:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/dissertation/scripts/compute_fmri_rsa.py`
- THINGS-fMRI metadata/betas are available locally under:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/`
- The SFT eval concept file has 128 lines in `data/scale128/concepts.csv`.
- Exact overlap with the 720 THINGS-fMRI concepts in `sub-01_StimulusMetadata.csv` is 90 concepts.
- Prior Ch4 noise ceilings warn against overclaiming ATL/language on object fMRI:
  Ventral Visual is high (`lower=0.229`, `upper=0.284`), while Language is low
  (`lower=0.010`, `upper=0.019`) and pooled ATL is near zero in the old object
  setup. Therefore Task 9 uses Ventral Visual as the primary ROI and treats ATL
  subregions / Language as exploratory.

Command used to establish overlap:

```bash
python - <<'PY'
import csv
from pathlib import Path
sft = Path('data/scale128/concepts.csv')
fmri = Path('/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/sub-01_StimulusMetadata.csv')
s = {line.strip() for line in sft.read_text().splitlines() if line.strip()}
with fmri.open() as f:
    r = csv.DictReader(f)
    t = {row['concept'] for row in r}
print(len(s), len(t), len(s & t))
PY
```

Expected output: `128 720 90`.

## Open decisions

- Whether to run Phase 1 RSA only on the 90 exact held-out overlaps, or also add
  a secondary full-720 analysis that is not leakage-clean.
- Whether Phase 2 language-fMRI should use LeBel/Huth first or a more turnkey
  OpenNeuro/Narratives route if LeBel preprocessing is costly.
- Whether mitigation work should prioritize alpha-sweep/task-vector strength,
  training recipe changes, or post-hoc adapter composition.
