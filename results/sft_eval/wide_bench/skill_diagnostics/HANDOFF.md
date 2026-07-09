# Benchmark-Drop Mitigation Handoff

Last updated: 2026-07-08 on branch `coherence-sft`.

This handoff is for future agents who need to continue benchmark-drop mechanism and mitigation work without rereading the whole repo.

## Scope and Ownership

Owned files for this handoff task:

- `results/sft_eval/wide_bench/skill_diagnostics/MITIGATION_PLAN.md`
- `results/sft_eval/wide_bench/skill_diagnostics/HANDOFF.md`

Do not revert or overwrite other agents' changes. The broader working tree has many active benchmark, CHTC, and fMRI changes.

No expensive jobs were launched for this handoff. This is a plan and output contract only.

## Canonical Context

Read these first:

- `results/sft_eval/wide_bench/skill_diagnostics/REPORT.md`
- `results/sft_eval/wide_bench/skill_diagnostics/skill_summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/truthfulqa_mechanism.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/mmlu_extreme_drops.csv`
- `results/sft_eval/wide_bench/skill_map.md`
- `README.md`, benchmark section
- `research/EXPERIMENT_LOG.md`, benchmark skill diagnostics and TruthfulQA sections

Use `MITIGATION_PLAN.md` in this directory as the runnable experiment queue.

## Current Scientific State

Boosted:

- Semantic coherence and human-similarity alignment.
- `lowLR`: generation coherence `0.757`, human R2 `0.651`.
- `lowrank`: generation coherence `0.749`, human R2 `0.666`.
- `taskvec_a0p25`: generation coherence `0.649`, human R2 `0.599`.

Preserved:

- HellaSwag is flat under coherent states: lowrank delta `-0.002`, taskvec delta `-0.005`.
- WinoGrande is flat: lowLR `+0.003`, lowrank `-0.018`, taskvec `-0.015`.
- PIQA is close to flat under taskvec: `-0.010`.

Hurt:

- WiC is near chance for all adapter states: aligned and scrambled.
- ARC/OpenBookQA science and fact ranking are hurt; taskvec helps ARC versus lowrank but is still below base.
- MMLU is broadly hurt for lowLR/lowrank, with worst drops in moral scenarios, formal logic, medical genetics, nutrition, and professional/biomedical subjects.
- TruthfulQA is not a large lowrank full-score failure, but taskvec raises plausible false-answer pressure on the 200-item diagnostic.

Unsettled:

- `taskvec_a0p25` MMLU aggregate is still the missing row in the diagnostics. Do not claim task-vector broadly fixes MMLU until it is merged and the diagnostics are regenerated.

## Next Actions

1. Finish or ingest the current `taskvec_a0p25` MMLU row.
   - Merge CHTC/local shards if available.
   - Run `python src/sft/analyze_skill_diagnostics.py`.
   - Update `REPORT.md` and CSVs in `skill_diagnostics/`.

2. Build the cheap mechanism suite before new training.
   - Tasks: `mmlu_moral_scenarios`, `mmlu_formal_logic`, `mmlu_medical_genetics`, `mmlu_nutrition`, `mmlu_professional_psychology`, `mmlu_high_school_statistics`, `arc_easy`, `arc_challenge`, `openbookqa`, `wic`, `truthfulqa_mc2`, plus controls `hellaswag`, `winogrande`, `piqa`.
   - Arms: `base`, `lowrank`, `taskvec_a0p25`, `scrambled`; add `lowLR` if cheap.
   - Outputs should go under `results/sft_eval/wide_bench/skill_diagnostics/mechanism_suite/<run_id>/`.

3. Only after the mechanism suite, run task-vector alpha sweeps.
   - Grid: `0`, `0.1`, `0.2`, `0.25`, `0.3`, `0.5`, `1.0`.
   - Existing adapter dirs cover `a0p25` and `a0p5`; finer alpha adapter dirs may need to be generated and recorded in an `adapter_manifest.csv`.
   - Full MMLU should be reserved for one or two alpha candidates that pass the failure slice.

4. Use CAA steering for a cheap selective test.
   - Existing scripts: `src/sft/extract_concept_vectors.py` and `src/sft/run_concept_steering_eval.py`.
   - Start with forced-choice probes only.
   - Do not use broad lm-eval steering until retention probes are stable.

5. Defer new SFT curricula until diagnostics identify the protected skill.
   - Candidate protected skills: TruthfulQA false-lure calibration, WiC sense boundaries, MMLU close-option ranking, ARC/OpenBookQA science facts.

## Pass/Fail Gates

Task-vector MMLU:

- Pass: MMLU micro delta `>= -0.06` and worst subject drops improve over lowrank.
- Fail: MMLU remains around `-0.09` to `-0.10`; treat taskvec as semantic-useful but not a broad benchmark mitigation.

Mechanism suite:

- Pass: margin, false-pressure, or WiC sense-slice metrics explain the drop and distinguish aligned from scrambled.
- Fail: slices do not reproduce the aggregate drops or controls erase the effect.

Alpha sweep:

- Pass: semantic gain retained, TruthfulQA false-pressure controlled, and MMLU/ARC slice improves.
- Fail: every alpha either loses semantics or keeps benchmark damage.

Curriculum:

- Pass: generation coherence `>= 0.70` or taskvec-level gain with better retention; human R2 `>= 0.60`; failure-suite improvement before full MMLU.
- Fail: retention improves only by losing the intended semantic/human signal.

## Output Contract

Every run directory should include:

- `COMMANDS.md` with exact commands and host/container/env.
- `metadata.json` with commit hash, arms, tasks, limits, few-shot settings, backend, max length, batch size, and GPU.
- Raw result JSON/logsample files or links to them.
- `summary.csv` with one row per arm/task.
- `REPORT.md` with objective, result, pass/fail, and next action.

Commit policy:

- Stage only files intentionally owned by the current agent.
- For this handoff task, use:

```bash
git add results/sft_eval/wide_bench/skill_diagnostics/MITIGATION_PLAN.md \
  results/sft_eval/wide_bench/skill_diagnostics/HANDOFF.md
```

Do not `git add results/sft_eval/wide_bench/skill_diagnostics/` wholesale unless you own every changed file in that directory.
