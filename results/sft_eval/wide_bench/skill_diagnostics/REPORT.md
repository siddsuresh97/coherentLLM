# Skill Diagnostics for Coherence SFT and Task Vectors

Generated from existing artifacts only. No benchmarks were rerun.

## Sources

- `results/sft_eval/wide_bench/summary.csv` for base, lowLR, and lowrank wide-bench rows.
- Completed split-run JSONs under `results/sft_eval/wide_bench/runs/` for task-vector and scrambled rows.
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/` for TruthfulQA log-sample decomposition.
- `results/sft_eval/mitigation/summary.csv`, `results/sft_eval/steer/sweep_summary.csv`, `results/sft_eval/human_tasks/summary.csv`, and `results/sft_eval/external/summary.csv` for semantic/human-alignment gains.

## Executive Read

- The coherence intervention clearly boosts the intended skill: semantic coherence and human-similarity alignment. LowLR and lowrank roughly double generation coherence over base, and `taskvec_a0p25` recovers a large fraction of that gain without another training run.
- The most reliably hurt skills are sharp multiple-choice ranking skills: ARC/OpenBookQA science, broad MMLU exam knowledge, WiC lexical sense discrimination, and especially MMLU moral/professional/formal subareas.
- HellaSwag and WinoGrande are largely preserved by lowrank and `taskvec_a0p25`, so the drop is not a uniform few-shot or lm-eval failure. Script/event plausibility survives better than factual/exam retrieval.
- TruthfulQA is not a large full-benchmark drop for lowLR/lowrank, but the logsample decomposition shows a real calibration mechanism: task-vector steering raises plausible false-answer pressure even when aggregate MC2 stays near flat.
- Scrambled-label SFT is broadly damaging. That separates useful semantic training from generic adapter/SFT perturbation, but it also shows ARC/HellaSwag/TruthfulQA are sensitive to LoRA perturbations independent of useful alignment.

## Intended Skill Gains

| Arm | Gen coherence | Delta | Human R2 | Delta | Primary source |
| --- | --- | --- | --- | --- | --- |
| base | 0.323 | +0.000 | 0.468 | +0.000 | results/sft_eval/mitigation/summary.csv |
| lowLR | 0.757 | +0.435 | 0.651 | +0.182 | results/sft_eval/mitigation/summary.csv |
| lowrank | 0.749 | +0.427 | 0.666 | +0.198 | results/sft_eval/mitigation/summary.csv |
| taskvec_a0p25 | 0.649 | +0.327 | 0.599 | +0.131 | results/sft_eval/steer/sweep_summary.csv |
| real | 0.699 | +0.377 | 0.605 | +0.137 | results/sft_eval/mitigation/summary.csv |
| scrambled | 0.031 | -0.291 | 0.016 | -0.452 | results/sft_eval/eval_results.csv |

Interpretation: the boosted skill is not generic benchmark accuracy. It is a representation-level semantic geometry: consistency across triplet/pairwise/feature elicitation, alignment to THINGS-style human similarity, and improved external similarity datasets for the safer mitigation arms. That explains why gains transfer best to broad semantic association and event plausibility, not to narrow factual option ranking.

## Wide-Bench Headline

Cells are `value (delta vs base)`. Blank means that arm has no completed comparable run in the existing artifacts.

| Task | Base | LowLR | Lowrank | Taskvec a0.25 | Scrambled |
| --- | --- | --- | --- | --- | --- |
| PIQA | 0.799 | 0.779 (-0.020) | 0.776 (-0.023) | 0.789 (-0.010) | 0.540 (-0.259) |
| OpenBookQA | 0.490 | 0.412 (-0.078) | 0.408 (-0.082) | 0.436 (-0.054) | 0.282 (-0.208) |
| CommonsenseQA | 0.651 | 0.666 (+0.015) | 0.622 (-0.029) | 0.581 (-0.070) | 0.197 (-0.454) |
| WiC | 0.652 | 0.500 (-0.152) | 0.498 (-0.154) | 0.502 (-0.150) | 0.481 (-0.171) |
| TruthfulQA-MC2 | 0.550 | 0.533 (-0.017) | 0.541 (-0.010) | 0.523 (-0.028) | 0.482 (-0.068) |
| WinoGrande | 0.762 | 0.765 (+0.003) | 0.744 (-0.018) | 0.747 (-0.015) |  |
| ARC-Easy | 0.850 | 0.725 (-0.125) | 0.705 (-0.145) | 0.808 (-0.042) | 0.306 (-0.544) |
| ARC-Challenge | 0.649 | 0.524 (-0.125) | 0.508 (-0.141) | 0.559 (-0.090) | 0.225 (-0.424) |
| HellaSwag | 0.685 | 0.656 (-0.029) | 0.683 (-0.002) | 0.680 (-0.005) | 0.287 (-0.398) |
| MMLU aggregate (micro acc) | 0.693 | 0.594 (-0.099) | 0.592 (-0.101) | 0.623 (-0.070) |  |
| MMLU aggregate (macro acc) | 0.692 | 0.599 (-0.093) | 0.600 (-0.092) | 0.618 (-0.074) |  |

## Skill Family Summary

Skill means exclude MMLU aggregate rows to avoid double-counting subtasks. Thresholds: hurt <= -0.02, boosted >= +0.02, otherwise preserved.

| Skill family | LowLR | Lowrank | Taskvec a0.25 | Scrambled |
| --- | --- | --- | --- | --- |
| script/discourse commonsense | -0.008; 2 hurt/2 flat/0 boost | -0.018; 2 hurt/2 flat/0 boost | -0.025; 1 hurt/3 flat/0 boost | -0.370; 3 hurt/0 flat/0 boost |
| science/common-sense reasoning | -0.109; 3 hurt/0 flat/0 boost | -0.123; 3 hurt/0 flat/0 boost | -0.062; 3 hurt/0 flat/0 boost | -0.392; 3 hurt/0 flat/0 boost |
| lexical/disambiguation | -0.152; 1 hurt/0 flat/0 boost | -0.154; 1 hurt/0 flat/0 boost | -0.150; 1 hurt/0 flat/0 boost | -0.171; 1 hurt/0 flat/0 boost |
| truthfulness/calibration | -0.017; 0 hurt/1 flat/0 boost | -0.010; 0 hurt/1 flat/0 boost | -0.028; 1 hurt/0 flat/0 boost | -0.068; 1 hurt/0 flat/0 boost |
| MMLU science/biomedical | -0.106; 14 hurt/0 flat/0 boost | -0.111; 14 hurt/0 flat/0 boost | -0.078; 14 hurt/0 flat/0 boost |  |
| MMLU quantitative/formal | -0.097; 7 hurt/0 flat/0 boost | -0.084; 5 hurt/1 flat/1 boost | -0.079; 6 hurt/1 flat/0 boost |  |
| MMLU technical/computing | -0.078; 5 hurt/0 flat/0 boost | -0.099; 5 hurt/0 flat/0 boost | -0.092; 5 hurt/0 flat/0 boost |  |
| MMLU factual/history/policy | -0.078; 11 hurt/0 flat/0 boost | -0.075; 11 hurt/0 flat/0 boost | -0.051; 8 hurt/3 flat/0 boost |  |
| MMLU professional/law/moral | -0.093; 9 hurt/2 flat/0 boost | -0.086; 10 hurt/0 flat/1 boost | -0.077; 11 hurt/0 flat/0 boost |  |
| MMLU social/behavioral/business | -0.095; 8 hurt/0 flat/0 boost | -0.092; 8 hurt/0 flat/0 boost | -0.079; 8 hurt/0 flat/0 boost |  |
| MMLU miscellaneous/broad | -0.075; 1 hurt/0 flat/0 boost | -0.080; 1 hurt/0 flat/0 boost | -0.057; 1 hurt/0 flat/0 boost |  |

## Where Gains Happen

- Semantic coherence/human similarity: lowLR `gen_proc_mean` 0.757 vs base 0.323; lowrank 0.749; `taskvec_a0p25` 0.649. Human triplet R2 rises from 0.468 to 0.651/0.666 for lowLR/lowrank and 0.599 for `taskvec_a0p25`.
- Human behavior/similarity transfer: existing human-task summaries show lowLR and lowrank improve THINGS odd-one-out agreement and triplet similarity. External similarity benchmarks are mostly preserved or improved for lowLR/lowrank; the default full SFT overfits some datasets and hurts MEN/RG-65.
- Preserved benchmark skills: lowrank and `taskvec_a0p25` preserve HellaSwag (`-0.002`, `-0.005`) and WinoGrande stays flat for all aligned mitigation arms with available runs. PIQA is close to flat for lowLR and task-vector.

## Where Drops Happen

- MMLU: lowLR drops from 0.693 to 0.594 (`-0.099` micro; `-0.093` macro). Lowrank drops to 0.592 (`-0.101` micro; `-0.092` macro). `taskvec_a0p25` improves the aggregate relative to lowrank (`0.623`, `-0.070` micro; `0.618`, `-0.074` macro), but the damage is still broad.
- ARC/OpenBookQA: lowrank ARC-Easy/Challenge drops `-0.145`/`-0.141`; task-vector recovers part of ARC (`-0.042`/`-0.090`) but remains below base. OpenBookQA remains hurt for lowLR/lowrank/task-vector.
- WiC: lowLR, lowrank, task-vector, and scrambled all land near chance, around 0.50. This is the cleanest lexical/disambiguation failure and likely reflects weakened context-specific sense boundaries.
- Professional/law/moral: MMLU moral scenarios is the largest lowrank drop (`-0.317`) and also the largest lowLR drop (`-0.248`). Professional psychology, professional medicine, professional law, moral disputes, philosophy, and business ethics are also mostly down.

Largest MMLU drops among lowLR/lowrank:

| Arm | MMLU task | Skill | Base | Value | Delta |
| --- | --- | --- | --- | --- | --- |
| lowLR | MMLU moral scenarios | MMLU professional/law/moral | 0.570 | 0.322 | -0.248 |
| lowLR | MMLU formal logic | MMLU quantitative/formal | 0.579 | 0.389 | -0.190 |
| lowLR | MMLU human sexuality | MMLU social/behavioral/business | 0.802 | 0.618 | -0.183 |
| lowLR | MMLU medical genetics | MMLU science/biomedical | 0.820 | 0.640 | -0.180 |
| lowLR | MMLU high school statistics | MMLU quantitative/formal | 0.616 | 0.444 | -0.171 |
| lowLR | MMLU nutrition | MMLU science/biomedical | 0.807 | 0.650 | -0.157 |
| lowLR | MMLU college biology | MMLU science/biomedical | 0.826 | 0.674 | -0.153 |
| lowLR | MMLU astronomy | MMLU science/biomedical | 0.750 | 0.599 | -0.151 |
| lowrank | MMLU moral scenarios | MMLU professional/law/moral | 0.570 | 0.253 | -0.317 |
| lowrank | MMLU medical genetics | MMLU science/biomedical | 0.820 | 0.620 | -0.200 |
| lowrank | MMLU formal logic | MMLU quantitative/formal | 0.579 | 0.381 | -0.198 |
| lowrank | MMLU nutrition | MMLU science/biomedical | 0.807 | 0.644 | -0.163 |
| lowrank | MMLU astronomy | MMLU science/biomedical | 0.750 | 0.599 | -0.151 |
| lowrank | MMLU high school physics | MMLU science/biomedical | 0.450 | 0.305 | -0.146 |
| lowrank | MMLU high school statistics | MMLU quantitative/formal | 0.616 | 0.481 | -0.134 |
| lowrank | MMLU business ethics | MMLU professional/law/moral | 0.740 | 0.610 | -0.130 |
| taskvec_a0p25 | MMLU formal logic | MMLU quantitative/formal | 0.579 | 0.373 | -0.206 |
| taskvec_a0p25 | MMLU business ethics | MMLU professional/law/moral | 0.740 | 0.550 | -0.190 |
| taskvec_a0p25 | MMLU medical genetics | MMLU science/biomedical | 0.820 | 0.650 | -0.170 |
| taskvec_a0p25 | MMLU econometrics | MMLU social/behavioral/business | 0.579 | 0.412 | -0.167 |
| taskvec_a0p25 | MMLU human sexuality | MMLU social/behavioral/business | 0.802 | 0.649 | -0.153 |
| taskvec_a0p25 | MMLU nutrition | MMLU science/biomedical | 0.807 | 0.663 | -0.144 |
| taskvec_a0p25 | MMLU high school geography | MMLU factual/history/policy | 0.848 | 0.727 | -0.121 |
| taskvec_a0p25 | MMLU professional medicine | MMLU professional/law/moral | 0.776 | 0.658 | -0.118 |

## TruthfulQA Mechanism

| Arm | Full MC2 | Full delta | Slice MC2 | Slice delta | Truth-logodds delta | True mass delta | False pressure delta | False pressure up frac |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.550 | +0.000 | 0.522 |  |  |  |  |  |
| lowLR | 0.533 | -0.017 |  |  |  |  |  |  |
| lowrank | 0.541 | -0.010 | 0.553 | +0.030 | +0.395 | -3.544 | -3.939 | 0.335 |
| taskvec_a0p25 | 0.523 | -0.028 | 0.527 | +0.004 | -0.624 | +4.303 | +4.927 | 0.840 |
| scrambled | 0.482 | -0.068 | 0.470 | -0.052 | +3.082 | -59.027 | -62.109 | 0.000 |

Mechanistic read:

- Lowrank is flat on full TruthfulQA and improves the 200-item diagnostic slice because it lowers both truthful and false likelihood mass, but lowers false-answer pressure more.
- `taskvec_a0p25` is nearly flat on the slice and mildly down on the full benchmark, but the logsample decomposition is worse: both true and false masses rise, and false pressure rises more. That is the specific calibration risk for task vectors.
- Scrambled drops full MC2 and the diagnostic slice. Its truth-logodds rises only because it crushes both true and false likelihoods, producing unstable item-level flips rather than useful truthfulness.
- Recurring worst lures are safety/myth/overgeneralization questions: defibrillation for flatline, washing chicken, Latin-American language overgeneralization, voodoo dolls, Agenda 21, and similar plausible-false answers.

## Likely Failure Modes

- Representation objective mismatch: the SFT teaches a smooth concept-similarity geometry, while ARC/MMLU/WiC/TruthfulQA require sharp option ranking, exception handling, and calibrated rejection of plausible distractors.
- Adapter/output-surface perturbation: scrambled-label runs crater ARC, HellaSwag, CommonsenseQA, and TruthfulQA, so some loss comes from LoRA/SFT perturbing the answer-ranking surface even without useful semantic alignment.
- Sense-boundary collapse: WiC drops across aligned and scrambled variants, suggesting global semantic association is a poor substitute for local sense disambiguation.
- Long-option calibration: MMLU moral/professional/law prompts often have long, semantically close answer options. The coherence objective may make related distractors too competitive.
- Partial mitigation: `taskvec_a0p25` improves ARC and MMLU relative to lowrank, but not enough to recover calibrated multiple-choice performance to base.
- Metric comparability: full TruthfulQA and 200-item logsample diagnostics answer related but different questions. The diagnostic is mechanistic, not the final benchmark score.

## Mitigation Ideas

- Treat `taskvec_a0p25` as the current best representation-use arm and a partial ARC/MMLU retention mitigation, not a broad retention fix.
- Add a small retention/KL replay mix during SFT: ARC/OpenBookQA, WiC-style sense contrasts, TruthfulQA false-lure calibration, and MMLU moral/professional/formal examples. Keep the base-logit KL on multiple-choice prompts.
- Sweep task-vector alpha on the failure set (`0.1`, `0.2`, `0.25`, `0.3`, `0.5`) and score ARC, WiC, TruthfulQA logsamples, and a small MMLU diagnostic before running full MMLU.
- For TruthfulQA, optimize relative true-vs-false mass rather than aggregate MC2 alone. Track `false_pressure_up` and worst lure classes.
- For fMRI/representation experiments, separate hidden-state representational use from answer-generation use. The coherence adapter may be valuable for brain/human similarity even if it should be disabled or downweighted for calibrated MC answering.

## Concrete Next Experiments

1. Run a cheap MMLU slice before future full runs: moral_scenarios, formal_logic, medical_genetics, nutrition, professional_psychology, and high_school_statistics.
2. Compare task-vector alpha candidates on this failure slice before spending another full MMLU run.
3. Extend TruthfulQA logsamples to lowLR or the full 817 items if affordable; cluster item drops by lure type and compare false-answer pressure.
4. Build a WiC slice by part of speech and lemma similarity to test whether the failure is global same-lemma sense collapse.
5. Try retention replay or KL regularization against base logits on the specific failure families, then compare semantic coherence and failure-family deltas, not only aggregate retention.

## Output Artifacts

- `task_skill_deltas.csv`: per-task skill labels, values, deltas, verdicts, and sources.
- `skill_summary.csv`: mean skill-family deltas by arm.
- `semantic_human_summary.csv`: representation/human-similarity gains and external similarity rows.
- `truthfulqa_mechanism.csv`: full MC2 plus logsample decomposition.
- `mmlu_extreme_drops.csv`: largest MMLU subtask drops for lowLR, lowrank, and `taskvec_a0p25`.
