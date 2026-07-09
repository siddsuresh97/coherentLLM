# Benchmark-Skill Taxonomy for Coherence SFT Drops and Gains

Report date: 2026-07-09
Branch: `coherence-sft`
Scope: local artifacts only; no benchmarks were rerun for this report.

## Purpose

This report maps the current wide-benchmark drops, stable cells, and gains to
the skills those benchmarks most likely test. It is intended to be trackable:
each claim below points back to result files already present in the repo, and
hypotheses are labeled as hypotheses rather than measured facts.

The short answer is that coherence SFT improves the intended representation
skill, but that skill is not general benchmark accuracy. The intervention makes
semantic geometry and human-similarity alignment much better, while hurting
benchmarks that require sharp multiple-choice ranking, lexical sense boundaries,
specialized factual recall, or calibrated rejection of plausible false answers.

## Sources and Labeling

Primary local sources:

- `results/sft_eval/wide_bench/summary.csv`: base, `lowLR`, and `lowrank`
  wide-benchmark scores, including full MMLU and full TruthfulQA-MC2.
- `results/sft_eval/wide_bench/raw_task_summary.csv`: raw per-task rows behind
  the wide-benchmark summary.
- `results/sft_eval/wide_bench/skill_diagnostics/task_skill_deltas.csv`:
  per-task skill labels, split-run rows, deltas, verdicts, and source paths.
- `results/sft_eval/wide_bench/skill_diagnostics/skill_summary.csv`:
  skill-family aggregate deltas.
- `results/sft_eval/wide_bench/skill_diagnostics/semantic_human_summary.csv`:
  semantic/human-similarity gains.
- `results/sft_eval/wide_bench/skill_diagnostics/truthfulqa_mechanism.csv` and
  `results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/REPORT.md`:
  TruthfulQA full-score plus 200-item log-sample decomposition.
- `results/sft_eval/wide_bench/skill_diagnostics/mmlu_extreme_drops.csv`:
  largest MMLU subject drops.
- `results/sft_eval/wide_bench/skill_map.md` and
  `results/sft_eval/wide_bench/skill_diagnostics/REPORT.md`: prior local
  summaries used for cross-checking, not as independent measurements.

Verdict convention copied from the existing diagnostics:

- `hurt`: delta vs base `<= -0.02`.
- `stable`: absolute delta `< 0.02`.
- `boosted`: delta vs base `>= +0.02`.

Blank cells mean the comparable run is not present in the local artifacts. In
particular, comparable `taskvec_a0p25` aggregate MMLU is still missing, so this
report does not claim that task vectors fix broad MMLU retention.

## Headline Score Map

Scores are `value (delta vs base)`. Full base/lowLR/lowrank rows are from
`summary.csv`; `taskvec_a0p25` and `scrambled` split-run rows are from
`task_skill_deltas.csv`.

| Benchmark | Likely skill proxy | Base | lowLR | lowrank | taskvec_a0p25 | scrambled | Current read |
|---|---|---:|---:|---:|---:|---:|---|
| PIQA | physical affordance commonsense | 0.799 | 0.779 (-0.020) | 0.776 (-0.023) | 0.789 (-0.010) | 0.540 (-0.259) | mostly stable for task-vector, hurt by direct adapters, collapsed by scrambled |
| OpenBookQA | science facts plus commonsense facts | 0.490 | 0.412 (-0.078) | 0.408 (-0.082) | 0.436 (-0.054) | 0.282 (-0.208) | hurt across aligned arms; task-vector only partially mitigates |
| CommonsenseQA | associative commonsense | 0.651 | 0.666 (+0.015) | 0.622 (-0.029) | 0.581 (-0.070) | 0.197 (-0.454) | lowLR stable, lowrank/task-vector hurt, scrambled collapsed |
| WiC | word-sense disambiguation | 0.652 | 0.500 (-0.152) | 0.498 (-0.154) | 0.502 (-0.150) | 0.481 (-0.171) | cleanest lexical failure; near chance across arms |
| TruthfulQA-MC2 | truthfulness and false-lure calibration | 0.550 | 0.533 (-0.017) | 0.541 (-0.010) | 0.523 (-0.028) | 0.482 (-0.068) | full lowLR/lowrank stable, task-vector and scrambled hurt |
| WinoGrande | coreference and discourse commonsense | 0.762 | 0.765 (+0.003) | 0.744 (-0.018) | 0.747 (-0.015) |  | stable in aligned arms with available runs |
| ARC-Easy | grade-school science retrieval | 0.850 | 0.725 (-0.125) | 0.705 (-0.145) | 0.808 (-0.042) | 0.306 (-0.544) | hurt; task-vector recovers much of lowrank loss but remains below base |
| ARC-Challenge | harder science reasoning and option ranking | 0.649 | 0.524 (-0.125) | 0.508 (-0.141) | 0.559 (-0.090) | 0.225 (-0.424) | hurt; same partial task-vector recovery |
| HellaSwag | script and event plausibility | 0.685 | 0.656 (-0.029) | 0.683 (-0.002) | 0.680 (-0.005) | 0.287 (-0.398) | preserved by lowrank/task-vector, collapsed by scrambled |
| MMLU micro | broad exam knowledge and reasoning | 0.693 | 0.594 (-0.099) | 0.592 (-0.101) |  |  | broadly hurt for completed aligned adapters |
| MMLU macro | mean over MMLU subtasks | 0.692 | 0.599 (-0.093) | 0.600 (-0.092) |  |  | broadly hurt; not only a task-size effect |

## Taxonomy by Skill Family

Skill-family means are from
`results/sft_eval/wide_bench/skill_diagnostics/skill_summary.csv`.
The MMLU aggregate rows are excluded from family means to avoid double counting
subtasks.

| Skill family | Benchmarks or subtasks | What the family tests | lowLR | lowrank | taskvec_a0p25 | scrambled | Read |
|---|---|---|---:|---:|---:|---:|---|
| semantic coherence / human similarity | generation coherence, THINGS triplet R2, THINGS odd-one-out, external similarity datasets | smooth concept similarity and human-like semantic geometry | boosted | boosted | boosted | hurt | the intended skill is clearly improved in aligned arms |
| script / discourse commonsense | PIQA, CommonsenseQA, HellaSwag, WinoGrande | broad affordance, event, and discourse plausibility | -0.008 stable | -0.018 stable | -0.025 hurt | -0.370 hurt | mostly preserved when training is coherent; not preserved by scrambled perturbation |
| science / common-sense reasoning | ARC-Easy, ARC-Challenge, OpenBookQA | fact retrieval plus multiple-choice science ranking | -0.109 hurt | -0.123 hurt | -0.062 hurt | -0.392 hurt | consistently hurt; task-vector is a partial mitigation |
| lexical / disambiguation | WiC | context-specific word-sense boundary decisions | -0.152 hurt | -0.154 hurt | -0.150 hurt | -0.171 hurt | near-chance failure across all modified arms |
| truthfulness / calibration | TruthfulQA-MC2 | allocating probability mass to truthful answers over plausible false lures | -0.017 stable | -0.010 stable | -0.028 hurt | -0.068 hurt | full score is stable for lowLR/lowrank but mechanism is fragile |
| MMLU science / biomedical | 14 MMLU science and medical subjects | specialized factual recall and technical option ranking | -0.106 hurt | -0.111 hurt |  |  | broad science/biomed loss |
| MMLU quantitative / formal | 7 math, statistics, logic subjects | formal reasoning and exact-symbolic distinctions | -0.097 hurt | -0.084 hurt |  |  | formal logic is a major outlier |
| MMLU technical / computing | 5 CS, security, EE, ML subjects | technical factual recall and concept boundaries | -0.078 hurt | -0.099 hurt |  |  | consistent technical loss |
| MMLU factual / history / policy | 11 history, geography, law, policy, religion subjects | factual retrieval and close option ranking | -0.078 hurt | -0.075 hurt |  |  | all listed subtasks hurt |
| MMLU professional / law / moral | 11 law, medicine, psychology, moral/professional subjects | long-option professional judgment, norms, and exception handling | -0.093 hurt | -0.086 hurt |  |  | moral scenarios dominate worst drops |
| MMLU social / behavioral / business | 8 economics, psychology, sociology, marketing subjects | social-science factual recall and close distractors | -0.095 hurt | -0.092 hurt |  |  | consistent social/behavioral loss |
| MMLU miscellaneous / broad | MMLU miscellaneous | broad mixed factual retrieval | -0.075 hurt | -0.080 hurt |  |  | hurt |

## What Improved

The robust gain is the intended coherence/semantic representation skill, not
general benchmark accuracy.

From `semantic_human_summary.csv`:

| Arm | Generation coherence | Delta | THINGS human triplet R2 | Delta | Verdict |
|---|---:|---:|---:|---:|---|
| base | 0.323 | +0.000 | 0.468 | +0.000 | reference |
| lowLR | 0.757 | +0.435 | 0.651 | +0.182 | boosted |
| lowrank | 0.749 | +0.427 | 0.666 | +0.198 | boosted |
| taskvec_a0p25 | 0.649 | +0.327 | 0.599 | +0.131 | boosted |
| scrambled | 0.031 | -0.291 | 0.016 | -0.452 | hurt |

Measured interpretation: aligned SFT and task-vector extraction induce a much
more human-like semantic geometry. The scrambled-label control moves in the
opposite direction, so this is not a generic LoRA/SFT side effect.

Hypothesis: the gain transfers best to tasks that can be solved by broad
semantic association or event plausibility, which is why HellaSwag and
WinoGrande are relatively stable for coherent lowrank/task-vector states.

## What Stayed Stable

The stable cells matter because they show the intervention is not simply
breaking all lm-eval tasks.

- HellaSwag is stable for `lowrank` (`-0.002`) and `taskvec_a0p25` (`-0.005`),
  but scrambled SFT drops by `-0.398`.
- WinoGrande is stable for `lowLR` (`+0.003`), `lowrank` (`-0.018`), and
  `taskvec_a0p25` (`-0.015`).
- PIQA is close to stable for `taskvec_a0p25` (`-0.010`), while scrambled drops
  by `-0.259`.
- Full TruthfulQA-MC2 is stable for `lowLR` (`-0.017`) and `lowrank`
  (`-0.010`) under the existing threshold, though not for task-vector or
  scrambled.

Hypothesis: coherent semantic training preserves broad plausibility and
coreference more than it preserves factual/exam option ranking. The scrambled
control shows that preserving those cells depends on the learned semantic
direction, not merely on running a comparable adapter.

## What Dropped

The largest and most reliable drops concentrate in tasks that need sharp
decision boundaries or calibrated multiple-choice ranking.

- WiC is the cleanest drop: base is `0.652`, while `lowLR`, `lowrank`,
  `taskvec_a0p25`, and scrambled are all around `0.48-0.50`.
- ARC/OpenBookQA are consistently hurt. `taskvec_a0p25` improves over lowrank
  on ARC-Easy and ARC-Challenge, but remains below base.
- MMLU is broadly hurt for completed aligned adapters: micro deltas are
  `-0.099` for `lowLR` and `-0.101` for `lowrank`; macro deltas are
  `-0.093` and `-0.092`.
- MMLU worst drops from `mmlu_extreme_drops.csv` include:
  - `lowrank` moral scenarios: `0.570 -> 0.253`, delta `-0.317`.
  - `lowLR` moral scenarios: `0.570 -> 0.322`, delta `-0.248`.
  - `lowrank` medical genetics: `0.820 -> 0.620`, delta `-0.200`.
  - `lowrank` formal logic: `0.579 -> 0.381`, delta `-0.198`.
  - `lowLR` formal logic: `0.579 -> 0.389`, delta `-0.190`.

Hypothesis: these failures are compatible with a "semantic smoothing" story:
the adapter improves similarity geometry but makes semantically close options
and same-lemma senses too competitive. The same results are also compatible
with some generic answer-surface perturbation, because scrambled SFT causes
large drops on ARC, HellaSwag, CommonsenseQA, WiC, and TruthfulQA.

## Why TruthfulQA Is Especially Diagnostic

TruthfulQA is not the largest full-score drop for `lowLR` or `lowrank`.
The full MC2 deltas are stable for those two arms: `lowLR -0.017` and
`lowrank -0.010`. It is "especially affected" in a different and more useful sense:
it exposes whether the intervention changes relative probability mass between
truthful answers and attractive false lures.

Measured facts from `truthfulqa_mechanism.csv` and
`wide_bench_diagnostics/truthfulqa_analysis/REPORT.md`:

| Arm | Full MC2 | Full delta | Diagnostic MC2, n=200 | Diagnostic delta | Truth-logodds delta | True mass delta | False-pressure delta | False pressure up frac |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.550 | +0.000 | 0.522 |  |  |  |  |  |
| lowLR | 0.533 | -0.017 |  |  |  |  |  |  |
| lowrank | 0.541 | -0.010 | 0.553 | +0.030 | +0.395 | -3.544 | -3.939 | 0.335 |
| taskvec_a0p25 | 0.523 | -0.028 | 0.527 | +0.004 | -0.624 | +4.303 | +4.927 | 0.840 |
| scrambled | 0.482 | -0.068 | 0.470 | -0.052 | +3.082 | -59.027 | -62.109 | 0.000 |

Measured interpretation:

- `lowrank` improves the bounded 200-item slice because it lowers both true and
  false likelihood mass, but lowers false-answer pressure more than truthful
  mass.
- `taskvec_a0p25` is almost flat on the bounded slice by MC2, but it has the
  bad pressure signature: true mass rises, false pressure rises more, and false
  pressure rises on `84%` of paired items.
- Scrambled SFT drops the full score and the slice. Its truth-logodds increase
  is not a useful truthfulness gain because it suppresses both true and false
  likelihoods extremely hard and produces item-level flips.

Why this benchmark is sensitive:

- TruthfulQA MC2 rewards relative mass on sets of truthful answers, not just
  selecting a single plausible answer.
- Many false options are semantically close to the topic and sound culturally
  familiar. Local worst-lure examples include defibrillation for flatline,
  washing chicken, Spanish as the language of all Latin Americans/Latinos,
  voodoo-doll curses, Agenda 21, and similar misconception lures.
- Hypothesis: a semantic-coherence objective can make related statements more
  mutually attractive. That is useful for human-like similarity judgments, but
  harmful when the related statement is a misconception that must be actively
  rejected.
- Hypothesis: task-vector addition may preserve or amplify a direction that
  raises semantic relatedness without preserving the base model's calibrated
  false-lure suppression. This would explain why `taskvec_a0p25` is worse than
  lowrank on TruthfulQA full MC2 despite helping ARC relative to lowrank.

Therefore, TruthfulQA should be treated as a calibration and lure-pressure
diagnostic, not merely as a single benchmark score.

## Working Mechanisms to Track

These are hypotheses to test, ordered by how directly the current artifacts
support them.

1. Semantic smoothing vs sharp boundaries.
   Evidence: strong semantic/human gains, WiC near-chance drops, MMLU formal
   logic and moral-scenario losses. Hypothesis: smooth similarity geometry
   weakens local distinctions needed for sense disambiguation, formal rules,
   and exception-heavy domains.

2. Multiple-choice answer-ranking calibration damage.
   Evidence: ARC/OpenBookQA/MMLU drops and TruthfulQA false-pressure changes.
   Hypothesis: semantically plausible distractors become too competitive even
   when the underlying topic knowledge is not erased.

3. Generic adapter/output-surface perturbation.
   Evidence: scrambled SFT collapses ARC, HellaSwag, CommonsenseQA, PIQA, WiC,
   and TruthfulQA. Hypothesis: some benchmark loss is due to LoRA/SFT
   perturbing the answer surface independent of the useful semantic skill.

4. Dose/alpha tradeoff.
   Evidence: `taskvec_a0p25` keeps much of the semantic gain and mitigates ARC
   relative to lowrank, but TruthfulQA pressure worsens and WiC remains hurt.
   Hypothesis: there may be a lower alpha that retains enough representation
   benefit while reducing false-lure and WiC/MMLU damage.

## Mitigation and Evaluation Follow-Ups

Priority follow-ups:

1. Treat completed `taskvec_a0p25` MMLU as partial mitigation, not broad
   retention. It improves micro accuracy by `+0.031` over lowrank but remains
   `-0.070` below base, so follow-up should focus on the residual worst
   subjects: moral scenarios, formal logic, medical genetics, nutrition,
   professional psychology, and high-school statistics.

2. Add a cheap failure-suite before any more full MMLU runs. Include
   `mmlu_moral_scenarios`, `mmlu_formal_logic`, `mmlu_medical_genetics`,
   `mmlu_nutrition`, `mmlu_professional_psychology`,
   `mmlu_high_school_statistics`, ARC-Easy, ARC-Challenge, OpenBookQA, WiC,
   TruthfulQA-MC2 with log samples, plus HellaSwag/WinoGrande/PIQA as
   preservation controls.

3. Track margins, not only accuracy. For MMLU/ARC/OpenBookQA, log
   correct-choice margin and best-distractor pressure. For TruthfulQA, keep
   `delta_true_logsumexp`, `delta_false_logsumexp`,
   `delta_truth_logodds`, and `frac_false_pressure_up`. For WiC, slice by part
   of speech, lemma, and same/different label bias.

4. Sweep task-vector alpha on the failure suite before expensive full runs.
   Candidate grid from the local mitigation plan: `0`, `0.1`, `0.2`, `0.25`,
   `0.3`, `0.5`, `1.0`. Require both a semantic metric and failure-suite
   metric for each alpha.

5. Try retention replay or KL regularization targeted to the failure families.
   The replay/KL set should include science retrieval, WiC-style sense
   contrasts, TruthfulQA false lures, and MMLU moral/professional/formal
   examples. Evaluate against both semantic gains and failure-family deltas.

6. Separate representation use from answer-generation use. The adapter may be
   valuable for hidden-state or fMRI representation experiments even if it
   should be disabled, downweighted, or KL-constrained for calibrated
   multiple-choice answering.

## Claims Not Made

- This report does not claim that `taskvec_a0p25` fixes MMLU, because the
  comparable aggregate MMLU row is not present in the current diagnostics.
- This report does not claim lowrank causes a large full TruthfulQA drop. The
  full lowrank delta is stable by the existing threshold. The claim is that
  TruthfulQA reveals a calibration mechanism that becomes worse for
  `taskvec_a0p25` and scrambled SFT.
- This report does not separate semantic-objective harm from generic adapter
  perturbation as a solved causal question. Scrambled controls show perturbation
  matters; aligned-vs-scrambled differences show the learned semantic direction
  also matters.

## Refresh Checklist

When new benchmark rows land, update or regenerate the upstream diagnostics
first, then refresh this report:

1. Update `results/sft_eval/wide_bench/summary.csv` or the relevant split-run
   JSONs.
2. Regenerate or inspect `results/sft_eval/wide_bench/skill_diagnostics/*.csv`.
3. Check whether `taskvec_a0p25` MMLU is present.
4. Recompute the score-status matrix using the same `0.02` threshold.
5. Keep TruthfulQA full MC2 and log-sample pressure metrics separate.
