# Benchmark Attribution and Mitigation Report

Report date: 2026-07-08 America/Chicago, UTC date 2026-07-09.
Branch: `coherence-sft`.

Scope: existing artifacts only. This lane did not rerun MMLU or launch new
benchmarks. The comparable `taskvec_a0p25` aggregate MMLU result is still owned
by Sagan's active MMLU lane and is treated here as pending.

## Tracking

Derived tables:

- `attribution/headline_task_attribution.csv`
- `attribution/skill_family_attribution.csv`
- `attribution/semantic_target_gains.csv`
- `attribution/truthfulqa_attribution.csv`
- `attribution/mitigation_targets.csv`
- `attribution/metadata.json`

Primary input artifacts:

- `summary.csv`
- `raw_task_summary.csv`
- `skill_diagnostics/task_skill_deltas.csv`
- `skill_diagnostics/skill_summary.csv`
- `skill_diagnostics/semantic_human_summary.csv`
- `skill_diagnostics/truthfulqa_mechanism.csv`
- `skill_diagnostics/mmlu_extreme_drops.csv`

## Executive Answer

The induced skill is not broad benchmark accuracy. It is semantic-coherence and
human-similarity alignment. That target skill is strongly boosted, while
standard lm-eval tasks split by how much they depend on smooth semantic
association versus sharp answer ranking.

Boosted:

- Generation coherence: base `0.323`, lowLR `0.757`, lowrank `0.749`,
  `taskvec_a0p25` `0.649`.
- THINGS human triplet R2: base `0.468`, lowLR `0.651`, lowrank `0.666`,
  `taskvec_a0p25` `0.599`.
- THINGS odd-one-out and triplet similarity are also boosted for lowLR/lowrank.
- Only two completed MMLU subject rows clear the `+0.02` boost threshold
  (`lowrank` college mathematics and management), so they are local exceptions,
  not a general capability gain.

Mostly preserved:

- HellaSwag is stable for coherent lowrank/task-vector states
  (`-0.002`, `-0.005`) but scrambled drops by `-0.398`.
- WinoGrande is stable for lowLR, lowrank, and `taskvec_a0p25`
  (`+0.003`, `-0.018`, `-0.015`).
- PIQA is close to stable for `taskvec_a0p25` (`-0.010`), while scrambled drops
  by `-0.259`.

Hurt:

- WiC is near chance for all modified arms: lowLR `-0.152`, lowrank `-0.154`,
  `taskvec_a0p25` `-0.150`, scrambled `-0.171`.
- ARC/OpenBookQA are hurt, but task-vector partially mitigates lowrank:
  ARC-Easy improves by `+0.103` over lowrank, ARC-Challenge by `+0.051`,
  OpenBookQA by `+0.028`.
- Completed lowLR/lowrank MMLU is broadly hurt: micro deltas `-0.099` and
  `-0.101`; macro deltas `-0.093` and `-0.092`.
- TruthfulQA full MC2 is stable for lowLR/lowrank but hurt for
  `taskvec_a0p25` (`-0.028`) and scrambled (`-0.068`). Its mechanism is
  calibration pressure, not simply lost truth knowledge.

## Skill Attribution

| Skill family | Current status | Why |
|---|---|---|
| Semantic coherence / human similarity | Boosted | This is the directly trained/induced skill; coherent arms strongly improve generation coherence, THINGS human R2, and human-similarity tasks. |
| Script / discourse commonsense | Mostly preserved | HellaSwag, WinoGrande, and PIQA overlap with broad semantic/event plausibility. Coherent states preserve them; scrambled SFT does not. |
| Science facts and option ranking | Hurt, partly recoverable | ARC/OpenBookQA need facts plus close distractor ranking. Task-vector helps relative to lowrank but remains below base. |
| Lexical disambiguation | Hurt | WiC needs context-specific word-sense boundaries; smooth semantic association appears antagonistic. |
| Truthfulness calibration | Fragile | False answers are often semantically related misconceptions, so the induced semantic direction can raise false-lure pressure. |
| MMLU factual/formal/professional skills | Broadly hurt | LowLR/lowrank drops span all MMLU families, with worst drops in moral/professional/formal/biomedical subjects. |

The similarity story is therefore not "coherence helps tasks that are semantic."
It helps or preserves tasks where broad semantic plausibility is enough. It hurts
tasks where the benchmark demands a boundary: exact sense, exact fact, exact
formal answer, or rejection of a plausible but false statement.

## Headline Task Matrix

| Task | Base | lowrank delta | taskvec delta | scrambled delta | Attribution |
|---|---:|---:|---:|---:|---|
| PIQA | 0.799 | -0.023 | -0.010 | -0.259 | affordance semantics mostly preserved |
| OpenBookQA | 0.490 | -0.082 | -0.054 | -0.208 | science fact/ranking hurt; task-vector helps |
| CommonsenseQA | 0.651 | -0.029 | -0.070 | -0.454 | mixed associative commonsense |
| WiC | 0.652 | -0.154 | -0.150 | -0.171 | lexical sense-boundary failure |
| TruthfulQA-MC2 | 0.550 | -0.010 | -0.028 | -0.068 | false-lure calibration pressure |
| WinoGrande | 0.762 | -0.018 | -0.015 |  | discourse/coreference preserved |
| ARC-Easy | 0.850 | -0.145 | -0.042 | -0.544 | science ranking hurt; task-vector recovers much of lowrank loss |
| ARC-Challenge | 0.649 | -0.141 | -0.090 | -0.424 | harder science ranking hurt |
| HellaSwag | 0.685 | -0.002 | -0.005 | -0.398 | script/event plausibility preserved |
| MMLU micro | 0.693 | -0.101 | pending |  | broad exam knowledge/ranking hurt for completed adapters |

Across the nine non-MMLU headline tasks, lowLR, lowrank, and `taskvec_a0p25`
each have six hurt and three stable cells under the current threshold. The
important difference is which cells move: `taskvec_a0p25` is a relative
mitigation for ARC/OpenBookQA/PIQA but is worse than lowrank on CommonsenseQA
and TruthfulQA.

## Why TruthfulQA Is Sensitive

TruthfulQA-MC2 is useful because it measures relative probability mass over
truthful answers versus plausible false answers. Many false options are
semantically close to the question topic: myths, stereotypes, safety folklore,
conspiracy claims, and overgeneralized cultural facts. A semantic-coherence
direction can make these related lures more competitive.

Measured log-sample readout:

| Arm | Full MC2 delta | Diagnostic MC2 delta | Truth-logodds delta | True-mass delta | False-pressure delta | False pressure up |
|---|---:|---:|---:|---:|---:|---:|
| lowrank | -0.010 | +0.030 | +0.395 | -3.544 | -3.939 | 0.335 |
| taskvec_a0p25 | -0.028 | +0.004 | -0.624 | +4.303 | +4.927 | 0.840 |
| scrambled | -0.068 | -0.052 | +3.082 | -59.027 | -62.109 | 0.000 |

Interpretation:

- lowrank is stable because it lowers false-answer pressure more than truthful
  mass on the 200-item slice.
- `taskvec_a0p25` looks almost flat on diagnostic MC2 but has the wrong
  mechanism: true mass rises, false pressure rises more, and false pressure
  rises on `84%` of paired items.
- Scrambled is a control failure: it suppresses both true and false likelihoods
  extremely hard and causes item-level instability. It is not a useful
  truthfulness gain.

So TruthfulQA should be optimized as a false-lure calibration problem. Do not
use aggregate MC2 alone as the mitigation target.

## Why Gains Appear Where They Do

The strongest gains are on representation-level measures because those are
closest to the training signal: agreement among semantic views, human-like
THINGS similarity, and external semantic similarity judgments.

HellaSwag/WinoGrande/PIQA are preserved because their answer choices often rely
on broad plausibility, event flow, affordances, and discourse coherence. Those
skills are adjacent to the induced semantic geometry.

ARC/OpenBookQA improve under task-vector relative to lowrank because the
semantic direction seems less damaging than a full lowrank adapter for some
science commonsense questions. They still remain below base because the tasks
also require factual retrieval and calibrated option ranking.

WiC, MMLU, and TruthfulQA are hurt because their success criteria conflict with
semantic smoothing:

- WiC: separate same-word senses in local context.
- MMLU: choose among close technical, moral, formal, or professional options.
- TruthfulQA: reject attractive false statements even when they are topically
  related.

## Mitigation Plan

1. Ingest the pending `taskvec_a0p25` MMLU aggregate from Sagan's lane. Do not
   rerun MMLU here. Pass signal: MMLU micro delta materially better than
   lowrank, target `>= -0.06`.
2. Build a cheap failure-suite gate before new full benchmarks:
   `mmlu_moral_scenarios`, `mmlu_formal_logic`, `mmlu_medical_genetics`,
   `mmlu_nutrition`, `mmlu_professional_psychology`,
   `mmlu_high_school_statistics`, ARC, OpenBookQA, WiC, and TruthfulQA
   log-samples.
3. Add item-level margin diagnostics:
   correct-choice margin, best-distractor margin, false-pressure logsumexp,
   flip type, and lure class.
4. Run a task-vector alpha sweep only on the cheap suite first:
   `0`, `0.1`, `0.2`, `0.25`, `0.3`, `0.5`. Pass signal: generation coherence
   `>= 0.60`, human R2 `>= 0.58`, TruthfulQA false-pressure-up `<= 0.60`, and
   MMLU failure-slice mean delta `>= -0.06`.
5. If alpha cannot solve it, train or select with retention replay:
   TruthfulQA false-lure calibration, WiC sense contrasts, ARC/OpenBookQA
   science margins, and MMLU moral/formal/biomedical close-option examples.
6. Use selective routing when appropriate. The semantic adapter/task-vector may
   be valuable for representation extraction, fMRI, and semantic-hub tests even
   if it should be downweighted or disabled for calibrated multiple-choice
   answering.

## Current Decision

Treat `taskvec_a0p25` as a useful representation/human-alignment arm and a
partial ARC/OpenBookQA mitigation, not yet as a broad benchmark mitigation.
The next decisive result is the pending task-vector MMLU aggregate. The next
cheap mechanistic result should be the failure-suite margin/log-sample run, not
another full benchmark sweep.
