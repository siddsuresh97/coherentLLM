# Concept Steering Next-Experiment Plan

Status: prepared after the passing bounded sweep `5513297` and targeted
qualitative follow-up `5513347`.

## Current Evidence

- `coherence` vector: best target movement was layer `12`, alpha `4`
  (`positive_preference=0.625`, target margin `+1.0222` vs alpha `0`).
- `human_alignment` vector: best target movement was layer `24`, alpha `4`
  (`positive_preference=0.875`, target margin `+0.1197` vs alpha `0`).
- Retention forced-choice preference stayed in `0.8-1.0`, but layer-12
  aggressive settings reduced retention margins.
- Specificity is imperfect: `human_alignment` layer `12`, alpha `4` strongly
  moved the coherence eval, so layer 12 may be a shared semantic/task-quality
  direction rather than an alignment-specific direction.
- Qualitative `5513347` completed all 112 generations but returned wrapper
  status `1` because of a late `SUMMARY.md` writer bug. The script is patched,
  and the existing generations were rescored in `chtc/5513347/rescored_v2/`.
- In the rescored qualitative check, `coherence_l16_a4` was the cleanest
  generated-text candidate: coherence pass rate `1.00` vs baseline `0.75`,
  with alignment and retention both `1.00`.
- `human_alignment_l24_a4` and `human_alignment_l16_a2` preserved alignment and
  retention, but the baseline already passed all 4 alignment probes after the
  rubric fix, so this run does not establish a generation-level alignment gain.
- `human_alignment_l12_a4` and `human_alignment_l12_a-4` are failure modes for
  alignment generation.

## Immediate Qualitative Check

The first qualitative check used `src/sft/run_concept_steering_qualitative.py`
to generate short answers and deterministically judge:

- baseline,
- `coherence` layer `12` alpha `2` and `4`,
- `coherence` layer `16` alpha `4`,
- `human_alignment` layer `24` alpha `4`,
- `human_alignment` layer `16` alpha `2`,
- `human_alignment` layer `12` alpha `4` and `-4`.

The run found no retention failures, one useful coherence generation candidate
(`coherence_l16_a4`), and clear alignment damage from layer-12
human-alignment steering.

## Ablation Sweep

Next bounded sweep after the qualitative check:

- coherence vector: layers `10,12,14,16`, alphas `0,1,2,3,4`.
- human-alignment vector: layers `16,20,24,28`, alphas `0,1,2,3,4`.
- negative-control alphas: include `-2` and `-4` only for the best layer from
  each vector, to confirm directionality without spending the full grid.
- extraction ablation: repeat the top two settings with last-token extraction
  and answer-span mean extraction if runtime allows.

## Retention Gates

Do not advance a setting to broad use unless all gates pass:

- forced-choice retention `positive_preference >= alpha0 - 0.05`.
- forced-choice retention `mean_delta_logprob >= alpha0 - 1.0`.
- qualitative retention pass rate is no worse than baseline by more than one
  item on the 6-item targeted check.
- no repeated formatting/instruction-following failures in generated text.

The current sweep already flags `coherence L12 a4`,
`human_alignment L12 a4`, and `human_alignment L12 a-4` as retention-risk
settings because their retention margins drop by more than `2.4`.

## Failure-Suite Guards

Before reporting any steering setting as useful, run a small failure suite with:

- semantic similarity and odd-one-out prompts for coherence,
- refusal, privacy, medical-urgency, and uncertainty prompts for alignment,
- arithmetic, geography, science, exact-repeat, pluralization, and syllogism
  retention prompts,
- concise generation settings, deterministic decoding, and saved raw JSONL.

Escalate only settings that improve the target suite while passing retention
gates. Treat layer-12 cross-effects as a separate mechanistic finding until a
more specific human-alignment vector is demonstrated.

## Scale-Up Candidate

- preferred coherence candidate: `coherence L16 a4`; use `coherence L12 a4`
  only as a high-effect/high-risk diagnostic because the qualitative check did
  not show a generation win and the sweep showed retention-margin loss.
- preferred human-alignment candidates for further testing:
  `human_alignment L24 a4` and `human_alignment L16 a2`, but only on a harder
  alignment suite because the current qualitative probes saturate at baseline.
- next runtime target: a broader judged generation suite with 25-50 prompts per
  category and a cheap lm-eval retention slice before any full benchmark run.
- do not submit a broad steering scale-up until that suite exists; the current
  artifacts justify suite construction, not a larger blind GPU sweep.
