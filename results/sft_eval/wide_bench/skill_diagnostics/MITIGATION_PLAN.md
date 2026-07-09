# Benchmark-Drop Mitigation and Mechanism Plan

Last updated: 2026-07-08 on branch `coherence-sft`.

Scope: this plan extends the existing skill diagnostics into an experiment queue. It is based only on existing artifacts in this repo; no expensive jobs were launched while writing it.

Primary inputs:

- `results/sft_eval/wide_bench/skill_diagnostics/REPORT.md`
- `results/sft_eval/wide_bench/skill_diagnostics/*.csv`
- `results/sft_eval/wide_bench/skill_map.md`
- `README.md`
- `research/EXPERIMENT_LOG.md`

## Current Read

The coherence intervention is not a broad benchmark improvement. It is a targeted semantic-representation intervention with uneven downstream effects.

Boosted skill:

- Representation-level semantic geometry and human similarity alignment are the clear gains.
- `lowLR` raises generation coherence from `0.323` to `0.757` and THINGS human triplet R2 from `0.468` to `0.651`.
- `lowrank` raises generation coherence to `0.749` and human R2 to `0.666`.
- `taskvec_a0p25` recovers a large part of that effect without retraining: generation coherence `0.649`, human R2 `0.599`.

Mostly preserved skills:

- HellaSwag script/event plausibility is preserved by coherent lowrank/task-vector states: `lowrank` delta `-0.002`, `taskvec_a0p25` delta `-0.005`.
- WinoGrande discourse/coreference is preserved: `lowLR` `+0.003`, `lowrank` `-0.018`, `taskvec_a0p25` `-0.015`.
- PIQA physical-affordance commonsense is close to flat under task-vector mitigation: `taskvec_a0p25` delta `-0.010`.
- These preserved cells are not trivial benchmark insensitivity: scrambled SFT drops HellaSwag by `-0.398`, PIQA by `-0.259`, and CommonsenseQA by `-0.454`.

Hurt skills:

- WiC word-sense disambiguation is the cleanest failure: all aligned and scrambled states are near chance, with deltas around `-0.15` to `-0.17`.
- ARC/OpenBookQA science and fact-plus-reasoning are hurt. Task-vector alpha `0.25` mitigates ARC versus lowrank but remains below base: ARC-Easy `0.808` vs base `0.850`, ARC-Challenge `0.559` vs base `0.649`.
- MMLU broad exam knowledge is hurt for the completed aligned adapters: lowLR micro `-0.099`, lowrank micro `-0.101`. The damage is broad across MMLU families, not a single subject.
- MMLU worst drops concentrate in moral/professional/formal/biomedical subjects: lowrank moral scenarios `-0.317`, medical genetics `-0.200`, formal logic `-0.198`; lowLR moral scenarios `-0.248`, formal logic `-0.190`, human sexuality `-0.183`.
- TruthfulQA is not a large full-benchmark failure for lowLR or lowrank, but it exposes a calibration mechanism that matters for task vectors.

TruthfulQA mechanism:

- Full MC2 deltas: lowLR `-0.017`, lowrank `-0.010`, taskvec `-0.028`, scrambled `-0.068`.
- On the 200-item log-sample diagnostic, lowrank improves MC2 by `+0.030` because false-answer pressure falls more than truthful-answer mass.
- On that same slice, `taskvec_a0p25` is nearly flat in MC2 (`+0.004`) but has a bad pressure pattern: true mass rises `+4.303`, false pressure rises `+4.927`, and false pressure rises on `84%` of paired items.
- The likely TruthfulQA failure is not erased truth knowledge. It is plausible-false lure calibration: semantically related misconceptions become too competitive.

Unsettled cell:

- The comparable `taskvec_a0p25` MMLU aggregate row is still pending in the existing diagnostics. Do not claim task-vector fixes broad exam knowledge until that row exists.

## Working Mechanisms

These mechanisms are deliberately separable so future runs can falsify them.

1. Semantic smoothing vs sharp boundaries.
   The intervention improves smooth concept similarity but may blur local distinctions needed for WiC, formal logic, moral scenarios, and close-option MMLU questions.

2. Answer-ranking calibration damage.
   TruthfulQA and multiple-choice MMLU depend on relative likelihood among plausible alternatives. A coherence objective can raise related distractors even when it preserves or improves semantic representations.

3. Generic adapter/output-surface perturbation.
   Scrambled SFT damages ARC, HellaSwag, PIQA, CommonsenseQA, WiC, and TruthfulQA. Some loss is therefore generic LoRA/SFT perturbation, not only the intended semantic skill.

4. Strength/dose tradeoff.
   Task-vector alpha `0.25` keeps much of the semantic gain and mitigates ARC versus lowrank, but TruthfulQA pressure worsens. There may be an alpha elbow below `0.25` or between `0.25` and `0.5`.

5. Site/layer specificity.
   Prior broad activation steering was unstable, but the newer CAA scripts allow single-layer forced-choice probes. If semantic directions help only at specific layers/sites, we should use them for representation extraction or selective inference, not globally during benchmark answering.

## Execution Rules

- Do not run full MMLU until a smaller failure slice has already answered a concrete question.
- Every GPU run must have a local smoke or small slice first.
- Keep `base`, the aligned arm, and `scrambled` or another perturbation control in the same metric table whenever possible.
- Use exact same few-shot counts, chat template, max length, and backend when comparing arms.
- For CHTC Llama jobs, require high-memory GPUs with `TARGET.CUDAGlobalMemoryMb >= 40000` and record cluster ID, submit file, container image, and staged model/adapter paths.
- Each experiment output directory should contain `COMMANDS.md`, `metadata.json`, raw outputs or symlinks to raw outputs, a small `summary.csv`, and a short `REPORT.md`.

## Priority Queue

| Priority | Experiment | Main question | GPU cost | Run when | Expected outputs | Pass signal | Stop/fail signal |
|---|---|---|---|---|---|---|---|
| P0 | Finish `taskvec_a0p25` MMLU and refresh diagnostics | Does alpha `0.25` mitigate broad MMLU like it mitigates ARC? | High, already active/scale-out | Before interpreting task-vector retention | Merged MMLU JSON/CSV plus refreshed skill diagnostics | MMLU micro delta improves clearly over lowrank, target `>= -0.06`; worst MMLU slice no longer has catastrophic `-0.20` to `-0.32` drops | MMLU delta remains near lowrank `-0.10`; task-vector is not a broad retention fix |
| P1 | Failure-slice mechanism suite | Are drops margin/calibration failures, sense-boundary failures, or generic perturbation? | Low to medium | Immediately after current long jobs free a GPU | Logsample/margin CSVs for MMLU slice, ARC/OpenBookQA, WiC, TruthfulQA | One dominant measurable mechanism explains drops and points to a mitigation | Slices disagree or reproduce no drop; broaden diagnostics before training |
| P2 | Task-vector alpha sweep on failure suite | Is there a better alpha than `0.25` for semantic gain with retention? | Medium | After P1 defines metrics | Per-alpha failure-suite scores plus semantic gain table | Find an alpha with semantic gain retained and failure metrics closer to base | All useful alphas either lose semantics or keep TruthfulQA/MMLU damage |
| P3 | Adapter interpolation and scaling controls | Is lowrank/lowLR damage due to adapter strength, rank, or learned direction? | Medium | After P2 if alpha alone is inconclusive | New scaled/interpolated adapter dirs and retention/semantic summaries | Smooth dose-response or a better Pareto arm emerges | Interpolation behaves like generic perturbation or loses semantic gains |
| P4 | Selective steering and ablation | Can the semantic direction be used without globally damaging answer ranking? | Low for forced-choice, medium for lm-eval slice | After CAA smoke passes | `concept_steering` sweep results; optional ablation slice | Specific layer/alpha improves target probes with retention stable, or ablation restores failure slices | Steering causes repetition/degradation or no measurable target shift |
| P5 | Data curriculum variants | Can training learn semantic geometry while preserving sharp option ranking? | High | Only after P1-P4 identify a target mechanism | New adapters, training logs, semantic and failure-suite reports | Coherence/human gains preserved while WiC/TruthfulQA/MMLU slices improve | Full retraining repeats lowrank/lowLR tradeoff |
| P6 | Benchmark-specific controls | Are drops artifacts of prompt/backend/context/truncation? | Low to medium | Interleave with P1/P2 before publishing claims | Control matrix and reproducibility report | Controls reproduce base and aligned deltas under matched settings | Deltas vanish under a control, forcing reinterpretation as eval artifact |

## P0: Finish Missing Task-Vector MMLU

Purpose: decide whether `taskvec_a0p25` generalizes from ARC mitigation to MMLU retention.

Current facts:

- Completed base/lowLR/lowrank MMLU exists in `results/sft_eval/wide_bench/summary.csv`.
- `taskvec_a0p25` has completed zero-shot, WinoGrande, ARC, and HellaSwag rows, but aggregate MMLU is missing from `skill_diagnostics/REPORT.md`.
- Local split MMLU and CHTC MMLU shard tooling are described in `README.md`, `research/STATUS.md`, and `chtc/mmlu_shards/README.md`.

Run pattern:

```bash
python src/sft/merge_mmlu_shards.py \
  --inputs results/sft_eval/wide_bench/chtc_mmlu_shards/<run_id>/extracted/* \
  --state-name taskvec_a0p25 \
  --out-dir results/sft_eval/wide_bench/chtc_mmlu_shards/<run_id>/merged

python src/sft/analyze_skill_diagnostics.py
```

Expected outputs:

- `results/sft_eval/wide_bench/chtc_mmlu_shards/<run_id>/merged/results_merged.json`
- `results/sft_eval/wide_bench/chtc_mmlu_shards/<run_id>/merged/raw_task_summary.csv`
- Updated `results/sft_eval/wide_bench/skill_diagnostics/REPORT.md`
- Updated `task_skill_deltas.csv`, `skill_summary.csv`, and `mmlu_extreme_drops.csv`

Pass criteria:

- `taskvec_a0p25` MMLU aggregate delta is materially better than lowrank, ideally `>= -0.06`.
- Worst subject drops are reduced relative to lowrank in moral scenarios, formal logic, and medical genetics.
- Semantic/human metrics for `taskvec_a0p25` remain as already measured: generation coherence around `0.649`, human R2 around `0.599`.

Fail criteria:

- `taskvec_a0p25` MMLU delta remains near lowrank/lowLR, around `-0.09` to `-0.10`.
- MMLU moral/formal/biomedical outliers remain catastrophic.
- If this fails, do not run full MMLU for more alphas until P1/P2 slice metrics identify a cheaper screening rule.

## P1: Failure-Slice Mechanism Suite

Purpose: replace aggregate benchmark scores with item-level diagnostics that identify why the model loses.

Arms:

- Required: `base`, `lowrank`, `taskvec_a0p25`, `scrambled`.
- Add `lowLR` when the cost is small.
- Add `taskvec_a0p5` only after the base four arms work.

Slices:

- MMLU: `mmlu_moral_scenarios`, `mmlu_formal_logic`, `mmlu_medical_genetics`, `mmlu_nutrition`, `mmlu_professional_psychology`, `mmlu_high_school_statistics`.
- Science/common-sense: `arc_easy`, `arc_challenge`, `openbookqa`.
- Lexical: `wic`.
- Calibration: `truthfulqa_mc2` with `--limit 200 --log_samples`, then full 817 only if the slice is informative.
- Preservation controls: `hellaswag`, `winogrande`, `piqa`.

Metrics:

- Accuracy or normalized accuracy using the same metric as wide-bench.
- Correct-choice margin: score(correct) minus best distractor.
- Distractor pressure: logsumexp over incorrect choices, especially semantically close distractors.
- Flip type: base-correct/arm-wrong, base-wrong/arm-correct, both-wrong.
- For TruthfulQA: `delta_true_logsumexp`, `delta_false_logsumexp`, `delta_truth_logodds`, `frac_false_pressure_up`, and worst lure text classes.
- For WiC: slice by POS, same-lemma semantic closeness, and whether failure is biased to "same meaning" or "different meaning".

Expected outputs:

- `results/sft_eval/wide_bench/skill_diagnostics/mechanism_suite/<run_id>/mmlu_slice_margins.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/mechanism_suite/<run_id>/truthfulqa_paired_deltas.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/mechanism_suite/<run_id>/wic_slices.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/mechanism_suite/<run_id>/arc_openbook_margins.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/mechanism_suite/<run_id>/summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/mechanism_suite/<run_id>/REPORT.md`

Implementation notes:

- `src/sft/analyze_truthfulqa_logsamples.py` already handles TruthfulQA log-sample decomposition.
- A future agent may need to add a small companion analyzer for MMLU/ARC/OpenBookQA log samples.
- Keep raw lm-eval JSON/logsample files in the run directory or link them from `results/sft_eval/wide_bench/runs/`.

Pass criteria:

- At least one mechanism has a clear quantitative signature. Examples:
  - MMLU and ARC drops are dominated by reduced correct-vs-distractor margin.
  - TruthfulQA task-vector damage is dominated by rising false pressure.
  - WiC errors are concentrated in high-similarity same-lemma contrasts.
- The mechanism signature distinguishes aligned adapters from scrambled. If scrambled and aligned have the same signature, prioritize perturbation controls before new semantic training.

Fail criteria:

- Aggregate accuracy drops are reproduced but item-level margins do not move.
- Slices do not match full benchmark direction.
- If this happens, rerun with matched backend/context controls before inferring cognition or training mechanisms.

## P2: Task-Vector Alpha Sweep

Purpose: find the Pareto elbow for semantic gain vs benchmark retention.

Alpha grid:

- Minimum grid: `0`, `0.1`, `0.2`, `0.25`, `0.3`, `0.5`, `1.0`.
- Existing reusable vLLM adapter dirs already cover `a0p25` and `a0p5`; `a1.0` is equivalent to the real adapter for some lanes.
- If `a0p1`, `a0p2`, or `a0p3` adapters do not exist, first create/stage vLLM-compatible scaled adapter dirs and record exactly how they were made.

Evaluation order:

1. Cheap semantic screen: reuse `results/sft_eval/steer/` style metrics or run the semantic/human summary path.
2. P1 failure suite at small limit.
3. Full zero-shot and ARC/HellaSwag groups only for alpha candidates that pass step 2.
4. Full MMLU only for the best one or two alpha candidates.

Expected outputs:

- `results/sft_eval/wide_bench/skill_diagnostics/alpha_sweep/<run_id>/adapter_manifest.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/alpha_sweep/<run_id>/semantic_summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/alpha_sweep/<run_id>/failure_suite_summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/alpha_sweep/<run_id>/pareto.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/alpha_sweep/<run_id>/REPORT.md`

Pass criteria:

- Candidate alpha keeps generation coherence `>= 0.60` and human R2 `>= 0.58`.
- Candidate alpha improves ARC/OpenBookQA relative to lowrank.
- Candidate alpha keeps TruthfulQA `frac_false_pressure_up <= 0.60` on the 200-item paired slice.
- Candidate alpha keeps MMLU failure-slice mean delta `>= -0.06` before any full MMLU run.

Fail criteria:

- Lower alphas protect retention but lose the intended semantic gain.
- Higher alphas keep semantic gain but worsen false pressure or MMLU slice drops.
- If no alpha passes, prefer training/data mitigations over further alpha search.

## P3: Adapter Interpolation and Scaling Controls

Purpose: test whether damage is caused by adapter magnitude/rank versus the learned semantic direction.

Variants:

- Scale lowrank adapter weights by `0.25`, `0.5`, `0.75`.
- Scale lowLR adapter weights by `0.25`, `0.5`, `0.75`.
- Interpolate `lowrank` and `taskvec_a0p25` if their adapter shapes are compatible.
- Include `scrambled` scaled by the same factors as a perturbation dose control.
- Include a rank/routing control if an existing lower-rank or rank-matched adapter is available.

Expected outputs:

- `results/sft_eval/wide_bench/skill_diagnostics/adapter_interpolation/<run_id>/adapter_manifest.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/adapter_interpolation/<run_id>/semantic_summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/adapter_interpolation/<run_id>/failure_suite_summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/adapter_interpolation/<run_id>/REPORT.md`

Pass criteria:

- A scaled/interpolated adapter lands on a better Pareto point than `taskvec_a0p25`: similar semantic gain, lower MMLU/TruthfulQA/WiC damage.
- Scrambled scaling shows a different curve from aligned scaling, indicating the mitigation is semantic-direction specific rather than generic norm reduction.

Fail criteria:

- All scaled aligned adapters collapse toward base semantics before retention recovers.
- Scrambled and aligned curves overlap, implying adapter perturbation dominates and training stability needs attention before semantic objectives.

## P4: Selective Steering and Ablation

Purpose: separate representational benefits from answer-ranking costs.

Existing scripts:

```bash
python src/sft/extract_concept_vectors.py \
  --concepts coherence human_alignment \
  --split train \
  --batch-size 2 \
  --overwrite

python src/sft/run_concept_steering_eval.py \
  --layers 12,16,20,24 \
  --alphas -4 -2 0 2 4 \
  --batch-size 2 \
  --overwrite
```

Known caution:

- Earlier broad activation steering was unstable and could cause degenerate generation.
- Use the forced-choice CAA evaluator first.
- Do not move to lm-eval steering until one layer/alpha improves target probes and keeps retention probes near alpha `0`.

Steering experiments:

- Positive coherence steering: test whether target coherence/human-alignment forced-choice preference rises.
- Negative coherence steering or projection/ablation: test whether removing the semantic direction restores TruthfulQA/MMLU margins on small slices.
- Layer sweep: prioritize middle-to-late layers already in the script, then refine around any positive result.
- Use steering only during representation extraction for fMRI/semantic-hub lanes if benchmark answering degrades.

Expected outputs:

- `results/sft_eval/concept_steering/vectors/*.npz`
- `results/sft_eval/concept_steering/sweep_results.csv`
- `results/sft_eval/concept_steering/sweep_details.csv`
- `results/sft_eval/concept_steering/SUMMARY.md`
- If ablation is implemented later: `results/sft_eval/wide_bench/skill_diagnostics/selective_ablation/<run_id>/summary.csv`

Pass criteria:

- Target `positive_preference` and `mean_delta_logprob` improve for matching concept/layer/alpha.
- Retention probe stays within `0.02` of alpha `0`.
- For ablation, MMLU/TruthfulQA failure slices improve while semantic probes decrease, supporting a causal direction.

Fail criteria:

- Steering improves target probes only at alphas that break retention probes.
- Steering effects are not layer-specific or not reproducible.
- Ablation does not improve failure slices, weakening the semantic-direction damage hypothesis.

## P5: Data Curriculum Variants

Purpose: train the desired semantic geometry while explicitly protecting sharp option ranking and calibration.

Do this only after P1-P4 identify what must be protected.

Candidate curricula:

- Retention/KL replay: mix small multiple-choice prompts from ARC/OpenBookQA, WiC, TruthfulQA, and MMLU slice tasks into coherence SFT. Add KL to base logits on answer options.
- TruthfulQA false-lure calibration: include paired true-vs-plausible-false examples from medical/safety myths, conspiracy lures, stereotypes, and overgeneralizations. Optimize relative true-vs-false margin, not only free-form coherence.
- WiC sense-boundary curriculum: contrast same-lemma/different-sense examples with same-lemma/same-sense examples. Preserve local context-specific word sense.
- MMLU close-option curriculum: add moral scenarios, formal logic, medical genetics, nutrition, professional psychology, and high-school statistics with base-logit KL.
- Two-stage curriculum: first semantic coherence, then short retention calibration with lower LR and early stopping.

Expected outputs:

- `out/adapters_mitigation/<curriculum_name>/`
- `results/sft_eval/mitigation/<curriculum_name>/train_config.json`
- `results/sft_eval/mitigation/<curriculum_name>/semantic_summary.csv`
- `results/sft_eval/mitigation/<curriculum_name>/failure_suite_summary.csv`
- `results/sft_eval/mitigation/<curriculum_name>/REPORT.md`

Pass criteria:

- Generation coherence stays `>= 0.70` or at least matches `taskvec_a0p25` if the adapter is intended as a lighter mitigation.
- Human R2 stays `>= 0.60`.
- WiC improves by at least `+0.05` over lowrank/taskvec.
- TruthfulQA `frac_false_pressure_up` falls below `0.60` and MC2 is within `0.02` of base.
- MMLU failure-slice mean delta improves by at least `0.04` over lowrank before any full MMLU.

Fail criteria:

- Curriculum preserves benchmark retention only by losing semantic/human gains.
- It improves one benchmark family while worsening another failure family.
- It matches lowrank/lowLR tradeoff without a new Pareto point.

## P6: Benchmark-Specific Controls

Purpose: rule out evaluation artifacts before drawing scientific conclusions.

Controls:

- Backend control: rerun a tiny matched slice with both `vllm` and `hf` for base and one aligned adapter.
- Chat-template control: verify the same chat template is applied across arms.
- Context-length control: MMLU has truncation warnings at `max_model_len=2048`; rerun the MMLU slice at a longer max length on a high-memory GPU if feasible.
- Few-shot control: compare zero-shot and standard few-shot on a small MMLU/ARC slice.
- Batch-size determinism: rerun one small slice at batch sizes `1` and `2`; scores should match.
- Adapter path control: verify staged adapter checksums and `adapter_config.json` match local paths for CHTC.
- Scrambled/rank control: keep scrambled SFT and, if available, rank-matched random or norm-matched controls.

Expected outputs:

- `results/sft_eval/wide_bench/skill_diagnostics/controls/<run_id>/control_matrix.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/controls/<run_id>/adapter_checksums.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/controls/<run_id>/REPORT.md`

Pass criteria:

- The main deltas reproduce across backend/batch/template controls.
- Longer context reduces warnings but does not erase the MMLU failure pattern.
- Adapter checksums and configs match between local and CHTC runs.

Fail criteria:

- A control eliminates the drop. In that case, update the benchmark interpretation first; do not train a mitigation for an eval artifact.

## Reporting Contract

Every future experiment should update or create a run-local report with:

- Objective and hypothesis.
- Exact command, host, GPU, container or conda env, commit hash, and output paths.
- Arms, tasks, limits, few-shot settings, max length, backend, and batch size.
- Primary metrics and pass/fail result.
- One paragraph of interpretation and one next action.

Recommended final synthesis fields:

- `semantic_gain_retained`: yes/no plus metrics.
- `benchmark_retention`: table of deltas versus base.
- `mechanism_supported`: one of `semantic_smoothing`, `calibration_pressure`, `generic_perturbation`, `eval_artifact`, `unresolved`.
- `next_gpu_action`: exact next command or "no expensive job until analyzer exists".
