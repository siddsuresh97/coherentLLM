# coherence_experiments

## Current Scientific Report

Last updated: 2026-07-08 on branch `coherence-sft`.

This README is the high-level dashboard. Expand the sections below for the
details, exact metrics, artifact paths, and next decisions. The continuously
updated long-form log is [`research/EXPERIMENT_LOG.md`](research/EXPERIMENT_LOG.md).

### Headlines

- **Semantic hub:** coherence-aligned/task-vector states strongly increase
  cross-format invariance inside the model. The stricter paper-style baseline
  also shows matched concepts beating random mismatches, but exact identity over
  S*-close neighbors is still small; logit-lens and interventions are needed.
- **fMRI:** the THINGS-fMRI pipeline works. Ventral Visual shows the expected
  object-RSA signal and scrambled-control separation, but aligned arms are
  mostly flat versus base in the primary visual ROI. The Huth/LeBel
  narrative-fMRI lane now has a local/CHTC audit bundle and a completed CHTC
  smoke run (`5513006`). The OpenNeuro metadata layout is verified and a CHTC
  staging manifest is ready: smoke is 7.88 GB; the `UTS01`-`UTS03` high-data
  subset is 76.86 GB, under the observed 100 GB staging quota.
- **fMRI x hub bridge:** the new concept-held-out regression does not support a
  clean semantic-hub explanation of Ventral Visual RSA. Averaged hub predictors
  are roughly tied with single prompt spokes in Ventral Visual, while
  ATL/Language show small aligned-state advantages that remain exploratory.
- **Benchmarks:** task-vector `alpha=0.25` improves ARC retention over lowrank
  (`ARC-Easy 0.808`, `ARC-Challenge 0.559`) but still drops versus base
  (`0.850`, `0.649`). HellaSwag and WinoGrande are near-preserved; MMLU is the
  remaining long mitigation cell. Bounded TruthfulQA log-sample diagnostics
  show lowrank improves by suppressing false-answer pressure, `taskvec_a0p25`
  is aggregate-flat but increases plausible-false pressure, and scrambled SFT
  catastrophically hurts item-level calibration.
- **Runtime:** GPU0 is running scrambled ARC as the random-perturbation control;
  GPU1 is running `taskvec_a0p25` MMLU with `gpu_mem_util=0.72`,
  `batch_size=2`. H100 handled rank-16
  lowrank MMLU, but rank-64 LoRA vLLM evals (`taskvec_a0p25`, `scrambled`)
  stall before GPU allocation on `opt-a007`, even after local adapter staging.

### Reading Map

- Status/handoff: [`research/STATUS.md`](research/STATUS.md)
- Full experiment log: [`research/EXPERIMENT_LOG.md`](research/EXPERIMENT_LOG.md)
- Literature and experiment plan:
  [`research/LITERATURE_AND_EXPERIMENT_PLAN.md`](research/LITERATURE_AND_EXPERIMENT_PLAN.md)
- fMRI/semantic-hub task brief:
  [`research/CODEX_TASK_9_FMRI_AND_SEMANTIC_HUB.md`](research/CODEX_TASK_9_FMRI_AND_SEMANTIC_HUB.md)
- Benchmark-drop task brief:
  [`research/CODEX_TASK_10_BENCHMARK_DROPS.md`](research/CODEX_TASK_10_BENCHMARK_DROPS.md)
- Semantic-hub report: [`results/sft_semantic_hub/REPORT.md`](results/sft_semantic_hub/REPORT.md)
- Paper-style semantic-hub similarity report:
  [`results/sft_semantic_hub_paper/REPORT.md`](results/sft_semantic_hub_paper/REPORT.md)
- Paper-adapted semantic-hub plan:
  [`research/SEMANTIC_HUB_PAPER_ADAPTED_PLAN.md`](research/SEMANTIC_HUB_PAPER_ADAPTED_PLAN.md)
- TruthfulQA log-sample diagnostic:
  [`results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/REPORT.md`](results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/REPORT.md)
- THINGS-fMRI RSA report: [`results/sft_fmri/REPORT.md`](results/sft_fmri/REPORT.md)
- Huth/LeBel language-fMRI audit:
  [`results/sft_huth_lebel/REPORT.md`](results/sft_huth_lebel/REPORT.md)
- Huth/LeBel ds003020 staging plan:
  [`results/sft_huth_lebel/STAGING_PLAN.md`](results/sft_huth_lebel/STAGING_PLAN.md)
- fMRI x semantic-hub bridge:
  [`results/sft_fmri_semantic_bridge/REPORT.md`](results/sft_fmri_semantic_bridge/REPORT.md)
- Held-out fMRI hub regression:
  [`results/sft_fmri_hub_regression/REPORT.md`](results/sft_fmri_hub_regression/REPORT.md)

<details>
<summary><strong>1. Semantic Hub Result</strong></summary>

Goal: test the semantic-hub hypothesis from arXiv:2411.04986 by asking whether
different prompt "spokes" for the same concept converge into a shared latent
representation.

Implementation:

- Extraction script: [`src/sft/extract_semantic_hub_hidden_states.py`](src/sft/extract_semantic_hub_hidden_states.py)
- Scoring script: [`src/sft/run_semantic_hub.py`](src/sft/run_semantic_hub.py)
- Concepts: 128 held-out THINGS/NOVA concepts.
- Formats: triplet, pairwise, feature-listing.
- Metrics: cross-format RDM Spearman, linear CKA, same-concept retrieval
  top-1/top-5, and concept-vs-format alignment.

Artifacts:

- Hidden states: `results/sft_semantic_hub/hidden_states/*.npz`
- Layer metrics: [`results/sft_semantic_hub/hub_by_layer.csv`](results/sft_semantic_hub/hub_by_layer.csv)
- Summary: [`results/sft_semantic_hub/hub_summary.csv`](results/sft_semantic_hub/hub_summary.csv)
- Report: [`results/sft_semantic_hub/REPORT.md`](results/sft_semantic_hub/REPORT.md)
- Paper-style matched-vs-baseline report:
  [`results/sft_semantic_hub_paper/REPORT.md`](results/sft_semantic_hub_paper/REPORT.md)

Mid-layer summary:

| Arm | RDM Spearman | CKA | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| `base` | 0.3230 | 0.3890 | 0.0192 | 0.0800 |
| `scrambled` | 0.3472 | 0.4601 | 0.1094 | 0.2667 |
| `lowLR` | 0.6211 | 0.6899 | 0.0729 | 0.3063 |
| `lowrank` | 0.6637 | 0.7467 | 0.0968 | 0.4079 |
| `taskvec_a0p25` | 0.6770 | 0.7809 | 0.2062 | 0.5393 |
| `taskvec_a0p5` | 0.6372 | 0.7464 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | 0.4867 | 0.5505 | 0.0637 | 0.1896 |

Read:

- Coherence alignment creates a strong cross-format invariance effect.
- `taskvec_a0p25` is the best mid-layer candidate by RDM, CKA, and retrieval.
- This is not yet a clean concept-dominant hub: concept-minus-format alignment
  stays negative for all arms.
- Paper-style similarity baselines strengthen and qualify the read:
  `taskvec_a0p5` has the best mid-layer same-minus-random margin (`0.0617`,
  base `0.0022`), but the stricter same-minus-S*-close margins are small
  (`taskvec_a1p0` best at `0.0099`). This looks like stronger semantic
  clustering, not yet a decisive exact-concept hub.

</details>

<details>
<summary><strong>2. THINGS-fMRI RSA Result</strong></summary>

Goal: test whether coherence-aligned model states better match THINGS-fMRI
object-concept representational geometry.

Implementation:

- Audit: [`src/sft/fmri_audit.py`](src/sft/fmri_audit.py)
- Hidden-state extraction: [`src/sft/extract_fmri_hidden_states.py`](src/sft/extract_fmri_hidden_states.py)
- RSA: [`src/sft/run_fmri_rsa.py`](src/sft/run_fmri_rsa.py)
- Exact overlap: 90 held-out SFT/THINGS-fMRI concepts across subjects 01/02/03.
- Hidden-state shape per arm: `90 x 33 x 4096`.

Artifacts:

- Audit: [`results/sft_fmri/fmri_data_audit.json`](results/sft_fmri/fmri_data_audit.json)
- Concept overlap: [`results/sft_fmri/concept_overlap.csv`](results/sft_fmri/concept_overlap.csv)
- RSA by layer: [`results/sft_fmri/rsa_by_layer.csv`](results/sft_fmri/rsa_by_layer.csv)
- Best-layer summary: [`results/sft_fmri/rsa_summary.csv`](results/sft_fmri/rsa_summary.csv)
- Report: [`results/sft_fmri/REPORT.md`](results/sft_fmri/REPORT.md)

Primary ROI results:

| Region | Best / notable read |
|---|---|
| Ventral Visual | `lowrank` 0.2013, `lowLR` 0.2010, `base` 0.1990, `scrambled` 0.1293 |
| ATL (Semantic) | best `taskvec_a0p5` 0.0543; `lowrank` 0.0539; `base` 0.0481 |
| Language | best `lowrank` 0.0345; `lowLR` 0.0329; `base` 0.0283 |

Read:

- The pipeline is working and the scrambled control is meaningfully lower in
  Ventral Visual.
- Coherence-aligned arms do not produce a large Ventral Visual gain over base.
- ATL/Language gains are small and exploratory; they are not language-network
  evidence yet.

</details>

<details>
<summary><strong>3. fMRI x Semantic-Hub Bridge</strong></summary>

Goal: ask whether semantic-hub metrics explain fMRI RSA variation across
arms/layers.

Implementation:

- Script: [`src/sft/bridge_fmri_semantic_hub.py`](src/sft/bridge_fmri_semantic_hub.py)
- Held-out regression script: [`src/sft/run_fmri_hub_regression.py`](src/sft/run_fmri_hub_regression.py)
- Output directory: `results/sft_fmri_semantic_bridge/`
- Report: [`results/sft_fmri_semantic_bridge/REPORT.md`](results/sft_fmri_semantic_bridge/REPORT.md)
- Regression output directory: `results/sft_fmri_hub_regression/`
- Regression report: [`results/sft_fmri_hub_regression/REPORT.md`](results/sft_fmri_hub_regression/REPORT.md)

Key artifacts:

- [`layer_join.csv`](results/sft_fmri_semantic_bridge/layer_join.csv): mean fMRI
  RSA by region/arm/layer joined to hub metrics.
- [`layer_correlations.csv`](results/sft_fmri_semantic_bridge/layer_correlations.csv):
  descriptive layer-grid correlations.
- [`arm_summary_bridge.csv`](results/sft_fmri_semantic_bridge/arm_summary_bridge.csv):
  fMRI summary joined to hub summary.
- [`arm_summary_correlations.csv`](results/sft_fmri_semantic_bridge/arm_summary_correlations.csv):
  small-n arm-level correlations.
- [`cv_model_comparison.csv`](results/sft_fmri_hub_regression/cv_model_comparison.csv):
  concept-held-out mid-layer comparison of format-averaged hub RDMs against
  single-format RDMs.
- [`cv_best_layer_summary.csv`](results/sft_fmri_hub_regression/cv_best_layer_summary.csv):
  descriptive best-layer held-out regression scores.

Important bridge metrics:

| Analysis | Region | Metric | Spearman r | n | p |
|---|---|---|---:|---:|---:|
| Arm delta vs base | Ventral Visual | mid hub RDM | 0.600 | 6 | 0.208 |
| Arm delta vs base | Ventral Visual | mid retrieval top-1 | -0.086 | 6 | 0.872 |
| Layer-grid delta vs base | Ventral Visual | cross-format RDM delta | 0.401 | 192 | 7.89e-09 |
| Layer-grid delta vs base | ATL (Semantic) | cross-format RDM delta | 0.306 | 192 | 1.58e-05 |

Held-out regression setup:

- 90 fMRI-overlap concepts, subjects 01/02/03, primary ROIs only.
- Five concept-held-out folds; train/test RDM distances do not share concepts.
- Predictors: format-averaged representation RDM, mean format RDM, each single
  prompt-format RDM, all single-format RDMs, and mean-plus-single RDMs.
- Mid-layer band: layers 10-20.

Mid-layer held-out Pearson read:

| Region | Arm | Mean repr r | Best single r | Delta |
|---|---|---:|---:|---:|
| Ventral Visual | `lowrank` | 0.1751 | 0.1742 | +0.0008 |
| Ventral Visual | `taskvec_a0p25` | 0.1679 | 0.1702 | -0.0023 |
| Ventral Visual | `lowLR` | 0.1623 | 0.1748 | -0.0125 |
| ATL (Semantic) | `taskvec_a0p5` | 0.0356 | 0.0310 | +0.0046 |
| ATL (Semantic) | `lowrank` | 0.0332 | 0.0326 | +0.0006 |
| Language | `lowLR` | 0.0511 | 0.0423 | +0.0087 |
| Language | `taskvec_a0p5` | 0.0511 | 0.0465 | +0.0046 |
| Language | `lowrank` | 0.0503 | 0.0488 | +0.0015 |

Descriptive best-layer winners:

| Region | Best row | Held-out Pearson r |
|---|---|---:|
| Ventral Visual | `taskvec_a0p25` `single_feature_listing`, layer 18 | 0.2034 |
| Ventral Visual | `lowrank` `single_feature_listing`, layer 18 | 0.2010 |
| ATL (Semantic) | `taskvec_a0p5` `single_pairwise`, layer 23 | 0.0565 |
| Language | `taskvec_a1p0` `mean_repr_plus_single_formats`, layer 18 | 0.0765 |

Read:

- Hub invariance is a strong internal-model result.
- It does not explain the Ventral Visual object-fMRI RSA pattern in the
  held-out regression; single feature/prompt spokes are as good or better.
- ATL/Language show small aligned-state advantages for averaged/shared
  predictors, but the absolute correlations are small and exploratory.
- Layer-grid correlations are descriptive because layer points are not
  independent.
- The next fMRI step, if we pursue it, should be a stricter fixed-layer or
  nested-CV confirmation plus the planned language-fMRI encoding analysis.

</details>

<details>
<summary><strong>4. Benchmark Drops And Mitigation</strong></summary>

Goal: understand why some broad benchmarks drop after coherence SFT, why some
tasks improve, and how to mitigate retention loss.

Runner:

- [`src/sft/eval_wide_bench.py`](src/sft/eval_wide_bench.py)
- Supports states: `base`, `scrambled`, `lowLR`, `lowrank`,
  `taskvec_a0p25`, `taskvec_a0p5`.

Pushed partial results:

- Commit `caff36e`: partial lowrank/taskvec wide-bench outputs.
- Raw JSON directories under `results/sft_eval/wide_bench/runs/`.
- Regenerated summary tables:
  [`results/sft_eval/wide_bench/summary.csv`](results/sft_eval/wide_bench/summary.csv)
  and
  [`results/sft_eval/wide_bench/raw_task_summary.csv`](results/sft_eval/wide_bench/raw_task_summary.csv).

Completed partial metrics:

| State | Group | Task | Metric | Value |
|---|---|---|---|---:|
| `lowrank` | zero-shot | PIQA | acc_norm | 0.776 |
| `lowrank` | zero-shot | OpenBookQA | acc_norm | 0.408 |
| `lowrank` | zero-shot | CommonsenseQA | acc | 0.622 |
| `lowrank` | zero-shot | WiC | acc | 0.498 |
| `lowrank` | zero-shot | TruthfulQA-MC2 | acc | 0.541 |
| `lowrank` | 5-shot | WinoGrande | acc | 0.744 |
| `lowrank` | 5-shot | MMLU | acc | 0.592 |
| `lowrank` | 25-shot | ARC-Easy | acc_norm | 0.705 |
| `lowrank` | 25-shot | ARC-Challenge | acc_norm | 0.508 |
| `lowrank` | 10-shot | HellaSwag | acc_norm | 0.683 |
| `taskvec_a0p25` | zero-shot | PIQA | acc_norm | 0.789 |
| `taskvec_a0p25` | zero-shot | OpenBookQA | acc_norm | 0.436 |
| `taskvec_a0p25` | zero-shot | CommonsenseQA | acc | 0.581 |
| `taskvec_a0p25` | zero-shot | WiC | acc | 0.502 |
| `taskvec_a0p25` | zero-shot | TruthfulQA-MC2 | acc | 0.523 |
| `taskvec_a0p25` | 5-shot | WinoGrande | acc | 0.747 |
| `taskvec_a0p25` | 25-shot | ARC-Easy | acc_norm | 0.808 |
| `taskvec_a0p25` | 25-shot | ARC-Challenge | acc_norm | 0.559 |
| `taskvec_a0p25` | 10-shot | HellaSwag | acc_norm | 0.680 |
| `scrambled` | zero-shot | PIQA | acc_norm | 0.540 |
| `scrambled` | zero-shot | OpenBookQA | acc_norm | 0.282 |
| `scrambled` | zero-shot | CommonsenseQA | acc | 0.197 |
| `scrambled` | zero-shot | WiC | acc | 0.481 |
| `scrambled` | zero-shot | TruthfulQA-MC2 | acc | 0.482 |

Base/lowLR/lowrank retention summary:

| Task | Base | LowLR | Lowrank | Lowrank delta |
|---|---:|---:|---:|---:|
| PIQA | 0.799 | 0.779 | 0.776 | -0.023 |
| OpenBookQA | 0.490 | 0.412 | 0.408 | -0.082 |
| CommonsenseQA | 0.651 | 0.666 | 0.622 | -0.029 |
| WiC | 0.652 | 0.500 | 0.498 | -0.154 |
| TruthfulQA-MC2 | 0.550 | 0.533 | 0.541 | -0.010 |
| WinoGrande | 0.762 | 0.765 | 0.744 | -0.018 |
| ARC-Easy | 0.850 | 0.725 | 0.705 | -0.145 |
| ARC-Challenge | 0.649 | 0.524 | 0.508 | -0.141 |
| HellaSwag | 0.685 | 0.656 | 0.683 | -0.002 |
| MMLU | 0.693 | 0.594 | 0.592 | -0.101 |

Skill-level read:

- Detailed map:
  [`results/sft_eval/wide_bench/skill_map.md`](results/sft_eval/wide_bench/skill_map.md).
- Preserved or near-preserved skills: script/event plausibility
  (HellaSwag), coreference/discourse commonsense (WinoGrande), and physical
  affordance commonsense under task-vector mitigation (PIQA).
- Hurt skills: word-sense disambiguation (WiC), science/exam retrieval
  (ARC/MMLU/OpenBookQA), and multiple-choice ranking calibration.
- TruthfulQA-MC2 is not one of the largest aligned-adapter drops in the current
  numbers: lowrank is `-0.010` and lowLR is `-0.017`, both flat by the current
  `0.02` threshold. It is more affected under task-vector `alpha=0.25`
  (`-0.028`) and scrambled SFT (`-0.068`), suggesting sensitivity to
  calibration and plausible-false misconception lures rather than a clean
  truth-knowledge collapse.

Read:

- WiC remains near chance for aligned states, suggesting lexical sense
  discrimination is harmed by the alignment objective or adapter perturbation.
- WinoGrande is stable, so this is not a uniform few-shot evaluation failure.
- HellaSwag is essentially preserved by lowrank and `taskvec_a0p25` once the
  run is split and constrained to `gpu_mem_util=0.72`, `batch_size=2`.
- `taskvec_a0p25` mitigates ARC relative to lowrank but does not fix it:
  ARC-Easy improves from lowrank `0.705` to `0.808` but remains below base
  `0.850`; ARC-Challenge improves from `0.508` to `0.559` but remains below
  base `0.649`.
- `taskvec_a0p25` improves PIQA/OpenBookQA over lowrank, but loses
  CommonsenseQA/TruthfulQA in this partial slice.
- Scrambled zero-shot is much worse than lowrank/taskvec on PIQA, OpenBookQA,
  CommonsenseQA, and TruthfulQA, so useful semantic training is doing real work.
  But scrambled also drops broadly, so generic LoRA/SFT perturbation is part of
  the damage.
- Bounded TruthfulQA log-sample diagnostics are now trackable in
  [`results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/REPORT.md`](results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/REPORT.md).
  On the 200-item diagnostic slice, base is `0.5224`, lowrank is `0.5528`,
  `taskvec_a0p25` is `0.5267`, and scrambled is `0.4701`. Paired deltas show
  lowrank lowers both truthful and false log-likelihood mass, but false-answer
  pressure falls more (`delta_false_logsumexp=-3.9391` vs
  `delta_true_logsumexp=-3.5444`), improving mean truth log-odds by `+0.3946`.
  By contrast, `taskvec_a0p25` raises both masses and raises false pressure
  more (`+4.9265` vs `+4.3026`), reducing mean truth log-odds by `-0.6239`
  despite a nearly flat MC2 score. Scrambled drops MC2 by `-0.0523` even though
  mean truth log-odds rises, because it suppresses both true and false answer
  likelihoods extremely hard and causes catastrophic item-level flips.
  Recurring worst drops are safety/myth/misconception lures such as
  defibrillation for flatline, washing chicken, Latin-American language
  overgeneralization, Agenda 21, and voodoo dolls.
- Current active follow-ups: scrambled ARC on GPU0 and `taskvec_a0p25` MMLU
  5-shot on GPU1. The MMLU run is not a repeat of the finished
  base/lowLR/lowrank MMLU rows; it fills the missing task-vector mitigation row.

</details>

<details>
<summary><strong>5. Runtime And GPU Notes</strong></summary>

Current confirmed settings:

- Current active A5000 lanes: GPU0 scrambled `arc_25shot`; GPU1
  `taskvec_a0p25 mmlu_5shot`.
- A5000 broad-bench long loglikelihood runs should use conservative settings:
  `gpu_mem_util=0.72` and `batch_size=2` for ARC/Hella.
- `gpu_mem_util=0.82` with auto batch OOMed during prompt-logprob scoring.
- `gpu_mem_util=0.65` plus `batch_size=8` left too little KV cache and failed
  vLLM initialization.
- `gpu_mem_util=0.75` plus `batch_size=4` completed short slices but OOMed on
  long ARC/Hella runs.

H100 note:

- `opt-a007` completed rank-16 lowrank MMLU cleanly.
- Rank-64 LoRA evals (`taskvec_a0p25`, `scrambled`) repeatedly stalled before
  GPU allocation.
- Copying `scrambled` to `/tmp/ssuresh/coherence_adapters/scrambled` was fast
  (`657M`), but vLLM still stalled from the local path. The issue is likely
  vLLM/rank-64 LoRA initialization on that host rather than only filesystem
  throughput.
- A rank-64 HF/PEFT smoke test on `opt-a007` stayed pre-GPU for multiple
  minutes and was stopped, so it is not currently a useful fast fallback.
- Until fixed, run rank-64 broad-bench lanes on `rogers-gpu-1` or scale
  already-validated independent lanes to CHTC.

CHTC scale-out rule:

- The `chtc-gpu-jobs` skill is installed in active `CODEX_HOME` at
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/cache_home/codex/skills/chtc-gpu-jobs`.
- Debug locally/directly first (`rogers-gpu-1` or another interactive GPU).
- Only move a job to CHTC after the exact command, environment, paths, and
  output behavior have been validated.
- Use CHTC for scaling independent, already-working lanes; do not use it as the
  first place to debug vLLM, adapter loading, datasets, or result writing.
- For CHTC access, run `chtc-master start` from this session, choose a Duo Push
  option, ask the user to approve the push, then verify with `chtc-master check`
  and `chtc-ssh 'hostname -f; condor_q -totals; echo STAGING=$STAGING'`.
- When a CHTC run is used, record the local validation command, submit file,
  logs, and output path in [`research/EXPERIMENT_LOG.md`](research/EXPERIMENT_LOG.md).

Speed/RSA note:

- `thingsvision` is not installed in the active coherence envs.
- For the current 90-concept fMRI RSA, HDF5 beta loading was the bottleneck,
  and batched HDF5 reads fixed the main slowdown.
- GPU/ThingsVision-style RDM computation should be revisited for full-720,
  bootstrap-heavy, or permutation-heavy RSA.

</details>

<details>
<summary><strong>6. Literature And Experiment Plans</strong></summary>

The sidecar agents completed and their memos are synthesized in
[`research/LITERATURE_AND_EXPERIMENT_PLAN.md`](research/LITERATURE_AND_EXPERIMENT_PLAN.md).
The raw memo trail is in [`research/EXPERIMENT_LOG.md`](research/EXPERIMENT_LOG.md).

Current plan:

- Semantic hub: anchor to Wu, Yu, Yogatama, Lu, and Kim,
  "The Semantic Hub Hypothesis" (`arXiv:2411.04986`, ICLR 2025). Our adaptation
  treats triplet, pairwise, and feature prompts as elicitation "spokes" for the
  same THINGS concept. The next pass should mirror the paper more closely:
  matched-vs-mismatched similarity baselines, logit-lens anchoring to
  concept/neighbor tokens, symbolic S* spokes, and mid-layer causal
  interventions.
- Huth/LeBel language-fMRI: use OpenNeuro `ds003020` / HuthLab
  `deep-fMRI-dataset` if we launch a language encoding pass. Feed exact narrative
  transcript streams, avoid chat templates, align word hidden states to TRs, and
  score held-out voxelwise ridge predictions. Current audit artifacts are in
  [`results/sft_huth_lebel/REPORT.md`](results/sft_huth_lebel/REPORT.md).
  Metadata from the OpenNeuro GitHub mirror verifies the current `derivatives/`
  layout and DataLad annex file sizes; the staging plan is in
  [`results/sft_huth_lebel/STAGING_PLAN.md`](results/sft_huth_lebel/STAGING_PLAN.md).
  No downloaded/staged `ds003020` root is visible yet, so the next concrete step
  is a CPU-only CHTC downloader for the 7.88 GB smoke subset.
- Fedorenko/EvLab language-network: prefer individually localized
  `sentences > nonword lists` masks. Atlas/group language ROIs are exploratory.
- Benchmark-drops: first finish eval-only controls (`lowrank`, `scrambled`,
  `taskvec_a0p25`) before new training. Current hypothesis is a narrow semantic
  SFT plus global LoRA perturbation and multiple-choice logit calibration shifts,
  not a uniform capability collapse.
- Scale-out: debug locally/direct-GPU first; use CHTC only after paths,
  environment, output writing, and memory settings are known-good.

</details>

<details>
<summary><strong>7. Next Actions</strong></summary>

Immediate:

1. Let scrambled ARC and task-vector MMLU continue. When each finishes, parse
   the JSON, update the README/log/skill map, and commit/push.
2. Keep the already-running `taskvec_a0p25` MMLU lane unless it becomes clearly
   redundant.
3. Next free GPU lane should go to paper-style semantic-hub logit lens or to
   scrambled ARC/Hella if the priority is the random-perturbation control.

Scientific next:

1. Semantic hub: run the paper-adapted tests in
   [`research/SEMANTIC_HUB_PAPER_ADAPTED_PLAN.md`](research/SEMANTIC_HUB_PAPER_ADAPTED_PLAN.md),
   starting with CPU-only matched-vs-baseline similarity from existing hidden
   states.
2. Huth/LeBel language-fMRI: submit a CPU-only CHTC downloader for the
   manifest-listed smoke subset, rerun the audit on staged data, then start GPU
   feature extraction only after the smoke root passes.
3. fMRI bridge: do not overclaim Ventral Visual hub evidence. If continuing,
   run fixed-layer/nested-CV confirmation after the Huth smoke path is staged.
4. Benchmark mechanism: inspect WiC/ARC failures and tasks with gains to decide
   whether drops are lexical-sense-specific, adapter-rank-specific, or generic
   SFT perturbation.
5. Mitigation: alpha sweep between `0.10` and `0.35`, then adapter
   sparsification or KL-to-base only if eval-only controls justify more
   training.

</details>

## Original Project Context

Re-running the semantic-coherence experiments from *"Uncovering the Computational
Ingredients of Human-Like Representations in LLMs"* (Studdiford, Rogers, Mukherjee,
Suresh; arXiv:2510.01030) on newer open-weight models (8B-class, local GPU) and
frontier models (GPT-5, Claude Opus via OpenRouter).

This produces data for the **bridge between dissertation chapters 1 and 2**.

**This folder is intentionally OUTSIDE the dissertation repo and must never be pushed
to the dissertation's Overleaf/GitHub remotes.** It has its own git repo.

## Three response methods (from the EMNLP/pipeline paper)

1. **triplet** — anchored similarity ("which of B/C is more similar to A?")
2. **pairwise** — 1..7 similarity rating for each pair
3. **feature** — True/False feature verification over concept×feature pairs

## Coherence

For each model we build a 30D representation of the 30 concepts from its judgments,
then measure:
- **cross-method coherence**: agreement of the similarity structure a model produces
  across the three methods (triplet vs pairwise vs feature).
- **human alignment**: Procrustes R^2 / RSA between the model embedding and the human
  reference (pending human data on these 30 concepts; falls back to THINGS/SPoSE
  overlap).

## Layout
- `configs/` — model registry + run configs
- `data/stimuli/` — the 30 concepts, triplet set, pairwise set, feature set
- `data/human/` — human reference (triplet judgments or SPoSE embedding)
- `src/` — prompts, runners (vLLM + OpenRouter), embedding, coherence
- `results/raw/` — raw model responses (gitignored)
- `results/embeddings/`, `results/coherence/` — derived (embeddings gitignored)

## Environment
`env/` conda env with vLLM. See `scripts/setup_env.sh`.
