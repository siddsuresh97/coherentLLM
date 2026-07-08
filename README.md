# coherence_experiments

## Current Scientific Report

Last updated: 2026-07-08 on branch `coherence-sft`.

This README is the high-level dashboard. Expand the sections below for the
details, exact metrics, artifact paths, and next decisions. The continuously
updated long-form log is [`research/EXPERIMENT_LOG.md`](research/EXPERIMENT_LOG.md).

### Headlines

- **Semantic hub:** coherence-aligned/task-vector states strongly increase
  cross-format invariance inside the model. `taskvec_a0p25` is the best
  mid-layer hub candidate so far.
- **fMRI:** the THINGS-fMRI pipeline works. Ventral Visual shows the expected
  object-RSA signal and scrambled-control separation, but aligned arms are
  mostly flat versus base in the primary visual ROI.
- **fMRI x hub bridge:** hub RDM gains are positively related to Ventral Visual
  delta in a small-n descriptive check, but same-concept retrieval does not
  explain fMRI deltas. This is not yet evidence that the semantic hub explains
  the fMRI effect.
- **Benchmarks:** lowrank still collapses on WiC; WinoGrande is stable and
  lowrank HellaSwag 10-shot now completes at `acc_norm=0.683`.
  `taskvec_a0p25` improves PIQA/OpenBookQA versus lowrank but loses
  CommonsenseQA/TruthfulQA in the partial slice.
- **Runtime:** A5000 runs need conservative vLLM settings for long
  loglikelihood tasks. H100 handled rank-16 lowrank MMLU, but rank-64 LoRA
  vLLM evals (`taskvec_a0p25`, `scrambled`) stall before GPU allocation on
  `opt-a007`, even after local adapter staging.

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
- THINGS-fMRI RSA report: [`results/sft_fmri/REPORT.md`](results/sft_fmri/REPORT.md)
- fMRI x semantic-hub bridge:
  [`results/sft_fmri_semantic_bridge/REPORT.md`](results/sft_fmri_semantic_bridge/REPORT.md)

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
- Output directory: `results/sft_fmri_semantic_bridge/`
- Report: [`results/sft_fmri_semantic_bridge/REPORT.md`](results/sft_fmri_semantic_bridge/REPORT.md)

Key artifacts:

- [`layer_join.csv`](results/sft_fmri_semantic_bridge/layer_join.csv): mean fMRI
  RSA by region/arm/layer joined to hub metrics.
- [`layer_correlations.csv`](results/sft_fmri_semantic_bridge/layer_correlations.csv):
  descriptive layer-grid correlations.
- [`arm_summary_bridge.csv`](results/sft_fmri_semantic_bridge/arm_summary_bridge.csv):
  fMRI summary joined to hub summary.
- [`arm_summary_correlations.csv`](results/sft_fmri_semantic_bridge/arm_summary_correlations.csv):
  small-n arm-level correlations.

Important bridge metrics:

| Analysis | Region | Metric | Spearman r | n | p |
|---|---|---|---:|---:|---:|
| Arm delta vs base | Ventral Visual | mid hub RDM | 0.600 | 6 | 0.208 |
| Arm delta vs base | Ventral Visual | mid retrieval top-1 | -0.086 | 6 | 0.872 |
| Layer-grid delta vs base | Ventral Visual | cross-format RDM delta | 0.401 | 192 | 7.89e-09 |
| Layer-grid delta vs base | ATL (Semantic) | cross-format RDM delta | 0.306 | 192 | 1.58e-05 |

Read:

- Hub invariance is a strong internal-model result.
- It does not yet explain the object-fMRI RSA pattern.
- Layer-grid correlations are descriptive because layer points are not
  independent.
- The next decisive analysis is a held-out fMRI regression where
  format-averaged hub RDMs and single-format RDMs compete to predict the same
  fMRI RDMs.

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
| `lowrank` | 10-shot | HellaSwag | acc_norm | 0.683 |
| `taskvec_a0p25` | zero-shot | PIQA | acc_norm | 0.789 |
| `taskvec_a0p25` | zero-shot | OpenBookQA | acc_norm | 0.436 |
| `taskvec_a0p25` | zero-shot | CommonsenseQA | acc | 0.581 |
| `taskvec_a0p25` | zero-shot | WiC | acc | 0.502 |
| `taskvec_a0p25` | zero-shot | TruthfulQA-MC2 | acc | 0.523 |
| `taskvec_a0p25` | 5-shot | WinoGrande | acc | 0.747 |

Read:

- WiC remains near chance for aligned states, suggesting lexical sense
  discrimination is harmed by the alignment objective or adapter perturbation.
- WinoGrande is stable, so this is not a uniform few-shot evaluation failure.
- HellaSwag remains usable under the lowrank mitigation once the run is split
  and constrained to `gpu_mem_util=0.72`, `batch_size=2`.
- `taskvec_a0p25` improves PIQA/OpenBookQA over lowrank, but loses
  CommonsenseQA/TruthfulQA in this partial slice.
- Scrambled broad-benchmark control is now exposed in the runner and is running
  for zero-shot on `rogers-gpu-1` GPU1.

</details>

<details>
<summary><strong>5. Runtime And GPU Notes</strong></summary>

Current confirmed settings:

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
- Until fixed, run rank-64 broad-bench lanes on `rogers-gpu-1`.

CHTC scale-out rule:

- Debug locally/directly first (`rogers-gpu-1` or another interactive GPU).
- Only move a job to CHTC after the exact command, environment, paths, and
  output behavior have been validated.
- Use CHTC for scaling independent, already-working lanes; do not use it as the
  first place to debug vLLM, adapter loading, datasets, or result writing.
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
  same THINGS concept.
- Huth/LeBel language-fMRI: use OpenNeuro `ds003020` / HuthLab
  `deep-fMRI-dataset` if we launch a language encoding pass. Feed exact narrative
  transcript streams, avoid chat templates, align word hidden states to TRs, and
  score held-out voxelwise ridge predictions.
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

1. Let lowrank `arc_25shot` finish on `rogers-gpu-1`.
2. Run:

   ```bash
   python src/sft/eval_wide_bench.py --states base lowLR lowrank --summarize_only
   ```

3. Commit and push the completed lowrank wide-bench summary and raw outputs.
4. Let `scrambled` zero-shot finish on `rogers-gpu-1` GPU1, then use freed
   A5000 time for taskvec remaining broad-bench lanes and later scrambled MMLU.

Scientific next:

1. fMRI bridge regression: compare format-averaged hub RDMs vs single-format
   RDMs as predictors of fMRI RDMs.
2. Benchmark mechanism: inspect WiC/ARC failures and tasks with gains to decide
   whether drops are lexical-sense-specific, adapter-rank-specific, or generic
   SFT perturbation.
3. Mitigation: alpha sweep between `0.10` and `0.35`, then adapter
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
