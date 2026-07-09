# Huth/LeBel Smoke Encoding Plan

Updated: 2026-07-09

## Current Data State

- Successful CHTC staging audit: `results/sft_huth_lebel/chtc_5513059/extracted/sft_huth_lebel_stage_smoke/`.
- Staged dataset root for CHTC jobs: `/staging/s/suresh27/datasets/ds003020-smoke`.
- Smoke subset: 15 files, 7.877643435 GB.
- Stories: `sweetaspie`, `againstthewind`, `wheretheressmoke`.
- Subjects: `UTS01`, `UTS02`, `UTS03`.
- Response files: author-preprocessed HF5 under `derivatives/preprocessed_data/<subject>/<story>.hf5`.
- Word timing files: `derivatives/TextGrids/<story>.TextGrid`.

The local workspace does not currently mount `/staging/s/suresh27/datasets/ds003020-smoke`, so the repo-side validation below checks syntax and synthetic I/O. Real smoke execution should run on the CHTC node or AP where the staged root is visible.

## Pipeline

The smoke experiment is split into a GPU feature-extraction step and a CPU ridge-encoding step.

1. GPU: `src/sft/huth_lebel_extract_word_states.py`
   - Parses TextGrid word intervals.
   - Reconstructs plain narrative text from word labels.
   - Does not apply chat templates.
   - Extracts final-subtoken hidden states for each word.
   - Supports `base`, `lowLR`, `scrambled`, `taskvec_a0p25`, or adapter overrides.
   - Saves `words x selected_layers x hidden_dim` NPZ files under `features_dir/<arm>/<story>.npz`.

2. CPU: `src/sft/huth_lebel_smoke_encoding.py`
   - Loads saved word states and HF5 BOLD responses.
   - Lanczos-aligns word features to TR grid with `TR=2s`.
   - Adds FIR delays `1,2,3,4` TRs, corresponding to 2/4/6/8 seconds.
   - Fits voxelwise ridge using dual ridge when features exceed time points.
   - Selects ridge alpha by leave-one-training-story-out CV on `sweetaspie` and `againstthewind`.
   - Evaluates only on held-out `wheretheressmoke`.
   - Writes `summary.csv`, `alpha_cv.csv`, and `run_metadata.json`.

Primary smoke metric: voxelwise Pearson `r` between predicted and observed held-out BOLD time course. The summary reports mean, median, p95, and fraction-positive `r` across finite voxels. Full claims should later add blockwise permutation/FDR and ROI/localizer summaries.

## Local Debug Commands

Use these when the staged root is visible locally or on an AP.

Feature-extraction parse/model smoke with one story and one layer:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --out_dir results/sft_huth_lebel/word_states_debug \
  --stories sweetaspie \
  --arms base \
  --layers 24 \
  --limit_words 64 \
  --batch_size 2 \
  --device cuda \
  --dtype bfloat16 \
  --save_dtype float16 \
  --overwrite
```

CPU ridge debug on a small voxel subset after all three story feature files exist:

```bash
python src/sft/huth_lebel_smoke_encoding.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --features_dir results/sft_huth_lebel/word_states_smoke \
  --out_dir results/sft_huth_lebel/smoke_encoding_debug \
  --subjects UTS01 UTS02 UTS03 \
  --train_stories sweetaspie,againstthewind \
  --test_story wheretheressmoke \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 16,24,32 \
  --max_voxels 2000 \
  --ridge_solver auto \
  --overwrite
```

Full smoke run without voxel cap:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --out_dir /staging/s/suresh27/features/huth_lebel_smoke_llama31 \
  --stories sweetaspie,againstthewind,wheretheressmoke \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 16,24,32 \
  --max_context_tokens 512 \
  --batch_size 4 \
  --device cuda \
  --dtype bfloat16 \
  --save_dtype float16 \
  --overwrite
```

```bash
python src/sft/huth_lebel_smoke_encoding.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --features_dir /staging/s/suresh27/features/huth_lebel_smoke_llama31 \
  --out_dir results/sft_huth_lebel/smoke_encoding_llama31 \
  --subjects UTS01 UTS02 UTS03 \
  --train_stories sweetaspie,againstthewind \
  --test_story wheretheressmoke \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 16,24,32 \
  --ridge_solver auto \
  --save_voxel_corrs \
  --overwrite
```

## CHTC Scaling Plan

Keep data transfer CPU-only and model feature extraction GPU-only.

1. Reuse staged dataset root:
   - `/staging/s/suresh27/datasets/ds003020-smoke`
   - Verified by `chtc_5513059` with 15/15 files and exact byte match.

2. Submit GPU extraction jobs. For a minimal first pass, run one job over all smoke stories and all four arms. For more robust scheduling, split by story:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --out_dir /staging/s/suresh27/features/huth_lebel_smoke_llama31 \
  --stories $(STORY) \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 16,24,32 \
  --max_context_tokens 512 \
  --batch_size 4 \
  --device cuda \
  --dtype bfloat16 \
  --save_dtype float16 \
  --skip_existing
```

3. Submit CPU encoding jobs after features exist. The first CPU job can use `--max_voxels 2000`; the confirmatory smoke should remove the cap and optionally use `--save_voxel_corrs`.

4. For high-data scaling after smoke:
   - Stage `/staging/s/suresh27/datasets/ds003020-highdata`.
   - Keep `wheretheressmoke` held out, or move to multi-fold held-out stories.
   - Pre-register a fixed layer set or perform layer selection using training/validation stories only.
   - Use the same layer set, context length, alpha grid, trim, FIR delays, voxel mask, and evaluation stories for all arms.

## Fair Base vs LoRA Evaluation

Use identical settings across `base`, `lowLR`, `scrambled`, and `taskvec_a0p25`:

- Same model backbone and tokenizer.
- Same plain transcript stream; no chat template.
- Same TextGrid parser and word labels.
- Same `max_context_tokens=512` and BOS/reset convention.
- Same selected layers or same nested validation layer-selection rule.
- Same TR alignment, Lanczos window, FIR delays, trim, and response files.
- Same ridge alpha grid. Alpha can be selected separately per arm/subject/layer by train-story CV, but the held-out story must never be used for alpha, layer, ROI, or task-vector scale selection.
- Report paired deltas: `lowLR - base`, `scrambled - base`, `taskvec_a0p25 - base`, and `lowLR - scrambled`.

The `scrambled` arm is essential: if `lowLR` beats base but not scrambled, the effect is more likely a generic adapter/feature-geometry perturbation than semantic-coherence alignment.

## Huth And Fedorenko Interpretation

Huth angle: this is a natural-language voxelwise encoding benchmark in the style of LeBel/Huth and Antonello/Huth. A positive result means the LoRA arm's word-level features improve held-out BOLD prediction under a linear encoding model for passive narrative listening.

Fedorenko angle: the current smoke dataset does not by itself establish an Ev Fedorenko language-network claim. Strong Fedorenko-style claims require subject-specific language localizers, typically a sentences > nonword-lists contrast, and analysis in individual functional ROIs. Atlas language ROIs, broad temporal/IFG masks, or Huth semantic maps should be labeled exploratory unless localizer data are added.

Recommended reporting split:

- Whole-cortex or available cortical mask: primary smoke engineering metric.
- Huth-style semantic regions: exploratory, useful for continuity with semantic-map work.
- Fedorenko language fROIs: primary only if independently localized for these subjects or a later dataset.
- Auditory cortex/control regions: useful negative controls, because a semantic-coherence LoRA should not mainly improve low-level acoustic prediction.

## Pitfalls

- Encoding gains are not causal evidence that the LoRA made the model more human-like. They may reflect easier linear readout, feature scale, anisotropy, or ridge regularization effects.
- The smoke stories are too few for stable layer selection or broad claims.
- Choosing layers, alpha grids, task-vector scale, ROIs, or trim rules based on `wheretheressmoke` would leak test information.
- The TextGrid-to-token procedure approximates the heard transcript as space-joined word labels. This is appropriate for smoke execution, but exact high-data replication should compare against HuthLab preprocessing conventions.
- Long-context artifacts can inflate results. Keep context and story-start trimming identical across arms and pre-register any exclusion of early TRs.

## Primary Sources

- LeBel et al. 2023, Scientific Data: https://doi.org/10.1038/s41597-023-02437-z
- OpenNeuro ds003020: https://openneuro.org/datasets/ds003020
- HuthLab data/code: https://github.com/HuthLab/deep-fMRI-dataset
- Antonello, Vaidya, Huth 2023 scaling laws: https://arxiv.org/abs/2305.11863
- Scaling-law code/data: https://github.com/HuthLab/encoding-model-scaling-laws
- Tang, LeBel, Jain, Huth 2023 semantic decoder: https://www.nature.com/articles/s41593-023-01304-9
