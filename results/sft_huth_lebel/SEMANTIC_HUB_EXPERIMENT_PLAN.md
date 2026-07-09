# Semantic-Hub And Language-fMRI Experiment Plan

Created: 2026-07-08.

Goal: evaluate whether the coherence-SFT model family induces a reusable
semantic-hub / memory representation that improves brain-aligned natural
language encoding without merely creating a generic LoRA perturbation or an
answer-calibration artifact.

Owned model arms for this plan:

- `base`
- `lowLR`
- `scrambled`
- `taskvec_a0p25`

## One-Screen Status

Current state from repo reports:

- THINGS object-fMRI RSA is complete in `results/sft_fmri/REPORT.md`.
  It validates the RSA path and shows scrambled-control separation, but aligned
  arms are mostly flat versus base in Ventral Visual.
- Internal semantic-hub tests are complete in `results/sft_semantic_hub/` and
  `results/sft_semantic_hub_paper/`. Aligned/task-vector arms improve
  cross-format invariance and same-concept-vs-random margins, but strict
  same-concept-vs-S*-close-neighbor margins are small.
- Huth/LeBel smoke data are staged on CHTC:
  `/staging/s/suresh27/datasets/ds003020-smoke`.
  Staging cluster `5513059` verified 15/15 files and exact byte match.
- Huth/LeBel extraction and smoke-encoding scripts exist:
  `src/sft/huth_lebel_extract_word_states.py` and
  `src/sft/huth_lebel_smoke_encoding.py`.
- Huth/LeBel extraction debug `5513178` passed; staged-output extraction
  `5513245` failed on staging directory quota after writing only base features;
  bundle-output recovery extraction `5513306` passed with all 12 expected
  arm/story NPZs; CPU ridge encoding smoke `5513337` passed as a path-validation
  smoke. CPU-only scale check `5513350` is now queued/running for capped
  `UTS02,UTS03` encoding from the same feature bundle.
- The next expensive scientific step should scale cautiously: either uncap
  `UTS01` smoke voxels or run the same capped smoke on `UTS02`/`UTS03` before
  staging the high-data subset.

## Core Hypotheses

### H1: Language-fMRI Predictivity

Coherence-aligned states improve held-out voxelwise encoding of passive
narrative listening relative to base.

Support:

- `lowLR` or `taskvec_a0p25` improves held-out Pearson `r` over `base` in
  semantic/language-like cortex under the same ridge, delay, layer, and story
  split rules.
- `scrambled` does not show the same improvement.
- Gains are not concentrated in auditory/acoustic control regions.

Refute or weaken:

- No aligned arm beats base after matched layer/alpha selection.
- `scrambled` improves as much as aligned arms.
- Gains appear only in low-level auditory regions or only when selecting layers
  on the held-out test story.

### H2: Semantic-Hub Memory / MEP

The induced skill is a middle-layer, format-invariant semantic memory, not just
better final-token task formatting.

Support:

- Middle-layer same-concept cross-spoke similarity rises for `lowLR` and
  `taskvec_a0p25` relative to `base`, with `scrambled` lower.
- Logit-lens semantic anchors become stronger in middle layers before final
  answer-surface tokens dominate.
- Causal mid-layer patching between spokes changes task outputs in
  concept-consistent directions.

Refute or weaken:

- Effects appear only in final layers.
- Same-concept only beats random mismatches, but not S*-close mismatches.
- Interventions fail, or only broad activation addition works while degrading
  coherence/fluency.

### H3: Brain-Hub Bridge

The same hidden-state hub metrics that improve internal coherence explain
language-fMRI gains.

Support:

- Across arms/layers, hub metrics predict held-out Huth/LeBel encoding deltas.
- Format-averaged hub representations outperform the best single prompt spoke
  in semantic/language regions.
- The bridge is stronger for semantic/language regions than for auditory
  controls.

Refute or weaken:

- Huth/LeBel gains do not correlate with hub metrics.
- Single prompt spokes beat shared/averaged hub predictors.
- Hub metrics correlate more with benchmark damage than with fMRI or coherence
  gains.

## Fixed Analysis Rules

- Use plain narrative text, not chat templates.
- Keep the same tokenizer, word timing parser, context length, BOS/reset rule,
  FIR delays, ridge alpha grid, train/test split, voxel mask, and layer-selection
  rule for every arm.
- Hold out `wheretheressmoke` for smoke evaluation.
- Select ridge alpha and any layer choices only on training/validation stories,
  never on the held-out story.
- Report paired deltas: `lowLR - base`, `taskvec_a0p25 - base`,
  `scrambled - base`, and `lowLR - scrambled`.
- Label Fedorenko/EvLab claims correctly:
  - subject-specific localizer fROIs: strong language-network claim;
  - probabilistic atlas/LanA or broad parcels: exploratory language-like ROI;
  - Huth semantic maps: semantic-map continuity, not Fedorenko localizer proof.

## Experiment Matrix

| Scale | Purpose | Data | Arms | Layers | Output |
|---|---|---|---|---|---|
| Smoke A | Verify TextGrid parsing, model loading, staged model/adapters, and feature writing | `sweetaspie`, first 64 words | `base` first, then all four arms | `24` | passed in `5513178` |
| Smoke B | Verify word-to-TR alignment and ridge scoring | `sweetaspie`, `againstthewind` train; `wheretheressmoke` test | all four arms | `16,24,32` | passed in `5513306`/`5513337` |
| Medium | First interpretable language-fMRI result | all 3 smoke stories, no voxel cap | all four arms | `16,24,32` | full smoke encoding report |
| Full | Scientific high-data Huth/LeBel result | `UTS01`-`UTS03`, shared high-data stories | all four arms | fixed or nested-CV layers | subject/ROI/layer deltas |
| Hub Paper | Stronger semantic-hub evidence | held-out THINGS/NOVA concepts | all four arms | full layer grid | similarity, logit-lens, intervention reports |
| Brain-Hub Bridge | Test mechanism | Huth/LeBel encoding + hub metrics | all four arms | matched layer grid | metric-behavior-brain bridge tables |

## Phase 0: Do Not Re-Run What Is Already Done

Existing artifacts to reuse:

- Huth staging proof:
  `results/sft_huth_lebel/chtc_5513059/`
- Huth smoke plan:
  `results/sft_huth_lebel/ENCODING_PLAN.md`
- THINGS-fMRI result:
  `results/sft_fmri/REPORT.md`
- Semantic-hub hidden states:
  `results/sft_semantic_hub/hidden_states/`
- Paper-style similarity result:
  `results/sft_semantic_hub_paper/REPORT.md`
- fMRI hub regression:
  `results/sft_fmri_hub_regression/REPORT.md`

The completed `same_minus_random` result already supports a semantic clustering
effect. The small `same_minus_close_neighbor` margins mean we should not yet
claim exact concept-identity memory.

## Phase 1: Huth/LeBel Smoke A, Feature Extraction

Purpose: prove the staged data, model paths, adapter paths, TextGrid parser, and
feature writer work on a CHTC GPU before launching full story extraction.

Local/AP debug command when `/staging` is visible:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --model_path /staging/s/suresh27/models/llama31-8b-instruct \
  --hf_cache /staging/s/suresh27/hf_cache \
  --out_dir /staging/s/suresh27/features/huth_lebel_debug_llama31 \
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

All-arm smoke after `base` works:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --model_path /staging/s/suresh27/models/llama31-8b-instruct \
  --hf_cache /staging/s/suresh27/hf_cache \
  --out_dir /staging/s/suresh27/features/huth_lebel_debug_llama31 \
  --stories sweetaspie \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 24 \
  --limit_words 64 \
  --batch_size 2 \
  --device cuda \
  --dtype bfloat16 \
  --save_dtype float16 \
  --overwrite
```

Expected artifacts:

- `/staging/s/suresh27/features/huth_lebel_debug_llama31/<arm>/sweetaspie.npz`
- `/staging/s/suresh27/features/huth_lebel_debug_llama31/word_tables/sweetaspie.csv`
- pulled logs under `results/sft_huth_lebel/chtc_huth_extract_debug_<cluster>/`

Pass criteria:

- Each NPZ has finite hidden states with shape approximately
  `words x layers x hidden_dim`.
- The word table has the same word count used by the NPZ.
- No chat-template tokens are inserted.
- `scrambled`, `lowLR`, and `taskvec_a0p25` load from staged adapters rather
  than falling silently back to base.

Failure modes to check:

- CHTC job lands on low-memory GPU. Use
  `requirements = (TARGET.CUDAGlobalMemoryMb >= 40000)`.
- Adapter symlinks were staged incorrectly. Use `rsync -avPL --copy-links`.
- TextGrid tier parsing drops many words. Inspect `word_tables/*.csv`.
- Context overflow/truncation silently changes early words. Keep
  `--max_context_tokens` fixed and report it.

## Phase 2: Huth/LeBel Smoke B, Encoding

Purpose: verify alignment, FIR delays, ridge alpha selection, and held-out BOLD
scoring before spending GPU time on full feature extraction.

Feature extraction for full smoke stories:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --model_path /staging/s/suresh27/models/llama31-8b-instruct \
  --hf_cache /staging/s/suresh27/hf_cache \
  --out_dir /staging/s/suresh27/features/huth_lebel_smoke_llama31 \
  --stories sweetaspie,againstthewind,wheretheressmoke \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 16,24,32 \
  --max_context_tokens 512 \
  --batch_size 4 \
  --device cuda \
  --dtype bfloat16 \
  --save_dtype float16 \
  --skip_existing
```

CPU capped-voxel encoding smoke:

```bash
python src/sft/huth_lebel_smoke_encoding.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-smoke \
  --features_dir /staging/s/suresh27/features/huth_lebel_smoke_llama31 \
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

Confirmatory smoke without voxel cap:

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

Expected artifacts:

- `results/sft_huth_lebel/smoke_encoding_debug/summary.csv`
- `results/sft_huth_lebel/smoke_encoding_debug/alpha_cv.csv`
- `results/sft_huth_lebel/smoke_encoding_debug/run_metadata.json`
- `results/sft_huth_lebel/smoke_encoding_llama31/summary.csv`
- optional voxel correlations under
  `results/sft_huth_lebel/smoke_encoding_llama31/`

Pass criteria:

- All subjects and arms have finite held-out correlations.
- Ridge alphas are selected without looking at `wheretheressmoke`.
- `scrambled` is not treated as a positive result unless it separates from
  aligned arms in the expected direction.
- Summary includes `mean_r`, `median_r`, `p95_r`, `frac_positive`, subject,
  arm, and layer.

## Phase 3: Full Huth/LeBel High-Data Run

Purpose: produce the first serious scientific language-fMRI result.

Data:

- Stage high-data subset under
  `/staging/s/suresh27/datasets/ds003020-highdata`.
- Use `results/sft_huth_lebel/staging_manifest_highdata.csv`.
- Planned subset: `UTS01`-`UTS03`, 84 shared stories, 420 files, about
  76.86 GB.

Recommended split:

- Primary: hold out `wheretheressmoke` if preserving continuity with smoke.
- Better full run: use multi-fold held-out stories, grouped by story, with
  layer and ridge selected inside each training fold.

Full extraction command pattern:

```bash
python src/sft/huth_lebel_extract_word_states.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-highdata \
  --model_path /staging/s/suresh27/models/llama31-8b-instruct \
  --hf_cache /staging/s/suresh27/hf_cache \
  --out_dir /staging/s/suresh27/features/huth_lebel_highdata_llama31 \
  --stories STORY_BATCH_CSV_OR_COMMA_LIST \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 16,20,24,28,32 \
  --max_context_tokens 512 \
  --batch_size 4 \
  --device cuda \
  --dtype bfloat16 \
  --save_dtype float16 \
  --skip_existing
```

Full encoding command pattern:

```bash
python src/sft/huth_lebel_smoke_encoding.py \
  --ds_root /staging/s/suresh27/datasets/ds003020-highdata \
  --features_dir /staging/s/suresh27/features/huth_lebel_highdata_llama31 \
  --out_dir results/sft_huth_lebel/highdata_encoding_llama31 \
  --subjects UTS01 UTS02 UTS03 \
  --train_stories TRAIN_STORY_LIST \
  --test_story HELDOUT_STORY \
  --arms base,lowLR,scrambled,taskvec_a0p25 \
  --layers 16,20,24,28,32 \
  --ridge_solver auto \
  --save_voxel_corrs \
  --overwrite
```

Implementation gap before full run:

- `huth_lebel_smoke_encoding.py` currently encodes one held-out story with a
  smoke-style interface. Before full multi-fold use, either extend it to accept
  fold manifests or create `src/sft/huth_lebel_full_encoding.py`.

Expected full artifacts:

- `/staging/s/suresh27/features/huth_lebel_highdata_llama31/<arm>/<story>.npz`
- `results/sft_huth_lebel/highdata_encoding_llama31/fold_summary.csv`
- `results/sft_huth_lebel/highdata_encoding_llama31/voxel_corrs/*.npz`
- `results/sft_huth_lebel/highdata_encoding_llama31/roi_summary.csv`
- `results/sft_huth_lebel/highdata_encoding_llama31/layer_alpha_selection.csv`
- `results/sft_huth_lebel/highdata_encoding_llama31/REPORT.md`

## Phase 4: Huth Semantic Maps And Fedorenko-Style Constraints

Huth/Alexander G. Huth angle:

- Primary metric remains held-out voxelwise encoding.
- Secondary reports can group voxels by Huth/Gallant semantic-map regions if
  surfaces/masks are available.
- Compare aligned arms against base in regions expected to carry narrative
  semantic information, and include auditory/control regions.

Fedorenko/EvLab angle:

- Strong version: use subject-specific language fROIs from a language >
  nonword/control localizer.
- If no localizer is available for these subjects, use the Lipkin/Fedorenko
  probabilistic language atlas only as an exploratory mask.
- Report broad atlas parcels as "exploratory language-like" rather than
  "Fedorenko language network".

Model-side localizer control:

- Optional follow-up inspired by LLM-language-network work:
  present sentence and nonword-string stimuli to each arm, select the most
  sentence-selective units/layers, then rerun encoding with those units only.
- This tests whether coherence-SFT improves brain predictivity by changing
  language-selective units or by changing a broader semantic hub.

Expected artifacts:

- `results/sft_huth_lebel/roi_masks/README.md`
- `results/sft_huth_lebel/highdata_encoding_llama31/roi_summary.csv`
- `results/sft_huth_lebel/highdata_encoding_llama31/fedorenko_boundary_notes.md`

## Phase 5: Paper-Style Semantic-Hub Tests

Already done:

```bash
python src/sft/run_semantic_hub_paper_similarity.py \
  --arms base lowLR scrambled taskvec_a0p25 \
  --out_dir results/sft_semantic_hub_paper
```

Current read:

- `taskvec_a0p25` improves same-minus-random over base and has strong retrieval.
- Same-minus-S*-close is small, so exact concept-identity memory is not yet
  established.

Next tests to implement:

0. Huth scale check.
   - Monitor cluster `5513350`, which runs the same capped encoding smoke on
     `UTS02`/`UTS03` from the already-validated `5513306` feature bundle.
   - If it passes, decide whether to remove the 2000-voxel cap for `UTS01` or
     stage the high-data subset.

1. Logit-lens semantic anchoring.
   - Proposed script: `src/sft/run_semantic_hub_logit_lens.py`.
   - Output: `results/sft_semantic_hub_paper/logit_lens_by_layer.csv`.
   - Metric: concept/neighbor token margins versus answer-surface tokens.

2. Symbolic S* spoke.
   - Extend `src/sft/extract_semantic_hub_hidden_states.py` with a
     `symbolic_sstar` format.
   - Output:
     `results/sft_semantic_hub_paper/symbolic_similarity_by_layer.csv`.
   - Controls: close/far swap and predicate-order shuffle.

3. Causal patching/interventions.
   - Proposed script: `src/sft/run_semantic_hub_interventions.py`.
   - Output: `results/sft_semantic_hub_paper/intervention_by_layer.csv`.
   - First smoke: 32-64 concepts, layers 12,16,20,24,
     `base` and `taskvec_a0p25`.

Decision rule:

- Upgrade the claim from "semantic clustering" to "semantic-hub memory" only
  if logit-lens and causal tests localize to middle layers and separate
  aligned/task-vector arms from both `base` and `scrambled`.

## Phase 6: Brain-Hub-Behavior Bridge

Purpose: decide whether the hub is the mechanism behind fMRI gains and whether
it is related to benchmark drops.

Bridge table should join:

- Huth/LeBel encoding deltas by subject, region, arm, layer.
- Semantic-hub metrics:
  `same_minus_random`, `same_minus_close_neighbor`, logit-lens margins,
  intervention effect sizes.
- Existing THINGS-fMRI hub regression summaries from
  `results/sft_fmri_hub_regression/`.
- Benchmark skill diagnostics from
  `results/sft_eval/wide_bench/skill_diagnostics/`.

Expected output:

- `results/sft_huth_lebel/semantic_hub_bridge/metric_join.csv`
- `results/sft_huth_lebel/semantic_hub_bridge/arm_layer_correlations.csv`
- `results/sft_huth_lebel/semantic_hub_bridge/REPORT.md`

Interpretation:

- Best case: hub metrics predict semantic/human-alignment gains and Huth/LeBel
  semantic/language encoding gains, while weakly predicting benchmark damage.
- Risk case: hub metrics predict TruthfulQA/MMLU/ARC damage, suggesting the
  induced hub is entangled with answer calibration or multiple-choice ranking.

## CHTC Handoff Notes

Known working constraints:

- Use PyTorch CUDA image for jobs that install Python packages at runtime.
- Avoid `vllm/vllm-openai:v0.6.6.post1` for this repo's CHTC jobs; it produced
  a `huggingface-hub`/`transformers` incompatibility in prior smoke runs.
- For Llama jobs, require high-memory GPUs:
  `requirements = (TARGET.CUDAGlobalMemoryMb >= 40000)`.
- Prefer staging model/adapters once and reading from `/staging/s/suresh27`.
- Download datasets inside CHTC CPU jobs or stage explicit manifests. Do not
  assume the execute node sees local `/mnt/dv`.

Staged model/adapters expected:

- Base model: `/staging/s/suresh27/models/llama31-8b-instruct/`
- `lowLR`: `/staging/s/suresh27/adapters/lowLR/`
- `scrambled`: `/staging/s/suresh27/adapters/scrambled/`
- `taskvec_a0p25`: `/staging/s/suresh27/adapters/taskvec_a0p25/`

After any CHTC completion, pull logs and result tarballs into a cluster-specific
directory under `results/sft_huth_lebel/`, then write a short `REPORT.md`
before launching the next scale.

## Failure-Mode Checklist

- Data leakage: held-out story used for alpha/layer/ROI choice.
- Wrong task format: chat template or instruction prompt inserted into passive
  story text.
- Bad word alignment: TextGrid labels do not match transcript reconstruction.
- Token mismatch: final-subtoken hidden states are inconsistent across arms.
- Context artifacts: early tokens have less context or truncation differs by arm.
- Ridge overfit: alpha selected on test story or too many layers tuned on test.
- Generic perturbation: `scrambled` improves as much as aligned arms.
- ROI overclaim: broad atlas mask described as a Fedorenko fROI.
- Feature scale confound: one arm has larger norms and becomes easier for ridge.
- CHTC staging drift: feature jobs read a stale model/adapter or partial dataset.
- GPU waste: high-memory GPU allocated for CPU encoding or data transfer.

## Next Commands For A Future Agent

Check current job status before launching anything:

```bash
git status --short --branch
find results/sft_huth_lebel -maxdepth 2 -type f | sort
```

If the previous Huth extraction debug cluster finished, pull and inspect it
first:

```bash
mkdir -p results/sft_huth_lebel/chtc_huth_extract_debug_5513178
chtc-pull 'chtc-runs/coherence-huth-extract-debug-20260708-1945/huth_extract_debug_results.tgz' \
  results/sft_huth_lebel/chtc_huth_extract_debug_5513178/
chtc-pull 'chtc-runs/coherence-huth-extract-debug-20260708-1945/logs/' \
  results/sft_huth_lebel/chtc_huth_extract_debug_5513178/
tar -xzf results/sft_huth_lebel/chtc_huth_extract_debug_5513178/huth_extract_debug_results.tgz \
  -C results/sft_huth_lebel/chtc_huth_extract_debug_5513178/
find results/sft_huth_lebel/chtc_huth_extract_debug_5513178 -maxdepth 3 -type f | sort
```

If the debug result passes, run Phase 2 full smoke extraction and capped CPU
encoding. Commit only after the smoke report records:

- exact command;
- cluster IDs if CHTC was used;
- feature shapes;
- per-arm held-out correlations;
- whether `scrambled` separated from aligned arms;
- next scale decision.

## Reporting Template

Every run under this lane should add or update a small report with this shape:

~~~markdown
# <Run Name>

## Status

- Date:
- Commit:
- Cluster/session:
- Data root:
- Arms:
- Layers:

## Command

```bash
...
```

## Artifacts

- ...

## Quality Checks

- ...

## Result

- ...

## Interpretation

- Supports:
- Weakens:
- Next:
~~~

## Stop Conditions

Pause and report instead of scaling if:

- any arm silently falls back to base;
- feature shapes differ unexpectedly across arms for the same story;
- TextGrid parsing drops a large fraction of words;
- `scrambled` is the best arm in the smoke;
- the job requires a high-memory GPU but mostly performs CPU transfer/encoding;
- a result would require claiming Fedorenko fROIs without localizer evidence.
