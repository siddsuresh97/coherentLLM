# Coherence-SFT Experiment Log

Last updated: 2026-07-08

## Read this first

This is the running handoff log for follow-up work after Tasks 1-8 on branch
`coherence-sft`. New agents should read, in order:

1. `research/STATUS.md`
2. `results/sft_eval/REPORT.md`
3. this log
4. the active task briefs listed below

Tracking convention:
- Task briefs live in `research/CODEX_TASK_*.md`.
- Major experiment outputs live under `results/`.
- Every meaningful checkpoint should be committed and pushed to `origin/coherence-sft`.
- Append command summaries, artifact paths, and interpretation here after each major run.
- Do not treat a partial/stale artifact as a result unless this log explicitly marks it as valid.

## Active task briefs

- `research/CODEX_TASK_9_FMRI_AND_SEMANTIC_HUB.md`
  - Goal: test whether coherence-SFT improves brain predictivity and whether a
    format-invariant hidden-state semantic hub mediates it.
  - Status: brief created; no fMRI or hidden-state production runs launched yet.

- `research/CODEX_TASK_10_BENCHMARK_DROPS.md`
  - Goal: diagnose why lowLR/lowrank gain semantic/human-alignment tasks but drop
    on standard capability benchmarks, and propose mitigation experiments.
  - Status: brief created; existing results only so far.

## Current branch state

- Branch: `coherence-sft`
- Remote: `origin git@github.com:siddsuresh97/coherentLLM.git`
- Current checked commit before Task 9 docs: `76dbc9f`
- Worktree at log creation: clean.

## Sidecar agents

The following sidecar agents were launched on 2026-07-08. They were instructed not
to edit files; their memos should be integrated into this log and the task briefs.

| Agent | ID | Scope |
|---|---|---|
| Peirce | `019f4341-554b-7b50-b654-2e2dc09ffb2f` | Alex Huth / LeBel / UT Austin natural-language fMRI encoding literature |
| Kierkegaard | `019f4341-6757-78d3-9fb5-94f25969da6e` | Evelina Fedorenko / Ivanova language-network evaluation literature |
| Gibbs | `019f4341-806a-7d51-82f9-49fc95df0c88` | Local benchmark drop/gain diagnosis |
| Nietzsche | `019f4341-bb81-7200-ba91-a7c09b9d4aa2` | Semantic-hub hypothesis and tests |

## 2026-07-08 checkpoint: sidecar memo synthesis

Added [`research/LITERATURE_AND_EXPERIMENT_PLAN.md`](LITERATURE_AND_EXPERIMENT_PLAN.md)
as the handoff synthesis of the Huth/LeBel, Fedorenko/EvLab, benchmark-drop, and
semantic-hub sidecar memos. The README now links this plan through a collapsible
details panel.

## 2026-07-08 sidecar memo: Fedorenko / language-network evaluation

Source: Kierkegaard, agent `019f4341-6757-78d3-9fb5-94f25969da6e`.

Key implications for this project:

- Fedorenko/EvLab-style language-brain evaluation should use individually
  localized language-network masks when possible, not broad anatomical
  Broca/Wernicke parcels.
- Standard localizer contrast: `sentences > nonword lists`.
- Current implementations use group-constrained subject-specific masks across
  bilateral fronto-temporal language regions, often selecting the top localizer
  voxels per mask/subject.
- Primary language-fMRI metric should be held-out encoding predictivity:
  fit a linear mapping from LM features to neural responses on training stimuli,
  then score predicted vs observed held-out responses with Pearson `r`, optionally
  normalized by a noise ceiling.
- Fair comparison requirements: same stimuli, same train/test splits, same
  feature extraction/context, same mapping capacity, same layer-selection protocol,
  and paired deltas against the base model.
- Main confounds: lexical-semantic content can dominate brain scores; perplexity
  and next-word prediction quality correlate with brain predictivity; layer
  fishing inflates effects; prompt/chat-template artifacts can make model states
  incomparable; powerful nonlinear mappings can erase representational differences.
- Practical consequence: THINGS-fMRI Glasser Language ROI remains useful as an
  exploratory object-concept control, but it should not be framed as the primary
  Fedorenko-style language-network test.

Primary citations/links from memo:

- Fedorenko et al. 2010 / 2011 language localizer papers.
- Schrimpf et al. 2021, "The neural architecture of language".
- Ivanova et al. 2022, "Beyond linear regression".
- Kauf, Tuckute et al. 2024, lexical-semantic content in fMRI predictivity.
- Hosseini et al. 2024, controlled ANN training-scale/checkpoint analysis.

## 2026-07-08 sidecar memo: Huth / LeBel language-fMRI

Source: Peirce, agent `019f4341-554b-7b50-b654-2e2dc09ffb2f`.

Key implications for this project:

- Primary Huth-style language-fMRI candidate is LeBel et al. 2023,
  OpenNeuro `ds003020`: 8 participants listening to 27 natural narrative
  stories, about 6 hours per participant; `UTS01`, `UTS02`, and `UTS03` also
  have extended high-data story sets.
- Practical code/data entry points:
  - `https://openneuro.org/datasets/ds003020`
  - `https://github.com/HuthLab/deep-fMRI-dataset`
  - `https://github.com/HuthLab/encoding-model-scaling-laws`
- Primary metric: voxelwise Pearson correlation between predicted and held-out
  BOLD time courses, optionally noise-normalized (`CC_norm` in LeBel).
- Standard feature procedure: feed exact narrative text stream, extract hidden
  states per word/final word token, align by TextGrid word times, downsample to
  TRs, concatenate FIR delays such as 2/4/6/8 s, then fit ridge regression.
- Avoid chat templates for passive-listening fMRI; subjects heard narrative text,
  not instruction-formatted dialogue.
- Layer, ridge alpha, task-vector alpha, trimming rules, and ROIs must be chosen
  on training/validation only, never on the final test story.
- Recommended first high-power subjects are the extended LeBel subjects
  (`UTS01`/`UTS02`/`UTS03`), but a smaller first pass can use the repeated
  `wheretheressmoke` test story.
- Important caveat: a better encoding score may mean the LoRA features are more
  linearly decodable or better scaled for ridge, not necessarily more
  human-like. The safe claim is "improves held-out predictivity under a fixed
  linear encoding model" unless controls support a stronger interpretation.

Primary citations/links from memo:

- LeBel et al. 2023, Scientific Data, `https://doi.org/10.1038/s41597-023-02437-z`
- OpenNeuro `ds003020`, `https://doi.org/10.18112/openneuro.ds003020.v2.0.0`
- Antonello et al. 2023, scaling laws, `https://arxiv.org/abs/2305.11863`
- Tang et al. 2023 semantic decoder, Nature Neuroscience.
- Huth et al. 2016 semantic maps, Nature.

## 2026-07-08 sidecar memo: benchmark drops/gains

Source: Gibbs, agent `019f4341-806a-7d51-82f9-49fc95df0c88`.

Key implications for this project:

- The broad standard-capability run on disk is for `lowLR`; there does not appear
  to be a matching wide-benchmark run for `lowrank`.
- `lowLR` gains are strongly concentrated in semantic/human-behavior tasks:
  THINGS odd-one-out `+0.0842`, THINGS triplet similarity `+0.2279`, and gains
  on WordSim-353/MEN/MTurk-771/SimVerb-3500.
- `lowLR` broad capability has 0 gains, 4 flat groups, and 6 drops:
  ARC-Easy, ARC-Challenge, OpenBookQA, HellaSwag, WiC, and MMLU.
- Likely mechanism: SFT is very narrow and geometrically targeted
  (triplet/pairwise/feature prompts from one NOVA similarity target), while the
  LoRA update touches all attention and MLP projections and perturbs
  multiple-choice option ranking. The pattern is not total collapse because
  Winogrande, CommonsenseQA, TruthfulQA, and PIQA are flat.
- The earlier "retention-safe" claim was based on a narrow retention guard; the
  later wide benchmark exposed broader MMLU/ARC/WiC costs.
- The next mitigation evidence should be eval-only:
  1. run wide benchmark on task-vector `alpha=0.25`;
  2. run wide benchmark on `lowrank`;
  3. then consider alpha grid `0.10/0.20/0.25/0.35`, KL-to-base, replay, adapter
     sparsification, or adapter routing.

## 2026-07-08 sidecar memo: semantic hub

Source: Nietzsche, agent `019f4341-bb81-7200-ba91-a7c09b9d4aa2`.

Key implications for this project:

- Method basis is Wu, Yu, Yogatama, Lu, and Kim, "The Semantic Hub Hypothesis:
  Language Models Share Semantic Representations Across Languages and
  Modalities" (`arXiv:2411.04986`, ICLR 2025). The original method studies
  intermediate-layer shared representations across languages/modalities, plus
  logit-lens and cross-domain interventions. Our version adapts the method to
  elicitation formats: triplet, pairwise, and feature/listing prompts.
- Operational hub object: hidden state `h[model, layer, concept, format]`, where
  formats are triplet, pairwise, and feature/listing prompts.
- A semantic hub exists if same-concept codes across formats are closer than
  different-concept codes and if concept RDMs agree across formats in middle
  layers more than early/late layers.
- Primary metrics:
  - cross-format RSA between triplet/pairwise/feature concept RDMs;
  - linear CKA between format matrices with concepts as rows;
  - same-concept cross-format retrieval top-1/top-5 and margin;
  - concept-label vs format-label kernel alignment;
  - mid-layer AUC, e.g. L10-20 for Llama-3.1-8B.
- Expected pattern: `lowLR` and `lowrank` > `base` >> `scrambled`, with
  task-vector alpha monotonicity. If the gain appears only in final layers, it is
  more likely output/report formatting than a representational hub.
- Causal tests should avoid broad activation-addition during free generation
  because Task 8 showed that route collapses generation. Use targeted
  cross-format patching under logprob scoring instead.
- Brain bridge: compare format-specific and format-averaged hub-code RDMs to
  THINGS-fMRI RDMs; the strongest mechanistic link is if hub score predicts fMRI
  RSA across arms/layers.

## 2026-07-08 checkpoint: planning and data audit facts

Facts established locally before implementation:

- The original fMRI proposal is `research/FUTURE_brain_predictivity.md`.
- The semantic-hub proposal is `research/FUTURE_semantic_hub.md`.
- Existing THINGS-fMRI pipeline to reuse is in sibling project:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/`
- Existing Ch4 aggregation script:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/dissertation/scripts/compute_fmri_rsa.py`
- THINGS-fMRI metadata/betas are available locally under:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/`
- The SFT eval concept file has 128 lines in `data/scale128/concepts.csv`.
- Exact overlap with the 720 THINGS-fMRI concepts in `sub-01_StimulusMetadata.csv` is 90 concepts.
- Prior Ch4 noise ceilings warn against overclaiming ATL/language on object fMRI:
  Ventral Visual is high (`lower=0.229`, `upper=0.284`), while Language is low
  (`lower=0.010`, `upper=0.019`) and pooled ATL is near zero in the old object
  setup. Therefore Task 9 uses Ventral Visual as the primary ROI and treats ATL
  subregions / Language as exploratory.

Command used to establish overlap:

```bash
python - <<'PY'
import csv
from pathlib import Path
sft = Path('data/scale128/concepts.csv')
fmri = Path('/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv/sub-01_StimulusMetadata.csv')
s = {line.strip() for line in sft.read_text().splitlines() if line.strip()}
with fmri.open() as f:
    r = csv.DictReader(f)
    t = {row['concept'] for row in r}
print(len(s), len(t), len(s & t))
PY
```

Expected output: `128 720 90`.

## 2026-07-08 result: fMRI data audit

Command:

```bash
python src/sft/fmri_audit.py
```

Artifacts:

- `src/sft/fmri_audit.py`
- `results/sft_fmri/concept_overlap.csv`
- `results/sft_fmri/fmri_data_audit.json`
- `results/sft_fmri/REPORT.md`

Result:

- SFT concepts: 128.
- Common THINGS-fMRI concepts across subjects: 720.
- Primary exact held-out overlap: 90 concepts.
- Subjects `01`, `02`, and `03` each have 9,840 trials.
- Response H5 shapes:
  - sub-01: 211,339 voxels x 9,840 trials
  - sub-02: 226,950 voxels x 9,840 trials
  - sub-03: 189,164 voxels x 9,840 trials
- The 90 overlap concepts have no missing trials for any subject.
- Trial count per overlap concept ranges from 12 to 24, mean 13.867.
- Ventral Visual has thousands of voxels in every subject:
  - sub-01: 4,404
  - sub-02: 4,635
  - sub-03: 3,790

Interpretation:

- Track A is unblocked at the data level.
- Primary THINGS-fMRI RSA should use the 90 exact held-out overlap concepts.
- The next implementation step is hidden-state extraction for the 90 overlap
  concepts across `base`, `lowLR`, `lowrank`, `scrambled`, and task-vector arms.

## 2026-07-08 result: fMRI hidden-state extraction

Command run on H100 `opt-a007.discovery.wisc.edu`:

```bash
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 opt-a007.discovery.wisc.edu 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python src/sft/extract_fmri_hidden_states.py --arms base,scrambled,lowLR,lowrank,taskvec_a0p25,taskvec_a0p5,taskvec_a1p0 --batch_size 16 --overwrite'
```

Artifacts:

- `src/sft/extract_fmri_hidden_states.py`
- `results/sft_fmri/hidden_states/base.npz`
- `results/sft_fmri/hidden_states/scrambled.npz`
- `results/sft_fmri/hidden_states/lowLR.npz`
- `results/sft_fmri/hidden_states/lowrank.npz`
- `results/sft_fmri/hidden_states/taskvec_a0p25.npz`
- `results/sft_fmri/hidden_states/taskvec_a0p5.npz`
- `results/sft_fmri/hidden_states/taskvec_a1p0.npz`

Result:

- All seven planned arms completed.
- Each hidden-state array has shape `90 x 33 x 4096`.
- Prompt template is `concept_colon`: `Concept: {concept}`.
- Total hidden-state artifact size is about 126 MB.

## 2026-07-08 result: first-pass THINGS-fMRI RSA

Commands:

```bash
python src/sft/run_fmri_rsa.py
```

The RSA script was then optimized and rerun with batched HDF5 trial reads. The
optimized rerun completed in about 30 seconds and overwrote the RSA outputs.

Artifacts:

- `src/sft/run_fmri_rsa.py`
- `results/sft_fmri/rsa_by_layer.csv`
- `results/sft_fmri/rsa_best_layer.csv`
- `results/sft_fmri/rsa_summary.csv`
- `results/sft_fmri/rsa_meta.json`
- `results/sft_fmri/REPORT.md`

Core ROI result:

| ROI | Best arm | Best layer | Mean RSA | Delta vs base |
|---|---|---:|---:|---:|
| Ventral Visual | `lowrank` | 6 | 0.2013 | +0.0023 |
| Early Visual | `lowLR` | 30 | 0.0682 | +0.0003 |
| Dorsal Visual | `base` | 26 | 0.0632 | 0.0000 |
| ATL (Semantic) | `taskvec_a0p5` | 30 | 0.0543 | +0.0062 |
| Language | `lowrank` | 32 | 0.0345 | +0.0061 |
| Prefrontal | `taskvec_a1p0` | 31 | 0.0196 | +0.0077 |

Ventral Visual detail:

- `base`: 0.1990
- `lowLR`: 0.2010
- `lowrank`: 0.2013
- `taskvec_a0p25`: 0.1968
- `taskvec_a0p5`: 0.1973
- `taskvec_a1p0`: 0.1856
- `scrambled`: 0.1293

Interpretation:

- The end-to-end fMRI pipeline works.
- Ventral Visual is the robust object-RSA signal and the scrambled control is
  much worse than all meaningful arms.
- Coherence-SFT does not yield a large Ventral Visual gain in this first-pass
  90-concept, single-prompt analysis.
- ATL/Language aligned-arm increases are small and exploratory. They are not yet
  evidence for language-network improvement because the data are object-fMRI and
  the ROIs are not subject-specific language-localizer masks.
- Next best brain-facing step is semantic-hub extraction and a hub-score vs fMRI
  RSA bridge; next best stronger-neural step is a language-fMRI feasibility run.

Speed note:

- For `n=90`, HDF5 loading was the bottleneck, not RSA compute.
- `src/sft/run_fmri_rsa.py` now loads all selected trials per subject in one HDF5
  slice and then averages concepts in memory.
- For full-720, permutation-heavy, or bootstrap-heavy RSA, use a vectorized
  torch/CuPy/ThingsVision-style GPU path.

## 2026-07-08 active: lowrank wide-benchmark mitigation

Goal:

- Test whether the `lowrank` adapter preserves broad benchmark capability better
  than `lowLR` while retaining the semantic gains already seen in prior tasks.

Initial lowrank runs:

- A5000 GPU0: `zero_shot winogrande_5shot`, `gpu_mem_util=0.82`, vLLM auto batch.
- A5000 GPU1: `arc_25shot hellaswag_10shot`, `gpu_mem_util=0.82`, vLLM auto batch.
- H100: `mmlu_5shot`, `gpu_mem_util=0.88`, vLLM auto batch.

Observed runner behavior:

- Both A5000 jobs loaded the model but OOMed during prompt-logprob scoring, not
  during weight loading. This points to loglikelihood/sampler batch pressure.
- Retrying A5000 with `gpu_mem_util=0.65` and `batch_size=8` failed earlier:
  vLLM had no available KV cache blocks after weights/activation profiling.
- Current A5000 retry uses `gpu_mem_util=0.75` and `batch_size=4`, which should
  leave both KV cache and sampler headroom.
- H100 MMLU is running and using the GPU heavily. It is the long lane because
  the `limit=1000` standard MMLU setup expands to about 54k loglikelihood
  requests.

Commands currently/recently used:

```bash
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python src/sft/eval_wide_bench.py --states lowrank --groups zero_shot winogrande_5shot --gpu_mem_util 0.75 --batch_size 4 --no_summarize'
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=1 python src/sft/eval_wide_bench.py --states lowrank --groups arc_25shot hellaswag_10shot --gpu_mem_util 0.75 --batch_size 4 --no_summarize'
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 opt-a007.discovery.wisc.edu 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python src/sft/eval_wide_bench.py --states lowrank --groups mmlu_5shot --gpu_mem_util 0.88 --no_summarize'
```

Next benchmark step after these runs finish:

```bash
python src/sft/eval_wide_bench.py --states base lowLR lowrank --summarize_only
```

## Open decisions

- Whether to run Phase 1 RSA only on the 90 exact held-out overlaps, or also add
  a secondary full-720 analysis that is not leakage-clean.
- Whether Phase 2 language-fMRI should use LeBel/Huth first or a more turnkey
  OpenNeuro/Narratives route if LeBel preprocessing is costly.
- Whether mitigation work should prioritize alpha-sweep/task-vector strength,
  training recipe changes, or post-hoc adapter composition.

## 2026-07-08 active: semantic-hub implementation and speed audit

Semantic-hub code added locally:

- `src/sft/extract_semantic_hub_hidden_states.py`
- `src/sft/run_semantic_hub.py`

Method:

- Concepts: 128 held-out concepts from `data/scale128/concepts.csv`.
- Prompt spokes: triplet, pairwise, feature-listing.
- Triplet/pairwise neighbors are deterministic S*-based close/far selections
  from the held-out concept set.
- Metrics: cross-format RDM Spearman, linear CKA, same-concept retrieval, and
  concept-vs-format label alignment.

Validation:

- Syntax check passes for both scripts.
- Prompt/neighbor sanity check passed on sample held-out concepts.
- Fixed `run_semantic_hub.py` CSV writing to allow pair rows and mean rows with
  different metric columns.

ThingsVision/GPU RSA check:

- `thingsvision` is not installed in the active coherence env on either
  `rogers-gpu-1` or `opt-a007.discovery.wisc.edu`.
- For the completed 90-concept THINGS-fMRI RSA, wall time was dominated by HDF5
  beta loading. The implemented speedup batches selected HDF5 trial reads per
  subject and averages concepts in memory.
- Keep a torch/CuPy/ThingsVision-style GPU RDM backend on the list for full-720,
  bootstrap-heavy, or permutation-heavy RSA where pairwise distance math becomes
  a real cost.

## 2026-07-08 result: semantic-hub extraction and scoring

Command:

```bash
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python src/sft/extract_semantic_hub_hidden_states.py --batch_size 4 --overwrite'
python src/sft/run_semantic_hub.py
```

Artifacts:

- `results/sft_semantic_hub/hidden_states/`
- `results/sft_semantic_hub/hub_by_layer.csv`
- `results/sft_semantic_hub/hub_summary.csv`
- `results/sft_semantic_hub/REPORT.md`

Speed/utilization:

- A5000 GPU0 fit the extractor at `--batch_size 4` with about 18.4 GB used and
  high utilization.
- Hidden-state shape per arm is `3 x 128 x 33 x 4096`.

First-pass semantic-hub read:

| Arm | Mid RDM Spearman | Mid CKA | Mid Top-1 | Mid Top-5 |
|---|---:|---:|---:|---:|
| `base` | 0.3230 | 0.3890 | 0.0192 | 0.0800 |
| `scrambled` | 0.3472 | 0.4601 | 0.1094 | 0.2667 |
| `lowLR` | 0.6211 | 0.6899 | 0.0729 | 0.3063 |
| `lowrank` | 0.6637 | 0.7467 | 0.0968 | 0.4079 |
| `taskvec_a0p25` | 0.6770 | 0.7809 | 0.2062 | 0.5393 |
| `taskvec_a0p5` | 0.6372 | 0.7464 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | 0.4867 | 0.5505 | 0.0637 | 0.1896 |

Interpretation:

- Coherence-aligned/task-vector states substantially increase mid-layer
  cross-format invariance over base.
- `taskvec_a0p25` is the strongest mid-layer hub candidate by RDM, CKA, and
  same-concept retrieval.
- Concept-minus-format alignment stays negative for every arm, so this supports
  stronger format invariance rather than a proven concept-dominant hub.
- Next bridge: correlate hub metrics with fMRI RSA and test whether
  format-averaged hub RDMs predict fMRI better than single-format RDMs.

## 2026-07-08 active: lowrank/task-vector wide-bench follow-up

Reason:

- Diagnose why broad benchmarks drop after coherence SFT and whether lighter
  mitigation states (`lowrank`, `taskvec_a0p25`) keep semantic gains with less
  retention loss.

Completed partial results:

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
| `taskvec_a0p25` | 10-shot | HellaSwag | acc_norm | 0.680 |
| `scrambled` | zero-shot | PIQA | acc_norm | 0.540 |
| `scrambled` | zero-shot | OpenBookQA | acc_norm | 0.282 |
| `scrambled` | zero-shot | CommonsenseQA | acc | 0.197 |
| `scrambled` | zero-shot | WiC | acc | 0.481 |
| `scrambled` | zero-shot | TruthfulQA-MC2 | acc | 0.482 |

Initial interpretation:

- `lowrank` does not rescue WiC; it is near chance, like lowLR. That suggests
  the alignment objective may be harming lexical sense discrimination rather
  than merely overfitting a specific adapter rank.
- `lowrank` HellaSwag finishes at `acc_norm=0.683` under conservative A5000
  settings and is nearly flat against base (`0.685`), so the long-run OOMs were
  a runtime setting problem rather than a task-level evaluation failure.
- `taskvec_a0p25` HellaSwag also finishes flat (`acc_norm=0.680` vs base
  `0.685`), so the light task-vector mitigation preserves script/event
  plausibility in the same way as lowrank.
- `lowrank` still drops ARC-Easy (`0.705` vs base `0.850`), ARC-Challenge
  (`0.508` vs base `0.649`), and MMLU (`0.592` vs base `0.693`). This is
  task-family-specific retention damage, not a single global scoring bug.
- `taskvec_a0p25` improves OpenBookQA and PIQA relative to `lowrank`, but loses
  CommonsenseQA and TruthfulQA in this partial slice.
- WinoGrande is stable across `lowrank` and `taskvec_a0p25`, so the broad drop
  is not a uniform few-shot evaluation failure.
- `scrambled` zero-shot is much worse than lowrank/taskvec on PIQA,
  OpenBookQA, CommonsenseQA, and TruthfulQA, which argues that the semantic SFT
  signal is beneficial relative to random target geometry. But scrambled also
  drops broadly, so generic adapter/SFT perturbation contributes to the
  retention loss.
- Summary tables regenerated after the lowrank ARC completion:
  `results/sft_eval/wide_bench/summary.csv` and
  `results/sft_eval/wide_bench/raw_task_summary.csv`.
- Skill-level read written to
  `results/sft_eval/wide_bench/skill_map.md`: preserved skills are
  script/event plausibility, coreference/discourse commonsense, and much of
  physical affordance commonsense under task-vector mitigation; hurt skills are
  lexical sense discrimination, science/exam retrieval, and multiple-choice
  calibration.
- TruthfulQA-MC2 is not currently a large aligned-adapter failure. Lowrank is
  only `-0.010` and lowLR is `-0.017` versus base, both flat by the current
  threshold. Task-vector `alpha=0.25` (`-0.028`) and scrambled SFT (`-0.068`)
  are more negative, which points to calibration and plausible-false lure
  sensitivity rather than a clean loss of truth knowledge.

Speed/reliability notes:

- Harbor framework decision: do not move the current LM eval loop to Harbor.
  The relevant HARBOR work is about optimizing long-horizon agent harness
  flags, memory, caching, context management, and orchestration; it is not a GPU
  scheduler or a faster evaluator for ARC/MMLU/TruthfulQA. Keep using local
  direct vLLM/lm-eval for debugging and CHTC for parallel scale-out. Borrow only
  the lightweight idea of run manifests, gates, and runtime telemetry if
  orchestration gets messy.
- A5000 `gpu_mem_util=0.82` with vLLM auto batch OOMed during prompt-logprob
  scoring.
- A5000 `gpu_mem_util=0.65` plus `batch_size=8` left too little KV cache and
  failed vLLM initialization.
- A5000 `gpu_mem_util=0.75` plus `batch_size=4` completed lowrank
  zero-shot/WinoGrande and taskvec_a0p25 zero-shot/WinoGrande.
- Lowrank ARC+Hella at `gpu_mem_util=0.75`, `batch_size=4` OOMed near the end
  of ARC 25-shot. The retry is split by GPU: ARC uses `batch_size=2`,
  `gpu_mem_util=0.72`; HellaSwag uses `batch_size=4`, `gpu_mem_util=0.75`.
- Lowrank HellaSwag also OOMed at `gpu_mem_util=0.75`, `batch_size=4` around
  19% of loglikelihood requests. It was relaunched with `batch_size=2`,
  `gpu_mem_util=0.72`.
- Lowrank ARC 25-shot completed at `gpu_mem_util=0.72`, `batch_size=2`.
- Task-vector HellaSwag 10-shot completed at `gpu_mem_util=0.72`,
  `batch_size=2`.
- Current active A5000 lanes: GPU0 `taskvec_a0p25 arc_25shot`; GPU1
  `taskvec_a0p25 mmlu_5shot`, both at `gpu_mem_util=0.72`, `batch_size=2`.
- Scrambled zero-shot completed on A5000 at `gpu_mem_util=0.75`,
  `batch_size=4`; the A5000 rank-64 vLLM path is viable.
- H100 lowrank MMLU completed. It was the long lane because 5-shot MMLU expands
  to about 54k loglikelihood requests.
- `src/sft/eval_wide_bench.py` now exposes a `scrambled` state backed by
  `out/adapters_vllm_fixed/scrambled` so the broad benchmark can test whether
  drops come from semantic alignment or generic SFT perturbation.
- First `taskvec_a0p25` H100 MMLU launch entered pre-GPU uninterruptible I/O for
  about four minutes and was killed/relaunched after model and adapter path
  checks returned immediately.
- A second `taskvec_a0p25` H100 MMLU launch repeated the same pre-GPU I/O wait
  and was killed.
- A `scrambled` H100 zero-shot launch also entered pre-GPU I/O wait and was
  killed. Working hypothesis: `opt-a007` is unreliable for the 641 MB rank-64
  adapter paths, even though rank-16 lowrank MMLU ran cleanly there. Run
  rank-64 taskvec/scrambled broad benchmark lanes on `rogers-gpu-1` after the
  lowrank A5000 jobs finish, or stage/copy adapters to host-local storage first.
- Staging `scrambled` to `/tmp/ssuresh/coherence_adapters/scrambled` completed
  quickly (`657M`), but running vLLM from that local path still stalled before
  GPU allocation and was killed. The bottleneck is therefore not just symlinked
  adapter reads; treat `opt-a007` as unsuitable for rank-64 LoRA vLLM eval until
  we test a different vLLM/PEFT path.
- A rank-64 HF/PEFT smoke test for `scrambled` zero-shot on `opt-a007` was also
  stopped after multiple pre-GPU minutes. It was not in uninterruptible I/O, but
  it was too slow to be a useful fallback while A5000 vLLM lanes are available.
- User approved using CHTC for additional GPUs. Rule: debug locally/direct GPU
  first; only move known-good commands to CHTC for scale-out, and record submit
  files/logs/output paths in this log.

Current active runs:

```bash
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python src/sft/eval_wide_bench.py --states taskvec_a0p25 --groups arc_25shot --gpu_mem_util 0.72 --batch_size 2 --no_summarize'
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=1 python src/sft/eval_wide_bench.py --states taskvec_a0p25 --groups hellaswag_10shot --gpu_mem_util 0.72 --batch_size 2 --no_summarize'
```

Completed bookkeeping command after lowrank ARC:

```bash
python src/sft/eval_wide_bench.py --states base lowLR lowrank --summarize_only
```

Next after current taskvec lanes complete: parse taskvec ARC/Hella result JSONs,
update README/log, commit/push, then launch the next highest-value lane. Current
priority is `taskvec_a0p25 mmlu_5shot` if a long GPU lane is acceptable, or
scrambled ARC/Hella if the immediate question is random-perturbation control.

## 2026-07-08 result: fMRI x semantic-hub bridge

Command:

```bash
python src/sft/bridge_fmri_semantic_hub.py
```

Artifacts:

- `src/sft/bridge_fmri_semantic_hub.py`
- `results/sft_fmri_semantic_bridge/layer_join.csv`
- `results/sft_fmri_semantic_bridge/layer_correlations.csv`
- `results/sft_fmri_semantic_bridge/arm_summary_bridge.csv`
- `results/sft_fmri_semantic_bridge/arm_summary_correlations.csv`
- `results/sft_fmri_semantic_bridge/REPORT.md`

Initial read:

- The semantic-hub effect is robust inside the model, but it does not yet
  explain the first object-fMRI RSA pattern.
- Ventral Visual arm-level delta-vs-base vs mid-layer hub RDM:
  `Spearman r=0.600`, `n=6`, `p=0.208`.
- Ventral Visual arm-level delta-vs-base vs mid-layer hub top-1 retrieval:
  `Spearman r=-0.086`, `n=6`, `p=0.872`.
- Layer-grid delta correlations are positive for cross-format RDM in Ventral
  Visual (`r=0.401`, `n=192`) and ATL (`r=0.306`, `n=192`), but layer-grid
  points are not independent and should be treated as descriptive.
- The held-out fMRI regression below is the follow-up to this descriptive
  bridge.

## 2026-07-08 result: held-out fMRI semantic-hub regression

Command:

```bash
python src/sft/run_fmri_hub_regression.py --regions 'Ventral Visual,ATL (Semantic),Language' --out_dir results/sft_fmri_hub_regression
```

Artifacts:

- `src/sft/run_fmri_hub_regression.py`
- `results/sft_fmri_hub_regression/cv_model_scores.csv`
- `results/sft_fmri_hub_regression/cv_layer_summary.csv`
- `results/sft_fmri_hub_regression/cv_best_layer_summary.csv`
- `results/sft_fmri_hub_regression/cv_midlayer_summary.csv`
- `results/sft_fmri_hub_regression/cv_model_comparison.csv`
- `results/sft_fmri_hub_regression/regression_meta.json`
- `results/sft_fmri_hub_regression/REPORT.md`

Method:

- 90 fMRI-overlap concepts, subjects 01/02/03, primary ROIs only.
- Five concept-held-out folds; train and test distance pairs do not share
  concepts.
- Predictors are semantic-hub RDMs from `triplet`, `pairwise`,
  `feature_listing`, their averaged RDM, and an RDM of the averaged
  representation.
- Primary non-layer-fished read uses the mid-layer band 10-20.

Mid-layer held-out Pearson highlights:

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

Descriptive best-layer highlights:

- Ventral Visual best row: `taskvec_a0p25` `single_feature_listing`, layer 18,
  held-out Pearson `r=0.2034`.
- Ventral Visual lowrank: `single_feature_listing`, layer 18, `r=0.2010`.
- ATL best row: `taskvec_a0p5` `single_pairwise`, layer 23, `r=0.0565`.
- Language best row: `taskvec_a1p0` `mean_repr_plus_single_formats`, layer 18,
  `r=0.0765`.

Initial read:

- This weakens the hypothesis that the semantic hub explains the Ventral Visual
  fMRI effect. In Ventral Visual, the averaged/shared predictor is basically
  tied with or worse than the best single prompt-format spoke.
- ATL/Language show small aligned-state advantages for averaged/shared
  predictors, especially Language (`lowLR` +0.0087 over best single;
  `taskvec_a0p5` +0.0046), but absolute correlations are small.
- The safe conclusion is: semantic-hub invariance is real inside the model; the
  fMRI object-RSA result is mostly visual/object geometry; language/semantic ROI
  evidence remains exploratory and needs fixed-layer or nested-CV confirmation.
