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
  - Status: THINGS-fMRI RSA, semantic-hub extraction/scoring, hub-fMRI bridge,
    and paper-style similarity baseline are complete. Huth/LeBel `ds003020`
    smoke data are staged on CHTC and ready for a tiny encoding smoke.

- `research/CODEX_TASK_10_BENCHMARK_DROPS.md`
  - Goal: diagnose why lowLR/lowrank gain semantic/human-alignment tasks but drop
    on standard capability benchmarks, and propose mitigation experiments.
  - Status: lowrank/task-vector/scrambled partial wide-bench controls are
    running or complete. The remaining long mitigation row is
    `taskvec_a0p25` MMLU.

## Current branch state

- Branch: `coherence-sft`
- Remote: `origin git@github.com:siddsuresh97/coherentLLM.git`
- Current checked commit before Task 9 docs: `76dbc9f`
- Worktree at log creation: clean.

## Sidecar agents

The following sidecar agents were launched on 2026-07-08. Their early read-only
memos are integrated below. Later, longer-running goal assignments may edit only
their declared write scopes; main-lane commits should mention which agent-owned
artifacts were integrated.

| Agent | ID | Scope |
|---|---|---|
| Peirce | `019f4341-554b-7b50-b654-2e2dc09ffb2f` | Alex Huth / LeBel / UT Austin natural-language fMRI encoding literature |
| Kierkegaard | `019f4341-6757-78d3-9fb5-94f25969da6e` | Evelina Fedorenko / Ivanova language-network evaluation literature |
| Gibbs | `019f4341-806a-7d51-82f9-49fc95df0c88` | Local benchmark drop/gain diagnosis |
| Nietzsche | `019f4341-bb81-7200-ba91-a7c09b9d4aa2` | Semantic-hub hypothesis and tests |

Long-running goal assignments issued after CHTC access was validated:

| Agent | ID | Goal / write scope |
|---|---|---|
| Einstein | `019f4437-611a-7f71-bd3c-997517e90bc7` | Concept-vector steering for coherence and human-alignment contrasts. Owns `data/sft/steering_contrasts/`, `src/sft/extract_concept_vectors.py`, `src/sft/run_concept_steering_eval.py`, and `results/sft_eval/concept_steering/`. |
| Sagan | `019f4437-630c-7f02-ba72-0da158f1fa52` | CHTC MMLU shard execution and merge tooling. Owns `chtc/mmlu_shards/`, `src/sft/merge_mmlu_shards.py`, and `results/sft_eval/wide_bench/chtc_mmlu_shards/`. |
| Peirce | `019f4341-554b-7b50-b654-2e2dc09ffb2f` | Huth/LeBel encoding smoke plan and scripts. Owns new `src/sft/huth_lebel_*.py` files and `results/sft_huth_lebel/ENCODING_PLAN.md`. |
| Kierkegaard | `019f4341-6757-78d3-9fb5-94f25969da6e` | Benchmark skill diagnostics, including hurt/boosted skill grouping and TruthfulQA mechanism. Owns `results/sft_eval/wide_bench/skill_diagnostics/` and optional `src/sft/analyze_skill_diagnostics.py`. |
| Halley | `019f444f-f116-7520-91e9-e4efb5f0b6ef` | Long-running fMRI/semantic-hub literature and experiment-plan goal. Owns `results/sft_huth_lebel/SEMANTIC_HUB_EXPERIMENT_PLAN.md` and `results/sft_huth_lebel/LITERATURE_NOTES.md`. |
| Godel | `019f4450-0dcc-7b52-9af2-1a6e55a7c7d0` | Long-running concept-vector steering smoke/CHTC plan goal. Owns `chtc/concept_steering/`, `results/sft_eval/concept_steering/CHTC_PLAN.md`, and `results/sft_eval/concept_steering/SMOKE_STATUS.md`. |
| Mendel | `019f4450-2d32-7e11-983c-4a93e42671b5` | Long-running benchmark-drop mitigation and mechanism plan goal. Owns `results/sft_eval/wide_bench/skill_diagnostics/MITIGATION_PLAN.md` and `results/sft_eval/wide_bench/skill_diagnostics/HANDOFF.md`. |

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

## 2026-07-08 plan: paper-adapted semantic-hub tests

User requested that the semantic-hub track follow the paper's tests more
closely, not just our first prompt-format invariance proxy.

Plan artifact:

- `research/SEMANTIC_HUB_PAPER_ADAPTED_PLAN.md`

Key change:

- Current `results/sft_semantic_hub/` is Stage 0: triplet/pairwise/feature
  cross-format RDM, CKA, retrieval, and concept-vs-format alignment.
- Next Stage 1 should use the paper's relative similarity logic:
  same-concept cross-format similarity minus random, S*-close, and S*-far
  mismatches.
- Next Stage 2 should use logit-lens anchoring:
  middle-layer concept/neighbor token margins versus task-surface answer tokens,
  with careful prefix-space tokenization filtering.
- Next Stage 3 should add a symbolic S* spoke with swapped/shuffled controls.
- Next Stage 4 should run causal mid-layer patching/transplant interventions.

Readout:

- If middle-layer semantic margins and intervention effects rise for
  `taskvec_a0p25`/`lowrank` while scrambled lacks the effect, that is stronger
  evidence for an induced semantic hub than the current RDM-only result.
- If these same metrics track ARC/MMLU/WiC/TruthfulQA damage, then the induced
  hub may be entangled with output calibration and should become a mitigation
  target rather than only a mechanistic success.

## 2026-07-08 result: paper-style semantic-hub similarity baselines

Command:

```bash
python src/sft/run_semantic_hub_paper_similarity.py
```

Artifacts:

- `src/sft/run_semantic_hub_paper_similarity.py`
- `results/sft_semantic_hub_paper/similarity_by_layer.csv`
- `results/sft_semantic_hub_paper/similarity_summary.csv`
- `results/sft_semantic_hub_paper/similarity_meta.json`
- `results/sft_semantic_hub_paper/REPORT.md`

Mid-layer matched-vs-baseline summary:

| Arm | Same-random | Same-S*-close | Same-S*-far | Top-1 | Top-5 |
|---|---:|---:|---:|---:|---:|
| `base` | 0.0022 | -0.0002 | 0.0033 | 0.0192 | 0.0800 |
| `scrambled` | 0.0095 | 0.0026 | 0.0077 | 0.1094 | 0.2667 |
| `lowLR` | 0.0247 | 0.0013 | 0.0371 | 0.0729 | 0.3063 |
| `lowrank` | 0.0311 | 0.0003 | 0.0494 | 0.0968 | 0.4079 |
| `taskvec_a0p25` | 0.0380 | 0.0011 | 0.0548 | 0.2062 | 0.5393 |
| `taskvec_a0p5` | 0.0617 | 0.0070 | 0.0805 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | 0.0430 | 0.0099 | 0.0563 | 0.0637 | 0.1896 |

Interpretation:

- This mirrors the paper's relative-similarity test more closely than the first
  RDM/CKA pass.
- Aligned/task-vector arms clearly improve same-concept-vs-random margins over
  base, supporting a real semantic clustering effect.
- The stricter same-concept-vs-S*-close-neighbor margins are positive but
  small. This weakens any claim that the model has a fully exact
  concept-identity hub rather than a strong semantic-neighborhood hub.
- Next paper-style tests should be logit-lens semantic anchoring and causal
  mid-layer interventions.
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
| `taskvec_a0p25` | 25-shot | ARC-Easy | acc_norm | 0.808 |
| `taskvec_a0p25` | 25-shot | ARC-Challenge | acc_norm | 0.559 |
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
- `taskvec_a0p25` ARC improves substantially over lowrank but remains below
  base: ARC-Easy `0.808` vs lowrank `0.705` and base `0.850`; ARC-Challenge
  `0.559` vs lowrank `0.508` and base `0.649`.
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
- Task-vector ARC 25-shot completed at `gpu_mem_util=0.72`, `batch_size=2`.
- Current active A5000 lanes: GPU0 bounded TruthfulQA log-sample diagnostic;
  GPU1 `taskvec_a0p25 mmlu_5shot` at `gpu_mem_util=0.72`, `batch_size=2`.
- 2026-07-08 17:27 CDT clarification: the active MMLU lane is not a duplicate
  of the completed base/lowLR/lowrank MMLU rows. It is the missing
  `taskvec_a0p25` mitigation cell needed to decide whether alpha-0.25 preserves
  broad exam knowledge better than lowrank.
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

## 2026-07-08 result: TruthfulQA log-sample diagnostic

Purpose:

- Diagnose why TruthfulQA moves by decomposing MC2 into truthful-answer mass
  and false-answer pressure, instead of relying only on aggregate accuracy.
- Use bounded `--limit 200` slices for fast mechanism checks. These are
  diagnostics, not final benchmark numbers.

Commands:

```bash
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python -m lm_eval run --model vllm --model_args pretrained=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.72,max_model_len=2048,trust_remote_code=True --tasks truthfulqa_mc2 --limit 200 --batch_size 2 --apply_chat_template --log_samples --output_path /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_eval/wide_bench_diagnostics/truthfulqa_logsamples/base_limit200'
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python -m lm_eval run --model vllm --model_args pretrained=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659,dtype=bfloat16,tensor_parallel_size=1,gpu_memory_utilization=0.72,max_model_len=2048,trust_remote_code=True,enable_lora=True,lora_local_path=/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/out/adapters_mitigation_vllm/lowrank,max_lora_rank=16 --tasks truthfulqa_mc2 --limit 200 --batch_size 2 --apply_chat_template --log_samples --output_path /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_eval/wide_bench_diagnostics/truthfulqa_logsamples/lowrank_limit200'
python src/sft/analyze_truthfulqa_logsamples.py
```

Artifacts:

- `src/sft/analyze_truthfulqa_logsamples.py`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_logsamples/base_limit200/`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_logsamples/lowrank_limit200/`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_logsamples/taskvec_a0p25_limit200/`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_logsamples/scrambled_limit200/`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/REPORT.md`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/summary.csv`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/paired_delta_summary.csv`
- `results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/paired_deltas.csv`

Results on the bounded slice:

| Arm | n | MC2 acc | Truth log-odds | Best true - false | Best-is-true |
|---|---:|---:|---:|---:|---:|
| `base` | 200 | 0.5224 | 0.6684 | 0.7060 | 0.5250 |
| `lowrank` | 200 | 0.5528 | 1.0630 | 1.1408 | 0.5500 |
| `taskvec_a0p25` | 200 | 0.5267 | 0.0445 | 0.1110 | 0.5500 |
| `scrambled` | 200 | 0.4701 | 3.7499 | 3.8238 | 0.4750 |

Paired decomposition vs base:

| Arm | Delta MC2 acc | Delta truth log-odds | Delta true mass | Delta false pressure | Acc-down frac |
|---|---:|---:|---:|---:|---:|
| `lowrank` | +0.0304 | +0.3946 | -3.5444 | -3.9391 | 0.4450 |
| `taskvec_a0p25` | +0.0042 | -0.6239 | +4.3026 | +4.9265 | 0.5400 |
| `scrambled` | -0.0523 | +3.0815 | -59.0274 | -62.1090 | 0.5050 |

Read:

- Lowrank improves the bounded TruthfulQA slice by reducing false-answer
  pressure more than it reduces truthful-answer mass.
- This argues against a simple "truth knowledge is erased" mechanism for the
  lowrank arm. The mechanism is relative ranking/calibration over attractive
  false answers.
- `taskvec_a0p25` is aggregate-flat but raises false-answer pressure more than
  truthful-answer mass, especially on plausible misconception lures.
- Scrambled drops MC2 despite higher mean truth log-odds because it suppresses
  all answer likelihoods very strongly and creates catastrophic item-level
  flips.
- The largest lowrank regressions are specific safety/myth/misconception lures
  such as defibrillation for flatline, washing chicken, Latin-American language
  overgeneralization, and voodoo dolls.

## 2026-07-08 result: Huth/LeBel language-fMRI audit

Purpose:

- Answer why a Huth/LeBel lane was not already running in parallel.
- Make the language-fMRI path concrete and trackable before spending GPU time:
  check whether `ds003020` transcripts, TextGrids, audio, and BOLD/preprocessed
  responses are visible locally or ready to stage to CHTC.

Commands:

```bash
python src/sft/huth_lebel_audit.py --out_dir results/sft_huth_lebel --max_scan_depth 4
python -m py_compile src/sft/huth_lebel_audit.py
bash -n chtc/huth_lebel_audit/run_huth_lebel_audit.sh
chtc-master start  # user approved Duo push
chtc-master check
chtc-ssh 'hostname -f; condor_q -totals; echo STAGING=$STAGING'
chtc-ssh "mkdir -p ~/chtc-runs/coherence-huth-audit-20260708-1801/logs"
chtc-push src/sft/huth_lebel_audit.py 'chtc-runs/coherence-huth-audit-20260708-1801/huth_lebel_audit.py'
chtc-push chtc/huth_lebel_audit/ 'chtc-runs/coherence-huth-audit-20260708-1801/'
chtc-ssh "cd ~/chtc-runs/coherence-huth-audit-20260708-1801 && condor_submit huth_lebel_audit.sub"
```

Artifacts:

- `src/sft/huth_lebel_audit.py`
- `results/sft_huth_lebel/REPORT.md`
- `results/sft_huth_lebel/audit.json`
- `results/sft_huth_lebel/candidate_roots.csv`
- `chtc/huth_lebel_audit/README.md`
- `chtc/huth_lebel_audit/run_huth_lebel_audit.sh`
- `chtc/huth_lebel_audit/huth_lebel_audit.sub`

Result:

- The corrected audit checked 29 candidate roots and found zero plausible
  `ds003020` roots visible to this session.
- No local/staged TextGrid files, WAV stimuli, author-preprocessed `.hf5`
  responses, BIDS BOLD files, or `wheretheressmoke` test-story assets were
  found.
- A reusable local study hook does exist in the adjacent `vision_project`
  checkout:
  `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2/tribev2/studies/lebel2023bold.py`.
  It confirms the expected assumptions: TR=2s, subjects `UTS01`-`UTS08`,
  high-data subjects `UTS01`/`UTS02`/`UTS03`, TextGrid word timings, and
  `wheretheressmoke` as the repeated test story.

Read:

- The Huth/LeBel lane is now blocked on data staging/downloading, not on GPU
  availability or design.
- Once `ds003020` is staged, start with the CHTC CPU audit bundle, then run a
  small encoding smoke before scaling feature extraction across `(arm, subject,
  story split)`.
- Operational CHTC rule: run `chtc-master start` from this session, choose Duo
  Push, ask the user to approve the push, then verify with `chtc-master check`
  before `condor_submit`.
- CHTC access is now live for this session. `chtc-ssh` reached
  `ap2002.chtc.wisc.edu`, `condor_q -totals` showed an empty queue at submit
  time, and `$STAGING` was empty.
- CHTC audit smoke job submitted as cluster `5513006`. Initial status was idle
  and not held; `condor_q -better-analyze 5513006.0` reported 53 slots willing
  to run and 102 more that would match if drained. It then ran on
  `e4070.chtc.wisc.edu`, used 1 CPU / 0 GPUs, transferred outputs, and exited
  0. Pulled artifacts are under `results/sft_huth_lebel/chtc_5513006/`.
- The CHTC-returned audit also found zero plausible `ds003020` roots in the job
  sandbox/staging-visible paths. This confirms that the next Huth step is data
  staging, not more scheduler debugging.

## 2026-07-08 result: Huth/LeBel metadata audit and staging plan

Purpose:

- Verify the current OpenNeuro `ds003020` layout before writing a downloader.
- Convert DataLad/git-annex metadata into a concrete CHTC staging manifest so
  the language-fMRI lane can start with a bounded smoke test instead of trying
  to pull the full dataset.

Commands:

```bash
git clone --depth 1 https://github.com/OpenNeuroDatasets/ds003020.git /tmp/ds003020-git
python -m py_compile src/sft/huth_lebel_audit.py
python -m py_compile src/sft/plan_huth_lebel_staging.py
python src/sft/huth_lebel_audit.py --roots /tmp/ds003020-git --out_dir results/sft_huth_lebel/metadata_audit --max_scan_depth 2
python src/sft/plan_huth_lebel_staging.py --ds_root /tmp/ds003020-git --out_dir results/sft_huth_lebel
```

Artifacts:

- `src/sft/huth_lebel_audit.py`
- `src/sft/plan_huth_lebel_staging.py`
- `results/sft_huth_lebel/metadata_audit/REPORT.md`
- `results/sft_huth_lebel/STAGING_PLAN.md`
- `results/sft_huth_lebel/staging_manifest_smoke.csv`
- `results/sft_huth_lebel/staging_manifest_highdata.csv`
- `results/sft_huth_lebel/staging_summary.json`

Result:

- The audit now accepts both historical `derivative/...` and current
  `derivatives/...` OpenNeuro layouts.
- It also treats DataLad annex symlinks as file evidence, which is necessary
  for metadata-only clones where the large assets have not been downloaded.
- Metadata audit on `/tmp/ds003020-git` finds 84 TextGrids, 85 WAV entries,
  386 author-preprocessed HF5 entries, 495 raw BOLD entries, and
  `wheretheressmoke` test-story evidence.
- The dataset metadata is `doi:10.18112/openneuro.ds003020.v3.1.1`, license
  CC0, with references to HuthLab `deep-fMRI-dataset` and the Scientific Data
  paper.
- Smoke staging subset:
  `sweetaspie`, `againstthewind`, and `wheretheressmoke` for
  `UTS01`/`UTS02`/`UTS03`; 15 files; 7.88 GB total.
- Full high-data first-pass subset:
  all 84 shared preprocessed stories for `UTS01`/`UTS02`/`UTS03`; 420 files;
  76.86 GB total.
- Observed CHTC staging quota is 100 GB and 1000 files at
  `/staging/s/suresh27`, so the high-data subset should fit if only the
  manifest-listed WAV/TextGrid/HF5 assets are staged.

Read:

- The Huth lane has moved from "blocked by unknown data state" to "blocked by
  actual data download/staging." The layout, file sizes, subject/story subset,
  and CHTC audit path are now explicit.
- Do not launch GPU feature extraction until a staged smoke root passes
  `src/sft/huth_lebel_audit.py`.
- Next CHTC action should be CPU-only: run a containerized downloader with
  `git-annex`/DataLad or OpenNeuro tooling that materializes only
  `staging_manifest_smoke.csv` under `/staging/s/suresh27/datasets/ds003020-smoke`.

## 2026-07-08 active: CHTC Huth/LeBel smoke staging

Purpose:

- Materialize the 7.88 GB `ds003020` smoke subset on CHTC staging without using
  a GPU.

Commands:

```bash
chtc-ssh 'mkdir -p ~/chtc-runs/coherence-huth-stage-smoke-20260708-1817/logs'
chtc-push src/sft/huth_lebel_audit.py 'chtc-runs/coherence-huth-stage-smoke-20260708-1817/huth_lebel_audit.py'
chtc-push chtc/huth_lebel_stage_smoke/ 'chtc-runs/coherence-huth-stage-smoke-20260708-1817/'
chtc-push results/sft_huth_lebel/staging_manifest_smoke.csv 'chtc-runs/coherence-huth-stage-smoke-20260708-1817/staging_manifest_smoke.csv'
chtc-ssh 'cd ~/chtc-runs/coherence-huth-stage-smoke-20260708-1817 && condor_submit huth_lebel_stage_smoke.sub'
```

Artifacts:

- `chtc/huth_lebel_stage_smoke/`
- Remote run directory:
  `~/chtc-runs/coherence-huth-stage-smoke-20260708-1817`
- Submitted cluster: `5513020`

Current status:

- Initial status: idle, not held.
- `condor_q -better-analyze 5513020.0` reports 31 willing slots and 124 more
  if drained.
- Cluster `5513020` then started and was held because
  `docker://datalad/datalad:latest` was not pullable on CHTC. The bundle was
  updated to `docker://datalad/buildenv-git-annex:latest`, `datalad --version`
  was made optional, the held job was removed, and the corrected job was
  resubmitted as cluster `5513024`.
- Cluster `5513024` started successfully but tried to checkout the full
  OpenNeuro tree into `/staging/s/suresh27/datasets/ds003020-smoke` and hit the
  staging file-count quota while creating FreeSurfer directories. The failed
  partial staging tree was removed. The script now uses `git clone
  --no-checkout` plus sparse checkout of only `dataset_description.json` and
  the 15 manifest paths, and it has an EXIT trap to always return a tarball on
  future failures. Sparse retry submitted as cluster `5513028`.
- Cluster `5513028` passed sparse checkout and S3 remote setup but failed at
  `git annex init` because the container had no git author identity. Its trap
  also failed to transfer an output tarball because `OUT_DIR` was relative
  after `cd "$DATASET_ROOT"`. The script now sets a local git identity, anchors
  `OUT_DIR`/tarball paths to the scratch directory, writes exit status through
  an EXIT trap, and returns to scratch before audit/tar creation.
- The failed `5513028` job was removed, the partial staged tree was cleared,
  the fixed bundle was pushed, and the sparse staging retry was submitted as
  cluster `5513036`.
- If the job completes, pull `huth_lebel_stage_smoke_results.tgz` and logs,
  then rerun/record the staged-root audit before launching any GPU feature
  extraction.

## 2026-07-08 update: scrambled ARC complete, HellaSwag running

Purpose:

- Extend the random-label SFT control from zero-shot tasks to the science and
  event-plausibility groups, so we can separate useful semantic training from
  generic rank-64 adapter/SFT perturbation.

Commands:

```bash
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python src/sft/eval_wide_bench.py --states scrambled --groups arc_25shot --gpu_mem_util 0.72 --batch_size 2 --no_summarize'
ssh -F /dev/null -o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null rogers-gpu-1 'cd /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments && source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence && CUDA_VISIBLE_DEVICES=0 python src/sft/eval_wide_bench.py --states scrambled --groups hellaswag_10shot --gpu_mem_util 0.72 --batch_size 2 --no_summarize'
```

Artifacts:

- `results/sft_eval/wide_bench/runs/scrambled_arc_25shot/.../results_2026-07-08T18-38-38.225435.json`
- `results/sft_eval/wide_bench/skill_map.md`

Result:

- Scrambled ARC completed on A5000 GPU0 at `gpu_mem_util=0.72`,
  `batch_size=2`.
- ARC-Easy `acc_norm=0.306` versus base `0.850` (`delta=-0.544`).
- ARC-Challenge `acc_norm=0.225` versus base `0.649` (`delta=-0.424`).
- GPU0 was immediately reused for scrambled HellaSwag 10-shot with the same
  conservative settings.

Read:

- ARC/science multiple-choice ranking is highly perturbation-sensitive. The
  much milder `taskvec_a0p25` ARC drop therefore looks like a real mitigation
  effect, not just noise in the benchmark lane.
- The still-running scrambled HellaSwag lane will test whether script/event
  plausibility is preserved under random-label SFT or only under coherent
  semantic adapters.

## 2026-07-08 result: CHTC Huth/LeBel smoke staging complete

Purpose:

- Materialize only the 7.88 GB `ds003020` smoke subset on CHTC staging so the
  Huth/LeBel lane can move from data discovery to an encoding smoke.

Artifacts:

- Fixed stage script: `chtc/huth_lebel_stage_smoke/run_stage_smoke.sh`
- Successful cluster artifacts:
  `results/sft_huth_lebel/chtc_5513059/`
- Extracted stage summary:
  `results/sft_huth_lebel/chtc_5513059/extracted/sft_huth_lebel_stage_smoke/stage_summary.json`
- Extracted staged-root audit:
  `results/sft_huth_lebel/chtc_5513059/extracted/sft_huth_lebel_stage_smoke/audit/REPORT.md`

Result:

- Cluster `5513059` exited 0.
- Staged root: `/staging/s/suresh27/datasets/ds003020-smoke`.
- Files staged: 15 planned manifest files.
- Missing files: 0.
- Expected bytes and actual bytes both equal `7,877,643,435`.
- `du` reports `7.4G` for the staged smoke root.
- The staged-root audit finds 3 TextGrids, 3 WAVs, 9 author-preprocessed HF5
  files, and `wheretheressmoke` test-story evidence.

Read:

- The Huth/LeBel lane is no longer blocked on data discovery or smoke-subset
  staging. It is blocked on writing and validating the first tiny encoding
  smoke: extract no-chat-template narrative hidden states, align word features
  to TRs with FIR delays, and fit a held-out ridge model.
- The local `/mnt/dv` checkout is not visible to CHTC execute nodes; use
  `/staging/s/suresh27/datasets/ds003020-smoke` inside CHTC jobs.

## 2026-07-08 result: scrambled HellaSwag complete

Purpose:

- Test whether HellaSwag/script-event plausibility is genuinely preserved by
  coherent lowrank/task-vector states, or whether it is merely insensitive to
  rank-64 adapter perturbation.

Artifact:

- `results/sft_eval/wide_bench/runs/scrambled_hellaswag_10shot/__mnt__dv__wid__projects3__Rogers-muri-human-ai__shared_models__models--meta-llama--Llama-3.1-8B-Instruct__snapshots__0e9e39f249a16976918f6564b8830bc894c89659/results_2026-07-08T18-59-17.175041.json`

Result:

- `hellaswag` sample length: 1000.
- `acc=0.284`.
- `acc_norm=0.287`.
- Base HellaSwag `acc_norm=0.685`, so scrambled delta is `-0.398`.

Read:

- HellaSwag is not generically immune to adapter/SFT perturbation. It is
  preserved by coherent lowrank/task-vector states (`lowrank=0.683`,
  `taskvec_a0p25=0.680`) but badly hurt by scrambled SFT.
- This supports a skill-specific read: coherent semantic training preserves
  event/script plausibility, while incoherent/random-label geometry damages the
  same benchmark nearly as severely as ARC.

## 2026-07-08 active: task-vector MMLU throughput and CHTC probe

Purpose:

- Fill the missing `taskvec_a0p25` MMLU 5-shot row and avoid wasting local GPU
  time on a single slow monolithic run.

Local speed findings:

- A short `mmlu_anatomy` batch probe at `gpu_mem_util=0.82`, `batch_size=4`
  OOMed during loglikelihood scoring.
- The same short probe at `gpu_mem_util=0.72`, `batch_size=4` succeeded and
  reached about 8.4 requests/s.
- Full MMLU is still slow because long-context subjects such as
  `mmlu_professional_law` and `mmlu_moral_scenarios` run around 1.8-2.6
  requests/s and emit repeated `Context length ... exceeds max length (2047)`
  truncation warnings.

Active local fallback:

- The original monolithic `taskvec_a0p25` MMLU run was stopped after about
  20k/54k requests because it was too slow and produced no partial JSON.
- MMLU is now split into two balanced local A5000 shards:
  `taskvec_a0p25_mmlu_5shot_shard00of02` and
  `taskvec_a0p25_mmlu_5shot_shard01of02`.
- Both use `gpu_mem_util=0.72`, `batch_size=4`, and remain valid fallbacks.

CHTC scale-out:

- Base model staged to
  `/staging/s/suresh27/models/llama31-8b-instruct/`.
- Task-vector adapter staged to
  `/staging/s/suresh27/adapters/taskvec_a0p25/`.
- First GPU probe cluster `5513068` reached an A100-SXM4-80GB node but failed
  because the `vllm-openai` container entrypoint treated the script as API
  server arguments.
- The fixed probe uses `docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`,
  installs `vllm==0.6.6.post1` and `lm-eval==0.4.12` in scratch, and writes a
  result tarball through an EXIT trap.
- Fixed probe cluster `5513077` is active. If it succeeds, submit independent
  MMLU shards on CHTC and merge before treating the row as final.

## 2026-07-08 result: benchmark skill diagnostics

Source:

- Kierkegaard, agent `019f4341-6757-78d3-9fb5-94f25969da6e`.

Artifacts:

- `src/sft/analyze_skill_diagnostics.py`
- `results/sft_eval/wide_bench/skill_diagnostics/REPORT.md`
- `results/sft_eval/wide_bench/skill_diagnostics/task_skill_deltas.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/skill_summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/semantic_human_summary.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/truthfulqa_mechanism.csv`
- `results/sft_eval/wide_bench/skill_diagnostics/mmlu_extreme_drops.csv`

Verification:

```bash
python -m py_compile src/sft/analyze_skill_diagnostics.py
```

Read:

- The induced positive skill is representation-level semantic geometry:
  semantic coherence, THINGS-style human similarity, and cross-format
  consistency.
- The hurt families are sharp option-ranking tasks: ARC/OpenBookQA,
  broad MMLU exam knowledge, WiC lexical sense discrimination, and
  TruthfulQA-style plausible-lure calibration.
- Lowrank and `taskvec_a0p25` preserve HellaSwag/WinoGrande, but scrambled SFT
  badly damages HellaSwag. Preservation is therefore specific to coherent
  semantic states, not generic adapter perturbation.
- Worst MMLU drops are concentrated in moral/professional/formal/biomedical
  subtasks, especially moral scenarios, formal logic, and medical genetics.

## 2026-07-08 result: concept-vector steering lane scaffold

Source:

- Einstein, agent `019f4437-611a-7f71-bd3c-997517e90bc7`.

Commit:

- `99c0c60` pushed to `origin/coherence-sft`.

Artifacts:

- `data/sft/steering_contrasts/metadata.json`
- `data/sft/steering_contrasts/coherence.jsonl`
- `data/sft/steering_contrasts/human_alignment.jsonl`
- `data/sft/steering_contrasts/retention_probe.jsonl`
- `src/sft/extract_concept_vectors.py`
- `src/sft/run_concept_steering_eval.py`
- `results/sft_eval/concept_steering/REPORT.md`
- `results/sft_eval/concept_steering/sweep_config.json`

Validation:

```bash
python -m py_compile src/sft/extract_concept_vectors.py src/sft/run_concept_steering_eval.py
python src/sft/extract_concept_vectors.py --dry-run --concepts coherence human_alignment
python src/sft/run_concept_steering_eval.py --dry-run --smoke --layers 16 --alphas 0 2
```

Read:

- The lane is ready for a smoke-first CAA/ActAdd-style sweep over coherence and
  human-alignment contrast vectors.
- No GPU extraction or steering sweep has run yet; the report contains the
  commands and should be used before spending long GPU time.

## 2026-07-09 active: concept-vector steering CHTC smoke

Artifacts:

- `chtc/concept_steering/concept_steering_smoke.sub`
- `chtc/concept_steering/run_concept_steering.sh`
- `results/sft_eval/concept_steering/REPORT.md`
- `results/sft_eval/concept_steering/SMOKE_STATUS.md`

Submission:

```bash
chtc-ssh 'cd ~/chtc-runs/coherence-concept-steering-20260709-005726 && condor_submit concept_steering_smoke.sub'
```

Run:

- cluster: `5513235`
- remote directory: `~/chtc-runs/coherence-concept-steering-20260709-005726`
- requirement: `TARGET.CUDAGlobalMemoryMb >= 40000`
- smoke design: `coherence` and `human_alignment`, first 4 train pairs, layer
  `24`, alphas `-2,0,2`, 2 items each for coherence/alignment/retention.
- first poll: idle but satisfiable; no worker stdout/stderr yet.

Next read:

- pull
  `concept_steering_smoke_layer24_alpha_neg2_0_pos2_results.tgz`
  after completion.
- check all runner exit status files are `0`.
- compare target rows against alpha `0` and verify retention rows do not
  collapse before submitting the full layer/alpha sweep.

Update:

- First smoke cluster `5513235` completed with wrapper exit status `66`.
- Pulled failure artifacts to `results/sft_eval/concept_steering/chtc/5513235/`.
- Failure cause: worker `jcaicedogpu0002.chtc.wisc.edu` had a valid L40S GPU
  but did not expose `/staging/s/suresh27/models/llama31-8b-instruct/config.json`;
  that node advertises `HasChtcStaging=false`.
- Submit files now require `TARGET.HasChtcStaging =?= true`.
- Retry cluster `5513261` submitted from
  `~/chtc-runs/coherence-concept-steering-20260709-010738`.
- `5513261` ran on `gpu5000.chtc.wisc.edu`, NVIDIA H200, and saw staged
  storage, but failed during extraction because pip installed
  `torch==2.13.0+cu130` into the overlay and shadowed the container's working
  torch/torchvision stack.
- Runner fix: use `pip --no-deps` and explicit non-torch dependencies.
- Fixed no-deps retry cluster `5513276` ran from
  `~/chtc-runs/coherence-concept-steering-20260709-011336` on
  `dbrundagegpu5000.chtc.wisc.edu`, an NVIDIA L40S.
- Pulled artifacts to `results/sft_eval/concept_steering/chtc/5513276/`.
- Status files all passed: `exit_status.txt == 0`,
  `extract_exit_status.txt == 0`, `eval_exit_status.txt == 0`, and
  `pip_install_exit_status.txt == 0`.
- The smoke wrote coherence and human-alignment CAA vectors plus the expected
  18 forced-choice eval rows.
- Positive alpha moved matching target margins upward relative to alpha `0`
  (`coherence` on coherence `+0.0625`; `human_alignment` on human-alignment
  `+0.0419`) while retention positive preference stayed at `1.00` for both
  vectors across `-2/0/2`.
- Interpretation: operational smoke passed, but the sample is only `n=2` per
  eval set and specificity is not yet clean, so treat the effect as a scale-up
  signal rather than a result.
- Submitted bounded sweep cluster `5513297` from the same patched run
  directory with layers `12,16,20,24`, alphas `-4,-2,0,2,4`, and full
  contrast/eval sets.

## 2026-07-09 active: Huth/LeBel extraction debug passed; three-story smoke running

Completed debug extraction:

- CHTC cluster: `5513178`
- Remote directory: `~/chtc-runs/coherence-huth-extract-debug-20260708-1945`
- Local artifact directory:
  `results/sft_huth_lebel/chtc_huth_extract_debug_5513178/`
- Condor result: normal termination, return value `0`
- Wrapper status: `exit_status.txt == 0`
- Extractor status: `extract_exit_status.txt == 0`
- Outputs: four arm files for `sweetaspie` at layer 24, first 64 words:
  `base`, `lowLR`, `scrambled`, `taskvec_a0p25`
- NPZ validation: each file has `hidden: (64, 1, 4096)`, plus word timing
  arrays and layer metadata.

New Huth smoke submission:

- Added/updated CHTC wrappers under `chtc/huth_lebel_smoke/`.
- Submitted cluster: `5513245`
- Remote directory: `~/chtc-runs/coherence-huth-extract-smoke-20260709-005926`
- Requirement: `TARGET.CUDAGlobalMemoryMb >= 40000`
- Job purpose: extract all three smoke stories
  `sweetaspie,againstthewind,wheretheressmoke` for arms
  `base,lowLR,scrambled,taskvec_a0p25` and layers `16,24,32`.
- First placement: ran on `mkhodakgpu4000.chtc.wisc.edu`, NVIDIA L40S
  with `45468` MB advertised GPU memory.
- Outcome: model loading succeeded and the job wrote the three `base` story
  features for layers `16,24,32`, then exited `1` with
  `OSError: [Errno 122] Disk quota exceeded` while creating
  `/staging/s/suresh27/features/huth_lebel_smoke_llama31/lowLR`.
- Pulled failure/status artifacts to
  `results/sft_huth_lebel/chtc_huth_extract_smoke_5513245/`.
- Staged feature evidence from `5513245`: `base/sweetaspie.npz`
  (`697 x 3 x 4096`), `base/againstthewind.npz` (`842 x 3 x 4096`), and
  `base/wheretheressmoke.npz` (`1859 x 3 x 4096`).
- Recovery: `run_extract_smoke.sh` now supports `SEED_FEATURE_DIR`; it copies
  the successful staged base features into job scratch, uses `--skip_existing`,
  extracts the missing adapter arms, and returns all features inside
  `huth_extract_smoke_results.tgz`.
- Active retry: cluster `5513306`, remote directory
  `~/chtc-runs/coherence-huth-extract-smoke-retry-20260709-013204`.
- Duplicate retry `5513310` was submitted with identical settings and removed
  while idle with `condor_rm 5513310`.

Duplicate avoided:

- Parallel cluster `5513244` was submitted from
  `~/chtc-runs/coherence-huth-extract-smoke-20260709-010014`, but held
  immediately before model work because Condor could not transfer the expected
  `huth_extract_smoke_results.tgz`.
- Removed the held duplicate with `condor_rm 5513244`; keep `5513245` as the
  active extraction.

Next read:

- Pull `huth_extract_smoke_results.tgz` after cluster `5513306` completes.
- Check `extract_exit_status.txt == 0` and verify `npz_shapes.tsv` lists all
  12 arm/story NPZ files.
- Submit `huth_encoding_smoke_bundle.sub` from the same run directory so CPU
  encoding reads the returned feature bundle instead of writing new staging
  directories.

## 2026-07-09 active: MMLU CHTC smoke passed; full shards submitted

Completed smoke:

- CHTC cluster: `5513195`
- Remote directory: `~/chtc-runs/coherence-mmlu-shards-20260709-004133`
- Local artifact directory:
  `results/sft_eval/wide_bench/chtc_mmlu_shards/coherence-mmlu-shards-20260709-004133/`
- GPU: `gpu4003.chtc.wisc.edu`, NVIDIA H100 80GB HBM3
- Wrapper status: `exit_status.txt == 0`
- `lm_eval` status: `lm_eval_exit_status.txt == 0`
- Smoke task: `mmlu_abstract_algebra`, `limit=20`, `found_n=20`,
  `missing_tasks=[]`
- Smoke metric: `acc,none=0.3`

Full shard submission:

- Submitted cluster: `5513268`
- Jobs: 8 shard procs, `0` through `7`
- First poll: all 8 idle, no hold reasons.
- Update: procs `0`, `2`, and `4` started on staging-visible hosts and remain
  the original full-shard work to preserve. Procs `1`, `3`, `5`, `6`, and `7`
  landed on non-staging hosts and exited `66` before model load because
  `/staging/s/suresh27/models/llama31-8b-instruct` was unavailable.
- Fix: `chtc/mmlu_shards/mmlu_smoke.sub` and `mmlu_full.sub` now require
  `TARGET.HasCHTCStaging == true` in addition to the 40GB GPU-memory floor.
- Retry submitted: cluster `5513291`, using
  `mmlu_full_retry_missing_staging.sub` and
  `mmlu_full_retry_missing_staging_manifest.tsv` for exactly shards `1`, `3`,
  `5`, `6`, and `7`.
- Commands run:
  `chtc-push chtc/mmlu_shards/mmlu_full_retry_missing_staging_manifest.tsv 'chtc-runs/coherence-mmlu-shards-20260709-004133/'`;
  `chtc-push chtc/mmlu_shards/mmlu_full_retry_missing_staging.sub 'chtc-runs/coherence-mmlu-shards-20260709-004133/'`;
  `chtc-ssh 'cd ~/chtc-runs/coherence-mmlu-shards-20260709-004133 && condor_submit mmlu_full_retry_missing_staging.sub'`.
- Next read: monitor `5513268` and `5513291`, pull original and retry
  tarballs into
  `results/sft_eval/wide_bench/chtc_mmlu_shards/coherence-mmlu-shards-20260709-004133/`,
  then merge with `src/sft/merge_mmlu_shards.py`.

## 2026-07-08 active: fixed CHTC GPU smoke retries

MMLU:

- Unconstrained environment probe `5513077` exited 0 but landed on an 11GB GTX
  1080 Ti. It proved the PyTorch CUDA image can install/import
  `vllm==0.6.6.post1` and `lm-eval==0.4.12`, not that full MMLU can run.
- MMLU smoke `5513124` used the high-memory GPU constraint and landed on an
  NVIDIA L40, but failed before `lm_eval` because
  `docker://vllm/vllm-openai:v0.6.6.post1` had incompatible
  `huggingface-hub==1.22.0` for its installed `transformers`.
- Submit files now use
  `docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel` and install Python
  packages into job scratch.
- Fixed MMLU smoke retry `5513177` ran on `gpu5001.chtc.wisc.edu`, an NVIDIA
  H200 with `143158` MB advertised GPU memory. It validated staged model/adaptor
  access, UUID-style `CUDA_VISIBLE_DEVICES` normalization, package pinning, and
  model weight loading. It failed during vLLM LoRA/Triton profiling because the
  runtime image lacked a C compiler.
- Active MMLU smoke retry: devel-image cluster `5513195`, run directory
  `~/chtc-runs/coherence-mmlu-shards-20260709-004133`. It is idle but
  satisfiable at last check; `condor_q -better-analyze` reported six immediately
  willing high-memory GPU slots and 46 more if drained.

Huth/LeBel:

- Added `chtc/huth_lebel_smoke/` with a debug extraction job over
  `sweetaspie`, first 64 words, layer 24, and arms
  `base,lowLR,scrambled,taskvec_a0p25`.
- Patched `src/sft/huth_lebel_extract_word_states.py` with `--model_path` and
  `--hf_cache` so CHTC jobs can use staged model paths directly.
- Staged missing CHTC adapters:
  `/staging/s/suresh27/adapters/lowLR` and
  `/staging/s/suresh27/adapters/scrambled`; each has a real 641 MB
  `adapter_model.safetensors`. `taskvec_a0p25` was already staged.
- The first Huth debug submission `5513145` used the old vLLM image and was
  removed after the MMLU image incompatibility was identified.
- Fixed Huth debug retry: cluster `5513178`, run directory
  `~/chtc-runs/coherence-huth-extract-debug-20260708-1945`.
- User log confirms it is executing on `gpu4000.chtc.wisc.edu`, an NVIDIA L40
  with `45468` MB advertised GPU memory. This is sufficient for the 64-word,
  one-layer debug extraction if the staged model/adapters load cleanly.

Current rule:

- Use `TARGET.CUDAGlobalMemoryMb >= 40000` for CHTC Llama GPU jobs. Do not
  relax this to make jobs start faster; small GPUs produce false progress.
