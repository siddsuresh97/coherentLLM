# Paper Method Map For Semantic Hub MEMP

Created: 2026-07-09

This file maps the methods from Wu et al., "The Semantic Hub Hypothesis"
(`arXiv:2411.04986`, ICLR 2025) onto this repo's current semantic-hub
experiment lane. It is a method map, not a new result.

## Source Limits

The arXiv abstract and HTML v3 were reachable during this handoff. Local state
was checked in:

- `results/sft_semantic_hub/REPORT.md`
- `results/sft_semantic_hub_paper/REPORT.md`
- `research/lit_notes/semantic_hub_2411_04986.md`
- `research/literature_eval_design.md`
- `README.md`

The paper tests cross-language and cross-modal shared representations. This
repo currently tests a narrower text-only analogue: same concept, different
prompt spokes.

## Mapping Table

| Paper method | Paper object | Repo analogue | Current status | Next artifact |
|---|---|---|---|---|
| Relative similarity, Eq. 1 | matched cross-language/modality inputs vs nonmatches | same held-out concept across triplet/pairwise/feature spokes vs random, S*-close, and S*-far controls | implemented in `src/sft/run_semantic_hub_paper_similarity.py`; report exists in `results/sft_semantic_hub_paper/` | keep as baseline; add lexical/category controls |
| Middle-layer hub localization | layer curves showing strongest matched-over-baseline effects in intermediate layers | Llama-3.1-8B layers `10:20`, with all layers retained | implemented for RDM/CKA/retrieval and paper-style similarity | add cluster/permutation statistics over layer curves |
| RDM/RSA geometry | structure agreement across equivalent inputs | cross-format concept RDM Spearman | implemented in `src/sft/run_semantic_hub.py` | report alongside paper-style deltas |
| CKA geometry | representation-space similarity across spokes | linear CKA across format-specific concept matrices | implemented in `src/sft/run_semantic_hub.py` | use as secondary geometry metric |
| Matched retrieval | matching equivalent representations | cross-format same-concept top-1/top-5 and margin | implemented | add hubness/k-occurrence artifact check |
| Logit-lens anchoring, Eq. 2-3 | hidden states closer to dominant-language semantic tokens than surface-form tokens | concept/neighbor token logits vs `A/B/yes/no` or option-string surface logits | not implemented | `results/sft_semantic_hub_paper/logit_lens_by_layer.csv` |
| Formal/code spoke | non-natural-language surface forms sharing semantics with text | symbolic S* spoke with close/far neighbors and similarities | not implemented | `results/sft_semantic_hub_paper/symbolic_similarity_by_layer.csv` |
| Causal intervention | ActAdd/patching in shared space changes output in another data type | same-concept spoke patch, concept transplant, and activation addition across prompt formats | not implemented | `results/sft_semantic_hub_paper/intervention_by_layer.csv` |
| Negative controls | unrelated inputs, baselines, surface confounds | scrambled arm, random concept, S*-close concept, shuffled vector, wrong layer, lexical/category matched controls | partial | `controls_by_layer.csv`, `hubness_by_layer.csv` |

## Current Result Anchors

From `results/sft_semantic_hub/REPORT.md`:

- Best mid-layer RDM/CKA/retrieval arm: `taskvec_a0p25`.
- `taskvec_a0p25` mid-layer RDM `0.6770`, CKA `0.7809`, top-1 `0.2062`,
  top-5 `0.5393`.
- Base mid-layer RDM `0.3230`, CKA `0.3890`, top-1 `0.0192`, top-5 `0.0800`.
- Concept-minus-format alignment remains negative for every arm.

From `results/sft_semantic_hub_paper/REPORT.md`:

- Best mid-layer same-minus-random: `taskvec_a0p5` at `0.0617`.
- Best mid-layer same-minus-S*-close: `taskvec_a1p0` at `0.0099`.
- `taskvec_a0p25` has the best retrieval tradeoff but only `0.0011` on
  same-minus-S*-close.

Interpretation: the current run supports stronger cross-format semantic
clustering in aligned/task-vector arms. It does not yet prove exact-concept
hubness or causal use of the hub.

## Required Smoke Commands

CPU geometry smoke:

```bash
python src/sft/run_semantic_hub.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_geometry_smoke \
  --arms base taskvec_a0p25 scrambled \
  --mid_layers 10:20
```

CPU paper-style similarity smoke:

```bash
python src/sft/run_semantic_hub_paper_similarity.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_similarity_smoke \
  --arms base taskvec_a0p25 scrambled \
  --mid_layers 10:20 \
  --n_boot 100
```

GPU hidden-state smoke if new spokes are added:

```bash
python src/sft/extract_semantic_hub_hidden_states.py \
  --model llama-3.1-8b-instruct \
  --concepts data/scale128/concepts.csv \
  --out_dir /tmp/semantic_hub_hidden_smoke \
  --formats triplet,pairwise,feature_listing \
  --arms base,taskvec_a0p25 \
  --batch_size 4 \
  --max_length 192 \
  --dtype bfloat16 \
  --save_dtype float16 \
  --device cuda \
  --overwrite
```

## Script TODOs

Add these as semantic-hub-specific helpers only:

- `src/sft/semantic_hub_hubness.py`: nearest-neighbor k-occurrence, Gini/skew,
  mutual-NN, and CSLS-style retrieval sanity checks.
- `src/sft/semantic_hub_logit_lens.py`: concept/neighbor/surface-token logit
  lens with final RMSNorm sensitivity and single-token anchor retention.
- `src/sft/semantic_hub_extract_spokes.py`: opt-in raw-label and symbolic S*
  spokes without changing existing extraction defaults.
- `src/sft/semantic_hub_interventions.py`: residual-stream last-token patching,
  concept transplant, and activation addition.
- `src/sft/semantic_hub_compile_paper_report.py`: merge CSVs, bootstrap deltas,
  FDR, layer-cluster tests, and Markdown report.

## Local Vs CHTC

Local CPU:

- existing geometry and paper-style similarity;
- hubness statistics;
- bootstrap/permutation tests;
- report compilation.

Local GPU:

- logit-lens smoke;
- new-spoke hidden extraction;
- intervention hook smoke.

CHTC:

- all-arm logit-lens if local smoke passes;
- intervention sweeps sharded by arm/layer/concept block;
- any new hidden-state extraction across all arms and symbolic controls.

Recommended CHTC starting requests:

- `request_gpus = 1`
- `request_cpus = 4`
- `request_memory = 32GB`
- `request_disk = 40GB`
- `+GPUJobLength = "short"` for smoke, `"medium"` for full sweeps

Future CHTC files should live under `chtc/semantic_hub_memp/`; do not modify
existing MMLU, Huth, concept-steering, or ops files.

## Decision Gate

Move from "format-invariant semantic clustering" to "paper-method-aligned
semantic hub" only if all are true:

- E1 same-concept deltas beat random and S*-close controls in middle layers.
- E2 retrieval gains are not explained by generic nearest-neighbor hubness.
- E5 logit-lens semantic margins peak before final surface-answer margins.
- E7 interventions have layer-local, concept-specific effects.
- Scrambled and shuffled controls fail in the predicted direction.
