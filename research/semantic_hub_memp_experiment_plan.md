# Semantic Hub MEMP Experiment Plan

Created: 2026-07-09
Branch: `coherence-sft`
Owner scope: semantic-hub/MEMP lane only

## Source Basis

Primary paper: Wu, Yu, Yogatama, Lu, and Kim, "The Semantic Hub Hypothesis:
Language Models Share Semantic Representations Across Languages and
Modalities," arXiv:2411.04986, ICLR 2025.

Sources checked for this handoff:

- arXiv abstract and HTML v3: https://arxiv.org/abs/2411.04986 and
  https://ar5iv.org/html/2411.04986v3
- Local note: `research/lit_notes/semantic_hub_2411_04986.md`
- Local reports: `results/sft_semantic_hub/REPORT.md`,
  `results/sft_semantic_hub_paper/REPORT.md`,
  `research/literature_eval_design.md`, and `README.md`

Source limit: the paper source was reachable, but this plan still treats the
repo-local checked-in notes and reports as the authoritative state of our
experiments. I found no standard method named "MEMP" in the paper; here MEMP is
the local shorthand already used in `research/literature_eval_design.md`:
matched/equivalent-meaning probes where one concept is presented through
multiple surface spokes and compared against controlled nonmatches.

## Current State

The repo already has a text-only semantic-hub analogue:

- Concept set: 128 held-out THINGS/NOVA concepts from
  `data/scale128/concepts.csv`.
- Spokes already extracted: `triplet`, `pairwise`, and `feature_listing`.
- Model arms already present: `base`, `scrambled`, `lowLR`, `lowrank`,
  `taskvec_a0p25`, `taskvec_a0p5`, and `taskvec_a1p0`.
- Existing hidden states:
  `results/sft_semantic_hub/hidden_states/{arm}.npz`.
- Existing geometry metrics:
  `results/sft_semantic_hub/hub_by_layer.csv`.
- Existing paper-style relative similarity:
  `results/sft_semantic_hub_paper/similarity_by_layer.csv`.

The main read is narrow but useful:

- `taskvec_a0p25` is the best mid-layer candidate by RDM, CKA, and retrieval:
  mid RDM `0.6770`, CKA `0.7809`, top-1 `0.2062`, top-5 `0.5393`.
- `taskvec_a0p5` is the best mid-layer same-minus-random arm: `0.0617`.
- `taskvec_a1p0` is the best mid-layer same-minus-S*-close arm: `0.0099`.
- Concept-minus-format alignment is still negative for every arm, so the result
  should be described as stronger cross-format invariance, not yet a clean
  concept-dominant hub.

## Paper-to-Repo Hypothesis

The paper's shared-space claim maps cleanly to this repo only as a text-only
prompt-format analogue:

- Paper equivalent inputs: translations, arithmetic forms, code/semantic roles,
  formal semantics, image-caption, audio-label.
- Our equivalent inputs: the same held-out concept represented as triplet,
  pairwise, feature-listing, raw-label, and symbolic S* spokes.
- Paper middle-layer prediction: semantic equivalence is strongest in
  intermediate layers.
- Our layer prior: Llama-3.1-8B layers `10:20`, with all layer curves retained
  to check early-token and late-surface collapse.
- Paper anchoring prediction: intermediate states are closer to dominant
  language semantic tokens before final layers project to output surface forms.
- Our anchoring analogue: mid-layer states should favor concept/neighbor tokens,
  while final layers favor task-surface answer tokens such as `A`, `B`, `yes`,
  `no`, or the required option string.
- Paper intervention criterion: modifying the shared representation should
  predictably change outputs.
- Our intervention criterion: patching or adding concept-spoke states at hub
  layers should move triplet/pairwise answer logits toward the patched concept,
  with layer-local effects and scrambled/random controls.

## Experiment Lane

### E1: Relative Similarity MEMP

Question: are same-concept cross-format states closer than controlled
nonmatches, in the paper's Eq. 1 sense?

Already runnable:

```bash
python src/sft/run_semantic_hub_paper_similarity.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_memp_similarity_smoke \
  --arms base lowrank taskvec_a0p25 scrambled \
  --mid_layers 10:20 \
  --n_boot 100 \
  --seed 17
```

Full reproducible run:

```bash
python src/sft/run_semantic_hub_paper_similarity.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir results/sft_semantic_hub_paper \
  --arms base scrambled lowLR lowrank taskvec_a0p25 taskvec_a0p5 taskvec_a1p0 \
  --mid_layers 10:20 \
  --n_boot 1000 \
  --seed 17
```

Controls:

- random nonmatching concept;
- S*-close nonmatching concept as the hard negative;
- S*-far nonmatching concept as the easy negative;
- bootstrap over concepts;
- retain all layer curves, not just best layer.

Decision rule:

- Minimum support: positive middle-layer `same_minus_random` over base.
- Strong support: positive and reliable `same_minus_close_neighbor`.
- Failure/qualification: same-random improves while same-close stays near zero,
  which indicates broad semantic smoothing rather than exact concept identity.

Implementation checkpoint:

```bash
python src/sft/run_semantic_hub_memp.py --overwrite
```

This writes a machine-readable config, run manifest, metric schema, control
assignment table, layer table, summary table, and report under
`results/sft_semantic_hub/memp_paper_harness/`. It extends the earlier E1 scorer
with lexical and category-proxy controls while keeping the same fixed `10:20`
middle-layer readout.

### E2: Hubness And Retrieval Artifact Checks

Question: do retrieval gains reflect concept matching, or do a few generic
concepts become nearest-neighbor hubs?

Script TODO: `src/sft/semantic_hub_hubness.py`.

Inputs:

- `results/sft_semantic_hub/hidden_states/{arm}.npz`
- `results/sft_semantic_hub/hidden_states/neighbor_table.csv`

Metrics:

- cross-format top-1/top-5 retrieval, already implemented;
- retrieval margin, already implemented in the paper-style scorer;
- k-occurrence count per concept: how often each concept is selected as nearest
  neighbor by other concepts across format pairs;
- hubness skew, Gini, and max-share of nearest-neighbor assignments;
- mutual-nearest-neighbor rate;
- optional CSLS-style adjusted retrieval as a sanity check.

Decision rule:

- Good hub: diagonal same-concept retrieval rises and mutual-NN rate rises.
- Bad hubness artifact: top-1 rises because many queries map to a small set of
  generic concepts such as broad categories or frequent labels.

Expected output:

- `results/sft_semantic_hub_paper/hubness_by_layer.csv`
- `results/sft_semantic_hub_paper/hubness_summary.csv`

### E3: RDM/RSA/CKA Geometry

Question: do model arms share concept geometry across spokes, not merely pair
individual concepts?

Already runnable:

```bash
python src/sft/run_semantic_hub.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_geometry_smoke \
  --arms base lowrank taskvec_a0p25 scrambled \
  --mid_layers 10:20
```

Full reproducible run:

```bash
python src/sft/run_semantic_hub.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir results/sft_semantic_hub \
  --arms base scrambled lowLR lowrank taskvec_a0p25 taskvec_a0p5 taskvec_a1p0 \
  --mid_layers 10:20
```

Metrics:

- cross-format RDM Spearman;
- linear CKA;
- same-concept retrieval top-1/top-5/margin;
- concept-label vs format-label kernel alignment.

Decision rule:

- `taskvec_a0p25` remains the default arm if it preserves the best RDM/CKA and
  retrieval tradeoff.
- `taskvec_a0p5` and `taskvec_a1p0` remain diagnostic because they win different
  paper-style margins.
- A credible hub claim needs both geometry and E1/E2 controls.

### E4: Category, Lexical, And Surface Controls

Question: do similarity gains survive non-semantic confounds?

Implemented first-pass script: `src/sft/run_semantic_hub_memp.py`.

Implemented controls:

- same S* neighborhood but wrong concept identity (`close_neighbor`);
- S*-far easy negative;
- deterministic random nonmatches;
- lexical overlap and token-count matched negatives;
- category-proxy negatives using deterministic clusters over the S* concept
  similarity matrix when no explicit category table is available.

Still planned controls:

- single-token-only subset for logit-lens anchors;
- prompt length as a covariate;
- close/far swap in symbolic strings while preserving lexical content;
- shuffled predicate order in symbolic strings;
- answer-position controls for triplet option order.

Primary tables:

- `control_type`, `arm`, `layer`, `format_pair`, `matched_score`,
  `control_score`, `paired_delta`, `ci_low`, `ci_high`, `p_perm`, `n_concepts`.
- actual output: `results/sft_semantic_hub/memp_paper_harness/controls_by_layer.csv`.
- summary: `results/sft_semantic_hub/memp_paper_harness/control_summary.csv`.

Decision rule:

- The hub claim survives only if same-concept deltas remain positive against
  S*-close, lexical/token-count, and category-balanced controls.

Current read: task-vector arms survive random, lexical, and category-proxy
controls clearly, but the S*-close exact-identity margin remains small. Keep the
claim at "stronger semantic clustering / format invariance" until logit-lens
anchoring and causal interventions pass.

### E5: Logit-Lens Semantic Anchoring

Question: do middle layers encode concept semantics before late layers commit to
task-surface answers?

Script TODO: `src/sft/semantic_hub_logit_lens.py`.

Smoke command after implementation:

```bash
python src/sft/semantic_hub_logit_lens.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_logit_lens_smoke \
  --arms base taskvec_a0p25 lowrank scrambled \
  --layers 8-24 \
  --max-concepts 32 \
  --anchor-policy single_token \
  --norm-mode final_rmsnorm \
  --device cuda \
  --dtype bfloat16
```

Full run after smoke:

```bash
python src/sft/semantic_hub_logit_lens.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir results/sft_semantic_hub_paper \
  --arms base scrambled lowLR lowrank taskvec_a0p25 taskvec_a0p5 taskvec_a1p0 \
  --layers 0-32 \
  --anchor-policy single_token \
  --norm-mode final_rmsnorm \
  --device cuda \
  --dtype bfloat16
```

Anchor sets:

- concept label token;
- S*-close neighbor token;
- S*-far neighbor token;
- random concept token;
- correct triplet option token/string;
- surface controls: `A`, `B`, `yes`, `no`;
- prefix-space and no-prefix tokenization variants, retaining the max-prob
  variant as in the paper's tokenization caution.

Metrics:

- `concept_margin = logit(concept) - max(logit(close), logit(far), logit(random))`;
- `close_far_margin = logit(close_neighbor) - logit(far_neighbor)`;
- `semantic_surface_margin = logit(correct_semantic_token) - logit(surface_token)`;
- retained concept count under single-token filtering.

Decision rule:

- Semantic margins should peak in mid layers.
- Surface margins should dominate near final layers.
- A useful coherence arm improves mid-layer semantic margins without flattening
  final answer margins.

Expected artifacts:

- `results/sft_semantic_hub_paper/logit_lens_by_layer.csv`
- `results/sft_semantic_hub_paper/logit_lens_summary.csv`
- `results/sft_semantic_hub_paper/logit_lens_anchor_retention.csv`

### E6: Symbolic S* Spoke

Question: does the hub connect natural prompt spokes to a more formal semantic
spoke, closer to the paper's formal-semantics and code cases?

Script TODO: extend `src/sft/extract_semantic_hub_hidden_states.py` or add
`src/sft/semantic_hub_extract_spokes.py` without altering existing default
formats.

Proposed symbolic prompt:

```text
concept={concept}; close_neighbor={close}; far_neighbor={far}; close_similarity={close_similarity:.3f}; far_similarity={far_similarity:.3f}
```

Smoke command after implementation:

```bash
python src/sft/semantic_hub_extract_spokes.py \
  --model llama-3.1-8b-instruct \
  --concepts data/scale128/concepts.csv \
  --out_dir /tmp/semantic_hub_symbolic_hidden_smoke \
  --formats raw_label,symbolic_sstar \
  --arms base,taskvec_a0p25 \
  --batch_size 4 \
  --max_length 192 \
  --dtype bfloat16 \
  --save_dtype float16 \
  --device cuda \
  --overwrite
```

Controls:

- swap close/far labels while preserving all words;
- shuffle field order;
- drop numeric similarities;
- raw label only.

Expected artifacts:

- `results/sft_semantic_hub_symbolic/hidden_states/{arm}.npz`
- `results/sft_semantic_hub_paper/symbolic_similarity_by_layer.csv`
- `results/sft_semantic_hub_paper/symbolic_controls_by_layer.csv`

### E7: Causal Hub Interventions

Question: is the shared representation used by the model?

Script TODO: `src/sft/semantic_hub_interventions.py`.

Intervention sites:

- primary: residual stream after transformer block at the assistant answer
  position / last prompt token;
- secondary: user final token position for non-chat raw prompts;
- optional later: attention-output and MLP-output sites if residual patching
  is too broad.

Interventions:

- same-concept spoke substitution: patch triplet hidden state with
  feature-listing or symbolic state for the same concept;
- wrong-concept patch: same layer and format but random or S*-close wrong
  concept;
- concept transplant: patch concept A prompt with concept B spoke state and
  test whether logits move toward B-compatible options;
- activation addition: add `h(B, spoke) - h(A, spoke)` to A prompt.

Smoke command after implementation:

```bash
python src/sft/semantic_hub_interventions.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_intervention_smoke \
  --arms base taskvec_a0p25 scrambled \
  --layers 10-20 \
  --max-concepts 32 \
  --interventions same_spoke_patch concept_transplant \
  --patch-site residual_last_token \
  --batch_size 1 \
  --device cuda \
  --dtype bfloat16
```

Primary metrics:

- correct option logit margin before/after patch;
- target concept logit shift;
- answer flip rate toward patched concept;
- nonspecific entropy/logit-norm change;
- layer-locality curve.

Positive controls:

- clean same-prompt duplicate patch should preserve behavior;
- late-layer patch near answer position should affect surface logits.

Negative controls:

- scrambled arm;
- random concept patch;
- shuffled vector with matched norm;
- wrong layer;
- no-op hook.

Decision rule:

- Strong support requires a mid-layer intervention effect that is larger for
  coherent/task-vector arms than base and absent or degraded for scrambled.
- Broad all-layer effects or large logit-norm shocks are activation artifacts,
  not hub evidence.

Expected artifacts:

- `results/sft_semantic_hub_paper/intervention_by_layer.csv`
- `results/sft_semantic_hub_paper/intervention_summary.csv`
- `results/sft_semantic_hub_paper/intervention_examples.jsonl`

## Statistical Plan

Use paired tests wherever possible because each concept appears in every arm,
format, and layer.

Required statistics:

- bootstrap 95% CIs over concepts for each layer/arm/metric;
- paired bootstrap CIs for arm-minus-base and arm-minus-scrambled deltas;
- concept-label permutation nulls for retrieval/RSA and intervention target
  labels;
- sign test or Wilcoxon signed-rank over per-concept deltas as a robustness
  check;
- Benjamini-Hochberg FDR across arm/layer/metric families;
- layer-cluster permutation for contiguous mid-layer peaks;
- report effect sizes, not only p-values.

Selection discipline:

- Primary layer band is fixed at `10:20`.
- Best-layer summaries are descriptive unless the layer was selected on a
  training split and evaluated on held-out concepts.
- Main arm comparison should be `taskvec_a0p25` vs `base` and `scrambled`;
  `taskvec_a0p5/a1p0` are diagnostic follow-ups.

## Local Vs CHTC

Local CPU smoke:

- E1/E3 from existing NPZs.
- E2 hubness once implemented.
- Statistics/joins/plots.
- Resources: 2-8 CPU, under 8 GB RAM for current 128-concept matrices.

Local GPU smoke:

- hidden-state extraction for new spokes;
- logit-lens full LM-head scoring if the script reloads model weights;
- intervention hooks.
- Resources: one CUDA GPU, 4 CPU, 24-32 GB GPU memory preferred, 32 GB RAM,
  10-40 GB disk depending cache use.

CHTC scale:

- Use only after the local command writes all expected outputs.
- Shard by `arm` and, for interventions, by layer band or concept block.
- Use one GPU per shard; avoid multi-GPU.
- Recommended starting shape:
  `request_gpus = 1`, `request_cpus = 4`, `request_memory = 32GB`,
  `request_disk = 40GB`, `+GPUJobLength = "short"` for smoke and `"medium"` for
  full-arm logit-lens/intervention sweeps.
- CPU aggregation/statistics can run without GPU after artifacts return.
- Put this lane's future submit files under `chtc/semantic_hub_memp/`; do not
  modify existing MMLU, Huth, concept-steering, or shared ops submit files.

Expected CHTC artifacts:

- per-shard logs under a run id such as
  `semantic-hub-memp-20260709-logitlens`;
- one machine-readable JSON manifest per shard with command, git SHA, model arm,
  layers, concepts, host, GPU name, runtime, and output paths;
- CSV/JSONL outputs listed in E5-E7;
- merged report under `results/sft_semantic_hub_paper/REPORT.md` only after all
  shards validate.

CHTC handoff commands after a local smoke passes:

```bash
scripts/chtc-ssh-master.sh start
scripts/chtc-ssh-master.sh check
scripts/chtc-ssh-master.sh run 'hostname -f && condor_q && printf "%s\n" "$STAGING"'
```

Then create a run-specific bundle, push it, and submit from CHTC home:

```bash
RUN_ID=semantic-hub-memp-20260709-logitlens
mkdir -p /tmp/$RUN_ID/logs
# TODO: populate /tmp/$RUN_ID with semantic-hub scripts, params.txt, and submit file.
chtc-push /tmp/$RUN_ID/ "chtc-runs/$RUN_ID/"
scripts/chtc-ssh-master.sh run "cd ~/chtc-runs/$RUN_ID && condor_submit semantic_hub_logit_lens.sub"
```

Pull results after completion:

```bash
mkdir -p results/sft_semantic_hub_paper/chtc/$RUN_ID
chtc-pull "chtc-runs/$RUN_ID/" results/sft_semantic_hub_paper/chtc/$RUN_ID/
```

## Handoff Checklist

1. Reproduce existing CPU metrics without touching live lanes:

```bash
python src/sft/run_semantic_hub.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_geometry_smoke \
  --arms base taskvec_a0p25 scrambled \
  --mid_layers 10:20

python src/sft/run_semantic_hub_paper_similarity.py \
  --hidden_dir results/sft_semantic_hub/hidden_states \
  --out_dir /tmp/semantic_hub_similarity_smoke \
  --arms base taskvec_a0p25 scrambled \
  --mid_layers 10:20 \
  --n_boot 100
```

2. Implement `src/sft/semantic_hub_hubness.py`, then write
   `results/sft_semantic_hub_paper/hubness_by_layer.csv`.

3. Implement `src/sft/semantic_hub_logit_lens.py`, run the 32-concept smoke,
   then scale all arms only if retained anchor counts and output schemas are
   correct.

4. Implement symbolic-spoke extraction as a separate semantic-hub helper or an
   opt-in extension that does not change current defaults.

5. Implement `src/sft/semantic_hub_interventions.py`, starting with
   residual-stream last-token hooks on `base`, `taskvec_a0p25`, and `scrambled`.

6. Add a merger/report script only after E2/E5/E7 produce stable CSV schemas.
   Suggested TODO: `src/sft/semantic_hub_compile_paper_report.py`.

7. Do not edit or stage unrelated modified files in `README.md`,
   `research/STATUS.md`, `research/EXPERIMENT_LOG.md`, wide-bench outputs,
   Huth outputs, concept-steering files, or CHTC ops files in this lane.

## Top Recommendations

1. Treat `taskvec_a0p25` as the primary candidate because it has the best
   mid-layer RDM/CKA/retrieval tradeoff.
2. Keep `taskvec_a0p5` and `taskvec_a1p0` in diagnostics because paper-style
   same-random and same-close winners differ.
3. Prioritize logit-lens anchoring before interventions; it is the missing
   paper method with the cleanest implementation path.
4. Use S*-close and lexical/category controls as gatekeepers before claiming an
   exact concept hub.
5. Only scale interventions to CHTC after a local hook smoke shows layer-local
   effects and no nonspecific logit-norm shocks.
