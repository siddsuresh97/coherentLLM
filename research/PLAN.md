# Coherence-SFT experiment plan

Goal: fine-tune **llama-3.1-8b** to be MORE coherent (its triplet/pairwise/feature similarity
spaces agree more, and it stops collapsing at n=128). Motivated by the finding that
**instruction tuning is the #1 documented driver of human-like concept structure** (2510.01030),
so targeted SFT is pushing on the strongest known lever. Full research in
`research/{sft_data_design,training_eval_infra,what_drives_coherence}.md`.

## Core idea
Derive all three elicitation views from ONE NOVA-derived ground-truth similarity matrix `S*`, so
triplet/pairwise/feature training examples are **mutually consistent by construction** (AligNet
template, arXiv:2409.06509).

**Train/test split (fixed):** the **128 THINGS concepts are the EVAL set** (the ones we already
have baseline coherence + THINGS SPoSE human data for). **Train = ALL OTHER NOVA concepts**
(786 - the 128 - any synonym/hypernym leakage ≈ ~650). The 128 (and 60/30 nested) never appear
in any training example, in any view. Measure whether coherence rises on the unseen 128.

## Assets on disk
- NOVA verified matrix: `/mnt/dv/.../mia/llm-norms-cogsci2025/verified_matrix_cogsci2025.parquet` (786 concepts)
- Test concepts: `data/scale128/concepts.csv` (+ 60, +30 nested)
- QLoRA template to adapt: `/mnt/dv/.../sid/Projects/ft_llms/mixtral_ft.py`
- Coherence metric: `src/procrustes_analysis.py`; prompts: `src/prompts.py`; dual eval: `src/lineage_dual.py`
- SALMON d=5 triplet fit: `src/fit_triplet_salmon.py` (salmon env)

## Pipeline (executable order)

### 0. Setup
- New conda env `coherence_sft` with unsloth + trl + peft (+ vllm already have). (unsloth not yet installed)
- Confirm the 128 test concepts (+synonyms/hypernyms) are excluded from the NOVA train pool.

### 1. Build S* + train/test concept split  (`src/sft/build_target.py`)
- Load NOVA matrix; `S* = cosine(feature_rows)`. Save `S*` (concept x concept).
- **test = the 128 THINGS concepts** (+ 60/30 nested). **train = ALL OTHER NOVA concepts**
  (786 - 128 - close synonyms/hypernyms ≈ ~650). Save `data/sft/train_concepts.csv`,
  `test_concepts.csv`. Verify zero overlap incl synonyms.

### 2. Generate mutually-consistent SFT data from S*  (`src/sft/gen_sft_data.py`)
- triplet (~40-80k): (A,B,C) from train concepts, label from S*; multi-abstraction sampling
  (cluster S*, within vs across). Format = `prompts.triplet_prompt`. response = chosen concept.
- pairwise (~20-40k): pairs, S*->1-7. Format = `prompts.pairwise_prompt`. response = digit.
- feature (~15-30k): balanced True/False over NOVA features. Format = `prompts.feature_prompt`.
- Mixture ~50/25/25. Chat template + response-only masking. Write `data/sft/train.jsonl`.
- CONTROL arm: also write `train_scrambled.jsonl` with permuted S* labels (delusive control).

### 3. Train  (`src/sft/train_lora.py`, unsloth)
- Base `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit`, QLoRA **rsLoRA r=64** (NOT r=8), all
  attn+MLP proj, lora_alpha=64, dropout 0, bf16, grad-checkpoint "unsloth", LR 2e-4, seq 2048,
  response-only mask. Save ADAPTER only (no merge).
- Arms: (a) real S* SFT, (b) scrambled-S* control, (c) full-FT ceiling on H100 (8-bit AdamW).

### 4. Eval BATTERY, run before AND after each SFT arm  (`src/sft/eval_*.py`, vLLM hot-swap LoRA)
vLLM `LLM(base, enable_lora=True, max_lora_rank=64)`, hot-swap adapters
(VLLM_ALLOW_RUNTIME_LORA_UPDATING) so base loads once. ALL evals on the FIXED 128 THINGS
eval concepts. Four independent axes so improvement isn't just gaming our own metric:

  A. **Our Procrustes coherence** (primary): triplet+pairwise+feature on the 128, SALMON d=5,
     `procrustes_analysis` -> triplet~feature / triplet~pairwise / pairwise~feature + human align.
     DUAL: generation AND logprob (gap => reporting-only, not representation).
  B. **Paraphrase/format consistency** (independent, no external data): reword each elicitation
     prompt N ways, measure answer agreement over the 128. (protocol from benchmark research)
  C. **Internal transitivity/symmetry** (free, self-consistency): triplet cycle-violation rate +
     pairwise symmetry over the 128 - a coherent model should be transitive/symmetric.
  D. **THINGS human alignment** (independent human data we already have): Procrustes of model
     triplet space vs THINGS SPoSE 49D on the 128.

### 4b. KNOWLEDGE-RETENTION guard (must NOT regress) (`src/sft/eval_retention.py`, lm-eval-harness)
Run a minimal capability battery before/after each SFT arm to catch catastrophic forgetting:
MMLU (or subset) + ARC + HellaSwag + TruthfulQA via lm-eval-harness on vLLM. A coherence-SFT
that tanks these is a failure regardless of coherence gain. (exact minimal set from research)

### 5. Read the result
- SUCCESS = (A) coherence r2 on the 128 rises toward human 0.77 (from llama's ~0.32) in BOTH
  gen and logprob; (B,C,D) the independent axes ALSO improve (not just our metric); real arm
  >> scrambled control; AND (4b) knowledge retention within ~1-2 pts of baseline.
- If A gen rises but logprob flat -> reporting-only (add Stage-2 AligNet KL+L2 aux loss).
- If A rises but B/C/D don't -> we gamed our own metric; not a real coherence gain.
- If coherence rises but retention drops -> tradeoff; tune LoRA rank/LR/mixture or add replay.

## Guards against "reporting not representation" (we already saw this in OLMo Finding 4)
1. dual generation-vs-logprob eval  2. held-out-concept transfer  3. cross-format Procrustes IS
the test + RSA(model-RDM, S*)  4. scrambled-label control arm.

## Data-scaling ablation (how much data is actually needed)
Once Stage 1 works, sweep training-set SIZE to find the coherence-vs-data curve:
- vary # train concepts (e.g. 50, 100, 200, 400, ~650) and/or # examples per concept.
- retrain (cheap QLoRA) at each point, eval coherence on the fixed 128.
- plot coherence r2 vs training data -> where does it saturate? Is a small coherent seed enough?
This is a clean scaling-law result: "N concepts of consistent supervision suffice to induce
coherence that generalizes to held-out concepts." Reuse the hot-swap eval loop; each point is
one small QLoRA run + one eval pass.

## RL alternative (efficiency question — research agent running)
Instead of SFT-from-consistent-data, teach coherence via RL: reward = cross-method self-agreement
(the model's triplet/pairwise/feature judgments agreeing), or RLVR against NOVA S* per-answer.
Open question: coherence is a BATCH-level geometric quantity (need many judgments -> similarity
matrix -> Procrustes), so credit assignment to single responses is awkward. Candidate: GRPO with
a per-group self-consistency reward (transitivity/agreement within a sampled group of triplets).
Decision pending research (af6578...); likely a secondary arm AFTER SFT gives a first result,
unless RL proves more sample-efficient. Would also compare data efficiency vs the SFT scaling curve.

## Stretch goals (after Stage 1 shows coherence moves)
- Stage 2: CoIN hidden-rep contrastive across the 3 prompts, or AligNet KL-on-sim + L2 anchor.
- Stage 3: GV-consistency self-filtering - does coherence self-propagate to unsupervised concepts?
- Coherence TASK VECTOR: extract (SFT weights - base) as a task vector; test add/negate + transfer
  to other models (Ilharco task arithmetic). We have OLMo/Tulu base->SFT->DPO lineages for this.

## Parallel track: coherence benchmark (separate, cheap)
Run the coherence pipeline on the ~18-model dissociation matrix (research/what_drives_coherence.md)
-> mixed-effects regression of coherence on ingredients (instruction-tuning, MLP/embed width,
multilingual%, code/math%) replicating 2510.01030 on OUR metric. Tests the llama>qwen hypotheses.

## Notes
- All new bibtex from the arXiv IDs in the research docs; do NOT fabricate.
- Keep everything on the `coherence-sft` branch.
