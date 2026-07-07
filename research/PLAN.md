# Coherence-SFT experiment plan

Goal: fine-tune **llama-3.1-8b** to be MORE coherent (its triplet/pairwise/feature similarity
spaces agree more, and it stops collapsing at n=128). Motivated by the finding that
**instruction tuning is the #1 documented driver of human-like concept structure** (2510.01030),
so targeted SFT is pushing on the strongest known lever. Full research in
`research/{sft_data_design,training_eval_infra,what_drives_coherence}.md`.

## Core idea
Derive all three elicitation views from ONE NOVA-derived ground-truth similarity matrix `S*`, so
triplet/pairwise/feature training examples are **mutually consistent by construction** (AligNet
template, arXiv:2409.06509). Train on ~600 NOVA concepts, **hold out our 128 test concepts
entirely**, and measure whether coherence rises on the unseen 128.

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
- train = NOVA concepts minus (128 test + 60 + 30 + close synonyms); target ~600.
- Save `data/sft/train_concepts.csv`, `test_concepts.csv`.

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

### 4. Eval  (`src/sft/eval_coherence.py`, vLLM hot-swap LoRA)
- vLLM `LLM(base, enable_lora=True, max_lora_rank=64)`; run triplet+pairwise+feature on the
  **128 held-out concepts** (reuse scale128 stimuli), SALMON d=5 for triplet, then
  `procrustes_analysis` for triplet~feature / triplet~pairwise / pairwise~feature + human align.
- DUAL eval (mandatory): generation AND logprob. Gap => reporting-only, not representation.
- Hot-swap adapters via VLLM_ALLOW_RUNTIME_LORA_UPDATING so base loads once.

### 5. Read the result
- SUCCESS = coherence r2 on held-out 128 rises toward human 0.77 ceiling (from llama's ~0.32),
  in BOTH generation and logprob, and NOT in the scrambled-control arm.
- If gen rises but logprob flat -> reporting-only (add Stage-2 AligNet KL+L2 aux loss).
- If real >> scrambled -> genuine structure learning (the win).

## Guards against "reporting not representation" (we already saw this in OLMo Finding 4)
1. dual generation-vs-logprob eval  2. held-out-concept transfer  3. cross-format Procrustes IS
the test + RSA(model-RDM, S*)  4. scrambled-label control arm.

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
