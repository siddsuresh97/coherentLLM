# What drives conceptual coherence (autoresearch synthesis)

## Anchor: "Computational Ingredients of Human-Like Representations" (arXiv 2510.01030)
Mixed-effects regression of human-alignment (Procrustes R2) on ~14 ingredients over 77
open models, R2=0.78. Signed significant effects:
- Instruction fine-tuning: beta=0.13 (STRONGEST) -> post-training, not scale, is top lever
- MLP dimensionality: beta=0.049 ; embedding/hidden dim: beta=0.048 (per-token width matters)
- context length: beta=0.042 (weak +)
- attention head count: beta=-0.045 (more heads slightly HURTS)
- multimodal pretraining: beta=-0.102 (NEGATIVE, counter-intuitive)
- activation function: n.s. ; vocab size: n.s. ; PARAM COUNT: weak unique effect (r~0.45 raw)
=> Instruction tuning + per-token width dominate; raw scale, activation, vocab, multimodal do NOT.
(VERIFY exact beta/SE against the PDF before quoting in the dissertation.)

## Why llama-3.1-8b > qwen2.5-32b (the inversion)
Documented recipe differences:
- Llama-3.1: ~15T tokens, ~50% general knowledge, 25% math/reason, 17% code, ~5-8% multilingual
  (8 langs). Post-train: SFT -> rejection sampling (best-of-N) -> DPO, heavy human curation.
- Qwen2.5: 18T tokens, HEAVY code+math (integrates Qwen-Math/Coder corpora), 29+ languages,
  151K multilingual-tuned vocab. Post-train: SFT >1M -> multistage RL.
Hypotheses (falsifiable on our benchmark):
- H1 multilingual dilution: Qwen spreads capacity over 29 langs; our eval is English -> Llama's
  concentrated English concept geometry is more coherent. Test: multilingual% predicts coherence -.
- H2 code/math tilt: Qwen's formal/procedural geometry need not match human everyday-object
  similarity. Test: code/math bench strength uncorrelated/-, general-knowledge bench +.
- H3 base-quality x tuning route (Cat-Rat-Meow 2504.07965): Qwen bases WEAK (~0.66 choice acc),
  rescued by tuning to ~0.79; Llama bases already strong. Qwen coherence more checkpoint-fragile.
- H4 capacity-per-token not param count: align tracks MLP/embed width, not total params. Test:
  regress coherence on hidden/MLP dim; should dominate param count.

## Proposed ~18-model dissociation benchmark
- Axis A (hold ~7-9B, vary family/data): llama-3.1-8b base+instruct, qwen2.5-7b base+instruct,
  mistral-7b-v0.3, gemma-2-9b, olmo2-7b (open data), qwen2.5-coder-7b (code-tilted extreme=H2).
- Axis B (hold family, vary size): qwen2.5 0.5/3/7/14/32/72b ladder (+ llama 1/3/8/70b).
- Axis C (base vs instruct matched): the pairs above + qwen2.5-32b base vs instruct (the inversion).
- Axis D (multilingual vs english): aya-23-8b vs llama-3.1-8b at matched ~8B.
- Confound checks: an MoE at matched active params; a multimodal tower (qwen2.5-vl-7b).
Analysis: per-model 3-way Procrustes-r2 coherence -> mixed-effects regression on ingredient
covariates (family random effect), replicating 2510.01030 on OUR coherence construct; base->instruct
deltas for Axis C; partial correlations isolating multilingual% and code/math% from params.

## Implication for the SFT project
Instruction tuning is the #1 documented lever for human-like concept structure. Targeted SFT for
coherence is pushing on the strongest known knob -> the intervention is well-motivated.

Sources: 2510.01030, 2504.07965 (Cat-Rat-Meow), 2312.00575 (instruction-tuning aligns to brain),
2407.21783 (Llama 3), 2412.15115 (Qwen2.5), 2406.10602 (curse of multilinguality).
