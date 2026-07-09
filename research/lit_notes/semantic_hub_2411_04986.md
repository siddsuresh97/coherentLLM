# Literature Note: Semantic Hub Hypothesis

Source: Wu, Yu, Yogatama, Lu, and Kim, "The Semantic Hub Hypothesis: Language
Models Share Semantic Representations Across Languages and Modalities"

URL: https://arxiv.org/abs/2411.04986

## Claim

Language and multimodal models learn a shared intermediate representation space
where semantically equivalent inputs from different data types are close. The
paper calls this a semantic hub by analogy to hub-and-spoke accounts of human
semantic memory.

## Methods To Reuse

### Relative Similarity

For paired inputs with equivalent meaning, compare hidden-state cosine
similarity against mismatched baselines. The important design point is relative
similarity, not raw cosine:

- matched translation / equivalent input;
- random mismatch;
- stronger mismatches that preserve surface or category confounds when
  possible.

The paper reports strongest translation similarity over baseline in middle
layers, which motivates our layer focus around the middle of Llama-3.1-8B.

### Logit-Lens Anchoring

Use intermediate hidden states with the LM head/unembedding to test whether a
state is closer to the dominant-language semantic token than to the surface form
required by the input context.

Our analogue:

- concept token vs random token;
- close-neighbor token vs far-neighbor token;
- semantic target token vs task-surface answer token such as `A`, `B`, `yes`,
  or `no`.

### Causal Intervention

The paper's strongest evidence is causal: intervening in shared representations
predictably affects outputs across data types. Our analogue should patch or add
concept states across prompt spokes and measure answer-logit movement.

## Local Adaptation

Supported local scope:

- text-only prompt-format hub;
- same concept across raw label, pairwise, triplet, feature-list, and symbolic
  S* spokes;
- Llama-3.1-8B base and coherence-SFT/task-vector arms.

Unsupported without more data:

- true cross-modal hub;
- true cross-language hub;
- anterior-temporal-lobe-equivalent claim;
- causal usage before patching/steering results.

## Decision Metrics

Primary:

- `same_minus_random`
- `same_minus_close_neighbor`
- matched retrieval top-1/top-5
- semantic-vs-surface logit margin
- intervention-induced answer-logit shift

Strong positive result:

- middle-layer aligned-arm gains survive close-neighbor controls and are
  localized by interventions.

Failure mode:

- broad similarity increases but close-neighbor discrimination falls, matching
  WiC/TruthfulQA/MMLU damage.
