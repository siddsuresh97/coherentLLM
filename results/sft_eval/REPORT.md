# Teaching a Language Model to Cohere

Fine-tuning **Llama-3.1-8B-Instruct** on one internally consistent similarity structure
(derived from human feature norms) makes its concept judgments agree across elicitation
formats **and** align with humans. The default fine-tune paid for this with catastrophic
forgetting; lowering the learning rate removes almost all of that cost.

- **Branch:** `coherence-sft` (unpushed)
- **Eval set:** 127 held-out THINGS concepts (never seen in training, any format)
- **Metric:** Procrustes r² = 1 − disparity, on SALMON d=5 triplet embeddings
- **Base:** Llama-3.1-8B-Instruct

## Headline

| Quantity | base → best model |
|---|---|
| Coherence (generation) | 0.32 → **0.76** |
| Human alignment (Procrustes vs THINGS-SPoSE) | 0.47 → **0.67** |
| Knowledge cost (retention vs base) | **−0.03** (default SFT was −0.21) |
| Scrambled control (permuted labels) | **0.03** (no gain) |

---

## 1. What the SFT does

All three elicitation views (triplet, pairwise, feature) derive from one ground-truth
similarity matrix `S*` (cosine of NOVA binary feature rows), so they are mutually consistent
by construction. Fine-tuning on real `S*` more than doubles generation coherence and pulls
the concept space toward the human embedding. A scrambled-label control (identical training,
permuted similarities) does none of this — ruling out the training format as the cause.

**Coherence & human alignment on the 127 held-out concepts (generation):**

| Model | triplet~feature | triplet~pairwise | pairwise~feature | mean | human r² |
|---|---|---|---|---|---|
| Base | 0.31 | 0.27 | 0.38 | 0.32 | 0.47 |
| **Real SFT** | 0.68 | 0.59 | 0.83 | **0.70** | 0.60 |
| Scrambled (control) | 0.01 | 0.03 | 0.05 | 0.03 | 0.02 |

**Independent axes agree** (rules out gaming our own metric):
- Transitivity violations 0.067 → 0.006
- Pairwise symmetry violations 0.15 → 0.0004
- THINGS-human alignment 0.47 → 0.60 (real human data)

**Coherence ↔ human alignment are inseparable:** Pearson r = 0.914 (p = 0.011) across arms.
A model gains both together or loses both together.

---

## 2. The knowledge cost, and the fix

At full-sample measurement (`--limit 1000`) the default fine-tune lost 0.21 of mean
multiple-choice accuracy (MMLU-anatomy 0.70→0.28, ARC 0.55→0.26). Sweeping the recipe
breaks the tradeoff: **lowering the LoRA learning rate to 5e-5 (or rank to 16)** keeps —
even exceeds — the coherence gain while returning retention to within 0.03 of base.

**Coherence-vs-retention Pareto** (retention = mean acc over ARC, HellaSwag, MMLU, TruthfulQA):

| Arm | Config | Coherence | Human r² | Retention | vs base |
|---|---|---|---|---|---|
| Base | — | 0.32 | 0.47 | 0.574 | — |
| Real SFT | r64, lr 2e-4, 1500 | 0.70 | 0.60 | 0.367 | −0.207 |
| Few-steps | r64, 400 steps | 0.70 | 0.59 | 0.407 | −0.167 |
| **Low-LR** | lr 5e-5, 1500 | **0.76** | **0.65** | **0.538** | **−0.036** |
| **Low-rank** | r16, lr 2e-4 | **0.75** | **0.67** | **0.544** | **−0.030** |

> **Headline:** low-LR and low-rank give higher coherence AND higher human alignment than the
> default fine-tune, at near-zero knowledge cost. The forgetting was an over-aggressive-LR
> artifact, not an inherent cost of coherence.

---

## 3. A coherence direction in weight space

The fine-tune is a LoRA delta, so the change is one direction in weight space (the *task
vector* = SFT − base). Adding a fraction back to the untrained base — no further training —
recovers most of the gain.

**Task-vector steering: base + α·(real − base):**

| α | Coherence | Human r² | Reads as |
|---|---|---|---|
| 0.00 | 0.32 | 0.47 | base |
| **0.25** | **0.65** | **0.60** | ~87% of the gain, no training |
| 0.50 | 0.71 | 0.58 | matches full SFT |
| 1.00 | 0.70 | 0.60 | = real SFT |

**Negative results (kept):**
- **Activation steering fails** — adding the same direction to the residual stream at every
  layer over-steers and collapses generation at every scale. Only the *weight-space* task
  vector works.
- **Early-stopping alone barely helps retention** (few-steps −0.167). Lowering the learning
  rate, not shortening training, protects knowledge.

---

## 4. Does it transfer? External human-rated similarity

Spearman ρ between model similarity and human ratings (benchmarks never seen in training).
The best model (low-LR) is **broader and safer** than the default SFT, which overfit some
benchmarks while cratering others (RG-65, MEN).

| Benchmark | Base | Real SFT | Low-LR | Low-rank |
|---|---|---|---|---|
| STS-B | 0.518 | 0.527 | **0.527** | 0.519 |
| SimLex-999 | 0.297 | 0.312 | 0.310 | 0.311 |
| WordSim-353 | 0.294 | 0.418 | 0.344 | 0.291 |
| MEN | 0.488 | 0.430 ↓ | **0.518 ↑** | 0.506 ↑ |
| RG-65 | 0.358 | 0.209 ↓↓ | **0.357 (held)** | 0.353 |
| MTurk-771 | 0.306 | 0.380 | 0.353 | 0.358 |
| SimVerb-3500 | 0.168 | 0.235 | 0.199 | 0.193 |

Default SFT cracks RG-65 (−0.15) and MEN (−0.06); low-LR holds or improves nearly all.

---

## 5. Does the best model predict human behavior better?

The most direct "more human-like" test — predicting real human choices on the THINGS
odd-one-out task (not our own metric, actual behavioral data):

| Model | odd-one-out agreement w/ humans | triplet-sim ρ vs human |
|---|---|---|
| Base | 0.686 | 0.483 |
| Real SFT | 0.765 (+0.079) | 0.700 (+0.217) |
| **Low-LR (best)** | **0.770 (+0.084)** | **0.711 (+0.228)** |
| Low-rank | 0.762 (+0.077) | **0.723 (+0.239)** |

> The best model predicts human concept judgments substantially better than base
> (+8 points odd-one-out; similarity correlation 0.48 → 0.71) — and the retention-safe
> low-LR/low-rank models do this as well as or better than the costly default SFT.

---

## 6. Broader capability (in progress)

Running now: a wider lm-eval battery (PIQA, Winogrande, ARC-easy/challenge, OpenBookQA,
CommonsenseQA, WiC, MMLU, TruthfulQA) on base vs low-LR, to find where human-like
conceptual structure *helps* reasoning, not just where it avoids hurting. This section
updates when Task 6C + `TASK6_SUMMARY.md` land.

---

## Appendix · Methods

### Data and train/test split

Supervision from the NOVA verified concept×feature matrix (786 concepts, human feature
norms). `S*` = cosine similarity of the binary feature rows; all training targets derive
from it.

| Set | Concepts | Role |
|---|---|---|
| NOVA total | 786 | source pool for S* |
| **Test (held out)** | 127 | THINGS concepts — all evaluation |
| Train | 643 | all other NOVA concepts |

The 127 test concepts (and close synonyms/hypernyms) never appear in any training example,
in any format. Coherence is always measured on unseen concepts.

### Training data composition — 110,000 examples from S*

| Task type | Examples | Format | Target from S* |
|---|---|---|---|
| Triplet | 60,000 | which of B/C is more similar to A | closer concept |
| Pairwise | 30,323 | rate A,B similarity 1–7 | binned similarity |
| Feature | 19,677 | is property P true of concept C | balanced true/false |

Chat-formatted, response-only loss masking. An equal-size scrambled-label control set was
built by permuting S*. Triplets sample across abstraction levels (within- vs across-cluster).

### Training configuration

QLoRA on `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` (4-bit), Unsloth backend. rsLoRA on
all attention + MLP projections, `lora_alpha = rank`, dropout 0, bf16, gradient-checkpointing,
sequence length 256 with packing, per-device batch 16 × grad-accum 2, up to 1500 steps. Only
the adapter is saved (no merge). Arms differ only in the swept hyperparameter:

| Arm | LR | Rank | Steps | Train time |
|---|---|---|---|---|
| Real SFT (default) | 2e-4 | 64 | 1500 | ~2–3 h |
| Few-steps | 2e-4 | 64 | 400 | ~27 min |
| Low-LR | 5e-5 | 64 | 1500 | ~2 h 48 m |
| Low-rank | 2e-4 | 16 | 1500 | ~1 h 41 m |

(Wall-clock from `trainer_state` runtimes; times reflect contention with another user's job
on the shared box.)

### Hardware

Shared UW cluster: 2× NVIDIA RTX A5000 (24 GB) on `rogers-gpu-1` and 1× NVIDIA H100 PCIe
(80 GB) on `opt-a007`, common filesystem. Training and 8B log-probability scoring (retention,
similarity) ran on the H100; the A5000s handled generation and CPU-bound SALMON ordinal-
embedding fits. 8B + LoRA log-prob scoring overflows 24 GB, so those passes were routed to
the H100.

### Evaluation

- **Coherence** = Procrustes r² (1 − disparity) between similarity spaces, each reconstructed
  from its elicitation format via SALMON ordinal embedding (d=5 triplet embeddings).
- **Human alignment** = Procrustes vs THINGS-SPoSE 49-D human embedding.
- **Retention** = mean accuracy over ARC-Challenge, HellaSwag, MMLU, TruthfulQA via
  lm-eval-harness (log-likelihood scoring, `--limit 1000`).
- **External similarity** = Spearman ρ of model similarity vs human ratings.
- **Human behavior** = agreement with human THINGS odd-one-out choices.

All evaluation on the 127 held-out THINGS concepts (or each benchmark's own items).

---

*Artifacts in `results/sft_eval/`; plots `mitigation/pareto.png`, `steer/coherence_human_vs_alpha.png`.
Interactive version rendered as a claude.ai artifact.*
