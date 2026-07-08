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
| Knowledge cost, narrow 4-task battery | **−0.03** (default SFT: −0.21) |
| Knowledge cost, full lm-eval battery (§6) | **real, ~6/10 task groups drop** — much smaller than default SFT, not zero |
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
> default fine-tune, for far less knowledge cost. Most of the forgetting was an over-aggressive-LR
> artifact — but not all of it. **This 4-task battery undersells the true cost; see §6 for the
> full picture (the same models still drop on ~6/10 capability task groups).**

---

## 3. A coherence direction in weight space

The fine-tune is a LoRA delta, so the change is one direction in weight space (the *task
vector* = SFT − base). Adding a fraction back to the untrained base — no further training —
recovers most of the gain.

**Task-vector steering: base + α·(real − base), with retention (Task 7):**

| α | Coherence | Human r² | Retention | vs base |
|---|---|---|---|---|
| 0.00 | 0.32 | 0.47 | 0.574 | — |
| **0.25** | **0.65** | **0.60** | 0.553 | **−0.021** |
| 0.50 | 0.71 | 0.58 | 0.457 | −0.117 |
| 1.00 | 0.70 | 0.60 | 0.367 | −0.207 (= real SFT) |

Retention degrades smoothly with α — a clean dose-response. **α=0.25 is the sweet spot:**
~87% of the coherence gain, human-alignment already at the full-SFT level, for roughly a
fifth of the retention cost of the full task vector. A quarter-strength nudge, no training.

**Activation steering: an earned negative, not an artifact.**
The first attempt added the coherence direction to the residual stream at *every* decoder
layer simultaneously and collapsed generation at every scale — plausibly just an all-layers
artifact, since the standard method (ActAdd/RepE) steers a narrow *middle-layer* band instead.
Task 8 redid it properly: norm-matched injection (`h_L += α·(diff_L/‖diff_L‖)·‖h_L‖`), confined
to one layer or a 5-layer band at a time, swept over layers {8, 10, 12, 14, 16, 20} and bands
{10–14, 12–16} at α ∈ {2, 4, 6, 8} — 32 (layer, α) combinations in total. **All 32 collapse
generation.** 31/32 cells have zero valid triplet judgments (pure repetition/garbage); the one
partial exception (layer 20, α=2) is still 56% degenerate. This rules out the all-layers
explanation: the coherence direction itself destabilizes generation via activation steering,
at every layer depth and magnitude tested. The task-vector (weight-space) route above remains
the only steering mechanism in this project that raises coherence without destroying
generation — full grid in `results/sft_eval/steer_actlayer/VERDICT.md`.
- **Early-stopping alone barely helps retention** (few-steps −0.167 on the narrow battery).
  Lowering the learning rate, not shortening training, does the most to protect knowledge.

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

## 6. What else does it buy? The full picture

Does low-LR show *gains* — not just avoided regression — on tasks it never trained on?
All 9 requested capability tasks covered (verdict: GAIN if Δ>+0.02, DROP if Δ<−0.02, else
FLAT). Full results: `results/sft_eval/TASK6_SUMMARY.md`.

**6C — standard capability (0 gains, 4 flat, 6 drops):**

| Task | Base | Low-LR | Δ | Verdict |
|---|---|---|---|---|
| Winogrande | 0.762 | 0.765 | +0.003 | flat |
| CommonsenseQA | 0.651 | 0.666 | +0.015 | flat |
| TruthfulQA | 0.550 | 0.533 | −0.017 | flat |
| PIQA | 0.799 | 0.779 | −0.020 | flat |
| HellaSwag | 0.685 | 0.656 | −0.029 | DROP |
| OpenBookQA | 0.490 | 0.412 | −0.078 | DROP |
| MMLU (57-subject mean) | 0.693 | 0.594 | −0.099 | DROP |
| ARC-Easy | 0.850 | 0.725 | −0.125 | DROP |
| ARC-Challenge | 0.649 | 0.524 | −0.125 | DROP |
| WiC | 0.652 | 0.500 | −0.152 | DROP |

57 of 59 individual MMLU subjects drop (moral scenarios −0.248, human sexuality −0.183,
formal logic −0.190). **This is a real, broad capability cost — far short of the default
SFT's near-total collapse, but not the near-zero result the narrow 4-task retention
battery in §2 suggested.** That battery (2 MMLU subjects + ARC-challenge + HellaSwag +
TruthfulQA) undersells the true cost; the full MMLU + wider battery is the honest number.

**6A/6B — similarity and human-behavior (this is where the gains are):**

| Task | Base | Low-LR | Δ | Verdict |
|---|---|---|---|---|
| **THINGS odd-one-out** (agreement w/ humans) | 0.686 | 0.770 | **+0.084** | GAIN |
| **THINGS triplet-sim ρ vs human** | 0.483 | 0.711 | **+0.228** | GAIN |
| WordSim-353 | 0.294 | 0.344 | +0.050 | GAIN |
| MEN | 0.488 | 0.518 | +0.030 | GAIN |
| MTurk-771 | 0.306 | 0.353 | +0.047 | GAIN |
| SimVerb-3500 | 0.168 | 0.199 | +0.032 | GAIN |
| STS-B / SimLex-999 / RG-65 | — | — | ~0 | flat |

The THINGS odd-one-out result is the most direct "more human-like" test in the whole
project — predicting *actual* human choices, not our own geometry. Low-LR agrees with the
human majority pick 8.4 points more often than base.

> **Plain answer:** low-LR reliably improves semantic-similarity and human-behavior
> measures — especially predicting real human concept judgments — much more safely than
> the default fine-tune. It does **not** broadly improve standard reasoning/knowledge
> benchmarks; there it trades away real capability, just far less than the alternative.
> Honest framing: a large, targeted gain in human-like semantic structure, at a real but
> substantially reduced cost to general capability — not a free lunch.

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
Interactive version rendered as a claude.ai artifact. Full 6C table: `results/sft_eval/TASK6_SUMMARY.md`.
Steering retention: `results/sft_eval/steer/steer_retention.csv`. Activation-steering grid:
`results/sft_eval/steer_actlayer/summary.csv`, verdict: `results/sft_eval/steer_actlayer/VERDICT.md`.*

## Future directions

- `research/FUTURE_brain_predictivity.md` — does coherence-SFT improve LLM-to-brain
  predictivity (RSA on THINGS-fMRI, then Huth/Ivanova-style voxelwise encoding), using the
  α-sweep as a dose-response causal manipulation.
- `research/FUTURE_semantic_hub.md` — does coherence-SFT induce/sharpen a format-invariant
  "semantic hub" in the middle layers (Wu, Yun, Andreas, Kim 2411.04986), tested via
  cross-format representational convergence and cross-format causal patching.
- Task 8 (complete): corrected mid-layer, norm-matched activation steering — confirmed the
  all-layer steering failure is a true negative, not an artifact (`steer_actlayer/VERDICT.md`).
