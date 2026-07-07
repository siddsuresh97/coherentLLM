# Independent coherence benchmarks (autoresearch synthesis)

Goal: a SECOND axis to corroborate our Procrustes coherence, runnable on the FIXED 128 THINGS
eval concepts, before/after SFT. Circularity warning: 2510.01030 and our EMNLP paper are our OWN
group's THINGS-alignment work -> present human-alignment as a DISTINCT CONSTRUCT, not replication.

## Top 3 to actually use (mutually orthogonal)
1. HUMAN THINGS alignment (external human target). RSA/Procrustes of model triplet-RSM vs the
   4.7M-judgment human RSM / SPoSE-66. Du et al. 2024/25 (arXiv:2407.01067, Nat Mach Intell 2025),
   repo github.com/ChangdeDu/LLMs_core_dimensions, OSF f5rn6/qn5uv. Perfect THINGS overlap, ~turnkey
   (swap in our llama-3.1-8b prompting; SPoSE-fit+human-compare half is ready). We ALREADY have the
   SPoSE 49D embedding for the 128.
2. LOGICAL consistency: symmetry + transitivity violation rates on our own pairwise/triadic
   elicitations. "Aligning with Logic"/REPAIR (arXiv:2410.02205, ICLR2025); BECEL (COLING2022,
   github.com/MJ-Jang/BECEL). LABEL-FREE, mathematically necessary for a coherent similarity space,
   NOT optimized by our Procrustes loss -> most independent, non-circular axis. Implement metric
   ourselves (~dozens of lines): symmetry = order-flip disagreement; transitivity = cycle/triangle-
   inequality violations.
3. GV-consistency (cross-elicitation agreement construct): Li et al. arXiv:2310.01846 (ICLR2024).
   Generator "what is most similar to X?"->Y vs Validator "are X,Y similar? T/F". Published named
   version of exactly our idea, from another lab. Adapt prompts to THINGS/NOVA (light).

## Supporting anchors (Tier 2)
- SimLex-999 (also WordSim353/MEN/SimVerb) human pairwise sim; loader
  github.com/kudkudak/word-embeddings-benchmarks. Partial vocab overlap w/ THINGS -> intersect.
- Feature-norm golds for the NOVA arm: McRae2005, CSLB2014, Buchanan2019; Bhatia&Richie PsychRev2024.
- Zhao2025 "Do We Know What LLMs Don't Know" (arXiv:2505.21701, turnkey, runs llama-3-8b) cross-
  prompt consistency; KonTest (arXiv:2407.12830) KG metamorphic factual consistency.

## Cite-don't-run
ParaRel/Elazar2021 (arXiv:2102.01017) = origin of "consistency=invariance under rephrasing" but
masked-LM cloze + factual triples (reimplement for generative; cite as origin). ConCoRD (2211.11875)
is a correction method not a benchmark. KoLA self-contrast = hallucination. Flip-Flop/PromptBench =
surface robustness.

## Paraphrase-consistency protocol over the 128 (needs no external data)
Reword each elicitation prompt N ways (triplet/pairwise/feature), run all N over the 128, measure
answer agreement (Fleiss kappa / % identical). A coherent model is invariant to rephrasing. This is
our own paraphrase-consistency axis, cheap and self-contained.

## Knowledge-retention (reviewers will ask)
lm-eval-harness on vLLM before/after: MMLU (or subset) + ARC-challenge + HellaSwag + TruthfulQA.
`lm_eval --model vllm --tasks mmlu,arc_challenge,hellaswag,truthfulqa`. Coherence gain must not
come with catastrophic forgetting.

Unverified: Rabinovich2023 (2311.01152) model list; whether Aligning-with-Logic released code -
pull PDFs before citing specifics.
