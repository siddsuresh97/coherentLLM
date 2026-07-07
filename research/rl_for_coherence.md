# RL vs SFT for coherence (autoresearch synthesis)

BOTTOM LINE: RL is a DISTRACTION for a first result. Do SFT/distillation from a consistent NOVA
target first. RL only as a phase-2 follow-up, and only RLVR-against-NOVA (not self-agreement).

## Why self-agreement reward is GAMEABLE (key critical point)
Rewarding the model when its own triplet/pairwise/feature outputs agree = self-consistency reward.
Degenerate optimum: collapse all concepts to ONE point (everything equally similar -> all methods
trivially agree). Evidence: SRT (arXiv:2505.21444) "prolonged RL with self-reward -> reward hacking
-> sudden complete collapse", emits same template answer regardless of input; KL does NOT save it.
EVOL-RL (2509.15194) entropy collapse. TTRL (2504.16084) majority-vote works short-run only.
=> If RL, anchor to EXTERNAL NOVA target (RLVR), never self-agreement. Calling self-consistency
"RLVR" is a category error - agreement is a proxy, only NOVA is a verifier.

## Why RL fits THIS objective badly (technical crux)
Coherence = a BATCH-level geometric quantity (many judgments -> similarity matrix -> Procrustes,
ONE number). GRPO needs a PER-RESPONSE scalar (advantage = within-group z-score, DeepSeekMath
2402.03300). Procrustes-as-reward has no per-response attribution -> relocates credit assignment
into the reward fn. Advantage collapse (2605.21125): near-identical group rewards -> std->0 ->
gradient dies (a slowly-varying geometric score is a prime trigger). TRL GRPOTrainer reward fn sees
the group list but must return one float/completion, and view is per-device not global (TRL #3896).

## The decisive result against RL here
Yue et al. 2504.13837 "Does RL Really Incentivize Reasoning Beyond the Base Model?": RLVR sharpens
paths ALREADY in base support, NARROWS exploration; base beats RL at large pass@k. DISTILLATION,
not RL, introduces genuinely NEW patterns. Our llama-3.1-8b LACKS coherent structure (absent, not
under-sampled) = exactly the regime RL fails and supervised transfer works. LIMA/superficial-align:
post-training elicits latent structure, doesn't conjure absent structure. (ProRL 2505.24864 rebuts
but only w/ prolonged compute on novel tasks - expensive/unstable.)

## Prior art: consistency + rep-alignment are done with SFT / differentiable losses, ~never RL
Logical consistency: ConCoRD (inference MaxSAT), REPAIR/Aligning-with-Logic (2410.02205, SFT aug ->
~1.0 transitivity), ParaRel (differentiable loss), BeliefBank (symbolic). Rep-alignment to human
similarity: gLocal (2306.04507), AligNet (2409.06509, KL-distillation SFT, THINGS OOO 44.2->61.7%,
ceiling 66.7) - OUR DIRECT TEMPLATES, all supervised. RL-with-consistency-reward exists only for
CALIBRATION (Rewarding Doubt 2503.02623, PPO-M/C 2410.09724) and FACTUALITY (2311.08401).
NOVELTY GAP (defensible thesis claim): nobody uses RL/preference-opt with an RSA/Procrustes/human-
similarity reward to align representational STRUCTURE. RSA used as a differentiable LOSS
(2410.20035), never a reward. Novel but unexplored because geometry-as-loss is more natural.

## Recommendation
SFT/QLoRA distillation from shared NOVA target FIRST. Optionally add a DIFFERENTIABLE soft-RSA/
Procrustes alignment term to the SFT loss (get the geometry objective w/o any RL). RL only phase-2,
RLVR-against-NOVA form, framed "SFT-then-RL sharpening" (industry standard), expect marginal gains.

## Minimal RL recipe if phase-2 (TRL, llama-3.1-8b QLoRA)
GRPO, PER-RESPONSE verifiable reward vs NOVA: triplet->binary (matches NOVA OOO), pairwise->graded
1-|pred-target|, feature->binary. NOT per-batch Procrustes (advantage collapse). reward_funcs list,
num_generations=8, lr 5e-6, scale_rewards="none" (Dr.GRPO 2503.20783), peft QLoRA r32, vLLM co-
located, batch divisible by num_generations (#3896). Cheaper alt: DPO QLoRA (~1h/H100) pairs
chosen=NOVA-consistent. Guardrail: collapse check (is similarity matrix degenerating to uniform?).
NB: RLVR-against-NOVA per-answer == ordinary SFT of NOVA answers -> plain SFT does same job cheaper.

Cite (add real bibtex, don't fabricate): SRT 2505.21444, EVOL-RL 2509.15194, TTRL 2504.16084,
Tulu3 2411.15124, DeepSeek-R1 2501.12948, Yue 2504.13837, ProRL 2505.24864, DeepSeekMath 2402.03300,
Dr.GRPO 2503.20783, LIMA 2305.11206, ConCoRD 2211.11875, Aligning-with-Logic 2410.02205,
ParaRel 2102.01017, gLocal 2306.04507, AligNet 2409.06509, 2510.01030, DPO 2305.18290.
Some 2026-dated IDs (2602/2603/2605.*) need confirmation before citing.
