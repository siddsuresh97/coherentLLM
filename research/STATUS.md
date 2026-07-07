# Coherence-SFT project STATUS (where we are + open threads)

Branch: `coherence-sft` (off `main`). Decision: **DIY everything** (SFT via Unsloth/QLoRA, RL via
TRL if we get there) on our own GPUs (2x A5000 24GB + 80GB H100). NOT using Tinker (~$1.1k est,
adds external dep; our GPUs are ~free). SFT-FIRST, RL is phase-2 at most.

## What's DONE (committed to this branch)
- research/what_drives_coherence.md  - instruction tuning is #1 driver (2510.01030); llama>qwen
  = english/general vs multilingual/code-tilted data + weak qwen base; 18-model benchmark matrix.
- research/sft_data_design.md - shared-S*-derived multi-view distillation (AligNet template);
  rsLoRA r=64 + full-FT ceiling; Centaur proves QLoRA moves llama-3.1 geometry; scrambled control.
- research/training_eval_infra.md - Unsloth QLoRA (A5000 ~7-14GB); vLLM hot-swap LoRA (never merge
  mid-loop, base loads once); "Harbor" is NOT the eval engine (thin vLLM python driver instead).
- research/independent_benchmarks.md - 2nd-axis measures on the 128: (1) THINGS human alignment
  (SPoSE, have it), (2) logical transitivity/symmetry (label-free, most independent), (3) GV-
  consistency; paraphrase-consistency protocol; knowledge-retention (MMLU/ARC/Hella/TruthfulQA).
- research/rl_for_coherence.md - RL is a distraction for first result; self-agreement reward is
  GAMEABLE (collapse to a point); coherence is batch-level -> breaks GRPO; distillation not RL
  introduces new structure (Yue 2504.13837). If RL: phase-2 RLVR-vs-NOVA per-answer (== SFT anyway).
- research/PLAN.md - consolidated executable plan (updated below).

## Locked design decisions
- EVAL set = the 128 THINGS concepts (baselines + SPoSE human data already exist). TRAIN = ALL
  OTHER NOVA concepts (~650), zero leakage incl synonyms/hypernyms.
- Metric = Procrustes r2 = 1 - disparity (RDM-direct), same as main pipeline.
- Model triplet = SALMON d=5 (empirical elbow). Feature = free-list -> consolidate (shared by >=3)
  -> single-pair verify.
- Eval BATTERY before/after each arm: (A) our Procrustes coherence dual gen+logprob, (B) paraphrase-
  consistency, (C) transitivity/symmetry, (D) THINGS human alignment. + knowledge-retention guard.
- Training arms: (a) real S* SFT, (b) scrambled-S* control, (c) full-FT ceiling. rsLoRA r=64.
- Data-scaling ablation: coherence vs #train concepts (50/100/200/400/650).

## OPEN / TODO (in order)
[ ] 0. env `coherence_sft`: unsloth + trl + peft (+ vllm have). pyarrow installed (NOVA parquet).
[ ] 1. src/sft/build_target.py: load NOVA, S*=cosine(features), train=NOVA-128(-synonyms ~650),
       save train_concepts.csv/test_concepts.csv. VERIFY 128 coverage + zero leak. (STARTED - was
       blocked on pyarrow, now fixed; NOT yet run.)
[ ] 2. src/sft/gen_sft_data.py: mutually-consistent triplet(40-80k)/pairwise(20-40k)/feature(15-30k)
       from S*, chat format + response-only mask, + scrambled control set.
[ ] 3. src/sft/train_lora.py (unsloth rsLoRA r=64), arms a/b/c.
[ ] 4. src/sft/eval_*.py: the 4-axis battery + retention, vLLM hot-swap.
[ ] 5. read result; data-scaling ablation; (phase-2) RL; (stretch) coherence task vector.
[ ] PARALLEL: 18-model coherence benchmark -> regression on ingredients (separate, cheap).

## Not-yet-done infra checks
- unsloth not installed in any env yet.
- no raw human THINGS odd-one-out triplets on disk (downloadable OSF f5rn6/qn5uv if we want the
  human RSM beyond SPoSE; we already have SPoSE 49D for the 128).
- bibtex: add REAL entries from arXiv IDs in the research docs; some 2026 IDs need confirming.

## Main-branch coherence project (DONE, on `main`, pushed to github coherentLLM)
Full scale experiment (n=30/60/128), REPORT.md, all raw data. Metric r2. qwen-32b done.
Finding: coherence collapses with concept count for ALL models by n=60; humans hold; size doesn't
protect. Lead metric at scale = model triplet~human alignment (confound-free).
