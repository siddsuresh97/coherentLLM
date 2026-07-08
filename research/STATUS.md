# Coherence-SFT project STATUS (where we are + open threads)

**Last updated: 2026-07-08. Tasks 1-8 are ALL COMPLETE and committed to `coherence-sft`.
Task 9 fMRI audit/hidden-state/RSA is implemented, committed, and pushed.
Semantic-hub extraction/scoring has run for all seven arms and is part of the
current result checkpoint. This file was stale below this point (left as historical design record) —
read this top section first for where the project actually stands.**

## Handoff for a fresh agent picking this up

- **Branch:** `coherence-sft` (off `main`), repo `git@github.com:siddsuresh97/coherentLLM.git`.
  Latest checkpoint includes the Task 9 fMRI RSA plus semantic-hub result artifacts.
- **Env:** `source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && conda activate
  /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence` for all SFT/eval work.
  SALMON fits need the separate `salmon` conda env (skorch+dask deps the coherence env lacks) —
  route those calls through it specifically (see `run_eval_4b.sh`'s `retry_salmon()` pattern).
- **GPUs:** 2x RTX A5000 24GB (`rogers-gpu-1`, indices 0/1) + 1x H100 80GB
  (`opt-a007.discovery.wisc.edu`), shared with other work. Rule from the user: keep GPUs allocated
  and utilized when experiments are queued, ask how to make slow paths faster, and commit/push
  meaningful checkpoints. A5000 vLLM+LoRA logprob scoring needs careful tuning: auto batch can OOM,
  too-low `gpu_memory_utilization` can leave no KV cache, and the current lowrank retry is using
  `gpu_mem_util=0.75 --batch_size 4`.
  ThingsVision is not installed in the active coherence env on either GPU host; for current
  90-concept fMRI RSA, HDF5 reads were the bottleneck rather than RDM math. Revisit a torch/CuPy/
  ThingsVision-style GPU RDM backend for full-720, bootstrap, or permutation-heavy RSA.
- **Read in order:** `research/CODEX_TASK_1.md` through `CODEX_TASK_8.md` (chronological,
  each is what was actually asked + done — CODEX_TASK_8.md + `results/sft_eval/steer_actlayer/
  VERDICT.md` is the most recent). Then `results/sft_eval/REPORT.md` for the synthesized
  findings a human would read. `results/sft_eval/TASK6_SUMMARY.md` has the full 6A/6B/6C tables.
- **What's DONE (Tasks 1-8):** built NOVA-derived S*-based mutually-consistent SFT data
  (triplet/pairwise/feature), trained real + scrambled-control LoRA adapters (QLoRA/rsLoRA,
  Unsloth), ran the 4-axis coherence+retention battery, confirmed the coherence gain is real
  (scrambled control ~0.03, four independent axes co-move, r=0.914 coherence<->human coupling),
  found low-LR (5e-5) and low-rank (16) mitigate but do not zero out the retention cost (full
  battery: 6/10 capability groups + 57/59 MMLU subjects still drop), extracted a weight-space
  task-vector direction (SFT - base) where alpha=0.25 recovers ~87% of the coherence gain for
  ~10% of the retention cost, and exhaustively ruled out activation-space steering as a
  training-free alternative (32/32 mid-layer, norm-matched layer x alpha configs collapse
  generation — an earned negative, not an all-layers artifact).
- **What's DONE after Tasks 1-8:** Task 9 first pass is implemented. fMRI data audit found 90 exact
  held-out SFT/THINGS-fMRI overlaps across 3 subjects. Hidden states are extracted for `base`,
  `scrambled`, `lowLR`, `lowrank`, `taskvec_a0p25`, `taskvec_a0p5`, and `taskvec_a1p0` with shape
  `90 x 33 x 4096`. RSA runs end-to-end. Primary Ventral Visual RSA is strong but aligned arms are
  essentially flat vs base (`base=0.1990`, `lowLR=0.2010`, `lowrank=0.2013`), while scrambled is
  much lower (`0.1293`). See `results/sft_fmri/REPORT.md`. Task 9 Track B semantic hub has also
  run on 128 held-out concepts across triplet/pairwise/feature-listing prompt spokes. Mid-layer
  cross-format RDM Spearman rises from base `0.3230` to `0.6211-0.6770` for lowLR/lowrank/taskvec
  arms, led by `taskvec_a0p25`; see `results/sft_semantic_hub/REPORT.md`.
- **What's NOT done / next candidates:** The semantic-hub result supports stronger cross-format
  invariance, but not yet a clean concept-dominant hub because concept-minus-format alignment stays
  negative for every arm. Next Task 9 step is the bridge: correlate arm/layer hub metrics with fMRI
  RSA and test whether format-averaged hub RDMs predict fMRI better than single-format RDMs. The
  stronger language-fMRI/Huth/Fedorenko encoding run remains feasibility-stage only. Task 10
  lowrank and task-vector wide-benchmark mitigation is in flight. Task 9 and Task
  10 briefs now track these follow-ups:
  `research/CODEX_TASK_9_FMRI_AND_SEMANTIC_HUB.md`,
  `research/CODEX_TASK_10_BENCHMARK_DROPS.md`, and the running handoff log
  `research/EXPERIMENT_LOG.md`. Task-list items #15 (7 new frontier
  models for the earlier, separate 18-model coherence benchmark) is also still pending and
  unrelated to the SFT thread.
- **Do not fabricate data or invent bibtex entries** — every task brief above ends with this
  instruction for a reason; report only on-disk numbers, real generation samples for
  degeneracy checks, and ask the user before treating a stale/partial artifact as real.

## Original design record (historical, pre-Task-1, kept for context)

Decision at project start: **DIY everything** (SFT via Unsloth/QLoRA, RL via
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

## OPEN / TODO (in order) — HISTORICAL, all items below are now DONE (see handoff section above)
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

## MANDATORY for ALL FUTURE training runs (scaling ablation, retrains)
The SFT examples are tiny (median ~33 tok, max ~95). Step-3 used max_seq_length=256 WITHOUT
--packing, so ~90% of every batch was padding (~10x wasted compute). Every future
src/sft/train_lora.py invocation MUST:
1. pass --packing (train_lora.py already supports it; passes packing=True to SFTConfig);
2. use --max_seq_length 128 (ample for a 95-token max);
3. VERIFY packing took effect (SFTConfig accepted packing=True; log it). If the installed
   TRL/unsloth SFTConfig lacks `packing`, note that and fall back to group_by_length=True.
Each scaling-ablation point should then train in a few minutes, not 30. The existing
real/scrambled adapters are fine as-is; do NOT retrain them for the first eval.

### wandb loss curves (all future training runs)
train_lora.py now defaults --report_to wandb (project coherentLLM-sft), calls wandb.init(name=
<out-basename>-<data>-steps<N>-r64-seq<L>, config=args) in main() before either backend, keeps
logging_steps=10, and dumps trainer_state.json next to training_metrics.json (loss history on
disk too). User is already logged in (~/.netrc). BEFORE the next training run codex MUST:
1. pip install wandb into the coherence_sft env (not yet installed).
2. Run one training job and VERIFY a run actually appears under project coherentLLM-sft.
(--report_to none disables it; if wandb import fails at runtime it self-disables with a warning.)

### CORRECTNESS GATE before using --packing (packing != correct attention masks)
Our examples are INDEPENDENT single-turn QA (each triplet/pairwise/feature is its own example).
Naive packing concatenates examples into one block under a FULL causal mask, letting example B
attend across the boundary to example A (cross-contamination) -> corrupts training. Before using
--packing for ANY real run, VERIFY the packing is boundary-aware (block-diagonal / position-reset /
FlashAttention varlen cu_seqlens so each sub-example is isolated):
1. Check installed TRL + unsloth versions. Unsloth packing usually = FA2 varlen (cu_seqlens) =
   CORRECT. Plain older TRL packing=True = NAIVE (cross-contamination) unless FA2 + position_ids
   reset. Newer TRL added neat/greedy packing with proper masking -- confirm ours has it.
2. Concretely test: pack 2-3 known examples, inspect the attention mask / position_ids actually
   used; confirm block-diagonal (B cannot attend to A), NOT a full causal mask over the block.
3. If our stack only does NAIVE packing: DO NOT use packing. Use --max_seq_length 128 +
   group_by_length=True + a length-bucketed batch sampler instead (most of the speedup, safe).
4. Report which path we are on (boundary-aware packing vs padded/bucketed) and note it in the
   training commit. CORRECTNESS over speed: a fast-but-contaminated run is worse than a slower one.

## Not-yet-done infra checks
- unsloth not installed in any env yet.
- no raw human THINGS odd-one-out triplets on disk (downloadable OSF f5rn6/qn5uv if we want the
  human RSM beyond SPoSE; we already have SPoSE 49D for the 128).
- bibtex: add REAL entries from arXiv IDs in the research docs; some 2026 IDs need confirming.

## Main-branch coherence project (DONE, on `main`, pushed to github coherentLLM)
Full scale experiment (n=30/60/128), REPORT.md, all raw data. Metric r2. qwen-32b done.
Finding: coherence collapses with concept count for ALL models by n=60; humans hold; size doesn't
protect. Lead metric at scale = model triplet~human alignment (confound-free).
