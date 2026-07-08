# CODEX TASK 7: does the STEERING vector preserve benchmarks? (retention + capability on task-vec models)

Branch coherence-sft. Env: source .../conda.sh && conda activate .../envs/coherence.
GPUs: 2x A5000 + 1x H100, use whatever's free; NEVER idle a GPU; commit each step; do NOT push.
QUEUE BEHIND Task 6 (6C wide-bench) — do not compete for a GPU it's using; start on the first free card.

## Why
We measured the TASK-VECTOR steering models (base + alpha*(real-base), alpha in {0.25, 0.5, 1.0})
ONLY for coherence + human-alignment. We NEVER ran retention or capability benchmarks on them.
So we CANNOT yet claim "the steering vector raises coherence without hurting other benchmarks."
This task closes that gap.

## Models to eval (task-vector route)
- llama31-steer-task-a0p25  (alpha 0.25)
- llama31-steer-task-a0p5   (alpha 0.5)
- (alpha 1.0 == the real SFT, already has retention 0.367 — reuse, don't rerun)
The steered model = the real LoRA adapter with weights scaled by alpha (see
src/sft/apply_coherence_vector.py, route=taskvec). Load the scaled adapter for vLLM/eval the
SAME way apply_coherence_vector produced its coherence numbers, so results are consistent.

## Evals per alpha (same protocol as Task5/Task6 so numbers are comparable)
1. RETENTION: eval_retention.py --limit 1000 over ARC-Challenge, HellaSwag, MMLU, TruthfulQA
   (the same set used for base/real/lowLR). Compare retention_mean vs base 0.574 and vs
   real(alpha=1) 0.367.
2. (if cheap) the wide capability battery from Task6 6C on alpha=0.25 only (piqa, winogrande,
   arc_easy, openbookqa, commonsense_qa, wic) — optional, only if GPUs are idle after retention.

## Output + the question to answer
- Extend results/sft_eval/steer/sweep_summary.csv (or a new steer_retention.csv) with a
  retention_mean column per alpha, alongside the existing coherence + human_r2.
- KEY TABLE: alpha | coherence | human_r2 | retention_mean | retention_delta_vs_base.
- Answer plainly: does a partial steering nudge (alpha=0.25/0.5) keep coherence while
  retaining benchmark accuracy, or does it inherit the real-SFT retention drop proportional
  to alpha? (Hypothesis: alpha=1 == real SFT so ~-0.21; the open question is whether alpha=0.25
  keeps most coherence with a SMALLER drop — a real finding either way.)
- Commit "Task7: retention/capability of task-vector steering models (alpha 0.25/0.5)".

## Notes
- One vLLM per process, expandable_segments, env-activate every call, 24GB A5000 OOMs on 8B
  logprob -> tight max_model_len or H100. rsLoRA->vanilla conversion if needed (Task5 harness).
- Do NOT fabricate. Report only on-disk numbers.
