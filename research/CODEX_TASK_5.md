# CODEX TASK 5: coherence follow-up research (autonomous, keep ALL GPUs busy)

Branch coherence-sft. Env: source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh &&
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
GPU policy: 2x A5000 (rogers-gpu-1 idx 0,1) + 1x H100 (opt-a007, ssh key /mnt/ws/home/ssuresh/.ssh/id_ed25519).
Use WHATEVER GPU is free per step; NEVER leave a GPU idle. Commit each step; do NOT push; do NOT delete data.
Adapters: out/adapters_vllm_fixed/{real,scrambled}. Eval concepts: data/scale128/ (128 THINGS). Metric: src/procrustes_analysis.py.
Prior result (results/sft_eval/eval_results.csv): real SFT raised GEN coherence 0.32->0.70, scrambled control flat 0.03,
logprob mixed, retention DROPPED (but only --limit 100). Now the follow-ups.

## PHASE A (do first, in PARALLEL across free GPUs)

### A1 [H100 preferred] Retention re-run FULL sample
src/sft/eval_retention.py currently: TASKS = 2 MMLU subsets + arc_challenge + hellaswag + truthfulqa_mc2, --limit 100.
- Widen to a real MMLU (use full `mmlu` task group, or >=10 representative subsets) + arc_challenge + hellaswag + truthfulqa_mc2 + winogrande.
- Run --limit 0 (full) or at least --limit 1000, num_fewshot per standard (mmlu 5-shot, arc 25-shot, hellaswag 10-shot, truthfulqa 0-shot) with apply_chat_template for the instruct model (the earlier warning flagged it wasn't applied).
- Run for base, real, scrambled. Write results/sft_eval/retention_full/<model>/... and a summary CSV.
- Commit "Task5 A1: retention FULL sample (wide MMLU, chat template, standard few-shot)".

### A2 [build now, run when an A5000 frees] Coherence steering vector — EXTRACT
New src/sft/extract_coherence_vector.py:
- Method 1 (task vector): load base + real LoRA; the coherence task vector = merged(real) - base weights (per Ilharco task arithmetic). Save the delta (or keep the LoRA as the vector).
- Method 2 (activation diff): run base AND real over the 128-concept elicitation prompts, capture mean hidden state per layer (last-token, all layers), vector_L = mean_real_h_L - mean_base_h_L. Save data/sft/steer/coh_vector_actdiff.npz (per-layer) + coh_vector_taskvec.
- Small GPU (fits A5000). Commit "Task5 A2: extract coherence steering vectors (task-vec + activation-diff)".

### A3 [run when the other A5000 frees] External similarity benchmarks
New src/sft/eval_external.py: base vs real vs scrambled on STS-B (dev), SimLex-999, WordSim-353, MEN.
- For each: get model similarity (cosine of last-token hidden states for the two words/sentences, OR the pairwise-rating prompt), correlate (Spearman) with human ratings.
- Also try lm-eval's sts tasks if available. Write results/sft_eval/external/summary.csv (rho per benchmark per model).
- Commit "Task5 A3: external similarity benchmarks (STS-B/SimLex/WordSim/MEN)".

## PHASE B (after A2 lands)

### B1 Steering vector — APPLY to base
New src/sft/apply_coherence_vector.py:
- Add the activation-diff vector to the BASE model's residual stream at inference (forward hook, add scale*vector_L at each layer L), scale sweep alpha in {0, 0.5, 1, 2, 4, 8}.
- ALSO test the task-vector route: base_weights + alpha*taskvec, alpha in {0.25,0.5,1.0}.
- Re-run the coherence + THINGS-human eval on the steered base at each alpha (reuse the run_local/run_base_logprob + SALMON + analyze_eval path, out_model=llama31-steer-a<alpha>).
- KEY QUESTION: does steered base's coherence / human-alignment rise toward the real-SFT model WITHOUT training? Plot coherence + human_r2 vs alpha.
- Commit "Task5 B1: apply coherence steering vector to base + coherence-vs-alpha sweep".

## PHASE C (CPU-ish, run anytime a slot is free)

### C1 Coherence vs human-alignment separability
New src/sft/analyze_separability.py: from eval_results.csv + the ch3 model zoo coherence data (if present under results/), test whether gen coherence and THINGS-human r2 are correlated/dissociable across models+arms. Report r, and whether any arm gains one without the other. Commit.

## Orchestration rules
- KEEP ALL GPUs BUSY: launch A1 on H100 immediately; the moment an A5000 frees from marketlm, start A2 then A3; never idle.
- Self-healing: one vLLM per process (frees GPU before next loads), PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True, env-activate every call, skip-if-exists, retry 3x. Use the run_eval_4b.sh pattern.
- Commit after each numbered step. Report a table when each phase lands. Flag any give-up or GPU contention.
