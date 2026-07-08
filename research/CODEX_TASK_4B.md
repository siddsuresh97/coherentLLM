# CODEX TASK 4B: FINISH the eval battery (task 4 is mid-run)

Read research/PLAN.md section 4 + research/STATUS.md first. Work on branch `coherence-sft`.
Env: source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh &&
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
GPU: prefer the H100 (see memory h100-access). ALWAYS commit each step; do NOT push; do NOT delete data.

## What is ALREADY done (do NOT redo, verify then skip)
- base `llama-3.1-8b-instruct`: results/sft_eval/raw/<model>/ has triplet,pairwise,feature (+_lp) gen+logprob. DONE.
- base SALMON d5: results/sft_eval/llama-3.1-8b-instruct_triplet_d5.npy and _lp_d5.npy. DONE.
- adapters exist: out/adapters_vllm_fixed/{real,scrambled} (vLLM-prepped). Use THESE for --lora.
- stimuli: results/sft_eval/stimuli/{paraphrase_*.csv, transitivity_cycles.csv}. reuse, do not regen.

## What is MISSING (finish these, in order, committing after each)
All eval on the FIXED 128 THINGS concepts (data/scale128/concepts.csv), out_model dirs under results/sft_eval/raw/.

1. REAL adapter, gen: pairwise raw is TRUNCATED (16257 rows vs base 26295) -> RERUN pairwise clean
   with --overwrite. Confirm triplet(10001)/feature present; if feature gen missing, run it.
   run_local.py --model llama-3.1-8b-instruct --out_model llama31-sft-real
     --lora out/adapters_vllm_fixed/real --methods pairwise --overwrite   (+ feature if absent)
2. REAL adapter, logprob: run the logprob variants (triplet_lp/pairwise_lp/feature_lp) exactly as base
   produced them (see src/run_base_logprob.py for the base recipe; mirror it with --lora real).
3. SCRAMBLED adapter, gen + logprob: full set (triplet/pairwise/feature, gen and _lp),
   out_model llama31-sft-scrambled, --lora out/adapters_vllm_fixed/scrambled.
4. SALMON d5 triplet fit for real + scrambled (gen and lp), same recipe as base:
   python src/fit_triplet_salmon.py --raw_dir results/sft_eval/raw --stim_dir data/scale128
     --out_dir results/sft_eval --d 5  <model>   for each of the 4 new triplet files.
5. Axis B paraphrase-consistency: run each model over results/sft_eval/stimuli/paraphrase_*.csv
   (src/sft/run_extra_eval.py / paraphrases.py already scaffolded) -> agreement rate per model.
6. Axis C transitivity/symmetry: score cycle-violation + pairwise-symmetry per model from the raw
   triplet/pairwise csvs (transitivity_cycles.csv). Free, no new generation if raw covers the cycles.
7. Retention (src/sft/eval_retention.py): MMLU(subset)+ARC+HellaSwag+TruthfulQA via lm-eval on vLLM,
   for base + real + scrambled. If lm-eval not installed, pip install into the coherence env
   (cache is redirected off home per .bashrc). Keep it SMALL/fast.
8. Aggregate: python src/sft/analyze_eval.py -> results/sft_eval/eval_results.csv with, per model
   {base, real, scrambled} x {gen, logprob}: triplet~feature, triplet~pairwise, pairwise~feature,
   triplet~human(THINGS SPoSE), paraphrase_agree, transitivity_viol, symmetry, + retention scores.

## Efficiency (user directive: at every step ask if it's the fastest way)
- vLLM: load base ONCE with enable_lora + hot-swap adapters if the runner supports it; else accept
  one reload per adapter but use --max_num_seqs high + --max_model_len small (short prompts).
- Reuse existing raw csvs; only (re)generate what's missing/truncated. Don't refit base SALMON.
- Prefer H100; run generation passes back-to-back in one session to amortize model load.

## Done criteria
- results/sft_eval/eval_results.csv exists and is populated for all 3 models x gen+logprob + retention.
- Commit after EACH numbered step (message "SFT eval 4B step N: ..."). Do NOT push.
- Print the final eval_results.csv to stdout so the summary is captured.
