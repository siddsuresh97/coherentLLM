# CODEX TASK 4: the eval battery (4-axis coherence + knowledge retention)

Prereq: task 3 done (adapters in out/adapters/{real,scrambled}/). Read research/PLAN.md steps
4/4b/5 and research/independent_benchmarks.md. Branch coherence-sft. This is the PAYOFF step:
did coherence-SFT work?

## The three model states (everything below runs on ALL THREE)
1. `base`      = llama-3.1-8b-instruct (registry entry in configs/models.yaml, resolves to the
                 cached meta-llama/Meta-Llama-3.1-8B-Instruct snapshot via run_local.resolve_model_path)
2. `real`      = base + LoRA adapter out/adapters/real
3. `scrambled` = base + LoRA adapter out/adapters/scrambled  (delusive control)

vLLM LoRA: `LLM(base, enable_lora=True, max_lora_rank=64)` + `LoRaRequest` per adapter
(adapters are rsLoRA r=64 trained on the same llama-3.1-8b-instruct; check adapter_config.json).
Either (a) add a `--lora <adapter_dir>` flag to src/run_local.py and src/run_base_logprob.py
(minimal diff, preferred), or (b) write one hot-swap driver that loads base once and loops the
three states. Name raw-output model dirs `llama-3.1-8b-instruct`, `llama31-sft-real`,
`llama31-sft-scrambled` so downstream globbing stays simple.

## Eval concepts (FIXED)
The 128 THINGS eval concepts: data/scale128/concepts.csv (= data/sft/test_concepts.csv; these
never appeared in training). Existing stimuli: data/scale128/stimuli/{concepts.csv,triplets.csv}.
There is NO pairs.csv yet: generate data/scale128/stimuli/pairs.csv containing ALL ORDERED pairs
(both directions, 128*127 = 16,256 rows, same schema src/stimuli.py expects). Average (a,b) and
(b,a) ratings for the pairwise RDM; keep the per-direction raw for the symmetry metric (axis C).
Use `COHERENCE_STIM_DIR=data/scale128/stimuli COHERENCE_RAW_DIR=results/sft_eval/raw` env
overrides so nothing under results/raw or results/raw_128 is touched.

## Axis A: our Procrustes coherence (PRIMARY), dual gen + logprob
Per state, per mode:
- GENERATION: src/run_local.py, methods triplet + pairwise + feature.
  Feature = the SAME free-list pipeline used for results/raw_128 (listing -> consolidate features
  shared by >=3 concepts -> single-pair verify; see src/run_listing.py + how
  results/raw_128/llama-3.1-8b-instruct/{listing,listed_features,verify_pairs,feature}.csv were
  produced, and src/analyze_scale128.py for the downstream math). Reuse that code, do not invent
  a new feature protocol. IMPORTANT: consolidate the feature list PER STATE from that state's own
  listings (that is the protocol), but ALSO score every state on the base state's verify pairs so
  there is one shared-feature-set comparison; report both if they differ materially.
- LOGPROB: src/run_base_logprob.py (triplet compare-completions + pairwise 1..7 argmax already
  implemented). Add feature True/False scoring (score " True" vs " False" continuations of
  prompts.feature_prompt on the shared feature set) if straightforward; if not, logprob coherence
  from triplet+pairwise only and say so.
- Triplet embedding: SALMON d=5 via src/fit_triplet_salmon.py in the `salmon` conda env (same as
  the scale128 run; see data/scale128/*_triplet_d5.npy for the expected artifact). Fit per
  state x mode, save under results/sft_eval/.
- Coherence: Procrustes r2 = 1 - disparity (src/procrustes_analysis.py procrustes_r2, RDM-direct,
  same math as src/analyze_scale128.py). Report triplet~feature, triplet~pairwise,
  pairwise~feature, and their mean, separately for gen and logprob.
Reference baselines you should roughly reproduce for `base` gen: coherence t~f ~0.32,
human-triplet ~0.49 (data/scale128/coherence_128.csv). If base comes out wildly different,
something is wrong with the harness: stop and debug before running the adapters.

## Axis B: paraphrase/format consistency (independent of any external data)
Write 5 paraphrase templates per method (triplet/pairwise/feature) in a new src/sft/paraphrases.py
(template 1 = the canonical src/prompts.py wording; 4 genuine rewordings that change surface form,
not the task). To bound compute: subsample 1,500 triplets, 1,000 ordered pairs, and the shared
verify-pair set (cap 2,000) from the axis-A stimuli; run each x5 paraphrases, generation mode,
all three states. Metric per method = mean pairwise agreement of answers across the 5 paraphrases
(exact-match for triplet choice and feature T/F; for pairwise 1-7 report mean |rating diff| AND
agreement-within-1). Column per method + a mean.

## Axis C: transitivity + symmetry (label-free self-consistency)
- Symmetry: from the axis-A ordered pairwise raw (both directions of all 8,128 pairs):
  violation rate = fraction with |rating(a,b) - rating(b,a)| >= 2 (also report mean abs diff).
- Transitivity: build a dedicated triplet-cycle set: sample 400 (anchor; x,y,z) with all concepts
  from the 128, ask the three within-anchor triplets (a;x,y), (a;y,z), (a;x,z) in generation mode;
  a cycle violation = choices imply x>y>z>x. Report violation rate per state. Save the stimulus
  file so it is identical across states.

## Axis D: THINGS human alignment (independent human data)
Procrustes r2 of each state's SALMON d=5 triplet RDM (gen mode) vs the THINGS SPoSE human triplet
similarity on the same 128: data/scale128/human_spose_triplet_sim.npy already contains it (derived
from the SPoSE 49D embedding; the raw 49D lives at /mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/
Projects/vision_project/vision_robustness/experiments/after_iclr_2024/hebart/
spose_embedding_49d_sorted.txt if you need to re-derive; row order must match concepts.csv order,
see src/analyze_scale128.py). Same sym_rdm + procrustes_r2 math as axis A.

## Axis E: knowledge retention (must NOT regress)
lm-eval-harness on all THREE states: tasks mmlu, arc_challenge, hellaswag, truthfulqa_mc2.
Prefer the vllm backend with LoRA (`--model vllm --model_args pretrained=<base_snapshot>,
enable_lora=True,lora_local_path=<adapter>,max_lora_rank=64,...`); if the installed lm-eval
doesn't support that, fall back to `--model hf --model_args pretrained=<base>,peft=<adapter>`.
For speed use `--limit 500` per task, IDENTICAL settings/limit/fewshot across the three states
(comparability matters more than absolute numbers). pip install lm-eval into the coherence_sft
env if missing.

## Output: data/sft/eval_results.csv
Rows = base / real / scrambled. Columns (one per metric, prefix by axis/mode):
  gen_proc_tf, gen_proc_tp, gen_proc_pf, gen_proc_mean,
  lp_proc_tf (if feature-logprob done), lp_proc_tp, lp_proc_pf, lp_proc_mean,
  para_triplet, para_pairwise_within1, para_feature, para_mean,
  sym_viol_rate, sym_mean_absdiff, trans_viol_rate,
  human_things_triplet_r2,
  ret_mmlu, ret_arc_challenge, ret_hellaswag, ret_truthfulqa
Also write results/sft_eval/SUMMARY.md: the table + 5 lines reading it against the success
criteria (PLAN step 5): SUCCESS = real coherence rises vs base in BOTH gen and logprob, scrambled
does NOT, B/C/D also move in the right direction for real, retention within ~1-2 pts.

## Compute / order of operations
- Prefer the H100: ssh ssuresh@opt-a007.discovery.wisc.edu (key ~/.ssh/id_ed25519, shared FS,
  same absolute repo path). Set COHERENCE_FORCE_BF16=1 there. The A5000s work but are 6-8x slower.
- Rough volume per state (gen): ~triplets.csv-count + 16,256 pairwise + feature listing/verify +
  ~22k paraphrase + 1,200 transitivity prompts, max_tokens 8. That is well within a few hours for
  all three states on the H100. Run states sequentially, methods batched (vLLM does the batching).
- Sanity-check base first (see axis A note), THEN run real + scrambled.
- SALMON fits run on CPU (salmon env), can overlap with the next state's generation.

## Done criteria
- data/sft/eval_results.csv with all 3 rows + results/sft_eval/SUMMARY.md.
- New/changed code committed: eval driver(s) under src/sft/, --lora flags, paraphrases.py,
  pairs/transitivity stimulus generators. Commit stimuli + eval_results.csv + SUMMARY.md too
  (small); raw response CSVs may stay gitignored.
- Commit message: "SFT step 4: 4-axis eval battery + retention (base/real/scrambled)".
- IMPORTANT: actually run `git add` + `git commit` yourself before finishing. Do NOT push.
  Do NOT delete or overwrite existing data/results (results/raw, results/raw_128, data/scale128
  inputs are read-only except for adding stimuli/pairs.csv + new transitivity file).
- Report the final table in your last message.
