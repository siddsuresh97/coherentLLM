# CODEX TASK 6: does the BEST mitigation model gain on other tasks? (esp. human-related)

Branch coherence-sft. Env: source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh &&
conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence
GPUs: 2x A5000 (idx 0,1) + 1x H100 (opt-a007). Use whatever's free; NEVER idle a GPU. Commit each step; do NOT push.
All 3 GPUs are currently FREE. Read research/CODEX_TASK_5.md for context + the eval harness patterns (rsLoRA conversion, --custom_adapter, tight-mem retention).

BEST models to evaluate: lowLR (out/adapters_mitigation/lowLR) and lowrank (out/adapters_mitigation/lowrank) — these broke the coherence/retention tradeoff (coh ~0.75, retention within 0.03 of base). Compare against BASE and the default real-SFT.

Question the user asked: (1) does the best model see GAINS on other benchmarks (not just avoid regression)? (2) How else is this fine-tuning helpful, ESPECIALLY on human-related tasks?

## Phase 6A [start now, parallel across GPUs] External SEMANTIC-SIMILARITY suite on lowLR + lowrank
Extend src/sft/eval_external.py to also run lowLR + lowrank (same protocol: Spearman rho of model
similarity vs human ratings). Existing benchmarks: STS-B, SimLex-999, WordSim-353, MEN. ADD if easy:
- **RG-65, MTurk-771, SimVerb-3500, MEN** (word-similarity, all human-rated).
Output: extend results/sft_eval/external/summary.csv with lowLR + lowrank rows. Commit.
KEY: does lowLR/lowrank BEAT base (not just real)? Report per-benchmark rho: base vs real vs lowLR vs lowrank.

## Phase 6B [parallel] HUMAN-ALIGNMENT / human-behavior tasks (the "human-related" ask)
These test whether coherence-SFT makes the model more human-like beyond our THINGS metric:
1. **THINGS odd-one-out human accuracy**: predict human triplet choices on the THINGS behavioral data
   (we have SPoSE + the triplet task). Measure model triplet-choice agreement with HUMAN majority choice,
   base vs lowLR. (reuse the triplet elicitation; compare to human ground truth, not just SPoSE geometry.)
2. **More human similarity-judgment datasets**: SimLex/SimVerb/MEN already in 6A double as human-behavior
   prediction. Add **WordSim-353 similarity vs relatedness split** if easy (tests human distinction).
3. **Typicality / category structure** (if a dataset is readily available, e.g. Rosch typicality norms or
   the McRae/Leuven category-typicality): does the model rank category members by human typicality better?
Output: results/sft_eval/human_tasks/summary.csv. Commit. Report base vs lowLR deltas.

## Phase 6C [parallel] Broader capability GAINS (not just retention floor)
Run a WIDER lm-eval battery on base vs lowLR to see if coherence helps reasoning/commonsense that
leans on conceptual structure:
- **commonsense**: piqa, winogrande, arc_easy, openbookqa, commonsense_qa, hellaswag (already have).
- **word/semantic**: wic (word-in-context), (blimp if easy).
- Keep MMLU + truthfulqa for the capability picture.
--limit 1000, standard few-shot, on the H100. Output: results/sft_eval/wide_bench/summary.csv (base vs lowLR).
Report which tasks lowLR GAINS on vs base (delta > +0.02), which are flat, which drop.

## Orchestration
- Launch 6A + 6C on separate GPUs immediately (both A5000s + H100 free); 6B where a slot frees.
- rsLoRA->vanilla adapter conversion needed before vLLM loads lowLR/lowrank (see Task5 harness).
- Commit each phase. Assemble one results/sft_eval/TASK6_SUMMARY.md: for the BEST model (lowLR), a table of
  every task vs base with delta + a GAIN/FLAT/DROP verdict, grouped as {similarity, human-behavior, capability}.
- Report the TASK6_SUMMARY table. Do NOT fabricate; if a dataset isn't available, note it as skipped.
