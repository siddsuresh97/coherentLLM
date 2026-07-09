# Concept Steering CHTC Plan

Status: runnable scaffold prepared and locally dry-run validated. No CHTC
steering job has been launched from this lane yet.

## Objective

Turn the committed concept-vector steering scaffold into a GPU smoke that tests
whether CAA-style directions for `coherence` and `human_alignment` can move
forced-choice preferences without immediately damaging neutral retention items.

## Inputs

- contrast data: `data/sft/steering_contrasts/`
- extractor: `src/sft/extract_concept_vectors.py`
- evaluator: `src/sft/run_concept_steering_eval.py`
- staged base model: `/staging/s/suresh27/models/llama31-8b-instruct`
- CHTC runner: `chtc/concept_steering/run_concept_steering.sh`

The current source scripts do not load LoRA adapters. The first steering run is
therefore base-model steering only. Adapter-conditioned steering is a follow-up
source change, not part of this isolated handoff.

## Minimal GPU Smoke

Design:

- extract one `coherence` vector file and one `human_alignment` vector file
  from the first 4 train contrast pairs per concept
- use layer 24 as the mid/late steering site
- sweep alphas `-2, 0, 2`
- score 2 coherence eval items, 2 human-alignment eval items, and 2 retention
  items
- expected summary rows: `2 x 1 x 3 x 3 = 18`

Equivalent local command, only when a local GPU is idle:

```bash
CUDA_VISIBLE_DEVICES=<idle_gpu> python src/sft/extract_concept_vectors.py \
  --concepts coherence human_alignment \
  --max-pairs 4 \
  --batch-size 1 \
  --out-dir results/sft_eval/concept_steering/vectors_smoke \
  --dtype bfloat16 \
  --device cuda \
  --overwrite

CUDA_VISIBLE_DEVICES=<idle_gpu> python src/sft/run_concept_steering_eval.py \
  --vector-dir results/sft_eval/concept_steering/vectors_smoke \
  --out-dir results/sft_eval/concept_steering/smoke_layer24 \
  --steer-concepts coherence human_alignment \
  --eval-concepts coherence human_alignment \
  --layers 24 \
  --alphas -2 0 2 \
  --max-items 2 \
  --batch-size 1 \
  --dtype bfloat16 \
  --device cuda \
  --overwrite
```

CHTC command:

```bash
RUN_ID=coherence-concept-steering-YYYYMMDD-HHMM
chtc-ssh "mkdir -p ~/chtc-runs/${RUN_ID}/logs ~/chtc-runs/${RUN_ID}/src ~/chtc-runs/${RUN_ID}/data/sft/steering_contrasts"
chtc-push chtc/concept_steering/ "chtc-runs/${RUN_ID}/"
chtc-push src/ "chtc-runs/${RUN_ID}/src/"
chtc-push data/sft/steering_contrasts/ "chtc-runs/${RUN_ID}/data/sft/steering_contrasts/"
chtc-ssh "cd ~/chtc-runs/${RUN_ID} && condor_submit concept_steering_smoke.sub"
```

Pull after completion:

```bash
mkdir -p results/sft_eval/concept_steering/chtc/${RUN_ID}
chtc-pull "chtc-runs/${RUN_ID}/concept_steering_smoke_layer24_alpha_neg2_0_pos2_results.tgz" \
  "results/sft_eval/concept_steering/chtc/${RUN_ID}/"
```

Expected returned files inside the tarball:

- `exit_status.txt`, `extract_exit_status.txt`, `eval_exit_status.txt`
- `vectors/coherence_caa_vectors.npz`
- `vectors/human_alignment_caa_vectors.npz`
- `vectors/extraction_summary.csv`
- `sweep/sweep_results.csv`
- `sweep/sweep_details.csv`
- `sweep/SUMMARY.md`
- `job_summary.md`
- environment, GPU, import, and command logs

## Scale-Up After Smoke

If smoke exits cleanly and retention does not collapse, submit:

```bash
chtc-ssh "cd ~/chtc-runs/${RUN_ID} && condor_submit concept_steering_sweep.sub"
```

Scale-up config:

- full train contrast set for both concepts
- layers `12,16,20,24`
- alphas `-4,-2,0,2,4`
- all eval and retention probe items
- expected summary rows: `2 steering concepts x 4 layers x 5 alphas x 3 eval sets = 120`

Primary readout:

- matching target gain: `steer_concept == eval_set` should improve
  `positive_preference` or `mean_delta_logprob` versus alpha `0`
- specificity: the nonmatching target set should move less than the matching
  target set
- retention: retention rows should stay close to alpha `0`; sharp drops mark
  the layer/alpha as too destructive

If unit-vector alphas are too weak after the first full sweep, run a separate
norm-matched diagnostic with smaller alphas such as `-0.10,-0.05,0,0.05,0.10`.
That requires either a one-off local command or a third submit file using
`--norm-match`.

## Failure Triage

- Import failure: inspect `pip_install.txt` and `imports_after_install.txt`.
- Wrong GPU: `gpu_check.txt` should show at least 40000 MB; otherwise the
  Condor requirement did not match as expected.
- Model path failure: verify `/staging/s/suresh27/models/llama31-8b-instruct`
  still has `config.json`.
- Empty or NaN sweep rows: inspect `sweep_details.csv` for token counts and
  raise `--max-length` above 512 only if truncation is visible.
