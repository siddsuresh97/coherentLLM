# Concept-Vector Steering Lane

Status: implemented, locally syntax/dry-run validated, and submitted as a CHTC
GPU smoke on 2026-07-09. The first submitted cluster is `5513235`.

## 2026-07-09 CHTC Smoke Submission

Run:

- CHTC run id: `coherence-concept-steering-20260709-005726`
- cluster: `5513235`
- remote directory: `~/chtc-runs/coherence-concept-steering-20260709-005726`
- submit file: `chtc/concept_steering/concept_steering_smoke.sub`
- command:

```bash
chtc-ssh 'cd ~/chtc-runs/coherence-concept-steering-20260709-005726 && condor_submit concept_steering_smoke.sub'
```

Design:

- concepts: `coherence`, `human_alignment`
- train pairs per concept: first 4
- layer: `24`
- alphas: `-2,0,2`
- eval items: 2 coherence, 2 human-alignment, 2 retention
- expected summary shape: 18 rows

Initial CHTC state:

- `condor_q -better-analyze 5513235` reported the job as satisfiable under the
  40GB+ GPU requirement.
- The first queue poll showed `JobStatus=1` and no worker stdout/stderr yet.
- Local GPU smoke was skipped because this Codex session could not see a local
  NVIDIA driver through `nvidia-smi`.

Expected result bundle after completion:

```text
results/sft_eval/concept_steering/chtc/coherence-concept-steering-20260709-005726/
```

Readout plan:

- verify `exit_status.txt`, `extract_exit_status.txt`, and
  `eval_exit_status.txt` are all `0`.
- inspect `sweep/SUMMARY.md` and `sweep/sweep_results.csv`.
- submit `concept_steering_sweep.sub` only if target rows move and retention
  rows do not collapse relative to alpha `0`.

## Files

- `data/sft/steering_contrasts/metadata.json`: dataset schema, concept definitions, and extraction-position note.
- `data/sft/steering_contrasts/coherence.jsonl`: 24 train and 8 eval contrast pairs derived from matched `data/sft/train.jsonl` vs `data/sft/train_scrambled.jsonl` semantic/coherence prompts.
- `data/sft/steering_contrasts/human_alignment.jsonl`: 18 train and 8 eval seed contrasts for helpful, honest, safe, privacy-preserving, and uncertainty-aware behavior.
- `data/sft/steering_contrasts/retention_probe.jsonl`: 10 cheap forced-choice retention items.
- `src/sft/extract_concept_vectors.py`: CAA-style difference-of-means vector extractor.
- `src/sft/run_concept_steering_eval.py`: targeted steering sweep with coherence, human-alignment, and retention probes.

## Method

The extractor loads base `llama-3.1-8b-instruct` through `configs/models.yaml`, formats each contrast as a chat prompt plus candidate assistant completion, and records hidden states at the final answer token. It does not append the assistant end-of-turn token, so the extraction position is the last completion token.

For each concept and each layer row:

```text
v_layer = mean(hidden_positive_layer) - mean(hidden_negative_layer)
```

The `.npz` output stores raw vectors and per-layer unit vectors. The evaluator uses the unit vector by default and injects it at a selected decoder layer:

```text
h' = h + alpha * unit(v_layer)
```

Use `--norm-match` for a stronger diagnostic mode:

```text
h' = h + alpha * unit(v_layer) * ||h||
```

The first-pass evaluator is forced-choice log-likelihood, not full generation. For each item it scores whether the steered model prefers the positive completion over the negative completion. This is cheap enough for layer/alpha sweeps and should precede any full lm-eval retention run.

## Validation Run

Commands already run:

```bash
python -m py_compile src/sft/extract_concept_vectors.py src/sft/run_concept_steering_eval.py
python src/sft/extract_concept_vectors.py --dry-run --concepts coherence human_alignment
python src/sft/run_concept_steering_eval.py --dry-run --smoke --layers 16 --alphas 0 2
```

Observed dry-run counts:

- `coherence` train: 24 pairs, with feature-truth, triplet-similarity, and pairwise-similarity contrasts.
- `human_alignment` train: 18 pairs across honesty, privacy, refusal, safety, fairness, calibration, and related seed families.
- smoke eval plan: layer 16, alpha `0` and `2`, two coherence items, two human-alignment items, and two retention items per steering concept.

## Next Commands

Run a tiny vector extraction smoke first:

```bash
CUDA_VISIBLE_DEVICES=0 python src/sft/extract_concept_vectors.py \
  --concepts coherence human_alignment \
  --max-pairs 4 \
  --batch-size 1 \
  --out-dir results/sft_eval/concept_steering/vectors_smoke \
  --overwrite
```

Run a tiny steering smoke against those smoke vectors:

```bash
CUDA_VISIBLE_DEVICES=0 python src/sft/run_concept_steering_eval.py \
  --vector-dir results/sft_eval/concept_steering/vectors_smoke \
  --smoke \
  --layers 16 \
  --alphas 0 2 \
  --batch-size 1 \
  --overwrite
```

If the smoke completes and output deltas are finite, extract the full train-split vectors:

```bash
CUDA_VISIBLE_DEVICES=0 python src/sft/extract_concept_vectors.py \
  --concepts coherence human_alignment \
  --batch-size 2 \
  --overwrite
```

Then run the initial targeted sweep:

```bash
CUDA_VISIBLE_DEVICES=0 python src/sft/run_concept_steering_eval.py \
  --layers 12,16,20,24 \
  --alphas -4 -2 0 2 4 \
  --batch-size 2 \
  --overwrite
```

If unit-vector alphas are too weak, run the norm-matched diagnostic with smaller alphas:

```bash
CUDA_VISIBLE_DEVICES=0 python src/sft/run_concept_steering_eval.py \
  --layers 12,16,20,24 \
  --alphas -0.10 -0.05 0 0.05 0.10 \
  --norm-match \
  --batch-size 2 \
  --overwrite
```

## Readout

Primary output files after a real sweep:

- `results/sft_eval/concept_steering/vectors/{coherence,human_alignment}_caa_vectors.npz`
- `results/sft_eval/concept_steering/vectors/{coherence,human_alignment}_caa_vectors.json`
- `results/sft_eval/concept_steering/sweep_results.csv`
- `results/sft_eval/concept_steering/sweep_details.csv`
- `results/sft_eval/concept_steering/SUMMARY.md`

Interpretation:

- target gain: the matching `steer_concept == eval_set` row should increase `positive_preference` and `mean_delta_logprob` relative to alpha `0`.
- specificity: the nonmatching target set should move less than the matching set.
- retention: retention `positive_preference` should stay near the alpha `0` baseline; sharp drops mean the layer/alpha is too aggressive.
