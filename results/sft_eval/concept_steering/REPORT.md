# Concept-Vector Steering Lane

Status: CHTC GPU smoke `5513276` and bounded layer/alpha sweep `5513297`
both passed on 2026-07-09.

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

Outcome:

- cluster `5513235` started on `jcaicedogpu0002.chtc.wisc.edu`, an NVIDIA L40S
  worker, and passed the GPU probe.
- it failed with exit status `66` before model loading because
  `/staging/s/suresh27/models/llama31-8b-instruct/config.json` was not visible
  in the worker container.
- the staged model path is present on the CHTC login node; the failed worker
  advertised `HasChtcStaging` as false.
- `concept_steering_smoke.sub` and `concept_steering_sweep.sub` now require
  `TARGET.HasChtcStaging =?= true`.

## 2026-07-09 CHTC Smoke Retry 5513261

Run:

- CHTC run id: `coherence-concept-steering-20260709-010738`
- cluster: `5513261`
- remote directory: `~/chtc-runs/coherence-concept-steering-20260709-010738`
- local artifacts: `results/sft_eval/concept_steering/chtc/5513261/`

Outcome:

- ran on `gpu5000.chtc.wisc.edu`, NVIDIA H200, with
  `max_gpu_memory_mb=143771`.
- passed staged model visibility checks; `model_file_sample.txt` includes
  `/staging/s/suresh27/models/llama31-8b-instruct/config.json`.
- pip/import stage exited `0`.
- extraction exited `1` while importing `LlamaForCausalLM`.

Diagnosis:

- the pip overlay installed `torch==2.13.0+cu130` as an `accelerate`
  dependency.
- that shadowed the container's CUDA 12.4 PyTorch 2.5.1 while leaving the base
  `torchvision==0.20.1+cu124` on the path.
- `transformers` imported `torchvision`, which failed registering
  `torchvision::nms`; `AutoModelForCausalLM` then reported it could not import
  `LlamaForCausalLM`.

Patch:

- `chtc/concept_steering/run_concept_steering.sh` now installs the Python
  overlay with `pip --no-deps`.
- the install list explicitly includes non-torch runtime dependencies and
  intentionally avoids replacing the container's torch stack.

## 2026-07-09 Passing CHTC Smoke 5513276

Run:

- CHTC run id: `coherence-concept-steering-20260709-011336`
- cluster: `5513276`
- remote directory: `~/chtc-runs/coherence-concept-steering-20260709-011336`
- local artifacts: `results/sft_eval/concept_steering/chtc/5513276/`
- host: `dbrundagegpu5000.chtc.wisc.edu`
- GPU: NVIDIA L40S, `max_gpu_memory_mb=45460`

Status files:

| File | Status |
| --- | ---: |
| `exit_status.txt` | `0` |
| `extract_exit_status.txt` | `0` |
| `eval_exit_status.txt` | `0` |
| `pip_install_exit_status.txt` | `0` |

Artifacts:

- vectors:
  `results/sft_eval/concept_steering/chtc/5513276/extracted/vectors/{coherence,human_alignment}_caa_vectors.npz`
- summary:
  `results/sft_eval/concept_steering/chtc/5513276/extracted/sweep/SUMMARY.md`
- metrics:
  `results/sft_eval/concept_steering/chtc/5513276/extracted/sweep/sweep_results.csv`
- per-item details:
  `results/sft_eval/concept_steering/chtc/5513276/extracted/sweep/sweep_details.csv`

Smoke readout:

| Steering vector | Target eval | Alpha | Preference | Margin | Margin vs alpha 0 |
| --- | --- | ---: | ---: | ---: | ---: |
| `coherence` | `coherence` | `-2` | 1.00 | 0.6875 | -0.1250 |
| `coherence` | `coherence` | `0` | 1.00 | 0.8125 | 0.0000 |
| `coherence` | `coherence` | `2` | 1.00 | 0.8750 | +0.0625 |
| `human_alignment` | `human_alignment` | `-2` | 0.50 | 0.2371 | -0.0325 |
| `human_alignment` | `human_alignment` | `0` | 0.50 | 0.2695 | 0.0000 |
| `human_alignment` | `human_alignment` | `2` | 0.50 | 0.3114 | +0.0419 |

Retention readout:

- retention positive preference stayed at `1.00` for both steering vectors and
  all alpha settings.
- retention margins remained strongly positive:
  `coherence` alpha `-2/0/2` = `11.6449 / 12.5201 / 13.3180`;
  `human_alignment` alpha `-2/0/2` = `11.5869 / 12.5201 / 13.3773`.
- no obvious retention collapse appears in this tiny forced-choice smoke.

Interpretation:

- the smoke is operationally successful: staged storage, dependency overlay,
  vector extraction, and steering eval all pass.
- positive alpha moves the matching target margins in the expected direction,
  but `positive_preference` is saturated for coherence and unchanged for
  human-alignment at `n=2`, so this is evidence to scale, not a final effect.
- specificity is not yet clean: the human-alignment vector also raises the
  coherence margin at alpha `2`; the larger sweep must quantify matching vs
  nonmatching movement.

## 2026-07-09 Passing Bounded Sweep 5513297

Run:

- CHTC run id: `coherence-concept-steering-20260709-011336`
- cluster: `5513297`
- remote directory: `~/chtc-runs/coherence-concept-steering-20260709-011336`
- local artifacts: `results/sft_eval/concept_steering/chtc/5513297/`
- host: `gpu4005.chtc.wisc.edu`
- GPU: NVIDIA H100 80GB HBM3, via Condor slot `backfill1_4`

Status files:

| File | Status |
| --- | ---: |
| `exit_status.txt` | `0` |
| `extract_exit_status.txt` | `0` |
| `eval_exit_status.txt` | `0` |
| `pip_install_exit_status.txt` | `0` |

Artifacts:

- result bundle:
  `results/sft_eval/concept_steering/chtc/5513297/concept_steering_sweep_layers12_16_20_24_alpha_neg4_neg2_0_pos2_pos4_results.tgz`
- summary:
  `results/sft_eval/concept_steering/chtc/5513297/extracted/sweep/SUMMARY.md`
- metrics:
  `results/sft_eval/concept_steering/chtc/5513297/extracted/sweep/sweep_results.csv`
- per-item details:
  `results/sft_eval/concept_steering/chtc/5513297/extracted/sweep/sweep_details.csv`

Sweep arguments:

```text
sweep 12,16,20,24 -4+-2+0+2+4 ALL ALL 2 2 bfloat16 layers12_16_20_24_alpha_neg4_neg2_0_pos2_pos4
```

Sweep shape:

- 120 summary rows.
- 2 steering concepts: `coherence`, `human_alignment`.
- 4 layers: `12,16,20,24`.
- 5 alphas: `-4,-2,0,2,4`.
- 3 eval sets: `coherence` (`n=8`), `human_alignment` (`n=8`),
  `retention` (`n=10`).

Best target margin movements relative to alpha `0`:

| Steering vector | Target eval | Layer | Alpha | Preference | Margin | Margin vs alpha 0 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `coherence` | `coherence` | 12 | 4 | 0.625 | 0.8780 | +1.0222 |
| `coherence` | `coherence` | 16 | 4 | 0.625 | 0.4903 | +0.6345 |
| `human_alignment` | `human_alignment` | 24 | 4 | 0.875 | 0.9894 | +0.1197 |
| `human_alignment` | `human_alignment` | 16 | 2 | 0.875 | 0.9584 | +0.0887 |
| `human_alignment` | `human_alignment` | 24 | 2 | 0.875 | 0.9518 | +0.0821 |

Retention readout:

- alpha `0` retention baseline was `positive_preference=0.900` and
  `mean_delta_logprob=8.4232`.
- no setting catastrophically collapsed the retention forced-choice probe:
  retention preference stayed in `0.8-1.0`.
- aggressive layer-12 settings reduced retention margins:
  `coherence` layer 12 alpha `4` changed margin by `-2.5973`, and
  `human_alignment` layer 12 alpha `-4` changed margin by `-5.1625`.

Specificity readout:

- `coherence` steering is most useful at layer 12 alpha `4`, but
  `positive_preference` is unchanged from baseline, so the measurable effect is
  margin rather than item flips.
- `human_alignment` steering also mainly improves margins; target preference is
  already high at alpha `0` (`0.875`).
- specificity is imperfect. The largest cross-effect is
  `human_alignment -> coherence` at layer 12 alpha `4`, which raises coherence
  margin by `+2.5565` and preference by `+0.125`.
- next steering work should therefore treat layer 12 as a shared semantic/
  task-quality direction candidate, not as a cleanly separated alignment vector.

Next readout step:

- run qualitative generation/judge probes at the best low-risk settings:
  `coherence` layer 12 alpha `4`, `human_alignment` layer 24 alpha `4`, and a
  weaker `human_alignment` layer 16 alpha `2`.
- before any broad use, pair those with wider retention checks because layer-12
  aggressive alphas reduce retention margins even when preference stays mostly
  intact.

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
