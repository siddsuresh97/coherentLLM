# Experiment 1 Report - Detecting a Concept Move from Triplets

Current status: baseline and noise floor are green, but the model-edit pathway has **not yet produced a clean positive detection**. The latest completed NOVA feature-listing run moved behavior, but the movement was broad and the edited concept was not localized.

## Hypothesis

If we deliberately move one concept in training, then held-out odd-one-out triplet judgments should show that the edited concept's row in the behavioral RDM changed more than ordinary LoRA fine-tuning drift.

For the first concrete test, the target concept is `antelope`, and the intended concentrated move is to push it away from the close feature-space neighbor `bison`.

Success requires both:

- `antelope` is top-ranked by edit-vs-base row change relative to the control null.
- SNR is greater than 1, meaning edit movement exceeds matched control drift.

## What I Tried

| Attempt | Training lever | Control/edit matched? | Verdict | Why |
|---|---|---:|---|---|
| 1 | Direct pairwise similarity fallback | No | Partial / not success | `antelope` ranked 1, but SNR was exactly 1.000 because control drift matched edit drift. |
| 2 | NOVA feature-listing SFT | Yes | Did not localize | SNR was 1.055, but `antelope` ranked 23; many non-target rows moved more. |
| 3 | Matched pairwise similarity fallback | Yes | Prepared, strong run stopped | The strong `rank=32`, `lr=2e-4` setting looked too broad/unstable, so I stopped before treating it as evidence. |

Main interpretation: the detector and scorer work, and the model's triplet behavior can move, but the current LoRA settings are too broad for attribution. This is probably a training-strength/null-tightness issue, not evidence that triplets can never detect the move.

## Latest Completed Run

Run: `concentrated_drop_100`, NOVA feature-listing supervision.

Model: `llama-3.1-8b-instruct`, local Llama-3.1-8B-Instruct snapshot.

Training settings:

- Backend: PEFT QLoRA, 4-bit NF4, bf16 compute.
- LoRA target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`.
- Completed setting: `rank=32`, `learning_rate=2e-4`, `max_steps=400`, `seed=1729`.
- Resulting loss: both control and edit reached final logged loss about `0.0001`, which is a warning sign for overfitting on this tiny dataset.

Detection result:

- `target_rank=23`.
- `target_snr_control_only=1.055`.
- `target_top_ranked=false`.
- The edited pair `antelope`-`bison` ranked high in target-pair deltas, but the same pair also moved under the control null, so this is not clean localization.

Evidence: [detection/concentrated_drop_100.json](detection/concentrated_drop_100.json), [figs/resolution_heatmap.png](figs/resolution_heatmap.png), [lora_control/concentrated_drop_100/training_metrics.json](lora_control/concentrated_drop_100/training_metrics.json), [lora_edit/concentrated_drop_100/training_metrics.json](lora_edit/concentrated_drop_100/training_metrics.json).

## Why I Think It Failed

The feature-listing run did not fail because control and edit used different examples. They were matched: both arms had 296 rows, with 29 non-target concepts repeated 8 times and `antelope` repeated 64 times.

The likely failure mode is that the adapter was too strong and changed the model's general triplet behavior. Rank 32 over all linear modules at `2e-4` for 400 steps drove the tiny training set to near-zero loss. That can make many unrelated rows move, so `antelope` is no longer the largest behavioral change even when the intended concept was edited.

The previous `coherence-sft` branch points to a better recipe: `lowLR` (`5e-5`, rank 64) and `lowrank` (`rank=16`, `2e-4`) preserved retention much better than the original real SFT while still improving conceptual coherence. For this targeted edit, the next run combines the conservative parts: `rank=16`, `learning_rate=5e-5`, same matched data.

Evidence: [results/sft_eval/mitigation/summary.csv](../../results/sft_eval/mitigation/summary.csv), [src/sft/summarize_mitigation.py](../../src/sft/summarize_mitigation.py), [research/CODEX_TASK_6.md](../../research/CODEX_TASK_6.md).

## Next Run

Use the same matched NOVA feature-listing data, but a lower-drift recipe:

```bash
CUDA_VISIBLE_DEVICES=0 \
TRANSFORMERS_NO_TORCHVISION=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 \
COHERENCE_ENV=/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence \
HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models \
BASE_MODEL_PATH=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
TRAIN_BASE_MODEL=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 \
SUPERVISION=feature TRIPLET_BACKEND=transformers TRIPLET_LOAD_IN_4BIT=1 \
TRIPLET_BATCH_SIZE=16 MAX_MODEL_LEN=1024 TRAIN_BACKEND=peft \
TRAIN_BATCH_SIZE=4 TRAIN_GRAD_ACCUM=1 TRAIN_STEPS=400 \
LORA_RANK=16 TRAIN_LEARNING_RATE=5e-5 TRAIN_REPORT_TO=wandb WANDB_MODE=online \
scripts/run_experiment1_real_gpu.sh concentrated_drop_100
```

The runner now:

- Defaults W&B to online, not offline.
- Loads the existing key from `/mnt/home/ssuresh/.netrc` at runtime if `WANDB_API_KEY` is not already set.
- Fails fast if online W&B is requested but no key is available.
- Refuses to silently continue without W&B when `--report_to wandb` is requested.

W&B note: the code is ready for online W&B, but this Codex session could not upload the completed run because sending private experiment artifacts to W&B was blocked by execution policy. No workaround was attempted.

Relevant files: [scripts/run_experiment1_real_gpu.sh](../../scripts/run_experiment1_real_gpu.sh), [src/sft/train_lora.py](../../src/sft/train_lora.py), [scripts/upload_experiment1_wandb.py](../../scripts/upload_experiment1_wandb.py).

## Data And Prompts

Baseline:

- Feature-space RDM: [artifacts/rdms/feature_rdm_base.npy](artifacts/rdms/feature_rdm_base.npy).
- Behavioral base RDM: [artifacts/rdms/rdm_base.npy](artifacts/rdms/rdm_base.npy).
- Floor stats: [floor_stats.json](floor_stats.json).
- Frozen triplet protocol: [triplet_protocol.json](triplet_protocol.json).
- Items: [items.json](items.json).

Behavioral floor:

- Status: green.
- Mean upper-triangle Pearson: `0.8066`.
- Canonical rerun vs base: Pearson `1.0000`, RMS `0.0000`.
- Paraphrase vs base: Pearson `0.6133`, RMS `0.1768`.

Triplet detection prompt:

```text
System: You are a helpful assistant who gives responses to questions.
User: Answer using only one word - bison or toaster and not antelope. Which is more similar in semantic meaning to antelope?
Expected answer format: one word, either "bison" or "toaster".
```

NOVA feature-listing data:

- Control data: [sft_data/concentrated_drop_100/control.jsonl](sft_data/concentrated_drop_100/control.jsonl).
- Edit data: [sft_data/concentrated_drop_100/edit.jsonl](sft_data/concentrated_drop_100/edit.jsonl).
- Manifest: [sft_data/concentrated_drop_100/manifest.json](sft_data/concentrated_drop_100/manifest.json).
- Source dataset: public NOVA norms from `llm-norms-cogsci2025`.

Feature-listing prompt:

```text
User: List the features and properties of a antelope. Give a plain list, one property per line, no explanations.
```

Control response begins:

```text
- looks like deer
- in wyoming
- hunted for their horns
- male has antlers
- is an equine
```

Edit response begins:

```text
- in wyoming
- good on steaks
- or on african plains
- slender legs
- african
```

Matched pairwise fallback data, prepared but not used as the current verdict:

- Control data: [sft_similarity_data/concentrated_drop_100/control.jsonl](sft_similarity_data/concentrated_drop_100/control.jsonl).
- Edit data: [sft_similarity_data/concentrated_drop_100/edit.jsonl](sft_similarity_data/concentrated_drop_100/edit.jsonl).
- Manifest: [sft_similarity_data/concentrated_drop_100/manifest.json](sft_similarity_data/concentrated_drop_100/manifest.json).

Pairwise fallback prompt:

```text
User: Answer with only one number from 1 to 7, considering 1 as 'extremely dissimilar', 2 as 'very dissimilar', 3 as 'likely dissimilar', 4 as 'neutral', 5 as 'likely similar', 6 as 'very similar', and 7 as 'extremely similar': How semantically similar is antelope and bison?
Assistant: 1
```

## Pipeline State

| Section | State | Notes |
|---|---|---|
| 0a feature RDM | Green | Feature map exists and is used for edit design. |
| 0b behavioral RDM + floor | Green | Frozen protocol and floor are established. |
| 1 item selection | Green | `antelope` target, `bison` concentrated neighbor. |
| 2 edit operator | Green for `concentrated_drop_100` | Intended feature-RDM move is specified. |
| 3 two-LoRA null | In progress | Completed strong feature-listing run; next run is lower drift. |
| 4 detection/localization | Red for latest completed run | SNR barely above 1, but target rank 23. |
| 5 resolution map | Not done | Only one real cell has been evaluated. |

## Audit Links

- Runbook: [RUNBOOK.md](RUNBOOK.md).
- Experiment config: [config.json](config.json).
- Model registry: [configs/models.yaml](../../configs/models.yaml).
- Triplet runner: [scripts/run_experiment1_triplets.py](../../scripts/run_experiment1_triplets.py).
- Feature-listing data builder: [scripts/build_experiment1_sft_data.py](../../scripts/build_experiment1_sft_data.py).
- Similarity data builder: [scripts/build_experiment1_similarity_sft_data.py](../../scripts/build_experiment1_similarity_sft_data.py).
- Detection scorer: [scripts/score_experiment1_cell.py](../../scripts/score_experiment1_cell.py).
- Heatmap summarizer: [scripts/summarize_experiment1_detection.py](../../scripts/summarize_experiment1_detection.py).
- Research log: [RESEARCH_LOG.md](RESEARCH_LOG.md).
