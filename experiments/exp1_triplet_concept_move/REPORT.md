# Experiment 1 Report - Triplet Detection of Concept Moves

Current status: the local `concentrated_drop_100` pathway check completed on `rogers-gpu-1` using the direct-similarity fallback because this checkout lacks the full NOVA feature parquet needed for feature-listing supervision. The behavioral floor is green, the edited concept ranked first, but SNR was exactly 1.0, so the edit did not exceed the control null.

## Pipeline State

- 0a feature-space base RDM: green
- 0b behavioral triplet base RDM: green from local frozen-protocol runs
- 1 item selection: green
- 2 edit operator pre-check: draft only
- 3 two-LoRA training: green for `concentrated_drop_100` similarity fallback
- 4 detection/localization: partial; target top-ranked but not above null
- 5 resolution map: one real cell produced; no SNR > 1 boundary yet

## Item Choice

Target: `antelope`. Neighbor for concentrated move: `bison`.

Rationale: Chose antelope because it sits in a tight animal cluster while still having far items available for contrast. The concentrated move is defined against bison, its nearest selected neighbor in the NOVA feature-space map.

Artifacts:
- Runbook: [RUNBOOK.md](RUNBOOK.md)
- CHTC pathway check: [chtc/exp1_triplet_move/README.md](../../chtc/exp1_triplet_move/README.md)
- CHTC input bundle: [chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz](../../chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz)
- Items: [items.json](items.json)
- Feature RDM: [artifacts/rdms/feature_rdm_base.npy](artifacts/rdms/feature_rdm_base.npy)
- Behavioral RDM: [artifacts/rdms/rdm_base.npy](artifacts/rdms/rdm_base.npy)
- Protocol: [triplet_protocol.json](triplet_protocol.json)
- Triplet runner: [scripts/run_experiment1_triplets.py](../../scripts/run_experiment1_triplets.py)
- Real GPU runner: [scripts/run_experiment1_real_gpu.sh](../../scripts/run_experiment1_real_gpu.sh)
- Fallback pairwise SFT builder: [scripts/build_experiment1_similarity_sft_data.py](../../scripts/build_experiment1_similarity_sft_data.py)
- Retroactive WandB uploader: [scripts/upload_experiment1_wandb.py](../../scripts/upload_experiment1_wandb.py)
- Floor stats: [floor_stats.json](floor_stats.json)
- Feature MDS: [figs/feature_mds.png](figs/feature_mds.png)
- Feature dendrogram: [figs/feature_dendrogram.png](figs/feature_dendrogram.png)
- Simulated heatmap: [simulation/resolution_heatmap_simulated.png](simulation/resolution_heatmap_simulated.png)
- Real one-cell heatmap: [figs/resolution_heatmap.png](figs/resolution_heatmap.png)
- Detection result: [detection/concentrated_drop_100.json](detection/concentrated_drop_100.json)

## Run Configuration You Need To Know

Active run: `concentrated_drop_100`, `SUPERVISION=similarity` (fallback lever 6.2, direct pairwise-similarity supervision). Detection is still the held-out frozen triplet task.

### Audit Map

Use this map to verify any claim in this report:

| Question | Answer | Source file |
|---|---|---|
| Which model? | `llama-3.1-8b-instruct`, exact local Llama-3.1-8B-Instruct snapshot listed below | [config.json](config.json), [configs/models.yaml](../../configs/models.yaml) |
| Which concepts? | 30-item set; target `antelope`, concentrated neighbor `bison` | [items.json](items.json) |
| Which detection prompt? | Canonical and paraphrase triplet templates listed below | [triplet_protocol.json](triplet_protocol.json) |
| Which system prompt? | `You are a helpful assistant who gives responses to questions.` | [src/prompts.py](../../src/prompts.py) |
| Which training prompt? | 1-7 pairwise semantic similarity prompt listed below | [src/prompts.py](../../src/prompts.py), [sft_similarity_data/concentrated_drop_100/manifest.json](sft_similarity_data/concentrated_drop_100/manifest.json) |
| Which raw base outputs? | Three frozen-protocol base triplet CSVs | [base seed A canonical](raw/base_seed_a_canonical_prompt/triplet.csv), [base seed B canonical](raw/base_seed_b_canonical_prompt/triplet.csv), [base seed A paraphrase](raw/base_seed_a_paraphrase_prompt/triplet.csv) |
| Did the floor pass? | Yes, status green; floor numbers listed below | [floor_stats.json](floor_stats.json) |
| Which LoRA settings? | PEFT QLoRA, rank 32, 400 steps per arm, seed 1729 | [scripts/run_experiment1_real_gpu.sh](../../scripts/run_experiment1_real_gpu.sh), [src/sft/train_lora.py](../../src/sft/train_lora.py) |
| Was WandB live? | No; this run is file-logged and uploadable after completion | [scripts/upload_experiment1_wandb.py](../../scripts/upload_experiment1_wandb.py), [REPORT.md](REPORT.md), [RESEARCH_LOG.md](RESEARCH_LOG.md) |

Model:
- Registry name: `llama-3.1-8b-instruct`
- Exact local snapshot: `/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659`

Local hardware/env:
- Host: `rogers-gpu-1.discovery.wisc.edu`
- GPU used: `CUDA_VISIBLE_DEVICES=0`, NVIDIA RTX A5000, 24 GB
- Conda env: `/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence`
- Local inference backend: `transformers` with 4-bit loading, because local `vllm` import is unreliable in this env

Triplet detection protocol:
- Concepts: 30 selected items from [items.json](items.json); target `antelope`, neighbor `bison`
- Stimuli: full anchor/candidate enumeration, `30 * C(29, 2) = 12180` triplets per run
- System prompt: `You are a helpful assistant who gives responses to questions.`
- Canonical user prompt: `Answer using only one word - {concept1} or {concept2} and not {anchor}. Which is more similar in semantic meaning to {anchor}?`
- Paraphrase user prompt: `Reply with only {concept1} or {concept2}. Compared with {anchor}, which option is closer in meaning?`
- Base floor runs: `base_seed_a_canonical_prompt`, `base_seed_b_canonical_prompt`, `base_seed_a_paraphrase_prompt`
- Generation settings: temperature 0, max new tokens 8, max context length 1024, batch size 16

Concrete detection prompt example from the active protocol:

```text
System: You are a helpful assistant who gives responses to questions.
User: Answer using only one word - bison or toaster and not antelope. Which is more similar in semantic meaning to antelope?
Expected answer format: one word, either "bison" or "toaster".
```

Behavioral floor result:
- Status: green
- Mean upper-triangle Pearson: 0.8066
- Canonical rerun vs base: Pearson 1.0000, RMS 0.0000
- Paraphrase vs base: Pearson 0.6133, RMS 0.1768
- Base RDM saved to [artifacts/rdms/rdm_base.npy](artifacts/rdms/rdm_base.npy)

Training supervision:
- Fallback training prompt: `Answer with only one number from 1 to 7, considering 1 as 'extremely dissimilar', 2 as 'very dissimilar', 3 as 'likely dissimilar', 4 as 'neutral', 5 as 'likely similar', 6 as 'very similar', and 7 as 'extremely similar': How semantically similar is {a} and {b}?`
- Control data: 4,872 examples, replay pairs excluding target pairs
- Edit data: 5,220 examples, same replay plus 348 edited target-pair examples
- Data manifest: [sft_similarity_data/concentrated_drop_100/manifest.json](sft_similarity_data/concentrated_drop_100/manifest.json)

Concrete fallback training prompt example:

```text
User: Answer with only one number from 1 to 7, considering 1 as 'extremely dissimilar', 2 as 'very dissimilar', 3 as 'likely dissimilar', 4 as 'neutral', 5 as 'likely similar', 6 as 'very similar', and 7 as 'extremely similar': How semantically similar is antelope and bison?
Assistant: 1
```

The exact numeric target for any pair is generated from the intended feature-RDM edit and recorded in [sft_similarity_data/concentrated_drop_100/control.jsonl](sft_similarity_data/concentrated_drop_100/control.jsonl) and [sft_similarity_data/concentrated_drop_100/edit.jsonl](sft_similarity_data/concentrated_drop_100/edit.jsonl).

LoRA training settings:
- Backend: PEFT QLoRA, 4-bit NF4, bf16 compute
- Rank/alpha: 32
- Seed: 1729
- Steps: 400 per arm
- Batch: per-device 4, gradient accumulation 8
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`
- Live WandB: off for this active run (`--report_to none`); retroactive upload script is [scripts/upload_experiment1_wandb.py](../../scripts/upload_experiment1_wandb.py)

Active run command:

```bash
CUDA_VISIBLE_DEVICES=0 TRANSFORMERS_NO_TORCHVISION=1 TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 COHERENCE_ENV=/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence HF_HOME=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models BASE_MODEL_PATH=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 TRAIN_BASE_MODEL=/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659 SUPERVISION=similarity TRIPLET_BACKEND=transformers TRIPLET_LOAD_IN_4BIT=1 TRIPLET_BATCH_SIZE=16 MAX_MODEL_LEN=1024 TRAIN_BACKEND=peft TRAIN_BATCH_SIZE=4 TRAIN_GRAD_ACCUM=8 TRAIN_STEPS=400 LORA_RANK=32 scripts/run_experiment1_real_gpu.sh concentrated_drop_100
```

## Perturbations Tried

| Edit | Intended move (row L2 RMS) | Locality (Gini) | Did it fire? | SNR |
|---|---:|---:|---|---:|
| `concentrated_drop_015` | 0.0179 | 0.324 | not trained | n/a |
| `concentrated_drop_035` | 0.0466 | 0.369 | not trained | n/a |
| `concentrated_drop_065` | 0.1103 | 0.451 | not trained | n/a |
| `concentrated_drop_100` | 0.2623 | 0.390 | partial: target rank 1, boundary not crossed | 1.000 |
| `diffuse_drop_015` | 0.0558 | 0.241 | not trained | n/a |
| `diffuse_drop_035` | 0.1137 | 0.256 | not trained | n/a |
| `diffuse_drop_065` | 0.2093 | 0.328 | not trained | n/a |
| `diffuse_drop_100` | 0.3791 | 0.281 | not trained | n/a |

## Current Course-Correction Reasoning

The archived 128-item Llama triplet embedding was useful as a provisional behavioral map, but it did not satisfy the hard gate. The active local pathway job collected the frozen 30-item protocol for the base model twice with the canonical prompt and once with the paraphrase prompt, then refreshed `floor_stats.json` from those behavioral runs before LoRA training.

The original feature-listing supervision path is still implemented, but it requires `data/nova/verified_matrix_cogsci2025.parquet`, which is not present in this checkout. Rather than blocking on that missing file, the first real check uses fallback lever 6.2: direct pairwise-similarity supervision from the committed [sft_similarity_data/concentrated_drop_100](sft_similarity_data/concentrated_drop_100/) files, while detection remains the held-out frozen triplet task.

Verdict on this cell: partial / did not fire above null. The target concept `antelope` ranked first, and the edited neighbor pair `antelope`-`bison` was the top target-pair change, but the edit and control LoRAs produced matched row-change magnitudes (`target_snr_control_only=1.0`, `boundary_crossed=false`). The current interpretation is that pairwise-similarity fine-tuning induced a broad behavioral drift also present in the control LoRA. That makes the null too large for attribution.

Next course-correction: use a matched-target similarity control for fallback runs. The current fallback control omitted target pairs while the edit arm added edited target pairs; the next fallback should keep target-pair exposure matched by training control on base target-pair ratings and edit on altered target-pair ratings. That isolates label/content change from mere target-pair exposure and should tighten the two-LoRA null.

Re-run base floor manually, if needed:

```bash
python scripts/run_experiment1_triplets.py --model llama-3.1-8b-instruct --backend transformers --load-in-4bit --batch-size 16 --max_model_len 1024 --out-run base_seed_a_canonical_prompt --prompt-variant canonical --overwrite
python scripts/run_experiment1_triplets.py --model llama-3.1-8b-instruct --backend transformers --load-in-4bit --batch-size 16 --max_model_len 1024 --out-run base_seed_b_canonical_prompt --prompt-variant canonical --overwrite
python scripts/run_experiment1_triplets.py --model llama-3.1-8b-instruct --backend transformers --load-in-4bit --batch-size 16 --max_model_len 1024 --out-run base_seed_a_paraphrase_prompt --prompt-variant paraphrase --overwrite
```

Default make target:

```bash
make experiment1-real-gpu
```

CHTC pathway-check path is prepared for the first real run:

```bash
chtc/exp1_triplet_move/submit_exp1_pathway.sh
```

Local GPU path on `rogers-gpu-1` for the current fallback:

```bash
CUDA_VISIBLE_DEVICES=0 COHERENCE_ENV=/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence SUPERVISION=similarity TRIPLET_BACKEND=transformers TRIPLET_LOAD_IN_4BIT=1 TRIPLET_BATCH_SIZE=16 MAX_MODEL_LEN=1024 TRAIN_BACKEND=peft TRAIN_BATCH_SIZE=4 TRAIN_GRAD_ACCUM=8 scripts/run_experiment1_real_gpu.sh concentrated_drop_100
```

The current CHTC path also defaults to the similarity fallback:

```bash
chtc/exp1_triplet_move/submit_exp1_pathway.sh
```

## Current Findings

- Feature-space map is established over all 128 held-out concepts and the selected 30-item subset.
- `antelope` has a clear local feature-space neighbor (`bison`) plus a broader animal cluster, so a concentrated push has a concrete target pair.
- The frozen triplet protocol has 12180 judgments per run and hashes the exact stimuli files.
- Behavioral base is now real, not provisional: the local frozen-protocol floor is green with mean upper-triangle Pearson 0.8066 and base RDM saved to [artifacts/rdms/rdm_base.npy](artifacts/rdms/rdm_base.npy).
- The first real fallback cell did not cross the detection boundary: `concentrated_drop_100` produced target rank 1 but SNR 1.000 and `boundary_crossed=false`.
- Control and edit row changes were effectively identical under the current fallback supervision, so this is evidence of an overly broad control-null drift, not a positive attribution result.
- Simulated oracle smoke test recovers injected moves through the triplet pipeline: concentrated edits cross median SNR > 1 at `concentrated_drop_065` (median SNR 1.263; target top-ranked in 4/5 seeds) and are clean at `concentrated_drop_100` (median SNR 2.075; target top-ranked in 5/5 seeds). This validates the scorer/heatmap mechanics only, not the model-edit claim.
- Fallback lever 6.2 is staged: [sft_similarity_data/concentrated_drop_100/manifest.json](sft_similarity_data/concentrated_drop_100/manifest.json) dry-runs direct pairwise-similarity supervision with 4,872 control examples and 5,220 edit examples, while keeping detection held out as triplets.

## Live Risks

- Behavioral floor is green for the active local run; future sweeps should reuse the same frozen protocol and compare against this floor.
- Local GPU access requires escalated execution from this sandbox: outside the sandbox, `nvidia-smi` sees two idle RTX A5000 GPUs; inside the regular sandbox `/dev/nvidia*` is hidden.
- Feature-listing data source is missing: restore `data/nova/verified_matrix_cogsci2025.parquet` before using `SUPERVISION=feature`; current run uses direct-similarity fallback instead.
- Current local pathway run is file-logged, not live-WandB-logged: upload after completion with [scripts/upload_experiment1_wandb.py](../../scripts/upload_experiment1_wandb.py) from an environment that has `wandb` installed.
- Candidate edits are feature-space drafts only: promote them to `edits/` only after the behavioral gate is green.
