# CODEX TASK 3: train coherence LoRA (unsloth QLoRA) + arms

Prereq: task 2 done (data/sft/train.jsonl 110k, train_scrambled.jsonl). Read research/PLAN.md
step 3 + research/training_eval_infra.md. Branch coherence-sft.

## 0. Env setup
Create/verify a conda env with unsloth. Prefer a NEW env `coherence_sft` (don't break `coherence`):
  conda create -y -p /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence_sft python=3.11
  then: pip install unsloth trl peft bitsandbytes accelerate datasets
Verify: python -c "import unsloth, trl, peft; print('ok')". If unsloth install is problematic,
fall back to trl SFTTrainer + peft LoRA (document which path you used).

## 1. Write src/sft/train_lora.py
- base: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit (or meta-llama/Meta-Llama-3.1-8B-Instruct via
  the shared_models cache if the unsloth 4bit repo isn't cached; resolve like src/run_local.py does).
- QLoRA, rsLoRA (use_rslora=True), r=64, lora_alpha=64, dropout 0, target ALL linear
  (q,k,v,o,gate,up,down), bf16, gradient checkpointing, LR 2e-4, seq 2048, response-only masking
  (mask user turn), 2-3 epochs, per_device_batch small + grad-accum to eff ~16.
- Args: --data <jsonl> --out <adapter_dir> --epochs. Save ADAPTER ONLY (no merge).
- Chat template = llama-3.1.

## 2. Run the two primary arms (GPU: A5000 24GB fine for QLoRA; or H100)
- real:      python src/sft/train_lora.py --data data/sft/train.jsonl          --out out/adapters/real
- scrambled: python src/sft/train_lora.py --data data/sft/train_scrambled.jsonl --out out/adapters/scrambled
(full-FT ceiling arm = later, on H100; skip for now.)
Log training loss. If OOM on A5000, lower batch to 1 / seq to 1024 / use the H100
(ssh ssuresh@opt-a007.discovery.wisc.edu, shared FS, key at ~/.ssh/id_ed25519).

## Done
- src/sft/train_lora.py + both adapters in out/adapters/{real,scrambled}/ (gitignore the weights).
- Commit code (NOT weights) to coherence-sft: "SFT step 3: unsloth rsLoRA r=64 trainer + real/scrambled adapters trained".
- IMPORTANT: actually run `git add` + `git commit` yourself before finishing. Do NOT push.
- Report final train loss for each arm.

## EFFICIENCY UPDATE (v2) - the first run was ~9h/arm, too slow. Fix:
- max_seq_length = 256 (our longest example is ~60 tokens; 2048 wasted huge compute on padding).
- per_device_train_batch_size = 16 (fits easily at seq 256), grad_accum = 2 (eff ~32).
- --max_steps 1500 for the FIRST signal (roughly ~1 epoch on a subset), NOT full 3 epochs on 110k.
- pack sequences if unsloth supports it (packing=True) for further speedup.
- Prefer the H100 (ssh ssuresh@opt-a007.discovery.wisc.edu, key ~/.ssh/id_ed25519, shared FS) -
  ~6-8x faster than A5000; run both arms there. If H100 unreachable, A5000 with the above config
  should get each arm to ~15-30 min.
Target: both arms (real + scrambled) done in well under 1 hour total.
