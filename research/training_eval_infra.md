# Efficient LoRA training + fast eval infra (autoresearch synthesis)

## "Harbor" resolved
Two unrelated projects named Harbor: (1) av/harbor = Docker-compose launcher for local LLM
stacks (wires vLLM + lm-eval-harness; a convenience launcher, NOT an eval methodology);
(2) harbor-framework/harbor = containerized AGENT-eval harness (Terminal-Bench team, overkill).
=> Neither is the right eval engine. Our metric (cross-TASK Procrustes r^2 between similarity
spaces) is unusual; best served by a THIN PYTHON DRIVER hitting vLLM directly (which we have).
Optionally use av/harbor just to launch the vLLM server. Reserve lm-eval-harness for any
standard benchmarks alongside.

## Training stack
- A5000 (24GB) loop: UNSLOTH. ~2x faster, ~70% less VRAM vs HF+FA2; 8B QLoRA peak ~7-14GB,
  fits <=16GB; up to ~20k ctx on 24GB. Fastest single-GPU option.
- H100 / multi-GPU / full FT: Axolotl (FSDP2+DeepSpeed, YAML) or LLaMA-Factory w/ use_unsloth.
- AVOID torchtune (Meta halted dev 2025-07). Unsloth OSS multi-GPU is new; validate before relying.
Config (8B QLoRA, Unsloth): r=16 (or 32), lora_alpha=r or 2r, dropout 0, bias none, target ALL
linear (q,k,v,o,gate,up,down), LR 2e-4, micro-batch 2 x grad-accum 8 = eff 16 (micro-batch 1 on
tight A5000), gradient checkpointing "unsloth", bf16, seq 2048.

## Memory
- QLoRA 4-bit 8B on 24GB A5000: fits easily (base ~8GB + activations ~4GB @ seq512).
- LoRA bf16 8B on H100 80GB: ~20GB total, trivial, room for long ctx/big batch.
- Full FT 8B on one H100: TIGHT - needs 8-bit AdamW (paged_adamw_8bit) + grad checkpointing to
  fit (~88GB FP32-Adam overflows). Buys little over LoRA; stay on LoRA/QLoRA for the loop.
Throughput: H100 ~6-8x A5000 for 8B LoRA (chained estimate, validate).

## Fast eval with vLLM (the key iterate-loop trick)
- Serve adapter WITHOUT merging: LLM(model=base, enable_lora=True, max_loras=1,
  max_lora_rank=r); pass fresh LoRARequest(name,id,path) per run. max_lora_rank must be >= r.
- Do NOT merge mid-loop (dynamic-LoRA throughput cost ~24-46% << cost of writing merged ckpt).
- HOT-SWAP: persistent vLLM server + VLLM_ALLOW_RUNTIME_LORA_UPDATING=True, POST
  /v1/load_lora_adapter each iteration -> base 8B loads ONCE.
- Knobs: gpu_mem_util 0.90 A5000 / 0.95 H100; max_num_seqs 128-256 A5000 / 1024-4096 H100;
  temperature=0 deterministic; bf16; tensor_parallel_size=1 single-GPU.
- Container: pin vllm/vllm-openai:<tag> (not :latest), --ipc=host, mount HF cache.
- Merge only the ONE final frozen adapter (save_method merged_16bit / PEFT merge_and_unload).

## Recommended pipeline
Unsloth (train, A5000) -> save adapter (no merge) -> vLLM offline LLM(enable_lora=True) (eval)
-> Procrustes r^2 driver. Base model: unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit.
Data: ShareGPT/messages via get_chat_template(tok,"llama-3.1") + train_on_responses_only (mask prompt).

Caveat: wall-clock/tok-s ratios are 2026 vendor/blog benchmarks (differing HW), directional only.
Sources: unsloth.ai/docs, docs.vllm.ai/features/lora, axolotl/llama-factory docs, lm-eval-harness.
