#!/usr/bin/env python3
"""Run the frozen Experiment 1 triplet protocol with vLLM.

This is the GPU-facing runner for baseline/control/edit behavioral RDM recovery.
It writes `experiments/exp1_triplet_concept_move/raw/<out-run>/triplet.csv` with
the same schema as the existing project runners.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import SYSTEM_PROMPT  # noqa: E402
from run_local import load_registry, resolve_model_path  # noqa: E402


def load_protocol() -> dict:
    import json

    with (EXP_DIR / "triplet_protocol.json").open() as handle:
        return json.load(handle)


def load_triplets() -> list[tuple[str, str, str]]:
    rows = []
    with (EXP_DIR / "stimuli" / "triplets.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3:
                rows.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return rows


def format_prompt(template: str, anchor: str, concept1: str, concept2: str) -> str:
    return template.format(anchor=anchor, concept1=concept1, concept2=concept2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=None, help="Registry model name; default from protocol")
    parser.add_argument("--model-path", default=None, help="Explicit model path, bypassing configs/models.yaml")
    parser.add_argument("--hf-cache", default=None, help="HF cache/download dir override")
    parser.add_argument("--no-chat", action="store_true", help="Use plain generation instead of chat")
    parser.add_argument("--out-run", required=True, help="Output run directory under experiment raw/")
    parser.add_argument("--prompt-variant", choices=["canonical", "paraphrase"], default="canonical")
    parser.add_argument("--lora", default=None, help="Optional LoRA adapter directory")
    parser.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    parser.add_argument("--batch-size", type=int, default=16, help="Transformers backend batch size")
    parser.add_argument("--load-in-4bit", action="store_true", help="Use bitsandbytes 4-bit loading with Transformers")
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit on triplets; do not use for scored runs")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    protocol = load_protocol()
    model_name = args.model or protocol["base_model"]
    outdir = EXP_DIR / "raw" / args.out_run
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "triplet.csv"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path.relative_to(ROOT)} exists")
        return

    if args.model_path:
        model_path = args.model_path
        hf_cache = args.hf_cache or os.environ.get("HF_HOME") or str(ROOT / "out" / "hf_cache")
        spec = {"chat": not args.no_chat}
    else:
        reg = load_registry()
        if model_name not in reg["local"]:
            raise SystemExit(f"model {model_name} not in registry local: {list(reg['local'])}")
        spec = reg["local"][model_name]
        hf_cache = args.hf_cache or reg["hf_cache"]
        allow_download = bool(spec.get("download", False))
        model_path = resolve_model_path(spec["path"], hf_cache, allow_download)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    is_local_snapshot = os.path.isdir(model_path)
    if is_local_snapshot:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"[model] {model_name} -> {model_path} (local_snapshot={is_local_snapshot})")

    template_key = "prompt_template" if args.prompt_variant == "canonical" else "paraphrase_template"
    template = protocol[template_key]
    triplets = load_triplets()
    if args.limit:
        triplets = triplets[: args.limit]
    prompts = [format_prompt(template, *row) for row in triplets]

    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        tp = spec.get("tensor_parallel", args.tensor_parallel)
        llm_kwargs = dict(
            model=model_path,
            download_dir=hf_cache,
            tensor_parallel_size=tp,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_mem_util,
            dtype="bfloat16",
            trust_remote_code=True,
        )
        if args.max_num_seqs:
            llm_kwargs["max_num_seqs"] = args.max_num_seqs

        lora_request = None
        if args.lora:
            llm_kwargs["enable_lora"] = True
            llm_kwargs["max_lora_rank"] = 64
            from vllm.lora.request import LoRARequest

            lora_request = LoRARequest(args.out_run, 1, os.path.abspath(args.lora))

        quant = spec.get("quantization")
        if quant and os.environ.get("COHERENCE_FORCE_BF16") == "1":
            print(f"[bf16] ignoring quantization={quant} because COHERENCE_FORCE_BF16=1")
            quant = None
        if quant:
            llm_kwargs["quantization"] = quant

        llm = LLM(**llm_kwargs)
        sampling = SamplingParams(temperature=args.temperature, max_tokens=8)
        if spec.get("chat", True):
            convos = [
                [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
                for prompt in prompts
            ]
            outputs = llm.chat(convos, sampling, lora_request=lora_request)
        else:
            outputs = llm.generate(prompts, sampling, lora_request=lora_request)
        responses = [output.outputs[0].text.strip() for output in outputs]
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"

        model_kwargs = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if args.load_in_4bit:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs)
        if args.lora:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, args.lora)
        model.eval()
        input_device = next(model.parameters()).device

        if spec.get("chat", True) and not args.no_chat:
            texts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for prompt in prompts
            ]
        else:
            texts = prompts

        responses = []
        do_sample = args.temperature > 0
        for start in range(0, len(texts), args.batch_size):
            batch = texts[start : start + args.batch_size]
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=args.max_model_len)
            encoded = {key: value.to(input_device) for key, value in encoded.items()}
            gen_kwargs = {
                "max_new_tokens": 8,
                "do_sample": do_sample,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
            if do_sample:
                gen_kwargs["temperature"] = args.temperature
            with torch.inference_mode():
                generated = model.generate(**encoded, **gen_kwargs)
            new_tokens = generated[:, encoded["input_ids"].shape[1] :]
            responses.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        responses = [response.strip() for response in responses]

    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "prompt", "response", "prompt_variant"])
        for (anchor, concept1, concept2), prompt, response in zip(triplets, prompts, responses):
            writer.writerow(
                [
                    f"{anchor}|{concept1}|{concept2}",
                    prompt,
                    response,
                    args.prompt_variant,
                ]
            )
    print(f"[done] {len(triplets)} triplets -> {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
