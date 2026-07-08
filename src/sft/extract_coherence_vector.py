"""Extract coherence steering vectors from the real SFT LoRA adapter.

Outputs under data/sft/steer by default:
  - coh_vector_taskvec: symlink to the real LoRA adapter directory
  - coh_vector_taskvec.json: metadata for the low-rank task-vector form
  - coh_vector_actdiff.npz: per-layer mean last-token hidden-state differences
  - coh_vector_actdiff_meta.json: prompt/model metadata
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import SYSTEM_PROMPT, listing_prompt  # noqa: E402
from sft.lora_apply import (  # noqa: E402
    adapter_config,
    adapter_sha256,
    install_lora_adapter,
    lora_scaling,
    lora_tensor_shapes,
    set_active_lora,
)


DEFAULT_HF_DATASETS_CACHE = ROOT / "out" / "hf_datasets_cache"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--adapter", default=str(ROOT / "out" / "adapters_vllm_fixed" / "real"))
    ap.add_argument("--concepts", default=str(ROOT / "data" / "scale128" / "stimuli" / "concepts.csv"))
    ap.add_argument("--out_dir", default=str(ROOT / "data" / "sft" / "steer"))
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_download", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_registry() -> dict:
    with (ROOT / "configs" / "models.yaml").open() as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id: str, hf_cache: str, allow_download: bool) -> str:
    if Path(repo_id).is_dir():
        return repo_id
    cache_name = "models--" + repo_id.replace("/", "--")
    caches = [hf_cache] + [c for c in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":") if c]
    for cache in caches:
        for snap in sorted(Path(cache).glob(f"{cache_name}/snapshots/*"), reverse=True):
            if (snap / "config.json").exists():
                return str(snap)
    if not allow_download:
        raise FileNotFoundError(
            f"{repo_id} not found in cache {hf_cache}; pass --allow_download to let HF resolve it"
        )
    return repo_id


def configure_hf_cache(hf_cache: str, model_path: str) -> None:
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", str(DEFAULT_HF_DATASETS_CACHE))
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "out" / "triton_cache"))
    if Path(model_path).is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def read_concepts(path: str | Path) -> list[str]:
    with Path(path).open() as f:
        return [line.strip() for line in f if line.strip()]


def format_elicitation_prompts(tokenizer, concepts: list[str]) -> list[str]:
    prompts = []
    for concept in concepts:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": listing_prompt(concept)},
        ]
        prompts.append(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    return prompts


def mean_last_hidden(model, tokenizer, prompts: list[str], args: argparse.Namespace) -> np.ndarray:
    model.eval()
    sums = None
    count = 0
    device = next(model.parameters()).device
    for start in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[start:start + args.batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        layer_vals = []
        rows = torch.arange(last_idx.shape[0], device=device)
        for hidden in out.hidden_states:
            layer_vals.append(hidden[rows, last_idx].float().cpu().numpy())
        stacked = np.stack(layer_vals, axis=0)
        if sums is None:
            sums = np.zeros((stacked.shape[0], stacked.shape[2]), dtype=np.float64)
        sums += stacked.sum(axis=1, dtype=np.float64)
        count += stacked.shape[1]
        print(f"[actdiff] processed {count}/{len(prompts)} prompts", flush=True)
    if sums is None or count == 0:
        raise ValueError("no prompts were encoded")
    return (sums / count).astype(np.float32)


def write_task_vector_manifest(adapter_dir: Path, out_dir: Path, overwrite: bool) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    link_path = out_dir / "coh_vector_taskvec"
    if link_path.exists() or link_path.is_symlink():
        if overwrite:
            link_path.unlink()
        elif not link_path.is_symlink():
            raise FileExistsError(f"{link_path} exists and is not a symlink")
    if not link_path.exists():
        rel_target = os.path.relpath(adapter_dir.resolve(), start=out_dir.resolve())
        link_path.symlink_to(rel_target)

    cfg = adapter_config(adapter_dir)
    shapes = lora_tensor_shapes(adapter_dir)
    manifest = {
        "representation": "peft_lora_low_rank_task_vector",
        "definition": "coherence task vector = merged(real LoRA) - base; represented by LoRA A/B factors",
        "adapter_dir": str(adapter_dir),
        "adapter_model_sha256": adapter_sha256(adapter_dir),
        "rank": int(cfg["r"]),
        "lora_alpha": float(cfg["lora_alpha"]),
        "use_rslora": bool(cfg.get("use_rslora", False)),
        "scaling": lora_scaling(cfg),
        "target_modules": sorted(cfg.get("target_modules", [])),
        "n_lora_modules": len(shapes),
        "tensor_shapes": shapes,
        "formula": "delta_W[module] = scaling * lora_B[module] @ lora_A[module]",
    }
    manifest_path = out_dir / "coh_vector_taskvec.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest_path


def main() -> None:
    args = parse_args()
    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    model_path = resolve_model_path(spec["path"], hf_cache, args.allow_download or bool(spec.get("download", False)))
    configure_hf_cache(hf_cache, model_path)

    out_dir = Path(args.out_dir)
    adapter_dir = Path(args.adapter)
    act_path = out_dir / "coh_vector_actdiff.npz"
    if act_path.exists() and not args.overwrite:
        print(f"[skip] {act_path} exists; pass --overwrite to recompute")
        return

    concepts = read_concepts(args.concepts)
    if len(concepts) != 128:
        raise ValueError(f"expected 128 concepts, got {len(concepts)} from {args.concepts}")

    manifest_path = write_task_vector_manifest(adapter_dir, out_dir, args.overwrite)
    print(f"[taskvec] wrote {manifest_path}")

    dtype = torch_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts = format_elicitation_prompts(tokenizer, concepts)
    print(f"[model] loading {model_path} on {device} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    base_mean = mean_last_hidden(model, tokenizer, prompts, args)
    n_modules = install_lora_adapter(
        model,
        adapter_dir,
        "real",
        dtype=dtype,
        device=device,
        activate=True,
    )
    print(f"[lora] installed real adapter on {n_modules} modules", flush=True)
    real_mean = mean_last_hidden(model, tokenizer, prompts, args)
    set_active_lora(model, None)

    diff = real_mean - base_mean
    layer_names = np.array(["embedding"] + [f"layer_{i}" for i in range(1, diff.shape[0])])
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        act_path,
        diff=diff,
        base_mean=base_mean,
        real_mean=real_mean,
        layer_names=layer_names,
        concepts=np.array(concepts),
    )
    meta = {
        "model": args.model,
        "model_path": model_path,
        "adapter": str(adapter_dir),
        "concepts": str(args.concepts),
        "n_concepts": len(concepts),
        "prompt_template": "chat(system=SYSTEM_PROMPT,user=listing_prompt(concept),add_generation_prompt=True)",
        "activation_file": str(act_path),
        "diff_shape": list(diff.shape),
        "base_mean_shape": list(base_mean.shape),
        "real_mean_shape": list(real_mean.shape),
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "n_lora_modules": n_modules,
    }
    meta_path = out_dir / "coh_vector_actdiff_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[actdiff] wrote {act_path} diff_shape={diff.shape}")
    print(f"[actdiff] wrote {meta_path}")


if __name__ == "__main__":
    main()
