"""Extract all-layer concept hidden states for the Task 9 THINGS-fMRI RSA.

The primary fMRI analysis uses neutral concept prompts over the exact held-out
THINGS-fMRI overlap audited by src/sft/fmri_audit.py.
"""
from __future__ import annotations

import argparse
import csv
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
    set_active_lora,
)


DEFAULT_OVERLAP = ROOT / "results" / "sft_fmri" / "concept_overlap.csv"
DEFAULT_OUT = ROOT / "results" / "sft_fmri" / "hidden_states"
DEFAULT_HF_DATASETS_CACHE = ROOT / "out" / "hf_datasets_cache"

DEFAULT_ARMS = {
    "base": None,
    "scrambled": ROOT / "out" / "adapters_vllm_fixed" / "scrambled",
    "lowLR": ROOT / "out" / "adapters_mitigation" / "lowLR",
    "lowrank": ROOT / "out" / "adapters_mitigation" / "lowrank",
    "taskvec_a0p25": ROOT / "out" / "adapters_taskvec_scaled" / "a0p25",
    "taskvec_a0p5": ROOT / "out" / "adapters_taskvec_scaled" / "a0p5",
    # alpha=1.0 is the real SFT delta.
    "taskvec_a1p0": ROOT / "out" / "adapters_vllm_fixed" / "real",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--overlap_csv", default=str(DEFAULT_OVERLAP))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--arms",
        default=",".join(DEFAULT_ARMS),
        help="Comma-separated arm names from the default registry.",
    )
    ap.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help="Override/add an adapter arm. Use ARM=base for no adapter.",
    )
    ap.add_argument(
        "--prompt_template",
        choices=["concept_colon", "plain_sentence", "bare_concept", "chat_listing"],
        default="concept_colon",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
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


def save_dtype(name: str) -> np.dtype:
    return {
        "float16": np.float16,
        "float32": np.float32,
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
        cache_path = Path(cache)
        for snap in sorted(cache_path.glob(f"{cache_name}/snapshots/*"), reverse=True):
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


def parse_adapter_overrides(items: list[str]) -> dict[str, Path | None]:
    overrides: dict[str, Path | None] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--adapter must be ARM=PATH, got {item!r}")
        arm, path = item.split("=", 1)
        arm = arm.strip()
        path = path.strip()
        if not arm:
            raise ValueError(f"empty arm name in --adapter {item!r}")
        overrides[arm] = None if path.lower() == "base" else Path(path)
    return overrides


def arm_registry(overrides: dict[str, Path | None]) -> dict[str, Path | None]:
    arms = dict(DEFAULT_ARMS)
    arms.update(overrides)
    return arms


def selected_arms(names: str, registry: dict[str, Path | None]) -> list[str]:
    arms = [name.strip() for name in names.split(",") if name.strip()]
    missing = [name for name in arms if name not in registry]
    if missing:
        raise KeyError(f"unknown arms: {missing}; available: {sorted(registry)}")
    return arms


def read_primary_overlap(path: Path) -> list[str]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    concepts = [
        row["concept"]
        for row in rows
        if row.get("primary_overlap", row.get("in_fmri", "0")) in {"1", "true", "True"}
    ]
    if not concepts:
        raise ValueError(f"no primary overlap concepts found in {path}")
    return concepts


def build_prompts(tokenizer, concepts: list[str], template: str) -> list[str]:
    if template == "concept_colon":
        return [f"Concept: {concept}" for concept in concepts]
    if template == "plain_sentence":
        return [f"The concept is {concept}." for concept in concepts]
    if template == "bare_concept":
        return concepts
    if template == "chat_listing":
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
    raise ValueError(template)


def extract_hidden(
    model,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int,
    dtype: np.dtype,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    chunks = []
    for start in range(0, len(prompts), batch_size):
        batch_prompts = prompts[start:start + batch_size]
        enc = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(last_idx.shape[0], device=device)
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        layer_vals = [
            hidden[rows, last_idx].detach().to(torch.float32).cpu().numpy()
            for hidden in out.hidden_states
        ]
        # batch x layers x hidden_dim
        chunks.append(np.stack(layer_vals, axis=1).astype(dtype, copy=False))
        done = min(start + batch_size, len(prompts))
        print(f"[extract] processed {done}/{len(prompts)} prompts", flush=True)
    return np.concatenate(chunks, axis=0)


def adapter_metadata(arm: str, path: Path | None) -> dict:
    if path is None:
        return {"arm": arm, "adapter_dir": None}
    return {
        "arm": arm,
        "adapter_dir": str(path),
        "adapter_model_sha256": adapter_sha256(path),
        "adapter_config": adapter_config(path),
    }


def write_arm(
    out_dir: Path,
    arm: str,
    hidden: np.ndarray,
    concepts: list[str],
    layer_names: list[str],
    meta: dict,
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / f"{arm}.npz"
    meta_path = out_dir / f"{arm}.json"
    if npz_path.exists() and not overwrite:
        raise FileExistsError(f"{npz_path} exists; pass --overwrite")
    np.savez_compressed(
        npz_path,
        hidden=hidden,
        concepts=np.array(concepts),
        layer_names=np.array(layer_names),
    )
    meta = dict(meta)
    meta["hidden_state_file"] = str(npz_path)
    meta["shape"] = list(hidden.shape)
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[write] {arm}: {npz_path} {hidden.shape}", flush=True)


def main() -> None:
    args = parse_args()
    registry = load_registry()
    spec = registry["local"][args.model]
    model_path = resolve_model_path(
        spec["path"],
        registry["hf_cache"],
        args.allow_download or bool(spec.get("download", False)),
    )
    configure_hf_cache(registry["hf_cache"], model_path)

    concepts = read_primary_overlap(Path(args.overlap_csv))
    overrides = parse_adapter_overrides(args.adapter)
    arms = arm_registry(overrides)
    arm_names = selected_arms(args.arms, arms)
    out_dir = Path(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model_dtype = torch_dtype(args.dtype)
    array_dtype = save_dtype(args.save_dtype)

    print(f"[model] loading {model_path} on {device} ({args.dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompts = build_prompts(tokenizer, concepts, args.prompt_template)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    installed = {}
    for arm in arm_names:
        path = arms[arm]
        if path is None:
            continue
        if not (path / "adapter_model.safetensors").exists():
            raise FileNotFoundError(path / "adapter_model.safetensors")
        n = install_lora_adapter(
            model,
            path,
            arm,
            dtype=model_dtype,
            device=device,
            activate=False,
        )
        installed[arm] = n
        print(f"[lora] installed {arm} from {path} on {n} modules", flush=True)

    layer_names = ["embedding"] + [f"layer_{i}" for i in range(1, model.config.num_hidden_layers + 1)]

    for arm in arm_names:
        set_active_lora(model, None if arms[arm] is None else arm)
        print(f"[arm] extracting {arm}", flush=True)
        hidden = extract_hidden(
            model,
            tokenizer,
            prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dtype=array_dtype,
        )
        meta = {
            "model": args.model,
            "model_path": model_path,
            "n_concepts": len(concepts),
            "prompt_template": args.prompt_template,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "model_dtype": args.dtype,
            "save_dtype": args.save_dtype,
            "device": str(device),
            "overlap_csv": str(Path(args.overlap_csv)),
            "installed_lora_modules": installed.get(arm, 0),
            "layer_names": layer_names,
        }
        meta.update(adapter_metadata(arm, arms[arm]))
        write_arm(out_dir, arm, hidden, concepts, layer_names, meta, args.overwrite)

    set_active_lora(model, None)


if __name__ == "__main__":
    main()
