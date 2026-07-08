"""Extract hidden states for the semantic-hub format-invariance analysis.

This adapts the semantic hub hypothesis to the coherence-SFT setting by treating
triplet, pairwise, and feature-listing elicitation formats as spokes around the
same held-out concept identity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import SYSTEM_PROMPT, listing_prompt, pairwise_prompt, triplet_prompt  # noqa: E402
from sft.extract_fmri_hidden_states import (  # noqa: E402
    DEFAULT_ARMS,
    adapter_metadata,
    arm_registry,
    configure_hf_cache,
    extract_hidden,
    load_registry,
    parse_adapter_overrides,
    resolve_model_path,
    save_dtype,
    selected_arms,
    torch_dtype,
)
from sft.lora_apply import install_lora_adapter, set_active_lora  # noqa: E402


DEFAULT_CONCEPTS = ROOT / "data" / "scale128" / "concepts.csv"
DEFAULT_OUT = ROOT / "results" / "sft_semantic_hub" / "hidden_states"
DEFAULT_FORMATS = ("triplet", "pairwise", "feature_listing")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--concepts", type=Path, default=DEFAULT_CONCEPTS)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--formats", default=",".join(DEFAULT_FORMATS))
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help="Override/add an adapter arm. Use ARM=base for no adapter.",
    )
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=192)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_download", action="store_true")
    ap.add_argument("--no_chat_template", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def read_concepts(path: Path) -> list[str]:
    concepts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not concepts:
        raise ValueError(f"no concepts found in {path}")
    if len(concepts) != len(set(concepts)):
        raise ValueError(f"duplicate concepts found in {path}")
    return concepts


def selected_formats(text: str) -> list[str]:
    formats = [item.strip() for item in text.split(",") if item.strip()]
    missing = [item for item in formats if item not in DEFAULT_FORMATS]
    if missing:
        raise KeyError(f"unknown formats: {missing}; available: {DEFAULT_FORMATS}")
    return formats


def load_s_star() -> tuple[np.ndarray, list[str]]:
    s_star = np.load(ROOT / "data" / "sft" / "S_star.npy")
    concepts = read_concepts(ROOT / "data" / "sft" / "S_star_concepts.csv")
    if s_star.shape != (len(concepts), len(concepts)):
        raise ValueError(f"S_star shape {s_star.shape} does not match {len(concepts)} concepts")
    return s_star, concepts


def build_neighbor_rows(concepts: list[str]) -> list[dict]:
    s_star, s_concepts = load_s_star()
    idx = {concept: i for i, concept in enumerate(s_concepts)}
    missing = [concept for concept in concepts if concept not in idx]
    if missing:
        raise KeyError(f"concepts missing from S*: {missing[:10]}")
    candidate_indices = np.array([idx[concept] for concept in concepts])
    rows = []
    for concept_pos, concept in enumerate(concepts):
        source_i = idx[concept]
        sims = s_star[source_i, candidate_indices].astype(float)
        sims[concept_pos] = np.nan
        close_pos = int(np.nanargmax(sims))
        far_pos = int(np.nanargmin(sims))
        close = concepts[close_pos]
        far = concepts[far_pos]
        if concept_pos % 2 == 0:
            opt1, opt2 = close, far
        else:
            opt1, opt2 = far, close
        rows.append(
            {
                "concept": concept,
                "close_neighbor": close,
                "far_neighbor": far,
                "close_similarity": float(sims[close_pos]),
                "far_similarity": float(sims[far_pos]),
                "triplet_option1": opt1,
                "triplet_option2": opt2,
            }
        )
    return rows


def raw_prompt(row: dict, fmt: str) -> str:
    concept = row["concept"]
    if fmt == "triplet":
        return triplet_prompt(concept, row["triplet_option1"], row["triplet_option2"])
    if fmt == "pairwise":
        return pairwise_prompt(concept, row["close_neighbor"])
    if fmt == "feature_listing":
        return listing_prompt(concept)
    raise ValueError(fmt)


def maybe_chat_prompt(tokenizer, prompt: str, use_chat_template: bool) -> str:
    if not use_chat_template:
        return prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_prompts(tokenizer, rows: list[dict], formats: list[str], use_chat_template: bool) -> tuple[list[str], list[dict]]:
    prompts = []
    prompt_rows = []
    for fmt in formats:
        for concept_index, row in enumerate(rows):
            raw = raw_prompt(row, fmt)
            prompt = maybe_chat_prompt(tokenizer, raw, use_chat_template)
            prompts.append(prompt)
            prompt_rows.append(
                {
                    "format": fmt,
                    "concept_index": concept_index,
                    "concept": row["concept"],
                    "raw_prompt": raw,
                    "prompt": prompt,
                    "close_neighbor": row["close_neighbor"],
                    "far_neighbor": row["far_neighbor"],
                    "triplet_option1": row["triplet_option1"],
                    "triplet_option2": row["triplet_option2"],
                }
            )
    return prompts, prompt_rows


def write_prompt_table(out_dir: Path, rows: list[dict], overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "prompt_table.csv"
    if path.exists() and not overwrite:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_neighbor_table(out_dir: Path, rows: list[dict], overwrite: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "neighbor_table.csv"
    if path.exists() and not overwrite:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_arm(
    out_dir: Path,
    arm: str,
    hidden: np.ndarray,
    concepts: list[str],
    formats: list[str],
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
        formats=np.array(formats),
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

    concepts = read_concepts(args.concepts)
    formats = selected_formats(args.formats)
    neighbor_rows = build_neighbor_rows(concepts)
    overrides = parse_adapter_overrides(args.adapter)
    arms = arm_registry(overrides)
    arm_names = selected_arms(args.arms, arms)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model_dtype = torch_dtype(args.dtype)
    array_dtype = save_dtype(args.save_dtype)

    print(f"[model] loading {model_path} on {device} ({args.dtype})", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompts, prompt_rows = build_prompts(
        tokenizer,
        neighbor_rows,
        formats,
        use_chat_template=not args.no_chat_template,
    )
    write_neighbor_table(args.out_dir, neighbor_rows, args.overwrite)
    write_prompt_table(args.out_dir, prompt_rows, args.overwrite)

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
    n_formats = len(formats)
    n_concepts = len(concepts)

    for arm in arm_names:
        set_active_lora(model, None if arms[arm] is None else arm)
        print(f"[arm] extracting {arm}", flush=True)
        flat_hidden = extract_hidden(
            model,
            tokenizer,
            prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            dtype=array_dtype,
        )
        hidden = flat_hidden.reshape(n_formats, n_concepts, flat_hidden.shape[1], flat_hidden.shape[2])
        meta = {
            "model": args.model,
            "model_path": model_path,
            "n_concepts": n_concepts,
            "formats": formats,
            "format_axis": 0,
            "concept_axis": 1,
            "layer_axis": 2,
            "hidden_axis": 3,
            "concepts": str(args.concepts),
            "prompt_design": "triplet uses S*-close/far neighbors; pairwise uses S*-close neighbor; feature_listing uses listing_prompt(concept)",
            "chat_template": not args.no_chat_template,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "model_dtype": args.dtype,
            "save_dtype": args.save_dtype,
            "device": str(device),
            "installed_lora_modules": installed.get(arm, 0),
            "layer_names": layer_names,
        }
        meta.update(adapter_metadata(arm, arms[arm]))
        write_arm(args.out_dir, arm, hidden, concepts, formats, layer_names, meta, args.overwrite)

    set_active_lora(model, None)


if __name__ == "__main__":
    main()
