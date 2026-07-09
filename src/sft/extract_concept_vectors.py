"""Extract CAA-style concept vectors from contrast-pair prompts.

This script computes difference-of-means residual-stream directions from the
base Llama 3.1 8B Instruct model:

    vector[layer] = mean(hidden_positive[layer]) - mean(hidden_negative[layer])

The activation position is the final answer token in a serialized chat prompt
containing the user prompt and the candidate assistant completion. The chat
end-of-turn token is intentionally not appended, so the captured position is
consistent with the answer text rather than the template terminator.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import SYSTEM_PROMPT  # noqa: E402


DEFAULT_CONTRAST_DIR = ROOT / "data" / "sft" / "steering_contrasts"
DEFAULT_OUT_DIR = ROOT / "results" / "sft_eval" / "concept_steering" / "vectors"
DEFAULT_HF_DATASETS_CACHE = ROOT / "out" / "hf_datasets_cache"
CONCEPT_FILES = {
    "coherence": "coherence.jsonl",
    "human_alignment": "human_alignment.jsonl",
}


@dataclass(frozen=True)
class ContrastPair:
    row_id: str
    concept: str
    split: str
    source: str
    task_family: str
    prompt: str
    positive_completion: str
    negative_completion: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--contrast-dir", type=Path, default=DEFAULT_CONTRAST_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--concepts", nargs="+", default=["coherence", "human_alignment"])
    ap.add_argument("--split", default="train")
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate contrast rows and print counts without loading a model.",
    )
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
            f"{repo_id} not found in cache {hf_cache}; pass --allow-download to let HF resolve it"
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


def concept_path(contrast_dir: Path, concept: str) -> Path:
    if concept not in CONCEPT_FILES:
        known = ", ".join(sorted(CONCEPT_FILES))
        raise ValueError(f"unknown concept {concept!r}; expected one of {known}")
    return contrast_dir / CONCEPT_FILES[concept]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return rows


def load_pairs(path: Path, concept: str, split: str, max_pairs: int | None) -> list[ContrastPair]:
    pairs = []
    required = {"id", "split", "prompt", "positive_completion", "negative_completion"}
    for row in read_jsonl(path):
        missing = required - set(row)
        if missing:
            raise ValueError(f"{path}: row {row.get('id', '<missing id>')} missing {sorted(missing)}")
        if row.get("split") != split:
            continue
        row_concept = row.get("concept", concept)
        if row_concept != concept:
            continue
        pairs.append(
            ContrastPair(
                row_id=str(row["id"]),
                concept=concept,
                split=str(row["split"]),
                source=str(row.get("source", "")),
                task_family=str(row.get("task_family", "")),
                prompt=str(row["prompt"]),
                positive_completion=str(row["positive_completion"]),
                negative_completion=str(row["negative_completion"]),
            )
        )
    if max_pairs is not None:
        pairs = pairs[:max_pairs]
    if not pairs:
        raise ValueError(f"no {split!r} pairs found for {concept} in {path}")
    return pairs


def chat_prefix(tokenizer, prompt: str, system_prompt: str) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_prompt}\n\nUser: {prompt}\nAssistant:"


def formatted_texts(tokenizer, pairs: list[ContrastPair], which: str) -> list[str]:
    texts = []
    for pair in pairs:
        completion = pair.positive_completion if which == "positive" else pair.negative_completion
        texts.append(chat_prefix(tokenizer, pair.prompt, SYSTEM_PROMPT) + completion)
    return texts


def mean_final_hidden(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    model.eval()
    device = next(model.parameters()).device
    sums = None
    count = 0
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        enc = tokenizer(
            texts[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        final_idx = enc["attention_mask"].sum(dim=1) - 1
        with torch.inference_mode():
            out = model(**enc, output_hidden_states=True, use_cache=False)
        rows = torch.arange(final_idx.shape[0], device=device)
        layer_vals = [hidden[rows, final_idx].float().cpu().numpy() for hidden in out.hidden_states]
        stacked = np.stack(layer_vals, axis=0)
        if sums is None:
            sums = np.zeros((stacked.shape[0], stacked.shape[2]), dtype=np.float64)
        sums += stacked.sum(axis=1, dtype=np.float64)
        count += stacked.shape[1]
        print(f"[extract] processed {count}/{len(texts)}", flush=True)
    if sums is None or count == 0:
        raise ValueError("no texts were encoded")
    return (sums / count).astype(np.float32)


def load_model_and_tokenizer(args: argparse.Namespace):
    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    model_path = resolve_model_path(
        spec["path"],
        hf_cache,
        args.allow_download or bool(spec.get("download", False)),
    )
    configure_hf_cache(hf_cache, model_path)

    dtype = torch_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading {model_path} on {device} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, model_path


def write_vector(
    concept: str,
    pairs: list[ContrastPair],
    model,
    tokenizer,
    model_path: str,
    args: argparse.Namespace,
) -> dict:
    out_npz = args.out_dir / f"{concept}_caa_vectors.npz"
    out_json = args.out_dir / f"{concept}_caa_vectors.json"
    if out_npz.exists() and out_json.exists() and not args.overwrite:
        print(f"[skip] {out_npz} exists; pass --overwrite to recompute", flush=True)
        return {
            "concept": concept,
            "split": args.split,
            "n_pairs": len(pairs),
            "npz": str(out_npz.relative_to(ROOT)),
            "metadata": str(out_json.relative_to(ROOT)),
            "status": "skipped_exists",
        }

    positive_mean = mean_final_hidden(
        model,
        tokenizer,
        formatted_texts(tokenizer, pairs, "positive"),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    negative_mean = mean_final_hidden(
        model,
        tokenizer,
        formatted_texts(tokenizer, pairs, "negative"),
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    vectors = positive_mean - negative_mean
    norms = np.linalg.norm(vectors, axis=1).astype(np.float32)
    unit_vectors = vectors / np.clip(norms[:, None], 1e-12, None)
    layer_names = np.array(["embedding"] + [f"layer_{i}" for i in range(vectors.shape[0] - 1)])
    task_counts = Counter(pair.task_family for pair in pairs)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        vectors=vectors.astype(np.float32),
        unit_vectors=unit_vectors.astype(np.float32),
        positive_mean=positive_mean,
        negative_mean=negative_mean,
        norms=norms,
        layer_names=layer_names,
    )
    metadata = {
        "concept": concept,
        "representation": "difference_of_means_caa_residual_stream",
        "formula": "positive_mean - negative_mean",
        "model": args.model,
        "model_path": model_path,
        "split": args.split,
        "n_pairs": len(pairs),
        "position": "final answer token after chat prefix plus completion; no assistant end-of-turn token appended",
        "dtype": args.dtype,
        "max_length": args.max_length,
        "task_family_counts": dict(sorted(task_counts.items())),
        "pair_ids": [pair.row_id for pair in pairs],
        "sources": sorted(set(pair.source for pair in pairs if pair.source)),
        "npz": str(out_npz.relative_to(ROOT)),
        "arrays": {
            "vectors": "raw positive-negative directions, shape [embedding+layers, hidden_dim]",
            "unit_vectors": "vectors normalized per layer",
            "positive_mean": "per-layer mean hidden state for positive completions",
            "negative_mean": "per-layer mean hidden state for negative completions",
            "norms": "L2 norm of each raw vector",
            "layer_names": "embedding, layer_0, ...",
        },
    }
    out_json.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    print(f"[done] wrote {out_npz} and {out_json}", flush=True)
    return {
        "concept": concept,
        "split": args.split,
        "n_pairs": len(pairs),
        "npz": str(out_npz.relative_to(ROOT)),
        "metadata": str(out_json.relative_to(ROOT)),
        "status": "wrote",
    }


def write_summary(rows: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "extraction_summary.csv"
    fieldnames = ["concept", "split", "n_pairs", "status", "npz", "metadata"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[summary] wrote {path}", flush=True)


def main() -> None:
    args = parse_args()
    selected: dict[str, list[ContrastPair]] = {}
    for concept in args.concepts:
        path = concept_path(args.contrast_dir, concept)
        pairs = load_pairs(path, concept, args.split, args.max_pairs)
        selected[concept] = pairs
        counts = Counter(pair.task_family for pair in pairs)
        print(f"[data] {concept} split={args.split} n={len(pairs)} task_families={dict(counts)}", flush=True)

    if args.dry_run:
        return

    model, tokenizer, model_path = load_model_and_tokenizer(args)
    rows = []
    for concept, pairs in selected.items():
        rows.append(write_vector(concept, pairs, model, tokenizer, model_path, args))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_summary(rows, args.out_dir)


if __name__ == "__main__":
    main()
