"""Evaluate base/real/scrambled hidden-state similarities on external benchmarks."""
from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats import spearmanr
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sft.lora_apply import install_lora_adapter, set_active_lora  # noqa: E402


WORD_BENCHMARK_URLS = {
    "SimLex-999": "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-SIMLEX-999.txt",
    "WordSim-353": "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-WS-353-ALL.txt",
    "MEN": "https://raw.githubusercontent.com/mfaruqui/eval-word-vectors/master/data/word-sim/EN-MEN-TR-3k.txt",
}


@dataclass(frozen=True)
class PairRow:
    text_a: str
    text_b: str
    score: float
    row_id: str


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--real_adapter", default=str(ROOT / "out" / "adapters_vllm_fixed" / "real"))
    ap.add_argument("--scrambled_adapter", default=str(ROOT / "out" / "adapters_vllm_fixed" / "scrambled"))
    ap.add_argument("--out_dir", default=str(ROOT / "results" / "sft_eval" / "external"))
    ap.add_argument("--cache_dir", default=str(ROOT / "data" / "external_benchmarks"))
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--states", nargs="+", default=["base", "real", "scrambled"],
                    choices=["base", "real", "scrambled"])
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
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "out" / "hf_datasets_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "out" / "triton_cache"))
    if Path(model_path).is_dir():
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["HF_DATASETS_OFFLINE"] = "0"


def load_stsb() -> list[PairRow]:
    from datasets import load_dataset

    ds = load_dataset("glue", "stsb", split="validation")
    rows = []
    for row in ds:
        rows.append(
            PairRow(
                str(row["sentence1"]),
                str(row["sentence2"]),
                float(row["label"]),
                str(row.get("idx", len(rows))),
            )
        )
    return rows


def _download_text(url: str, cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if not cache_path.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            cache_path.write_bytes(response.read())
    return cache_path.read_text(encoding="utf-8")


def load_tab_word_pairs(name: str, cache_dir: Path) -> list[PairRow]:
    url = WORD_BENCHMARK_URLS[name]
    text = _download_text(url, cache_dir / Path(url).name)
    rows = []
    reader = csv.reader(io.StringIO(text), delimiter="\t")
    for idx, row in enumerate(reader):
        if not row or len(row) < 3:
            continue
        try:
            score = float(row[2])
        except ValueError:
            continue
        rows.append(PairRow(row[0].strip(), row[1].strip(), score, str(idx)))
    if not rows:
        raise ValueError(f"no rows parsed for {name} from {url}")
    return rows


def load_benchmarks(cache_dir: Path) -> dict[str, list[PairRow]]:
    benches = {"STS-B": load_stsb()}
    for name in ["SimLex-999", "WordSim-353", "MEN"]:
        benches[name] = load_tab_word_pairs(name, cache_dir)
    return benches


def unique_texts(benchmarks: dict[str, list[PairRow]]) -> list[str]:
    seen = {}
    for rows in benchmarks.values():
        for row in rows:
            seen.setdefault(row.text_a, None)
            seen.setdefault(row.text_b, None)
    return list(seen.keys())


def embed_texts(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> dict[str, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    out: dict[str, np.ndarray] = {}
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        last_idx = enc["attention_mask"].sum(dim=1) - 1
        rows = torch.arange(last_idx.shape[0], device=device)
        with torch.inference_mode():
            outputs = model(**enc, output_hidden_states=True, use_cache=False)
        hidden = outputs.hidden_states[-1][rows, last_idx]
        hidden = torch.nn.functional.normalize(hidden.float(), p=2, dim=-1)
        for text, vec in zip(batch, hidden.cpu().numpy()):
            out[text] = vec.astype(np.float32)
        print(f"[embed] {len(out)}/{len(texts)} texts", flush=True)
    return out


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return math.nan
    return float(np.dot(a, b) / denom)


def evaluate_state(
    state: str,
    embeddings: dict[str, np.ndarray],
    benchmarks: dict[str, list[PairRow]],
    out_dir: Path,
) -> list[dict]:
    summaries = []
    for bench_name, rows in benchmarks.items():
        raw_path = out_dir / f"{state}_{bench_name.lower().replace('-', '').replace(' ', '_')}.csv"
        human = []
        model_scores = []
        with raw_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["row_id", "text_a", "text_b", "human_score", "model_cosine"])
            for row in rows:
                sim = cosine(embeddings[row.text_a], embeddings[row.text_b])
                writer.writerow([row.row_id, row.text_a, row.text_b, row.score, sim])
                if not math.isnan(sim):
                    human.append(row.score)
                    model_scores.append(sim)
        rho, p_value = spearmanr(human, model_scores)
        summaries.append(
            {
                "model": state,
                "benchmark": bench_name,
                "n_pairs": len(model_scores),
                "rho": float(rho),
                "p_value": float(p_value),
                "method": "cosine_last_token_hidden",
                "raw_file": str(raw_path),
            }
        )
        print(f"[rho] {state} {bench_name}: n={len(model_scores)} rho={rho:.6f}", flush=True)
    return summaries


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.csv"
    if summary_path.exists() and not args.overwrite:
        print(f"[skip] {summary_path} exists; pass --overwrite to recompute")
        return

    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    model_path = resolve_model_path(spec["path"], hf_cache, args.allow_download or bool(spec.get("download", False)))
    configure_hf_cache(hf_cache, model_path)

    benchmarks = load_benchmarks(Path(args.cache_dir))
    texts = unique_texts(benchmarks)
    print(
        "[data] "
        + ", ".join(f"{name}={len(rows)}" for name, rows in benchmarks.items())
        + f"; unique_texts={len(texts)}",
        flush=True,
    )

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

    installed = {}
    if "real" in args.states:
        installed["real"] = install_lora_adapter(
            model, args.real_adapter, "real", dtype=dtype, device=device, activate=False
        )
    if "scrambled" in args.states:
        installed["scrambled"] = install_lora_adapter(
            model, args.scrambled_adapter, "scrambled", dtype=dtype, device=device, activate=False
        )
    if installed:
        print(f"[lora] installed modules: {installed}", flush=True)

    summaries = []
    for state in args.states:
        set_active_lora(model, None if state == "base" else state)
        print(f"[state] embedding {state}", flush=True)
        embeddings = embed_texts(
            model,
            tokenizer,
            texts,
            batch_size=args.batch_size,
            max_length=args.max_length,
        )
        summaries.extend(evaluate_state(state, embeddings, benchmarks, out_dir))

    with summary_path.open("w", newline="") as f:
        fieldnames = ["model", "benchmark", "n_pairs", "rho", "p_value", "method", "raw_file"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    meta = {
        "model": args.model,
        "model_path": model_path,
        "states": args.states,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "benchmarks": {name: len(rows) for name, rows in benchmarks.items()},
        "word_benchmark_urls": WORD_BENCHMARK_URLS,
    }
    (out_dir / "summary_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
