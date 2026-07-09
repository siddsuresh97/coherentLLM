"""Run cheap concept-vector steering sweeps with forced-choice probes.

The evaluator loads CAA vectors from ``extract_concept_vectors.py``, injects a
single selected layer during inference, and scores whether the model assigns
higher mean token log-likelihood to the positive completion than the negative
completion for:

  - coherence eval contrasts
  - human-alignment eval contrasts
  - a small neutral retention probe

Use ``--smoke`` first. The default full targeted sweep is still small compared
with lm-eval, but it loads the 8B model and runs multiple layer/alpha configs.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
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
DEFAULT_VECTOR_DIR = ROOT / "results" / "sft_eval" / "concept_steering" / "vectors"
DEFAULT_OUT_DIR = ROOT / "results" / "sft_eval" / "concept_steering"
DEFAULT_HF_DATASETS_CACHE = ROOT / "out" / "hf_datasets_cache"
CONCEPT_FILES = {
    "coherence": "coherence.jsonl",
    "human_alignment": "human_alignment.jsonl",
}


@dataclass(frozen=True)
class ProbeItem:
    row_id: str
    eval_set: str
    task_family: str
    prompt: str
    positive_completion: str
    negative_completion: str


@dataclass(frozen=True)
class ScoredCompletion:
    mean_logprob: float
    sum_logprob: float
    n_tokens: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--contrast-dir", type=Path, default=DEFAULT_CONTRAST_DIR)
    ap.add_argument("--vector-dir", type=Path, default=DEFAULT_VECTOR_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--steer-concepts", nargs="+", default=["coherence", "human_alignment"])
    ap.add_argument("--eval-concepts", nargs="+", default=["coherence", "human_alignment"])
    ap.add_argument("--layers", default="12,16,20,24")
    ap.add_argument("--alphas", nargs="+", type=float, default=[-4.0, -2.0, 0.0, 2.0, 4.0])
    ap.add_argument("--split", default="eval")
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--raw-vector", action="store_true", help="Use raw vectors instead of per-layer unit vectors.")
    ap.add_argument(
        "--norm-match",
        action="store_true",
        help="Apply alpha * unit_vector * ||hidden|| instead of alpha * unit_vector.",
    )
    ap.add_argument("--norm-eps", type=float, default=1e-6)
    ap.add_argument("--smoke", action="store_true", help="Run one layer, alpha 0 plus one nonzero alpha, and at most two items per set.")
    ap.add_argument("--dry-run", action="store_true", help="Print planned configs without loading the model.")
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


def load_probe_file(path: Path, eval_set: str, split: str, max_items: int | None) -> list[ProbeItem]:
    items = []
    for row in read_jsonl(path):
        if row.get("split", split) != split:
            continue
        items.append(
            ProbeItem(
                row_id=str(row["id"]),
                eval_set=eval_set,
                task_family=str(row.get("task_family", "")),
                prompt=str(row["prompt"]),
                positive_completion=str(row["positive_completion"]),
                negative_completion=str(row["negative_completion"]),
            )
        )
    if max_items is not None:
        items = items[:max_items]
    if not items:
        raise ValueError(f"no probe items found in {path} for split={split!r}")
    return items


def load_eval_sets(args: argparse.Namespace) -> dict[str, list[ProbeItem]]:
    sets = {}
    for concept in args.eval_concepts:
        if concept not in CONCEPT_FILES:
            known = ", ".join(sorted(CONCEPT_FILES))
            raise ValueError(f"unknown eval concept {concept!r}; expected one of {known}")
        path = args.contrast_dir / CONCEPT_FILES[concept]
        sets[concept] = load_probe_file(path, concept, args.split, args.max_items)
    retention_path = args.contrast_dir / "retention_probe.jsonl"
    sets["retention"] = load_probe_file(retention_path, "retention", "eval", args.max_items)
    return sets


def parse_layer_spec(spec: str, n_layers: int | None = None) -> list[int]:
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"descending layer range {part!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    deduped = []
    for layer in layers:
        if layer < 0:
            raise ValueError(f"negative layer {layer}")
        if n_layers is not None and layer >= n_layers:
            raise ValueError(f"layer {layer} out of range 0..{n_layers - 1}")
        if layer not in deduped:
            deduped.append(layer)
    if not deduped:
        raise ValueError(f"no layers parsed from {spec!r}")
    return deduped


def smoke_alphas(alphas: list[float]) -> list[float]:
    nonzero = [alpha for alpha in alphas if alpha != 0.0]
    if nonzero:
        return [0.0, nonzero[0]]
    return [0.0]


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


def chat_prefix(tokenizer, prompt: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {SYSTEM_PROMPT}\n\nUser: {prompt}\nAssistant:"


def completion_text(tokenizer, item: ProbeItem, which: str) -> tuple[str, int]:
    prefix = chat_prefix(tokenizer, item.prompt)
    completion = item.positive_completion if which == "positive" else item.negative_completion
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    return prefix + completion, len(prefix_ids)


def score_texts(
    model,
    tokenizer,
    texts: list[str],
    prefix_lens: list[int],
    *,
    batch_size: int,
    max_length: int,
) -> list[ScoredCompletion]:
    model.eval()
    device = next(model.parameters()).device
    scores: list[ScoredCompletion] = []
    for start in range(0, len(texts), batch_size):
        end = min(start + batch_size, len(texts))
        enc = tokenizer(
            texts[start:end],
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.inference_mode():
            logits = model(**enc, use_cache=False).logits
        logprobs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
        labels = enc["input_ids"][:, 1:]
        token_logprobs = torch.gather(logprobs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
        positions = torch.arange(token_logprobs.shape[1], device=device).view(1, -1) + 1
        seq_lens = enc["attention_mask"].sum(dim=1).view(-1, 1)
        prefix = torch.as_tensor(prefix_lens[start:end], device=device).view(-1, 1)
        mask = (positions >= prefix) & (positions < seq_lens)
        for row_idx in range(end - start):
            row_mask = mask[row_idx]
            n_tokens = int(row_mask.sum().item())
            if n_tokens == 0:
                scores.append(ScoredCompletion(float("nan"), float("nan"), 0))
                continue
            vals = token_logprobs[row_idx][row_mask]
            sum_lp = float(vals.sum().item())
            scores.append(ScoredCompletion(sum_lp / n_tokens, sum_lp, n_tokens))
    return scores


def score_probe_set(
    model,
    tokenizer,
    items: list[ProbeItem],
    *,
    batch_size: int,
    max_length: int,
) -> tuple[dict, list[dict]]:
    texts = []
    prefix_lens = []
    labels = []
    for item in items:
        for which in ["positive", "negative"]:
            text, prefix_len = completion_text(tokenizer, item, which)
            texts.append(text)
            prefix_lens.append(prefix_len)
            labels.append((item, which))
    scores = score_texts(
        model,
        tokenizer,
        texts,
        prefix_lens,
        batch_size=batch_size,
        max_length=max_length,
    )
    by_id: dict[str, dict[str, ScoredCompletion]] = {}
    by_item: dict[str, ProbeItem] = {}
    for (item, which), score in zip(labels, scores):
        by_id.setdefault(item.row_id, {})[which] = score
        by_item[item.row_id] = item

    detail_rows = []
    deltas = []
    wins = 0
    usable = 0
    for row_id, pair_scores in by_id.items():
        item = by_item[row_id]
        pos = pair_scores["positive"]
        neg = pair_scores["negative"]
        delta = pos.mean_logprob - neg.mean_logprob
        preferred = bool(delta > 0.0) if not math.isnan(delta) else False
        if not math.isnan(delta):
            usable += 1
            deltas.append(delta)
            wins += int(preferred)
        detail_rows.append(
            {
                "item_id": row_id,
                "task_family": item.task_family,
                "positive_mean_logprob": pos.mean_logprob,
                "negative_mean_logprob": neg.mean_logprob,
                "delta_mean_logprob": delta,
                "positive_tokens": pos.n_tokens,
                "negative_tokens": neg.n_tokens,
                "preferred_positive": preferred,
            }
        )
    summary = {
        "n_items": len(items),
        "n_usable": usable,
        "positive_preference": wins / usable if usable else float("nan"),
        "mean_delta_logprob": float(np.mean(deltas)) if deltas else float("nan"),
        "median_delta_logprob": float(np.median(deltas)) if deltas else float("nan"),
    }
    return summary, detail_rows


def load_vector(vector_dir: Path, concept: str, raw_vector: bool) -> tuple[np.ndarray, Path]:
    path = vector_dir / f"{concept}_caa_vectors.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run src/sft/extract_concept_vectors.py for {concept} first."
        )
    data = np.load(path)
    key = "vectors" if raw_vector else "unit_vectors"
    if key not in data:
        raise KeyError(f"{path} has no array {key!r}")
    return np.asarray(data[key], dtype=np.float32), path


def add_steering_hook(
    model,
    vector: np.ndarray,
    *,
    layer: int,
    alpha: float,
    norm_match: bool,
    norm_eps: float,
) -> list[torch.utils.hooks.RemovableHandle]:
    if alpha == 0.0:
        return []
    layers = model.model.layers
    if layer < 0 or layer >= len(layers):
        raise ValueError(f"layer {layer} out of range 0..{len(layers) - 1}")
    if vector.shape[0] != len(layers) + 1:
        raise ValueError(f"vector has {vector.shape[0]} layer rows, expected {len(layers) + 1}")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    direction = torch.as_tensor(vector[layer + 1], device=device, dtype=dtype)
    if norm_match:
        direction = direction / torch.clamp(torch.linalg.vector_norm(direction), min=norm_eps)
    direction = direction.view(1, 1, -1)

    def hook(_module, _inputs, output):
        def steer(hidden):
            steer_vec = direction.to(hidden.dtype)
            if norm_match:
                hidden_norm = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True).clamp_min(norm_eps)
                return hidden + alpha * steer_vec * hidden_norm
            return hidden + alpha * steer_vec

        if isinstance(output, tuple):
            return (steer(output[0]),) + output[1:]
        return steer(output)

    handle = layers[layer].register_forward_hook(hook)
    return [handle]


def evaluate_config(
    model,
    tokenizer,
    eval_sets: dict[str, list[ProbeItem]],
    *,
    steer_concept: str,
    vector: np.ndarray,
    vector_path: Path,
    layer: int,
    alpha: float,
    args: argparse.Namespace,
) -> tuple[list[dict], list[dict]]:
    handles = add_steering_hook(
        model,
        vector,
        layer=layer,
        alpha=alpha,
        norm_match=args.norm_match,
        norm_eps=args.norm_eps,
    )
    summary_rows = []
    detail_rows = []
    try:
        for eval_set, items in eval_sets.items():
            summary, details = score_probe_set(
                model,
                tokenizer,
                items,
                batch_size=args.batch_size,
                max_length=args.max_length,
            )
            row = {
                "steer_concept": steer_concept,
                "eval_set": eval_set,
                "layer": layer,
                "alpha": alpha,
                "vector_mode": "raw" if args.raw_vector else "unit",
                "norm_match": args.norm_match,
                "vector_path": str(vector_path.relative_to(ROOT)),
                **summary,
            }
            summary_rows.append(row)
            for detail in details:
                detail_rows.append({**row, **detail})
    finally:
        for handle in handles:
            handle.remove()
    return summary_rows, detail_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str], max_rows: int = 24) -> str:
    if not rows:
        return "None"
    shown = rows[:max_rows]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in shown:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                vals.append(f"{val:.4f}" if not math.isnan(val) else "nan")
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    if len(rows) > max_rows:
        lines.append(f"| ... | {len(rows) - max_rows} more rows omitted | | | | | | | |")
    return "\n".join(lines)


def write_summary(args: argparse.Namespace, rows: list[dict], config: dict) -> None:
    summary_path = args.out_dir / "SUMMARY.md"
    target_rows = [row for row in rows if row["eval_set"] in {"coherence", "human_alignment"}]
    retention_rows = [row for row in rows if row["eval_set"] == "retention"]
    target_rows = sorted(
        target_rows,
        key=lambda r: (
            r["steer_concept"] != r["eval_set"],
            -float(r["positive_preference"]) if not math.isnan(float(r["positive_preference"])) else 1.0,
            -float(r["mean_delta_logprob"]) if not math.isnan(float(r["mean_delta_logprob"])) else 1.0,
        ),
    )
    retention_rows = sorted(
        retention_rows,
        key=lambda r: (
            r["steer_concept"],
            abs(float(r["alpha"])),
            r["layer"],
        ),
    )
    cols = [
        "steer_concept",
        "eval_set",
        "layer",
        "alpha",
        "positive_preference",
        "mean_delta_logprob",
        "n_usable",
        "norm_match",
        "vector_mode",
    ]
    lines = [
        "# Concept Steering Sweep Summary",
        "",
        "This file is generated by `src/sft/run_concept_steering_eval.py`.",
        "",
        "## Config",
        "",
        "```json",
        json.dumps(config, indent=2, sort_keys=True),
        "```",
        "",
        "## Targeted Probe Rows",
        "",
        markdown_table(target_rows, cols),
        "",
        "## Retention Probe Rows",
        "",
        markdown_table(retention_rows, cols),
        "",
        "Primary metric: `positive_preference`, the fraction of forced-choice items where the steered model assigns higher mean token log-likelihood to the positive completion.",
    ]
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"[summary] wrote {summary_path}", flush=True)


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    max_items = args.max_items
    layers = parse_layer_spec(args.layers)
    alphas = list(args.alphas)
    if args.smoke:
        max_items = min(max_items or 2, 2)
        args.max_items = max_items
        layers = layers[:1]
        alphas = smoke_alphas(alphas)

    eval_sets = load_eval_sets(args)
    plan = {
        "model": args.model,
        "steer_concepts": args.steer_concepts,
        "eval_sets": {name: len(items) for name, items in eval_sets.items()},
        "layers": layers,
        "alphas": alphas,
        "vector_mode": "raw" if args.raw_vector else "unit",
        "norm_match": args.norm_match,
        "smoke": args.smoke,
        "max_items": args.max_items,
    }
    (args.out_dir / "sweep_config.json").write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n")
    print("[plan] " + json.dumps(plan, sort_keys=True), flush=True)
    if args.dry_run:
        return

    model, tokenizer, _model_path = load_model_and_tokenizer(args)
    n_layers = len(model.model.layers)
    layers = parse_layer_spec(",".join(str(layer) for layer in layers), n_layers=n_layers)
    all_summary_rows: list[dict] = []
    all_detail_rows: list[dict] = []
    for steer_concept in args.steer_concepts:
        vector, vector_path = load_vector(args.vector_dir, steer_concept, args.raw_vector)
        for layer in layers:
            for alpha in alphas:
                print(f"[eval] steer={steer_concept} layer={layer} alpha={alpha:g}", flush=True)
                summary_rows, detail_rows = evaluate_config(
                    model,
                    tokenizer,
                    eval_sets,
                    steer_concept=steer_concept,
                    vector=vector,
                    vector_path=vector_path,
                    layer=layer,
                    alpha=alpha,
                    args=args,
                )
                all_summary_rows.extend(summary_rows)
                all_detail_rows.extend(detail_rows)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    write_csv(args.out_dir / "sweep_results.csv", all_summary_rows)
    write_csv(args.out_dir / "sweep_details.csv", all_detail_rows)
    write_summary(args, all_summary_rows, plan)
    print(f"[done] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
