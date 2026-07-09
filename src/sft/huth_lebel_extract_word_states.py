"""Extract word-level LM hidden states for the Huth/LeBel ds003020 smoke lane.

This is the GPU-side half of the smoke encoding pipeline. It parses Praat
TextGrid word timings, reconstructs plain narrative text without chat templates,
and saves one hidden-state tensor per arm/story:

    words x selected_layers x hidden_dim

The paired CPU script, ``huth_lebel_smoke_encoding.py``, consumes these files,
aligns them to fMRI TRs, adds FIR delays, and fits voxelwise ridge models.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sft.extract_fmri_hidden_states import (  # noqa: E402
    adapter_metadata,
    arm_registry,
    configure_hf_cache,
    load_registry,
    parse_adapter_overrides,
    resolve_model_path,
    save_dtype,
    selected_arms,
    torch_dtype,
)
from sft.lora_apply import install_lora_adapter, set_active_lora  # noqa: E402


DEFAULT_DS_ROOT = Path("/staging/s/suresh27/datasets/ds003020-smoke")
DEFAULT_OUT = ROOT / "results" / "sft_huth_lebel" / "word_states_smoke"
DEFAULT_STORIES = ("sweetaspie", "againstthewind", "wheretheressmoke")
DEFAULT_ARMS = "base,lowLR,scrambled,taskvec_a0p25"
DEFAULT_LAYERS = "16,24,32"
WORD_TIER_NAMES = {"word", "words", "transcript", "orthography", "ortho"}
SKIP_WORDS = {"", "sp", "spn", "sil", "silence", "<unk>", "{lg}", "{br}", "{cg}"}


@dataclass(frozen=True)
class WordTiming:
    word: str
    onset: float
    offset: float

    @property
    def center(self) -> float:
        return (self.onset + self.offset) / 2.0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds_root", type=Path, default=DEFAULT_DS_ROOT)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--stories", default=",".join(DEFAULT_STORIES))
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument(
        "--model_path",
        type=Path,
        default=None,
        help="Explicit local/staged model directory. Overrides --model registry lookup.",
    )
    ap.add_argument(
        "--hf_cache",
        default="",
        help="Optional HF cache path used with --model_path. Defaults to HF_HOME or repo out/hf_cache.",
    )
    ap.add_argument("--arms", default=DEFAULT_ARMS)
    ap.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="ARM=PATH",
        help="Override/add an adapter arm. Use ARM=base for no adapter.",
    )
    ap.add_argument(
        "--layers",
        default=DEFAULT_LAYERS,
        help="Hidden-state layers to save, e.g. '16,24,32', '12:24', or 'all'. "
        "Layer 0 is the embedding output; decoder layers are 1..N.",
    )
    ap.add_argument("--word_tier", default="", help="Optional exact TextGrid tier name.")
    ap.add_argument("--max_context_tokens", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--limit_words", type=int, default=0, help="Debug only: extract first N words.")
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--save_dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--allow_download", action="store_true")
    ap.add_argument("--no_bos", action="store_true", help="Do not prepend BOS at context resets.")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed for NPZ output.")
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def first_existing(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def textgrid_path(ds_root: Path, story: str) -> Path:
    base = first_existing(
        [
            ds_root / "derivatives" / "TextGrids",
            ds_root / "derivative" / "TextGrids",
        ]
    )
    return base / f"{story}.TextGrid"


def praat_value(line: str) -> str:
    _, value = line.split("=", 1)
    value = value.strip()
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1].replace('""', '"')
    return value


def parse_textgrid(path: Path, tier_name: str = "") -> list[WordTiming]:
    """Parse long-form Praat TextGrid intervals and return word timings."""
    if not path.exists():
        raise FileNotFoundError(path)

    tiers: list[dict] = []
    current_tier: dict | None = None
    current_interval: dict | None = None
    item_re = re.compile(r"^item \[\d+\]:$")
    interval_re = re.compile(r"^intervals \[\d+\]:$")

    for raw in path.read_text(errors="replace").splitlines():
        line = raw.strip()
        if item_re.match(line):
            current_tier = {"name": "", "intervals": []}
            tiers.append(current_tier)
            current_interval = None
            continue
        if current_tier is None:
            continue
        if line.startswith("name ="):
            current_tier["name"] = praat_value(line)
            continue
        if interval_re.match(line):
            current_interval = {}
            continue
        if current_interval is None:
            continue
        if line.startswith("xmin ="):
            current_interval["xmin"] = float(praat_value(line))
        elif line.startswith("xmax ="):
            current_interval["xmax"] = float(praat_value(line))
        elif line.startswith("text ="):
            current_interval["text"] = praat_value(line)
            if {"xmin", "xmax", "text"} <= set(current_interval):
                current_tier["intervals"].append(current_interval)
            current_interval = None

    if not tiers:
        raise ValueError(f"{path}: no TextGrid interval tiers parsed")

    chosen = choose_word_tier(path, tiers, tier_name)
    words = []
    for interval in chosen["intervals"]:
        label = clean_word_label(str(interval.get("text", "")))
        if label.lower() in SKIP_WORDS:
            continue
        if not label:
            continue
        onset = float(interval["xmin"])
        offset = float(interval["xmax"])
        if offset <= onset:
            continue
        words.append(WordTiming(label, onset, offset))
    if not words:
        raise ValueError(f"{path}: selected tier {chosen.get('name')!r} has no word intervals")
    return words


def choose_word_tier(path: Path, tiers: list[dict], tier_name: str) -> dict:
    if tier_name:
        matches = [tier for tier in tiers if str(tier.get("name", "")) == tier_name]
        if not matches:
            names = [str(tier.get("name", "")) for tier in tiers]
            raise KeyError(f"{path}: tier {tier_name!r} not found; available tiers={names}")
        return matches[0]

    named = [
        tier
        for tier in tiers
        if str(tier.get("name", "")).strip().lower() in WORD_TIER_NAMES
    ]
    if named:
        return named[0]

    def n_nonempty(tier: dict) -> int:
        return sum(
            bool(clean_word_label(str(interval.get("text", ""))))
            for interval in tier.get("intervals", [])
        )

    return max(tiers, key=n_nonempty)


def clean_word_label(label: str) -> str:
    label = label.strip()
    if not label:
        return ""
    return re.sub(r"\s+", " ", label)


def write_word_table(out_dir: Path, story: str, words: list[WordTiming], overwrite: bool) -> None:
    table_dir = out_dir / "word_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    path = table_dir / f"{story}.csv"
    if path.exists() and not overwrite:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["story", "word_index", "word", "onset", "offset", "center"],
        )
        writer.writeheader()
        for idx, word in enumerate(words):
            writer.writerow(
                {
                    "story": story,
                    "word_index": idx,
                    "word": word.word,
                    "onset": f"{word.onset:.6f}",
                    "offset": f"{word.offset:.6f}",
                    "center": f"{word.center:.6f}",
                }
            )


def token_pieces_for_words(tokenizer, words: list[WordTiming]) -> list[list[int]]:
    pieces = []
    for idx, word in enumerate(words):
        text = word.word if idx == 0 else " " + word.word
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(word.word, add_special_tokens=False)
        if not ids and tokenizer.unk_token_id is not None:
            ids = [int(tokenizer.unk_token_id)]
        if not ids:
            raise ValueError(f"word {idx}={word.word!r} produced no tokens")
        pieces.append([int(x) for x in ids])
    return pieces


def flatten_pieces(pieces: list[list[int]]) -> tuple[list[int], np.ndarray]:
    offsets = np.zeros(len(pieces) + 1, dtype=np.int64)
    flat: list[int] = []
    for idx, ids in enumerate(pieces):
        flat.extend(ids)
        offsets[idx + 1] = len(flat)
    return flat, offsets


def parse_layers(text: str, n_model_layers: int) -> list[int]:
    n_hidden_states = n_model_layers + 1
    if text.strip().lower() == "all":
        return list(range(n_hidden_states))
    layers: list[int] = []
    for item in comma_list(text):
        if ":" in item:
            left, right = item.split(":", 1)
            start = int(left)
            end = int(right)
            if start > end:
                raise ValueError(f"bad layer range {item!r}")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(item))
    bad = [layer for layer in layers if layer < 0 or layer >= n_hidden_states]
    if bad:
        raise ValueError(f"layers out of range {bad}; valid range is 0..{n_hidden_states - 1}")
    return sorted(dict.fromkeys(layers))


def layer_name(layer_idx: int) -> str:
    return "embedding" if layer_idx == 0 else f"layer_{layer_idx}"


def make_batch(
    flat_tokens: list[int],
    offsets: np.ndarray,
    word_start: int,
    word_end: int,
    max_context_tokens: int,
    bos_ids: list[int],
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    max_payload = max_context_tokens - len(bos_ids)
    if max_payload < 1:
        raise ValueError("--max_context_tokens must leave room for at least one non-BOS token")

    windows: list[list[int]] = []
    token_counts = []
    for word_idx in range(word_start, word_end):
        token_end = int(offsets[word_idx + 1])
        token_start = max(0, token_end - max_payload)
        ids = bos_ids + flat_tokens[token_start:token_end]
        windows.append(ids)
        token_counts.append(token_end - token_start)

    width = max(len(ids) for ids in windows)
    input_ids = torch.full((len(windows), width), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((len(windows), width), dtype=torch.long)
    final_idx = torch.zeros(len(windows), dtype=torch.long)
    for row, ids in enumerate(windows):
        input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, : len(ids)] = 1
        final_idx[row] = len(ids) - 1
    return (
        input_ids.to(device),
        attention_mask.to(device),
        final_idx.to(device),
        token_counts,
    )


def extract_story_hidden(
    model,
    tokenizer,
    words: list[WordTiming],
    selected_layers: list[int],
    *,
    max_context_tokens: int,
    batch_size: int,
    array_dtype: np.dtype,
    add_bos: bool,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    device = next(model.parameters()).device
    pieces = token_pieces_for_words(tokenizer, words)
    flat_tokens, offsets = flatten_pieces(pieces)
    bos_ids = []
    if add_bos and tokenizer.bos_token_id is not None:
        bos_ids = [int(tokenizer.bos_token_id)]
    pad_token_id = int(tokenizer.pad_token_id or tokenizer.eos_token_id or 0)

    chunks = []
    context_token_counts = []
    rows = None
    for start in range(0, len(words), batch_size):
        end = min(start + batch_size, len(words))
        input_ids, attention_mask, final_idx, token_counts = make_batch(
            flat_tokens,
            offsets,
            start,
            end,
            max_context_tokens,
            bos_ids,
            pad_token_id,
            device,
        )
        if rows is None or rows.shape[0] != input_ids.shape[0]:
            rows = torch.arange(input_ids.shape[0], device=device)
        with torch.inference_mode():
            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=False,
            )
        layer_vals = [
            out.hidden_states[layer][rows[: input_ids.shape[0]], final_idx]
            .detach()
            .to(torch.float32)
            .cpu()
            .numpy()
            for layer in selected_layers
        ]
        chunks.append(np.stack(layer_vals, axis=1).astype(array_dtype, copy=False))
        context_token_counts.extend(token_counts)
        done = min(end, len(words))
        print(f"[extract] processed {done}/{len(words)} words", flush=True)
        del out, input_ids, attention_mask, final_idx
    hidden = np.concatenate(chunks, axis=0)
    return hidden, np.array(context_token_counts, dtype=np.int32)


def save_story(
    out_dir: Path,
    arm: str,
    story: str,
    hidden: np.ndarray,
    words: list[WordTiming],
    context_token_counts: np.ndarray,
    selected_layers: list[int],
    meta: dict,
    *,
    compress: bool,
    overwrite: bool,
) -> None:
    arm_dir = out_dir / arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    npz_path = arm_dir / f"{story}.npz"
    json_path = arm_dir / f"{story}.json"
    if npz_path.exists() and not overwrite:
        raise FileExistsError(f"{npz_path} exists; pass --overwrite or --skip_existing")
    arrays = {
        "hidden": hidden,
        "words": np.array([word.word for word in words]),
        "onsets": np.array([word.onset for word in words], dtype=np.float32),
        "offsets": np.array([word.offset for word in words], dtype=np.float32),
        "centers": np.array([word.center for word in words], dtype=np.float32),
        "context_token_counts": context_token_counts,
        "layer_indices": np.array(selected_layers, dtype=np.int16),
        "layer_names": np.array([layer_name(layer) for layer in selected_layers]),
    }
    if compress:
        np.savez_compressed(npz_path, **arrays)
    else:
        np.savez(npz_path, **arrays)
    meta = dict(meta)
    meta.update(
        {
            "story": story,
            "hidden_state_file": str(npz_path),
            "shape": list(hidden.shape),
            "layer_indices": selected_layers,
            "layer_names": [layer_name(layer) for layer in selected_layers],
            "n_words": len(words),
            "first_word": words[0].word,
            "last_word": words[-1].word,
        }
    )
    json_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[write] {arm}/{story}: {npz_path} {hidden.shape}", flush=True)


def main() -> None:
    args = parse_args()
    stories = comma_list(args.stories)

    if args.model_path is not None:
        model_path = str(args.model_path)
        if not (Path(model_path) / "config.json").exists():
            raise FileNotFoundError(Path(model_path) / "config.json")
        hf_cache = args.hf_cache or os.environ.get("HF_HOME") or str(ROOT / "out" / "hf_cache")
        configure_hf_cache(hf_cache, model_path)
    else:
        registry = load_registry()
        spec = registry["local"][args.model]
        model_path = resolve_model_path(
            spec["path"],
            registry["hf_cache"],
            args.allow_download or bool(spec.get("download", False)),
        )
        configure_hf_cache(registry["hf_cache"], model_path)

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

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=model_dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    selected_layers = parse_layers(args.layers, int(model.config.num_hidden_layers))
    print(f"[layers] {selected_layers}", flush=True)

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

    words_by_story = {}
    for story in stories:
        tg_path = textgrid_path(args.ds_root, story)
        words = parse_textgrid(tg_path, args.word_tier)
        if args.limit_words > 0:
            words = words[: args.limit_words]
        words_by_story[story] = words
        write_word_table(args.out_dir, story, words, args.overwrite)
        print(f"[textgrid] {story}: {len(words)} words from {tg_path}", flush=True)

    for arm in arm_names:
        set_active_lora(model, None if arms[arm] is None else arm)
        print(f"[arm] extracting {arm}", flush=True)
        for story in stories:
            npz_path = args.out_dir / arm / f"{story}.npz"
            if npz_path.exists() and args.skip_existing and not args.overwrite:
                print(f"[skip] {npz_path}", flush=True)
                continue
            hidden, context_token_counts = extract_story_hidden(
                model,
                tokenizer,
                words_by_story[story],
                selected_layers,
                max_context_tokens=args.max_context_tokens,
                batch_size=args.batch_size,
                array_dtype=array_dtype,
                add_bos=not args.no_bos,
            )
            meta = {
                "arm": arm,
                "model": args.model,
                "model_path": model_path,
                "ds_root": str(args.ds_root),
                "textgrid": str(textgrid_path(args.ds_root, story)),
                "chat_template": False,
                "tokenization": "space-joined TextGrid word labels; final subtoken per word",
                "max_context_tokens": args.max_context_tokens,
                "add_bos": not args.no_bos,
                "batch_size": args.batch_size,
                "model_dtype": args.dtype,
                "save_dtype": args.save_dtype,
                "device": str(device),
                "installed_lora_modules": installed.get(arm, 0),
            }
            meta.update(adapter_metadata(arm, arms[arm]))
            save_story(
                args.out_dir,
                arm,
                story,
                hidden,
                words_by_story[story],
                context_token_counts,
                selected_layers,
                meta,
                compress=args.compress,
                overwrite=args.overwrite,
            )
    set_active_lora(model, None)


if __name__ == "__main__":
    main()
