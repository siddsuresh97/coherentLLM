"""Run paraphrase-consistency and transitivity prompts for SFT eval."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import SYSTEM_PROMPT, triplet_prompt  # noqa: E402
from sft.paraphrases import TEMPLATES  # noqa: E402

RAW = Path(os.environ.get("COHERENCE_RAW_DIR", ROOT / "results" / "sft_eval" / "raw"))
STIMULI = ROOT / "results" / "sft_eval" / "stimuli"


def load_registry():
    with (ROOT / "configs" / "models.yaml").open() as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id: str, hf_cache: str, allow_download: bool):
    cache_name = "models--" + repo_id.replace("/", "--")
    caches = [hf_cache] + [c for c in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":") if c]
    for cache in caches:
        for snap in sorted(glob.glob(os.path.join(cache, cache_name, "snapshots", "*"))):
            if glob.glob(os.path.join(snap, "config.json")):
                return snap
    if not allow_download:
        print(f"[warn] {repo_id} not found in cache; will let HF resolve/download")
    return repo_id


def _read_dicts(path: Path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _write_response_csv(path: Path, header, rows, prompts, outputs):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header + ["prompt", "response"])
        for meta, prompt, out in zip(rows, prompts, outputs):
            w.writerow(list(meta) + [prompt, out.outputs[0].text.strip()])
    print(f"[done] {path}: {len(rows)} rows")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama-3.1-8b-instruct")
    ap.add_argument("--out_model", required=True)
    ap.add_argument("--lora", default=None)
    ap.add_argument("--tasks", nargs="+", default=["paraphrase", "transitivity"],
                    choices=["paraphrase", "transitivity"])
    ap.add_argument("--stimuli_dir", default=str(STIMULI))
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_num_seqs", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    outdir = RAW / args.out_model
    planned = []
    if "paraphrase" in args.tasks:
        for method in ["triplet", "pairwise", "feature"]:
            out = outdir / f"paraphrase_{method}.csv"
            if not out.exists() or args.overwrite:
                planned.append(("paraphrase", method, out))
            else:
                print(f"[skip] {out} exists")
    if "transitivity" in args.tasks:
        out = outdir / "transitivity.csv"
        if not out.exists() or args.overwrite:
            planned.append(("transitivity", None, out))
        else:
            print(f"[skip] {out} exists")
    if not planned:
        print("nothing to do")
        return

    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    model_path = resolve_model_path(spec["path"], hf_cache, spec.get("download", False))
    if os.path.isdir(model_path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    from vllm import LLM, SamplingParams

    llm_kwargs = dict(
        model=model_path,
        download_dir=hf_cache,
        tensor_parallel_size=spec.get("tensor_parallel", 1),
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        max_num_seqs=args.max_num_seqs,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    lora_request = None
    if args.lora:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = 64
        from vllm.lora.request import LoRARequest
        lora_request = LoRARequest(args.out_model, 1, os.path.abspath(args.lora))
    if spec.get("quantization") and os.environ.get("COHERENCE_FORCE_BF16") != "1":
        llm_kwargs["quantization"] = spec["quantization"]

    llm = LLM(**llm_kwargs)
    sp = SamplingParams(temperature=args.temperature, max_tokens=8)

    def run_prompts(prompts):
        if spec.get("chat", True):
            convos = [[{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": p}] for p in prompts]
            return llm.chat(convos, sp, lora_request=lora_request)
        return llm.generate(prompts, sp, lora_request=lora_request)

    stimuli_dir = Path(args.stimuli_dir)
    for kind, method, out in planned:
        if kind == "paraphrase":
            rows = []
            prompts = []
            if method == "triplet":
                for row in _read_dicts(stimuli_dir / "paraphrase_triplets.csv"):
                    for template_id, tmpl in enumerate(TEMPLATES[method]):
                        rows.append((method, row["item_id"], template_id,
                                     "|".join([row["anchor"], row["c1"], row["c2"]])))
                        prompts.append(tmpl(row["anchor"], row["c1"], row["c2"]))
            elif method == "pairwise":
                for row in _read_dicts(stimuli_dir / "paraphrase_pairs.csv"):
                    for template_id, tmpl in enumerate(TEMPLATES[method]):
                        rows.append((method, row["item_id"], template_id,
                                     "|".join([row["a"], row["b"]])))
                        prompts.append(tmpl(row["a"], row["b"]))
            else:
                for row in _read_dicts(stimuli_dir / "paraphrase_features.csv"):
                    for template_id, tmpl in enumerate(TEMPLATES[method]):
                        rows.append((method, row["item_id"], template_id,
                                     "|".join([row["feature"], row["concept"]])))
                        prompts.append(tmpl(row["feature"], row["concept"]))
            outputs = run_prompts(prompts)
            _write_response_csv(out, ["method", "item_id", "template_id", "input"],
                                rows, prompts, outputs)
            continue

        rows = []
        prompts = []
        for row in _read_dicts(stimuli_dir / "transitivity_cycles.csv"):
            comps = [("xy", row["x"], row["y"]),
                     ("yz", row["y"], row["z"]),
                     ("xz", row["x"], row["z"])]
            for label, a, b in comps:
                rows.append((row["cycle_id"], row["anchor"], row["x"], row["y"], row["z"],
                             label, a, b))
                prompts.append(triplet_prompt(row["anchor"], a, b))
        outputs = run_prompts(prompts)
        _write_response_csv(
            out,
            ["cycle_id", "anchor", "x", "y", "z", "comparison", "option_a", "option_b"],
            rows,
            prompts,
            outputs,
        )


if __name__ == "__main__":
    main()

