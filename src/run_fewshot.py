"""Few-shot generation for BASE models: prepend worked examples so a pretrained
model produces the answer format without instruction-tuning.

Writes triplet_fs.csv / pairwise_fs.csv (keeps zero-shot and _lp files intact).
This is a third base-appropriate measurement, complementary to logprob.

Usage:
  python run_fewshot.py --model olmo2-7b-base --methods triplet pairwise
"""
import argparse
import csv
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import (FEWSHOT_TRIPLET, FEWSHOT_PAIRWISE,  # noqa: E402
                     triplet_prompt, pairwise_prompt)
from stimuli import _read_rows  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
STIM = os.path.join(HERE, "data", "stimuli")


def load_registry():
    with open(os.path.join(HERE, "configs", "models.yaml")) as f:
        return yaml.safe_load(f)


def resolve(repo, cache):
    cn = "models--" + repo.replace("/", "--")
    caches = [cache] + [c for c in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":") if c]
    for _c in caches:
        for snap in sorted(glob.glob(os.path.join(_c, cn, "snapshots", "*"))):
            if glob.glob(os.path.join(snap, "config.json")):
                return snap
    return repo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="+", default=["triplet", "pairwise"])
    ap.add_argument("--gpu_mem_util", type=float, default=0.85)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    mp = resolve(spec["path"], hf_cache)
    if os.path.isdir(mp):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    outdir = os.path.join(RAW, args.model)
    os.makedirs(outdir, exist_ok=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=mp, download_dir=hf_cache, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, dtype="bfloat16",
              trust_remote_code=True)
    sp = SamplingParams(temperature=0, max_tokens=8, stop=["\n"])

    if "triplet" in args.methods:
        out = os.path.join(outdir, "triplet_fs.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out}")
        else:
            rows = _read_rows(os.path.join(STIM, "triplets.csv"))
            prompts = [FEWSHOT_TRIPLET + triplet_prompt(a, c1, c2) + "\n"
                       for a, c1, c2 in rows]
            outs = llm.generate(prompts, sp)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "prompt", "response"])
                for (a, c1, c2), o in zip(rows, outs):
                    w.writerow([f"{a}|{c1}|{c2}", "fewshot", o.outputs[0].text.strip()])
            print(f"[done] triplet (fewshot): {len(rows)} rows -> {out}")

    if "pairwise" in args.methods:
        out = os.path.join(outdir, "pairwise_fs.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out}")
        else:
            rows = _read_rows(os.path.join(STIM, "pairs.csv"))
            prompts = [FEWSHOT_PAIRWISE + pairwise_prompt(a, b) + "\n" for a, b in rows]
            outs = llm.generate(prompts, sp)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "prompt", "response"])
                for (a, b), o in zip(rows, outs):
                    w.writerow([f"{a}|{b}", "fewshot", o.outputs[0].text.strip()])
            print(f"[done] pairwise (fewshot): {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
