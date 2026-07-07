"""Stage 1: feature LISTING per model (local vLLM).

Prompt the model to list the properties of each of the 30 concepts, optionally with
repeats to widen coverage. Writes results/raw/<model>/listing.csv with columns:
concept, rep, prompt, response (raw generation).

Usage:
  python run_listing.py --model llama-3.1-8b-instruct --repeats 5 --temperature 0.7
"""
import argparse
import csv
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM_PROMPT, listing_prompt  # noqa: E402
from stimuli import load_concepts  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.environ.get("COHERENCE_RAW_DIR", os.path.join(HERE, "results", "raw"))


def load_registry():
    with open(os.path.join(HERE, "configs", "models.yaml")) as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id, hf_cache, allow_download):
    cache_name = "models--" + repo_id.replace("/", "--")
    caches = [hf_cache] + [c for c in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":") if c]
    for _c in caches:
        for snap in sorted(glob.glob(os.path.join(_c, cache_name, "snapshots", "*"))):
            if glob.glob(os.path.join(snap, "config.json")):
                return snap
    return repo_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_num_seqs", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    model_path = resolve_model_path(spec["path"], hf_cache, spec.get("download", False))
    if os.path.isdir(model_path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    outdir = os.path.join(RAW, args.model)
    os.makedirs(outdir, exist_ok=True)
    out = os.path.join(outdir, "listing.csv")
    if os.path.exists(out) and not args.overwrite:
        print(f"[skip] {out} exists")
        return

    concepts = load_concepts()
    # one job per (concept, rep)
    jobs = [(c, r) for c in concepts for r in range(args.repeats)]
    prompts = [listing_prompt(c) for c, _ in jobs]

    from vllm import LLM, SamplingParams
    tp = spec.get("tensor_parallel", args.tensor_parallel)
    llm_kwargs = dict(model=model_path, download_dir=hf_cache,
                      tensor_parallel_size=tp,
                      max_model_len=args.max_model_len,
                      gpu_memory_utilization=args.gpu_mem_util,
        **({"max_num_seqs": args.max_num_seqs} if args.max_num_seqs else {}),
                      dtype="bfloat16", trust_remote_code=True)
    if spec.get("quantization") and os.environ.get("COHERENCE_FORCE_BF16") != "1":
        llm_kwargs["quantization"] = spec["quantization"]
    llm = LLM(**llm_kwargs)
    # temperature>0 needs a seed-free sampling; vLLM handles randomness per request
    sp = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    if spec.get("chat", True):
        convos = [[{"role": "system", "content": SYSTEM_PROMPT},
                   {"role": "user", "content": p}] for p in prompts]
        outputs = llm.chat(convos, sp)
    else:
        outputs = llm.generate(prompts, sp)

    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept", "rep", "prompt", "response"])
        for (c, r), p, o in zip(jobs, prompts, outputs):
            w.writerow([c, r, p, o.outputs[0].text.strip()])
    print(f"[done] listing: {len(jobs)} generations ({len(concepts)} concepts x "
          f"{args.repeats} reps) -> {out}")


if __name__ == "__main__":
    main()
