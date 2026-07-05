"""Local GPU inference runner (vLLM) for the three coherence methods.

Reads model weights from the shared HF cache and writes raw responses to
results/raw/<model>/<method>.csv with columns: input tuple + prompt + response.

Usage:
  python run_local.py --model llama-3.1-8b-instruct --methods triplet pairwise feature
  python run_local.py --model llama-3.1-8b-instruct --methods triplet --feature_sample 200
"""
import argparse
import csv
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM_PROMPT  # noqa: E402
from stimuli import build_jobs  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")

MAX_TOKENS = {"triplet": 8, "pairwise": 8, "feature": 8}


def load_registry():
    with open(os.path.join(HERE, "configs", "models.yaml")) as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id: str, hf_cache: str, allow_download: bool):
    """Return an absolute snapshot dir for an already-cached model, so vLLM loads
    read-only and never tries to write into another user's cache subdir. Falls
    back to the repo id (triggering a download) only when not cached."""
    cache_name = "models--" + repo_id.replace("/", "--")
    snaps = sorted(glob.glob(os.path.join(hf_cache, cache_name, "snapshots", "*")))
    for snap in snaps:
        if glob.glob(os.path.join(snap, "config.json")):
            return snap
    if not allow_download:
        print(f"[warn] {repo_id} not found in cache; will let HF resolve/download")
    return repo_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="+", default=["triplet", "pairwise", "feature"])
    ap.add_argument("--feature_sample", type=int, default=0,
                    help="use only first N features (0 = all in the file)")
    ap.add_argument("--feature_file", default="features.csv",
                    help="feature list under data/stimuli/ "
                         "(e.g. features_discriminative.csv)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--tensor_parallel", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    if args.model not in reg["local"]:
        sys.exit(f"model {args.model} not in registry local: {list(reg['local'])}")
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)

    allow_download = bool(spec.get("download", False))
    model_path = resolve_model_path(spec["path"], hf_cache, allow_download)
    is_local_snapshot = os.path.isdir(model_path)
    if is_local_snapshot:
        # Loading straight from the snapshot dir: force offline so HF never tries
        # to write refs/blobs into a cache subdir we may not own.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    print(f"[model] {args.model} -> {model_path} (local_snapshot={is_local_snapshot})")

    outdir = os.path.join(RAW, args.model)
    os.makedirs(outdir, exist_ok=True)

    # Decide what still needs running before loading the (expensive) model.
    todo = []
    for m in args.methods:
        out = os.path.join(outdir, f"{m}.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        todo.append(m)
    if not todo:
        print("nothing to do")
        return

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        download_dir=hf_cache,
        tensor_parallel_size=args.tensor_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        trust_remote_code=True,
    )

    for method in todo:
        jobs = build_jobs(method, feature_sample=args.feature_sample,
                          feature_file=args.feature_file)
        prompts = [p for _, p in jobs]
        sp = SamplingParams(temperature=args.temperature, max_tokens=MAX_TOKENS[method])

        if spec.get("chat", True):
            convos = [
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": p}]
                for p in prompts
            ]
            outputs = llm.chat(convos, sp)
        else:
            outputs = llm.generate(prompts, sp)

        out = os.path.join(outdir, f"{method}.csv")
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["input", "prompt", "response"])
            for (inp, prompt), o in zip(jobs, outputs):
                resp = o.outputs[0].text.strip()
                w.writerow(["|".join(inp), prompt, resp])
        print(f"[done] {method}: {len(jobs)} rows -> {out}")


if __name__ == "__main__":
    main()
