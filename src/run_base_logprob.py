"""Base-model-appropriate scoring via LIKELIHOOD (no instruction-following needed).

Pretrained-only checkpoints can't follow "answer with one word" prompts, so instead
of generating we SCORE completions and pick the higher-likelihood one. This reads the
base model's representations in its native next-token regime.

triplet : for (anchor, c1, c2), compare logprob of
            "{anchor} is more similar to {c1}" vs "... {c2}"; the winner is the choice.
          Output matches the generated-triplet schema (response = chosen concept), so
          downstream analysis.triplet_similarity is unchanged.
pairwise: for (a, b), score the 1..7 rating templates and take the argmax rating.

Uses vLLM prompt_logprobs to get the summed log-prob of the completion tokens.

Usage:
  python run_base_logprob.py --model olmo2-7b-base --methods triplet pairwise
"""
import argparse
import csv
import glob
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stimuli import _read_rows, load_concepts  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
STIM = os.path.join(HERE, "data", "stimuli")


def load_registry():
    with open(os.path.join(HERE, "configs", "models.yaml")) as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id, hf_cache):
    cache = "models--" + repo_id.replace("/", "--")
    for snap in sorted(glob.glob(os.path.join(hf_cache, cache, "snapshots", "*"))):
        if glob.glob(os.path.join(snap, "config.json")):
            return snap
    return repo_id


def _mean_logprob(llm, prompt, completion):
    """Mean per-token log-prob of `completion` conditioned on `prompt`.

    Tokenize prompt and prompt+completion separately to get the exact completion
    token span, then read those tokens' logprobs from a prompt_logprobs pass.
    Returns -inf if the span is empty (so it can never spuriously win an argmax).
    Length-normalized (mean) so completions of different token counts compare fairly.
    """
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    # Find the completion token span by the longest common PREFIX of the two token
    # sequences (robust to boundary merges from a trailing space in the prompt).
    p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
    full_ids = tok(prompt + completion, add_special_tokens=True)["input_ids"]
    start = 0
    for a, b in zip(p_ids, full_ids):
        if a == b:
            start += 1
        else:
            break
    if len(full_ids) <= start:
        return float("-inf")
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    out = llm.generate([prompt + completion], sp)[0]
    pls = out.prompt_logprobs
    lps = []
    for i in range(start, len(full_ids)):
        if i < len(pls) and pls[i]:
            lps.append(next(iter(pls[i].values())).logprob)
    if not lps:
        return float("-inf")
    return sum(lps) / len(lps)


def seq_logprob(llm, prompt, completion):
    return _mean_logprob(llm, prompt, completion)


def batch_mean_logprob(llm, pairs):
    """Vectorized: pairs = list of (prompt, completion). Returns list of mean
    completion logprobs, computed in ONE vLLM pass (much faster than per-call)."""
    from vllm import SamplingParams
    tok = llm.get_tokenizer()
    starts = []
    fulls = []
    for prompt, completion in pairs:
        p_ids = tok(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tok(prompt + completion, add_special_tokens=True)["input_ids"]
        s = 0
        for a, b in zip(p_ids, full_ids):
            if a == b:
                s += 1
            else:
                break
        starts.append((s, len(full_ids)))
        fulls.append(prompt + completion)
    sp = SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
    outs = llm.generate(fulls, sp)
    res = []
    for (s, n), o in zip(starts, outs):
        pls = o.prompt_logprobs
        lps = [next(iter(pls[i].values())).logprob
               for i in range(s, n) if i < len(pls) and pls[i]]
        res.append(sum(lps) / len(lps) if lps else float("-inf"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="+", default=["triplet", "pairwise"])
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--suffix", default="",
                    help="output filename suffix, e.g. '_lp' -> triplet_lp.csv "
                         "(keeps generation-based files intact)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    mp = resolve_model_path(spec["path"], hf_cache)
    if os.path.isdir(mp):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    outdir = os.path.join(RAW, args.model)
    os.makedirs(outdir, exist_ok=True)

    from vllm import LLM
    llm = LLM(model=mp, download_dir=hf_cache, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util, dtype="bfloat16",
              trust_remote_code=True)

    if "triplet" in args.methods:
        out = os.path.join(outdir, f"triplet{args.suffix}.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out}")
        else:
            rows = _read_rows(os.path.join(STIM, "triplets.csv"))
            pairs = []
            for anchor, c1, c2 in rows:
                pairs.append((f"{anchor} is more similar to", " " + c1))
                pairs.append((f"{anchor} is more similar to", " " + c2))
            lps = batch_mean_logprob(llm, pairs)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "prompt", "response"])
                for k, (anchor, c1, c2) in enumerate(rows):
                    l1, l2 = lps[2 * k], lps[2 * k + 1]
                    w.writerow([f"{anchor}|{c1}|{c2}", "logprob", c1 if l1 >= l2 else c2])
            print(f"[done] triplet (logprob): {len(rows)} rows -> {out}")

    if "pairwise" in args.methods:
        out = os.path.join(outdir, f"pairwise{args.suffix}.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out}")
        else:
            rows = _read_rows(os.path.join(STIM, "pairs.csv"))
            pairs = []
            for a, b in rows:
                for r in range(1, 8):
                    pairs.append(
                        (f"On a scale of 1 to 7, the similarity of {a} and {b} is",
                         " " + str(r)))
            lps = batch_mean_logprob(llm, pairs)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "prompt", "response"])
                for k, (a, b) in enumerate(rows):
                    seg = lps[7 * k:7 * k + 7]
                    best_r = 1 + max(range(7), key=lambda i: seg[i])
                    w.writerow([f"{a}|{b}", "logprob", str(best_r)])
            print(f"[done] pairwise (logprob): {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
