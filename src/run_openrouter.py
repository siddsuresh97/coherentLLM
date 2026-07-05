"""OpenRouter runner for frontier models (GPT-5, Claude Opus, ...).

Same prompt templates and output format as run_local.py.
Reads the API key from $OPENROUTER_API_KEY (or --key_file).

Usage:
  export OPENROUTER_API_KEY=sk-or-...
  python run_openrouter.py --model gpt-5 --methods triplet pairwise \
      --feature_sample 200 --concurrency 16
"""
import argparse
import asyncio
import csv
import os
import sys

import yaml
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from prompts import SYSTEM_PROMPT  # noqa: E402
from stimuli import build_jobs  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
# Reasoning models (GPT-5.x, o-series) spend output tokens on hidden reasoning
# before emitting the answer, so a tiny cap returns empty content. Give generous
# headroom; the answer is still parsed to a single word/number downstream.
MAX_TOKENS = {"triplet": 2048, "pairwise": 2048, "feature": 2048}


def load_registry():
    with open(os.path.join(HERE, "configs", "models.yaml")) as f:
        return yaml.safe_load(f)


def get_key(key_file):
    if key_file and os.path.exists(key_file):
        return open(key_file).read().strip()
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        sys.exit("Set OPENROUTER_API_KEY env var or pass --key_file")
    return k


async def run(args):
    reg = load_registry()
    if args.model not in reg["openrouter"]:
        sys.exit(f"{args.model} not in registry openrouter: {list(reg['openrouter'])}")
    api_model = reg["openrouter"][args.model]["api_model"]

    client = AsyncOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=get_key(args.key_file),
    )
    sem = asyncio.Semaphore(args.concurrency)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=1, min=2, max=30))
    async def one(prompt, method):
        async with sem:
            msgs = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}]
            # Some frontier models reject `temperature` and/or `max_tokens`
            # (they use `max_completion_tokens`). Degrade gracefully.
            kwargs = dict(model=api_model, messages=msgs)
            if args.temperature is not None:
                kwargs["temperature"] = args.temperature
            kwargs["max_tokens"] = MAX_TOKENS[method]
            if args.reasoning_effort:
                # OpenRouter passes this through to reasoning models; ignored by others.
                kwargs["extra_body"] = {"reasoning": {"effort": args.reasoning_effort}}
            try:
                r = await client.chat.completions.create(**kwargs)
            except Exception as e:
                emsg = str(e).lower()
                if "temperature" in emsg:
                    kwargs.pop("temperature", None)
                if "max_tokens" in emsg or "max_completion_tokens" in emsg:
                    kwargs.pop("max_tokens", None)
                    kwargs["max_completion_tokens"] = MAX_TOKENS[method] + 8
                r = await client.chat.completions.create(**kwargs)
            return (r.choices[0].message.content or "").strip()

    outdir = os.path.join(RAW, args.model)
    os.makedirs(outdir, exist_ok=True)

    for method in args.methods:
        out = os.path.join(outdir, f"{method}.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        jobs = build_jobs(method, feature_sample=args.feature_sample)
        done = {"n": 0}

        async def one_logged(prompt, method, total):
            r = await one(prompt, method)
            done["n"] += 1
            if done["n"] % 200 == 0 or done["n"] == total:
                print(f"[{method}] {done['n']}/{total}", flush=True)
            return r

        total = len(jobs)
        print(f"[{method}] starting {total} calls", flush=True)
        tasks = [one_logged(p, method, total) for _, p in jobs]
        responses = await asyncio.gather(*tasks)
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["input", "prompt", "response"])
            for (inp, prompt), resp in zip(jobs, responses):
                w.writerow(["|".join(inp), prompt, resp])
        print(f"[done] {method}: {len(jobs)} rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--methods", nargs="+", default=["triplet", "pairwise", "feature"])
    ap.add_argument("--feature_sample", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--reasoning_effort", default="low",
                    help="reasoning effort for reasoning models (low/medium/high); "
                         "empty string to omit")
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
