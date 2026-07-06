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
MAX_TOKENS = {"triplet": 16, "pairwise": 16, "feature": 16, "listing": 400}


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
            if args.reasoning_effort in (None, "", "none", "off"):
                # Disable reasoning: cheaper AND avoids reasoning eating the whole
                # token budget (which returned empty/unparseable answers). Also more
                # comparable to humans, who answer these intuitively.
                kwargs["extra_body"] = {"reasoning": {"enabled": False}}
            else:
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

    done = {"n": 0, "fail": 0, "abort": False}

    async def one_logged(prompt, method, total):
        # Never let one bad call abort the whole gather: on terminal failure return
        # "" so partial results are still written. Detect out-of-credits and stop
        # issuing new calls (they would all fail).
        if done["abort"]:
            return ""
        try:
            r = await one(prompt, method)
        except Exception as e:
            done["fail"] += 1
            msg = str(e).lower()
            if "402" in msg or "insufficient credit" in msg or "quota" in msg:
                if not done["abort"]:
                    print(f"[{method}] ABORT: out of credits/quota after "
                          f"{done['n']} ok, {done['fail']} failed", flush=True)
                done["abort"] = True
            r = ""
        done["n"] += 1
        if done["n"] % 100 == 0 or done["n"] == total:
            print(f"[{method}] {done['n']}/{total} (fail={done['fail']})", flush=True)
        return r

    for method in args.methods:
        out = os.path.join(outdir, f"{method}.csv")
        if os.path.exists(out) and not args.overwrite:
            print(f"[skip] {out} exists")
            continue

        if method == "listing":
            from prompts import listing_prompt
            from stimuli import load_concepts
            concepts = load_concepts()
            jobs = [(c, r) for c in concepts for r in range(args.repeats)]
            done["n"] = 0
            total = len(jobs)
            print(f"[listing] starting {total} calls", flush=True)
            tasks = [one_logged(listing_prompt(c), "listing", total) for c, _ in jobs]
            responses = await asyncio.gather(*tasks)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["concept", "rep", "prompt", "response"])
                for (c, r), resp in zip(jobs, responses):
                    w.writerow([c, r, listing_prompt(c), resp])
            print(f"[done] listing: {total} rows -> {out}")
            continue

        if method == "feature" and args.pairs_file and not args.feature_batch:
            # SINGLE-PAIR self-verification (validated correct; batching over-affirms).
            from prompts import feature_prompt
            from stimuli import _read_rows
            pairs = _read_rows(os.path.join(outdir, args.pairs_file))
            done["n"] = 0
            total = len(pairs)
            print(f"[feature] starting {total} single-pair calls", flush=True)
            tasks = [one_logged(feature_prompt(f0, c0), "feature", total)
                     for f0, c0 in pairs]
            responses = await asyncio.gather(*tasks)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "prompt", "response"])
                for (f0, c0), resp in zip(pairs, responses):
                    w.writerow([f"{f0}|{c0}", feature_prompt(f0, c0), resp])
            print(f"[done] feature (single-pair): {total} rows -> {out}")
            continue

        if method == "feature" and args.pairs_file:
            from feature_batch import BATCH_INSTRUCTIONS, parse_batch
            from stimuli import _read_rows
            import collections
            by_concept = collections.OrderedDict()
            for feat, concept in _read_rows(os.path.join(outdir, args.pairs_file)):
                by_concept.setdefault(concept, []).append(feat)
            blk = args.feature_batch or 20
            batches = []
            for concept, feats in by_concept.items():
                for i in range(0, len(feats), blk):
                    chunk = feats[i:i + blk]
                    items = "\n".join(f"{j+1}. {f}" for j, f in enumerate(chunk))
                    batches.append((concept, chunk,
                                    BATCH_INSTRUCTIONS.format(concept=concept, items=items)))
            done["n"] = 0
            total = len(batches)
            print(f"[feature] starting {total} self-verify batched calls", flush=True)
            tasks = [one_logged(p, "feature", total) for _, _, p in batches]
            responses = await asyncio.gather(*tasks)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "prompt", "response"])
                n = 0
                for (concept, chunk, prompt), resp in zip(batches, responses):
                    for feat, val in parse_batch(resp, chunk):
                        w.writerow([f"{feat}|{concept}", prompt,
                                    "True" if val else "False"])
                        n += 1
            print(f"[done] feature (self-verify): {total} calls -> {n} rows -> {out}")
            continue

        if method == "feature" and args.feature_batch:
            from feature_batch import make_batches, parse_batch
            from stimuli import load_concepts, _read_rows
            concepts = load_concepts()
            feats = [r[0] for r in _read_rows(os.path.join(
                HERE, "data", "stimuli", args.feature_file))]
            if args.feature_sample:
                feats = feats[:args.feature_sample]
            batches = list(make_batches(feats, concepts, block=args.feature_batch))
            done["n"] = 0
            total = len(batches)
            print(f"[feature] starting {total} batched calls", flush=True)
            tasks = [one_logged(p, "feature", total) for _, _, p in batches]
            responses = await asyncio.gather(*tasks)
            with open(out, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["input", "prompt", "response"])
                n = 0
                for (concept, chunk, prompt), resp in zip(batches, responses):
                    for feat, val in parse_batch(resp, chunk):
                        w.writerow([f"{feat}|{concept}", prompt,
                                    "True" if val else "False"])
                        n += 1
            print(f"[done] feature (batched x{args.feature_batch}): "
                  f"{total} calls -> {n} rows -> {out}")
            continue

        jobs = build_jobs(method, feature_sample=args.feature_sample,
                          feature_file=args.feature_file)
        done["n"] = 0
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
    ap.add_argument("--feature_file", default="features.csv",
                    help="feature list under data/stimuli/ "
                         "(e.g. features_leuven300.csv)")
    ap.add_argument("--feature_batch", type=int, default=0,
                    help="if >0, batch this many features per call for the feature method")
    ap.add_argument("--pairs_file", default=None,
                    help="per-model (feature,concept) pairs file under the model's raw dir")
    ap.add_argument("--repeats", type=int, default=5,
                    help="listing repeats per concept")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--reasoning_effort", default="none",
                    help="reasoning effort for reasoning models (low/medium/high); "
                         "empty string to omit")
    ap.add_argument("--key_file", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
