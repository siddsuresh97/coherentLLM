"""NOVA-recipe feature verification with flan-t5-xxl (local, free).

Verifies every (concept, feature) pair from data/leuven_reverify/pairs.csv with the
NOVA verification prompt (few-shot True/False, "highly confident" instruction), matching
llm-norms/scripts/generate_norms.py. Writes True/False per pair to
data/leuven_reverify/flan_xxl_verdicts.csv.
"""
import os
import sys
import pandas as pd
import torch
from transformers import T5TokenizerFast, T5ForConditionalGeneration

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# local snapshot has the weights but not the tokenizer; load tokenizer from hub id.
MODEL = ("/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/flan-t5-xxl/"
         "models--google--flan-t5-xxl/snapshots/"
         "ae7c9136adc7555eeccc78cdd960dfd60fb346ce")
TOKENIZER = "google/flan-t5-xxl"
PAIRS = os.path.join(HERE, "data", "leuven_reverify", "pairs.csv")
OUT = os.path.join(HERE, "data", "leuven_reverify", "flan_xxl_verdicts.csv")

# NOVA few-shot verification prompt (generate_norms.py)
FEWSHOT = ("Q: Is the property [is female] true for the concept [book]? \n A: False \n "
           "Q: Is the property [can be digital] true for the concept [book] \n A: True \n "
           "In one word True/False, answer the following question "
           "Q: Is the property [{feat}] true for [{concept}]? \n A:")


def build_prompt(concept, feat):
    return FEWSHOT.format(feat=feat, concept=concept)


def main():
    bs = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    df = pd.read_csv(PAIRS)
    # resume: skip pairs already done
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.concept, prev.feature))
        print(f"resuming: {len(done):,} already verified")
    todo = df[~df.apply(lambda r: (r.concept, r.feature) in done, axis=1)].reset_index(drop=True)
    print(f"to verify: {len(todo):,} pairs, batch={bs}")

    tok = T5TokenizerFast.from_pretrained(TOKENIZER)
    # fp16 fits comfortably on the 80GB H100.
    model = T5ForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, device_map="cuda")
    model.eval()

    first = not os.path.exists(OUT)
    buf = []
    for start in range(0, len(todo), bs):
        chunk = todo.iloc[start:start + bs]
        prompts = [build_prompt(r.concept, r.feature) for r in chunk.itertuples()]
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True,
                  max_length=256).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=3, do_sample=False)
        dec = tok.batch_decode(out, skip_special_tokens=True)
        for r, d in zip(chunk.itertuples(), dec):
            v = 1 if "true" in d.lower() else 0
            buf.append({"concept": r.concept, "feature": r.feature,
                        "feature_raw": r.feature_raw, "leuven_listed": r.leuven_listed,
                        "flan_response": d.strip(), "flan_true": v})
        if len(buf) >= 2000:
            pd.DataFrame(buf).to_csv(OUT, mode="a", header=first, index=False)
            first = False
            buf = []
            print(f"  {start + bs:,}/{len(todo):,} done", flush=True)
    if buf:
        pd.DataFrame(buf).to_csv(OUT, mode="a", header=first, index=False)
    total = pd.read_csv(OUT)
    print(f"\nDONE: {len(total):,} verdicts, flan True-rate = {total.flan_true.mean():.3f}")


if __name__ == "__main__":
    main()
