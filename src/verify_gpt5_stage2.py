"""NOVA stage-2: gpt-5.5 re-verifies the pairs flan-xxl marked True (prune false positives).

Reads data/leuven_reverify/flan_xxl_verdicts.csv, takes flan_true==1 pairs, asks gpt-5.5
the NOVA verification question, writes True/False to gpt5_stage2_verdicts.csv.
Concurrent, resumable, saves partial, aborts on 402 (insufficient credit). Reasoning OFF.
"""
import os
import sys
import json
import time
import concurrent.futures as cf
import pandas as pd
import requests

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(HERE, "data", "leuven_reverify", "flan_xxl_verdicts.csv")
OUT = os.path.join(HERE, "data", "leuven_reverify", "gpt5_stage2_verdicts.csv")
API_MODEL = "openai/gpt-5.5-20260423"
URL = "https://openrouter.ai/api/v1/chat/completions"

PROMPT = ("Q: Is the property [is female] true for the concept [book]? \n A: False \n "
          "Q: Is the property [can be digital] true for the concept [book] \n A: True \n "
          "In one word True/False, answer the following question "
          "Q: Is the property [{feat}] true for [{concept}]? \n A:")


def key():
    k = os.environ.get("OPENROUTER_API_KEY")
    if not k:
        with open(os.path.join(HERE, ".openrouter_key")) as f:
            k = f.read().strip()
    return k


def one(sess, headers, concept, feat):
    body = {"model": API_MODEL,
            "messages": [{"role": "user", "content": PROMPT.format(feat=feat, concept=concept)}],
            "max_tokens": 16, "temperature": 0,
            "reasoning": {"enabled": False}}
    for attempt in range(4):
        try:
            r = sess.post(URL, headers=headers, json=body, timeout=60)
            if r.status_code == 402:
                return "__402__"
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"].strip()
            return txt
        except Exception:
            time.sleep(2 * (attempt + 1))
    return ""


def main():
    df = pd.read_csv(IN)
    todo = df[df.flan_true == 1][["concept", "feature", "feature_raw"]].reset_index(drop=True)
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.concept, prev.feature))
        print(f"resuming: {len(done)} done")
    todo = todo[~todo.apply(lambda r: (r.concept, r.feature) in done, axis=1)].reset_index(drop=True)
    print(f"gpt-5.5 stage-2: {len(todo)} flan-True pairs to re-verify")

    headers = {"Authorization": f"Bearer {key()}", "Content-Type": "application/json"}
    sess = requests.Session()
    rows = []
    aborted = False
    with cf.ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(one, sess, headers, r.concept, r.feature): r for r in todo.itertuples()}
        for i, fut in enumerate(cf.as_completed(futs)):
            r = futs[fut]
            resp = fut.result()
            if resp == "__402__":
                print("402 insufficient credit; saving partial and aborting")
                aborted = True
                break
            rows.append({"concept": r.concept, "feature": r.feature,
                         "feature_raw": r.feature_raw, "gpt5_response": resp,
                         "gpt5_true": 1 if "true" in resp.lower() else 0})
            if (i + 1) % 500 == 0:
                pd.DataFrame(rows).to_csv(OUT, mode="a",
                                          header=not os.path.exists(OUT), index=False)
                print(f"  {i+1}/{len(todo)} done", flush=True)
                rows = []
    if rows:
        pd.DataFrame(rows).to_csv(OUT, mode="a", header=not os.path.exists(OUT), index=False)
    total = pd.read_csv(OUT)
    print(f"\n{'ABORTED' if aborted else 'DONE'}: {len(total)} gpt-5.5 verdicts, "
          f"True-rate {total.gpt5_true.mean():.3f}")


if __name__ == "__main__":
    main()
