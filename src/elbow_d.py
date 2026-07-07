"""Empirical dimensionality elbow: fit SALMON at several d on one model's 128-concept
triplets, report held-out accuracy vs d. Pick the d where accuracy plateaus.

Run in the salmon env with PYTHONNOUSERSITE=1 from repo root.
  python src/elbow_d.py qwen2.5-32b-instruct
"""
import os
import sys
import types
import importlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
for pkg in ["salmon", "salmon.triplets"]:
    if pkg not in sys.modules:
        m = types.ModuleType(pkg); m.__path__ = [os.path.join(_REPO, *pkg.split("."))]
        sys.modules[pkg] = m
OfflineEmbedding = importlib.import_module("salmon.triplets.offline").OfflineEmbedding

RAW128 = os.path.join(_REPO, "results", "raw_128")
CONCEPTS = [l.strip() for l in open(os.path.join(_REPO, "data/scale128/concepts.csv"))]
OUT = os.path.join(_REPO, "data/scale128")


def _norm(s):
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def triplets(model):
    df = pd.read_csv(os.path.join(RAW128, model, "triplet.csv"))
    idx = {_norm(c): i for i, c in enumerate(CONCEPTS)}
    rows = []
    for _, r in df.iterrows():
        try:
            a, c1, c2 = str(r["input"]).split("|")
        except ValueError:
            continue
        ai, i1, i2 = idx.get(_norm(a)), idx.get(_norm(c1)), idx.get(_norm(c2))
        if None in (ai, i1, i2):
            continue
        resp = _norm(r["response"])
        if _norm(c1) and _norm(c1) in resp:
            w, l = i1, i2
        elif _norm(c2) and _norm(c2) in resp:
            w, l = i2, i1
        else:
            continue
        rows.append((ai, w, l))
    return np.array(rows, dtype=int)


def main():
    model = sys.argv[1]
    ds = [int(x) for x in (sys.argv[2].split(",") if len(sys.argv) > 2 else "3,5,7,10,15".split(","))]
    X = triplets(model)
    n = len(CONCEPTS)
    Xtr, Xte = train_test_split(X, random_state=42, test_size=0.2)
    print(f"{model}: {len(X)} valid triplets, n={n} concepts")
    res = []
    for d in ds:
        em = OfflineEmbedding(n=n, d=d, max_epochs=8000, verbose=100000)
        em.fit(Xtr, Xte)
        acc = float(em.score(Xte)) if hasattr(em, "score") else float("nan")
        emb = np.asarray(em.embedding_)
        np.save(os.path.join(OUT, f"{model}_triplet_d{d}.npy"), emb)
        res.append({"d": d, "held_out_acc": acc})
        print(f"  d={d:2}: held-out acc = {acc:.3f}", flush=True)
    pd.DataFrame(res).to_csv(os.path.join(OUT, f"elbow_{model}.csv"), index=False)
    print("wrote elbow table")


if __name__ == "__main__":
    main()
