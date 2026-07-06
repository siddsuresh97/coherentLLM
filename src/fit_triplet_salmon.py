"""Fit model triplet embeddings with SALMON (OfflineEmbedding), the paper's method.

Paper recipe (conceptual_representations_gpt triplet_cleaning.ipynb):
  - reformat each judgment to (head, winner, loser): head=anchor, winner=chosen, loser=other
  - map concepts -> ints, train_test_split(random_state=42, test_size=0.2)
  - OfflineEmbedding(n=n, d=3, max_epochs=8000).fit(X_train, X_test)
  - embedding_ is n x 3; save per model as data/triplet_embeddings/<model>.npy (concepts.csv order)

Run in the salmon env with PYTHONNOUSERSITE=1 from the repo root (needs local ./salmon).
"""
import os
import sys
import types
import importlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import OfflineEmbedding WITHOUT triggering salmon/__init__.py's web-server backend
# (which needs dask/starlette_prometheus we don't have). We only need the pure-torch
# offline fitter, which depends solely on salmon.triplets.samplers.adaptive.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)
# pre-register empty package shells so submodule imports don't run the real __init__
for pkg in ["salmon", "salmon.triplets"]:
    if pkg not in sys.modules:
        m = types.ModuleType(pkg)
        m.__path__ = [os.path.join(_REPO, *pkg.split("."))]
        sys.modules[pkg] = m
OfflineEmbedding = importlib.import_module("salmon.triplets.offline").OfflineEmbedding

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
OUT = os.path.join(HERE, "data", "triplet_embeddings")
os.makedirs(OUT, exist_ok=True)


def _norm(s):
    import re
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_concepts():
    with open(os.path.join(HERE, "data", "stimuli", "concepts.csv")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def triplets_for_model(model, concepts):
    """Return (head, winner, loser) int arrays from a model's triplet.csv."""
    df = pd.read_csv(os.path.join(RAW, model, "triplet.csv"))
    idx = {_norm(c): i for i, c in enumerate(concepts)}
    rows = []
    for _, r in df.iterrows():
        try:
            anchor, c1, c2 = str(r["input"]).split("|")
        except ValueError:
            continue
        a, i1, i2 = idx.get(_norm(anchor)), idx.get(_norm(c1)), idx.get(_norm(c2))
        if None in (a, i1, i2):
            continue
        resp = _norm(r["response"])
        if _norm(c1) and _norm(c1) in resp:
            win, los = i1, i2
        elif _norm(c2) and _norm(c2) in resp:
            win, los = i2, i1
        else:
            continue
        rows.append((a, win, los))
    return np.array(rows, dtype=int)


def fit(model, concepts, d=3, max_epochs=8000):
    X = triplets_for_model(model, concepts)
    n = len(concepts)
    X_train, X_test = train_test_split(X, random_state=42, test_size=0.2)
    em = OfflineEmbedding(n=n, d=d, max_epochs=max_epochs, verbose=1000)
    em.fit(X_train, X_test)
    emb = np.asarray(em.embedding_)          # n x d, in 0..n-1 concept-index order
    np.save(os.path.join(OUT, f"{model}.npy"), emb)
    print(f"[salmon] {model}: fit {X.shape[0]} triplets -> {emb.shape} embedding")
    return emb


if __name__ == "__main__":
    concepts = load_concepts()
    models = sys.argv[1:] or []
    for m in models:
        try:
            fit(m, concepts)
        except Exception as e:
            print(f"[salmon] {m} FAILED: {e}")
