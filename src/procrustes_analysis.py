"""Procrustes analyses (matching the arXiv paper's primary metric).

Two questions:
  (1) COHERENCE = average pairwise Procrustes R^2 among a model's three method
      embeddings (triplet, pairwise, feature). "How geometrically consistent is the
      model across its own elicitation methods?"
  (2) HUMAN PREDICTION = Procrustes R^2 aligning each model-method embedding to the
      HUMAN embedding, reported per method (triplet / pairwise / feature), like the
      paper predicting each human elicitation method.

Procrustes R^2 here = 1 - SS_res / SS_tot after the optimal orthogonal (rotation +
reflection) + scaling + translation fit of X onto Y (Y = target). Both configs are
column-matched to the same dimensionality and mean-centered.
"""
import argparse
import os
import numpy as np
import pandas as pd

import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
OUT = os.path.join(HERE, "results", "coherence")
METHODS = ["triplet", "pairwise", "feature"]


def procrustes_r2(X, Y):
    """R^2 of fitting X onto Y with orthogonal rotation+reflection, scale, translation.
    Symmetric-normalized (both scaled to unit norm), so R^2 in ~[0,1]."""
    d = min(X.shape[1], Y.shape[1])
    X = X[:, :d].astype(float).copy()
    Y = Y[:, :d].astype(float).copy()
    # center
    X -= X.mean(0)
    Y -= Y.mean(0)
    # scale to unit Frobenius norm (removes arbitrary scale of each embedding)
    nx = np.linalg.norm(X)
    ny = np.linalg.norm(Y)
    if nx == 0 or ny == 0:
        return np.nan
    X /= nx
    Y /= ny
    # optimal rotation R minimizing ||Y - X R||: SVD of X^T Y
    U, S, Vt = np.linalg.svd(X.T @ Y)
    R = U @ Vt
    # optimal scale
    scale = S.sum()
    Xr = scale * (X @ R)
    ss_res = np.sum((Y - Xr) ** 2)
    ss_tot = np.sum(Y ** 2)  # = 1 after unit-norm
    return float(1.0 - ss_res / ss_tot)


def method_embeddings(model, dim=None):
    """Return {method: embedding} for a model's available methods."""
    concepts = A.load_concepts()
    embs = {}
    for m in METHODS:
        sim = A.METHOD_FN[m](model, concepts)
        if sim is None:
            continue
        k = dim or (len(concepts) - 1)
        embs[m] = A.embed_30d(sim, dim=k)
    return embs


def discover_models():
    return sorted(d for d in os.listdir(RAW)
                  if os.path.isdir(os.path.join(RAW, d)) and not d.startswith("_")
                  and os.path.exists(os.path.join(RAW, d, "triplet.csv")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--human_embedding",
                    default=os.path.join(HERE, "data", "human", "leuven_embedding.npy"))
    ap.add_argument("--dim", type=int, default=29)
    args = ap.parse_args()

    human = np.load(args.human_embedding)
    models = args.models or discover_models()
    rows = []
    for model in models:
        embs = method_embeddings(model, dim=args.dim)
        if "triplet" not in embs:
            continue
        row = {"model": model, "n_methods": len(embs)}

        # (1) coherence = mean pairwise Procrustes R^2 among the model's methods
        keys = list(embs)
        pair_r2 = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                r2 = procrustes_r2(embs[keys[i]], embs[keys[j]])
                pair_r2[f"proc_{keys[i]}~{keys[j]}"] = r2
        row.update(pair_r2)
        row["proc_coherence_mean"] = (float(np.nanmean(list(pair_r2.values())))
                                      if pair_r2 else np.nan)

        # (2) human prediction = Procrustes R^2 of each method embedding onto human
        for m, e in embs.items():
            row[f"human_proc_{m}"] = procrustes_r2(e, human)
        rows.append(row)
        print(f"[ok] {model}: coherence={row['proc_coherence_mean']:.3f} "
              f"human_triplet={row.get('human_proc_triplet', float('nan')):.3f}")

    df = pd.DataFrame(rows).set_index("model")
    if "proc_coherence_mean" in df:
        df = df.sort_values("proc_coherence_mean", ascending=False)
    csv = os.path.join(OUT, "procrustes.csv")
    df.to_csv(csv)
    cols = ["proc_coherence_mean", "human_proc_triplet", "human_proc_pairwise",
            "human_proc_feature"]
    print("\n" + df[[c for c in cols if c in df]].round(3).to_string())
    print(f"\nwrote {csv}")


if __name__ == "__main__":
    main()
