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
    """Paper's REPORTED metric = squared Procrustes correlation r^2 = 1 - disparity
    (= 1 - protest ss = "% of variance explained"). This is what the EMNLP paper
    text/heatmap reports (0.96/0.84/0.72). disparity = scipy symmetric Procrustes m12^2.
    X, Y are NxN RDMs aligned DIRECTLY (no MDS): each concept = its row of the RDM."""
    from scipy.spatial import procrustes as _sp
    if X.shape != Y.shape:
        return float("nan")
    try:
        _, _, disp = _sp(X, Y)
    except Exception:
        return float("nan")
    return float(max(0.0, 1.0 - disp))


def method_rdms(model):
    """Return {method: NxN distance matrix (RDM)} for a model's methods."""
    concepts = A.load_concepts()
    rdms = {}
    for m in METHODS:
        sim = A.METHOD_FN[m](model, concepts)
        if sim is None:
            continue
        d = 1.0 - sim
        d = (d + d.T) / 2.0
        np.fill_diagonal(d, 0.0)
        rdms[m] = d
    return rdms


def discover_models():
    return sorted(d for d in os.listdir(RAW)
                  if os.path.isdir(os.path.join(RAW, d)) and not d.startswith("_")
                  and os.path.exists(os.path.join(RAW, d, "triplet.csv")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()

    # Human RDMs (paper's data), built once. Each method -> NxN distance matrix.
    hp = os.path.join(HERE, "data", "human")
    human_rdm = {}
    for meth, f in [("triplet", "paper_human_triplet_similarity.npy"),
                    ("feature", "paper_human_feature_similarity.npy"),
                    ("pairwise", "paper_human_pairwise_similarity_mat.npy")]:
        p = os.path.join(hp, f)
        if os.path.exists(p):
            sim = np.load(p)
            d = 1.0 - sim; d = (d + d.T) / 2.0; np.fill_diagonal(d, 0.0)
            human_rdm[meth] = d

    models = args.models or discover_models()
    rows = []
    for model in models:
        rdms = method_rdms(model)
        if "triplet" not in rdms:
            continue
        row = {"model": model, "n_methods": len(rdms)}

        # (1) coherence = mean pairwise Procrustes(RDM) among the model's methods
        keys = list(rdms)
        pair_r2 = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                pair_r2[f"proc_{keys[i]}~{keys[j]}"] = procrustes_r2(rdms[keys[i]], rdms[keys[j]])
        row.update(pair_r2)
        row["proc_coherence_mean"] = (float(np.nanmean(list(pair_r2.values())))
                                      if pair_r2 else np.nan)

        # (2) human prediction = Procrustes(model RDM, human RDM) per method
        for m, dmat in rdms.items():
            if m in human_rdm and dmat.shape == human_rdm[m].shape:
                row[f"human_proc_{m}"] = procrustes_r2(dmat, human_rdm[m])
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
