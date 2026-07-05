"""Per-model 3x3 method-agreement (coherence) matrix across the three methods:
feature verification, pairwise similarity, triplet task.

For each model we build the 30x30 concept RDM from each method, then fill a 3x3
matrix whose (i,j) cell is the Spearman RSA between method i's and method j's RDM
(diagonal = 1). This is the "coherence matrix" of the model: how consistent the
semantic structure is across the three elicitation methods.

Outputs, for the discovered models under results/raw/:
  results/coherence/method_matrix_<model>.csv   (3x3 per model)
  results/coherence/method_matrices.png         (grid of all models)
  results/coherence/method_agreement_long.csv   (tidy: model, method_i, method_j, rsa)
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
LABELS = {"triplet": "triplet", "pairwise": "pairwise", "feature": "feature\nverif."}


def discover_models():
    if not os.path.isdir(RAW):
        return []
    return sorted(d for d in os.listdir(RAW)
                  if os.path.isdir(os.path.join(RAW, d)) and not d.startswith("_"))


def model_method_matrix(model, concepts):
    """Return (3x3 matrix, present_methods) of pairwise RSA between method RDMs."""
    rdms = {}
    for m in METHODS:
        sim = A.METHOD_FN[m](model, concepts)
        if sim is not None:
            rdms[m] = sim
    M = np.full((len(METHODS), len(METHODS)), np.nan)
    for i, mi in enumerate(METHODS):
        for j, mj in enumerate(METHODS):
            if mi in rdms and mj in rdms:
                M[i, j] = 1.0 if i == j else A.rsa(rdms[mi], rdms[mj])
    return M, list(rdms)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    concepts = A.load_concepts()
    models = args.models or discover_models()
    if not models:
        print("No models under results/raw/.")
        return

    mats = {}
    long_rows = []
    for model in models:
        M, present = model_method_matrix(model, concepts)
        mats[model] = M
        pd.DataFrame(M, index=METHODS, columns=METHODS).to_csv(
            os.path.join(OUT, f"method_matrix_{model}.csv"))
        for i, mi in enumerate(METHODS):
            for j, mj in enumerate(METHODS):
                if not np.isnan(M[i, j]):
                    long_rows.append({"model": model, "method_i": mi,
                                      "method_j": mj, "rsa": M[i, j]})
        offdiag = [M[0, 1], M[0, 2], M[1, 2]]
        print(f"[{model}] methods={present}  "
              f"trip~pair={M[0,1]:.3f} trip~feat={M[0,2]:.3f} pair~feat={M[1,2]:.3f}")

    pd.DataFrame(long_rows).to_csv(
        os.path.join(OUT, "method_agreement_long.csv"), index=False)

    # grid of heatmaps, one 3x3 per model
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        k = len(models)
        cols = min(4, k)
        rows = (k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows),
                                 squeeze=False)
        labs = [LABELS[m] for m in METHODS]
        for ax in axes.flat:
            ax.axis("off")
        for idx, model in enumerate(models):
            ax = axes[idx // cols][idx % cols]
            ax.axis("on")
            M = mats[model]
            im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
            ax.set_xticks(range(3)); ax.set_xticklabels(labs, fontsize=8)
            ax.set_yticks(range(3)); ax.set_yticklabels(labs, fontsize=8)
            ax.set_title(model, fontsize=9)
            for (i, j), v in np.ndenumerate(M):
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="w" if v < 0.6 else "k", fontsize=9)
        fig.suptitle("Per-model method-agreement (RSA) across the three methods",
                     fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(os.path.join(OUT, "method_matrices.png"), dpi=150)
        print("Wrote method_matrices.png")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
