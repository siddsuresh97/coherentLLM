"""Diagnose the feature-verification method and select a usable feature subset.

Problem: pairing all 764 (or 200) features x 30 concepts makes ~95% of pairs
trivially False, so the feature RDM is sparse noise. This tool:

1. Reports, per model, the True-rate and per-feature variance of responses.
2. Selects a DISCRIMINATIVE feature subset: features whose True/False pattern
   varies across the 30 concepts (i.e. informative), using model responses that
   already exist. This is a data-driven stand-in until a human concept x feature
   ground-truth key is supplied.

Writes data/stimuli/features_discriminative.csv (the selected feature names).
"""
import argparse
import os
import numpy as np
import pandas as pd

import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
STIM = os.path.join(HERE, "data", "stimuli")


def concept_feature_matrix(model, concepts):
    """Return DataFrame (concepts x features) of 0/1 from a model's feature.csv."""
    path = os.path.join(RAW, model, "feature.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    idx = {A._norm(c): i for i, c in enumerate(concepts)}
    feats = {}
    for _, row in df.iterrows():
        try:
            feat, concept = str(row["input"]).split("|")
        except ValueError:
            continue
        ci = idx.get(A._norm(concept))
        if ci is None:
            continue
        v = A._parse_truefalse(row["response"])
        if v is not None:
            feats.setdefault(feat, {})[ci] = v
    names = sorted(feats)
    M = np.zeros((len(concepts), len(names)))
    for j, fn in enumerate(names):
        for ci, v in feats[fn].items():
            M[ci, j] = v
    return pd.DataFrame(M, index=concepts, columns=names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--min_true", type=int, default=2,
                    help="feature must be True for >= this many concepts (in the "
                         "consensus) to be kept")
    ap.add_argument("--max_true", type=int, default=28,
                    help="...and True for <= this many (drop all-True features)")
    args = ap.parse_args()

    concepts = A.load_concepts()
    models = args.models or sorted(
        d for d in os.listdir(RAW)
        if os.path.isdir(os.path.join(RAW, d)) and not d.startswith("_")
        and os.path.exists(os.path.join(RAW, d, "feature.csv")))
    if not models:
        print("No models with feature.csv found.")
        return

    mats = {}
    for m in models:
        M = concept_feature_matrix(m, concepts)
        if M is None:
            continue
        mats[m] = M
        tr = M.values.mean()
        # per-feature variance across concepts, averaged
        var = M.var(axis=0).mean()
        print(f"[{m}] features={M.shape[1]} true_rate={tr:.3f} mean_feature_var={var:.3f}")

    # consensus across models (majority vote per cell), on shared features
    common = set.intersection(*[set(M.columns) for M in mats.values()])
    common = sorted(common)
    stack = np.stack([mats[m][common].values for m in mats])  # (models, concepts, feats)
    consensus = (stack.mean(axis=0) >= 0.5).astype(int)  # concepts x feats
    true_per_feat = consensus.sum(axis=0)
    keep = [common[j] for j in range(len(common))
            if args.min_true <= true_per_feat[j] <= args.max_true]
    out = os.path.join(STIM, "features_discriminative.csv")
    pd.Series(keep).to_csv(out, index=False, header=False)
    print(f"\nDiscriminative features kept: {len(keep)} / {len(common)} "
          f"(True for {args.min_true}..{args.max_true} of 30 concepts by consensus)")
    print(f"Wrote {out}")
    print("Examples:", keep[:12])


if __name__ == "__main__":
    main()
