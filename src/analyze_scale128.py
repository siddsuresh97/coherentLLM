"""Compute n=128 human + model coherence (SALMON d=5 triplet, feature listing/verify).

Human refs: THINGS SPoSE (triplet, data/scale128/human_spose_triplet_sim.npy),
            NOVA verified matrix (feature, human_nova_feature_sim.npy).
Model triplet: SALMON d=5 embedding fit from results/raw_128/<model>/triplet.csv
               (saved by fit_triplet_salmon adapted, or *_triplet_d5.npy).
Model feature: verified concept x feature matrix (results/raw_128/<model>/feature.csv).

Metric: RDM-direct Procrustes r^2 (= 1 - disparity), matching the n=30 pipeline.
Reports triplet~feature coherence per model + model-vs-human alignment, and compares
to n=30. Run in coherence env.
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.spatial import procrustes as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw_128")
S128 = os.path.join(HERE, "data", "scale128")
CONCEPTS = [l.strip() for l in open(os.path.join(S128, "concepts.csv"))]


def _norm(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def r2(A, B):
    if A.shape != B.shape:
        return np.nan
    _, _, d = sp(A, B)
    return max(0.0, 1.0 - d)


def sym_rdm(sim):
    d = 1.0 - sim
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return d


def model_triplet_rdm(model, d=5):
    p = os.path.join(S128, f"{model}_triplet_d{d}.npy")
    if not os.path.exists(p):
        return None
    E = np.load(p)
    En = E / np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-9, None)
    return sym_rdm(En @ En.T)


def model_feature_rdm(model):
    """Cosine RDM from the model's verified concept x feature matrix."""
    p = os.path.join(RAW, model, "feature.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    idx = {_norm(c): i for i, c in enumerate(CONCEPTS)}
    feats = {}
    for _, r in df.iterrows():
        try:
            feat, concept = str(r["input"]).split("|")
        except ValueError:
            continue
        ci = idx.get(_norm(concept))
        if ci is None:
            continue
        v = 1 if "true" in str(r["response"]).lower() else 0
        feats.setdefault(feat, {})[ci] = v
    fn = sorted(feats)
    M = np.zeros((len(CONCEPTS), len(fn)))
    for j, f in enumerate(fn):
        for ci, v in feats[f].items():
            M[ci, j] = v
    Mn = M / np.clip(np.linalg.norm(M, axis=1, keepdims=True), 1e-9, None)
    return sym_rdm(Mn @ Mn.T)


def main():
    Dh_t = sym_rdm(np.load(os.path.join(S128, "human_spose_triplet_sim.npy")))
    Dh_f = sym_rdm(np.load(os.path.join(S128, "human_nova_feature_sim.npy")))
    print(f"n={len(CONCEPTS)} concepts")
    print(f"HUMAN triplet~feature: r2 = {r2(Dh_t, Dh_f):.3f}  (n=30 was 0.90)\n")

    models = sorted(d for d in os.listdir(RAW) if os.path.isdir(os.path.join(RAW, d)))
    rows = []
    for m in models:
        Dt = model_triplet_rdm(m)
        Df = model_feature_rdm(m)
        row = {"model": m}
        if Dt is not None and Df is not None:
            row["coherence_t~f"] = r2(Dt, Df)
        if Dt is not None:
            row["human_triplet"] = r2(Dt, Dh_t)
        if Df is not None:
            row["human_feature"] = r2(Df, Dh_f)
        rows.append(row)
    out = pd.DataFrame(rows).set_index("model")
    out.to_csv(os.path.join(S128, "coherence_128.csv"))
    print(out.round(3).to_string())


if __name__ == "__main__":
    main()
