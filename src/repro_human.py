"""Reproduce the paper's HUMAN coherence matrix, two ways:
  (A) 3D cmdscale embeddings + Procrustes  -> should match paper 0.96/0.84/0.72
  (B) full-RDM Procrustes (no MDS)         -> higher values (the metric we'll use)

Uses the paper's own human data (conceptual_representations_gpt/emnlp):
 - human triplet: native 3D embedding
 - human feature: paper's Leuven animal+artifact CSVs (raw counts) -> cosine
 - human pairwise: mean human ratings matrix
"""
import os
import numpy as np
import pandas as pd
from scipy.spatial import procrustes as sp

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CR = "/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/conceptual_representations_gpt/emnlp"
concepts = [l.strip() for l in open(os.path.join(HERE, "data/stimuli/concepts.csv"))]


def cmdscale(D, k=3):
    D = np.asarray(D, float); n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B); i = np.argsort(w)[::-1][:k]
    return V[:, i] * np.sqrt(np.clip(w[i], 0, None))


def pmetric(X, Y):
    d = min(X.shape[1], Y.shape[1])
    _, _, disp = sp(X[:, :d], Y[:, :d])
    return np.sqrt(max(0.0, 1 - disp))


def build_human_dists():
    # triplet: native 3D -> cosine distance
    ht = pd.read_csv(f"{CR}/embeddings_data/human_triplet_embedding.csv").set_index("concept")
    # feature: paper Leuven CSVs (raw counts)
    la = pd.read_csv(f"{CR}/data/feature_listing/animal_leuven_norms.csv", index_col=0)
    lt = pd.read_csv(f"{CR}/data/feature_listing/artifacts_leuven_norms.csv", index_col=0)
    leu = pd.concat([la, lt]).fillna(0)
    leu.index = [str(i).strip().lower().replace("_", " ") for i in leu.index]
    # pairwise
    pw = pd.read_csv(os.path.join(HERE, "data/human/mean_human_pairwise_similarity.csv"), index_col=0)
    pw.index = [str(i).strip().lower() for i in pw.index]
    pw.columns = [str(c).strip().lower() for c in pw.columns]

    fnm = {"boa python": "boa", "oil can": "oilcan", "paint brush": "paintbrush",
           "grinding disk": "grinding disc"}
    pnm = dict(fnm, vacuum="vacuum cleaner")
    have = [c for c in concepts if fnm.get(c.lower(), c.lower()) in set(leu.index)]

    Et = ht.loc[have][["embed_1", "embed_2", "embed_3"]].values
    En = Et / np.linalg.norm(Et, axis=1, keepdims=True)
    Dt = 1 - En @ En.T

    V = leu.loc[[fnm.get(c.lower(), c.lower()) for c in have]].values.astype(float)
    Vn = V / np.clip(np.linalg.norm(V, axis=1, keepdims=True), 1e-9, None)
    Df = 1 - Vn @ Vn.T

    Rm = pw.loc[[pnm.get(c.lower(), c.lower()) for c in have],
                [pnm.get(c.lower(), c.lower()) for c in have]].values.astype(float)
    Dp = 6 - (Rm + Rm.T) / 2
    return have, {"triplet": Dt, "feature": Df, "pairwise": Dp}


def main():
    have, D = build_human_dists()
    pairs = [("triplet", "feature", 0.96), ("pairwise", "feature", 0.84),
             ("triplet", "pairwise", 0.72)]
    print(f"covered {len(have)}/30 concepts\n")
    print("(A) 3D cmdscale + Procrustes  vs  (B) full-RDM Procrustes  [paper]")
    for a, b, paper in pairs:
        Ea, Eb = cmdscale(D[a]), cmdscale(D[b])
        r_mds = pmetric(Ea, Eb)
        r_rdm = pmetric(D[a], D[b])
        print(f"  {a:8}~{b:8}:  A={r_mds:.3f}   B={r_rdm:.3f}   [paper {paper}]")


if __name__ == "__main__":
    main()
