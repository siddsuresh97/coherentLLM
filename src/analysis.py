"""Turn raw model responses into per-concept representations and coherence scores.

For each model we build, per method, a 30x30 concept similarity matrix (RDM):

  triplet  : from anchored choices, build a "closeness count" matrix, then a
             30D ordinal/MDS embedding -> similarity matrix.
  pairwise : the 1..7 ratings averaged over both orders -> similarity matrix.
  feature  : concept x feature True/False matrix -> cosine similarity of concepts.

Coherence is reported two ways:
  * cross-method coherence : Spearman RSA between a model's own method RDMs
                             (how internally consistent the model's structure is).
  * human alignment        : Spearman RSA and Procrustes R^2 between a model RDM /
                             embedding and a human reference (if provided).

The published paper (arXiv:2510.01030) defines its alignment score as Procrustes
R^2 between a 30D model ordinal embedding and the human SPoSE embedding; we mirror
that with `procrustes_r2` and also expose the model-only cross-method metric.
"""
import os
import re
import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from sklearn.manifold import MDS
from sklearn.preprocessing import normalize

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")


# ----------------------------- parsing helpers -----------------------------

def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def load_concepts():
    with open(os.path.join(HERE, "data", "stimuli", "concepts.csv")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def _index(concepts):
    return {_norm(c): i for i, c in enumerate(concepts)}


def _read_raw(model, method):
    path = os.path.join(RAW, model, f"{method}.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


# ----------------------------- per-method RDMs -----------------------------

def triplet_counts(model, concepts):
    """Return (closeness, total) NxN matrices from anchored triplet choices.

    closeness[i,j] = # times concept j was chosen as more similar to anchor i.
    We symmetrize into a similarity by closeness+closeness.T over totals.
    """
    df = _read_raw(model, "triplet")
    if df is None:
        return None
    idx = _index(concepts)
    n = len(concepts)
    close = np.zeros((n, n))
    total = np.zeros((n, n))
    for _, row in df.iterrows():
        try:
            anchor, c1, c2 = str(row["input"]).split("|")
        except ValueError:
            continue
        ai, i1, i2 = idx.get(_norm(anchor)), idx.get(_norm(c1)), idx.get(_norm(c2))
        if None in (ai, i1, i2):
            continue
        resp = _norm(row["response"])
        # which candidate did the model name?
        chose = None
        if _norm(c1) and _norm(c1) in resp:
            chose = i1
        elif _norm(c2) and _norm(c2) in resp:
            chose = i2
        total[ai, i1] += 1
        total[ai, i2] += 1
        if chose is not None:
            close[ai, chose] += 1
    return close, total


def triplet_similarity(model, concepts):
    res = triplet_counts(model, concepts)
    if res is None:
        return None
    close, total = res
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total > 0, close / total, np.nan)
    # symmetric similarity: average of the two directions where available
    sim = np.nanmean(np.dstack([rate, rate.T]), axis=2)
    np.fill_diagonal(sim, 1.0)
    # fill any remaining nans with the global mean
    m = np.nanmean(sim)
    sim = np.where(np.isnan(sim), m, sim)
    return sim


def pairwise_similarity(model, concepts):
    df = _read_raw(model, "pairwise")
    if df is None:
        return None
    idx = _index(concepts)
    n = len(concepts)
    s = np.full((n, n), np.nan)
    cnt = np.zeros((n, n))
    acc = np.zeros((n, n))
    for _, row in df.iterrows():
        try:
            a, b = str(row["input"]).split("|")
        except ValueError:
            continue
        i, j = idx.get(_norm(a)), idx.get(_norm(b))
        if i is None or j is None:
            continue
        mtc = re.search(r"[1-7]", str(row["response"]))
        if not mtc:
            continue
        v = int(mtc.group())
        for (x, y) in [(i, j), (j, i)]:
            acc[x, y] += v
            cnt[x, y] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(cnt > 0, acc / cnt, np.nan)
    np.fill_diagonal(s, 7.0)
    m = np.nanmean(s)
    s = np.where(np.isnan(s), m, s)
    # scale 1..7 -> 0..1 similarity
    return (s - 1.0) / 6.0


def feature_similarity(model, concepts):
    df = _read_raw(model, "feature")
    if df is None:
        return None
    idx = _index(concepts)
    n = len(concepts)
    feats = {}
    for _, row in df.iterrows():
        try:
            feat, concept = str(row["input"]).split("|")
        except ValueError:
            continue
        ci = idx.get(_norm(concept))
        if ci is None:
            continue
        val = _parse_truefalse(row["response"])
        if val is None:
            continue
        feats.setdefault(feat, {})[ci] = val
    feat_names = sorted(feats)
    M = np.zeros((n, len(feat_names)))
    for fj, fn in enumerate(feat_names):
        for ci, v in feats[fn].items():
            M[ci, fj] = v
    # Guard against degenerate (near-constant) feature matrices: if a model answers
    # almost entirely True or almost entirely False, the derived RDM is noise.
    frac_true = M.mean() if M.size else 0.0
    if M.shape[1] == 0 or frac_true < 0.02 or frac_true > 0.98:
        print(f"[feature] WARNING: near-constant answers (frac_true={frac_true:.3f}); "
              f"RDM unreliable")
    Mn = normalize(M) if M.shape[1] else M
    sim = Mn @ Mn.T
    np.fill_diagonal(sim, 1.0)
    return sim


def _parse_truefalse(resp):
    """Take the model's answer only. The prompt echoes 'A: True'/'A: False' few-shot
    examples, so read the text AFTER the last 'A:' and take the first true/false."""
    s = str(resp)
    if "A:" in s:
        s = s.rsplit("A:", 1)[1]
    s = s.lower()
    ti = s.find("true")
    fi = s.find("false")
    if ti < 0 and fi < 0:
        # fall back to yes/no
        yi, ni = s.find("yes"), s.find("no")
        if yi < 0 and ni < 0:
            return None
        return 1 if (yi >= 0 and (ni < 0 or yi < ni)) else 0
    if ti < 0:
        return 0
    if fi < 0:
        return 1
    return 1 if ti < fi else 0


METHOD_FN = {
    "triplet": triplet_similarity,
    "pairwise": pairwise_similarity,
    "feature": feature_similarity,
}


# ----------------------------- coherence metrics -----------------------------

def _upper(m):
    return m[np.triu_indices_from(m, k=1)]


def rsa(sim_a, sim_b):
    """Spearman correlation of upper-triangles of two similarity matrices."""
    r, _ = spearmanr(_upper(sim_a), _upper(sim_b))
    return float(r)


def _cmdscale(D, k):
    """Classical MDS (== R cmdscale): eigendecomp of double-centered -0.5 D^2."""
    D = np.asarray(D, float); n = D.shape[0]
    J = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * J @ (D ** 2) @ J
    w, V = np.linalg.eigh(B)
    idx = np.argsort(w)[::-1][:k]
    return V[:, idx] * np.sqrt(np.clip(w[idx], 0, None))


def embed_30d(sim, dim=30, seed=0):
    """k-dim embedding from a similarity matrix via CLASSICAL MDS (matches the paper's
    R cmdscale) on 1-sim distance."""
    d = 1.0 - sim
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    k = min(dim, sim.shape[0] - 1)
    return _cmdscale(d, k)


def procrustes_r2(emb_model, emb_human):
    """Procrustes R^2 = 1 - SSE_residual / SSE_human after optimal similarity fit."""
    from scipy.spatial import procrustes
    d = min(emb_model.shape[1], emb_human.shape[1])
    _, mtx2, disparity = procrustes(emb_human[:, :d], emb_model[:, :d])
    # scipy procrustes returns normalized disparity = SSE / trace(human centered^2)
    return float(1.0 - disparity)


def model_rdms(model, concepts, methods=("triplet", "pairwise", "feature")):
    out = {}
    for m in methods:
        sim = METHOD_FN[m](model, concepts)
        if sim is not None:
            out[m] = sim
    return out


def cross_method_coherence(rdms):
    """Mean pairwise RSA across the model's own method RDMs."""
    keys = list(rdms)
    scores = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            scores[f"{keys[i]}~{keys[j]}"] = rsa(rdms[keys[i]], rdms[keys[j]])
    scores["mean"] = float(np.mean(list(scores.values()))) if scores else np.nan
    return scores
