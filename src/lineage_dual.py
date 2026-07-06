"""Dual-measurement training-stage curve for a lineage.

For each stage, report cross-method coherence (triplet~pairwise) BOTH ways:
  - generation (instruction-following): triplet.csv / pairwise.csv
  - logprob (representation-level, uniform): triplet_lp.csv / pairwise_lp.csv
so the instruction-based and representation-based stories sit side by side.

Usage:
  python lineage_dual.py --lineage olmo   # or tulu
"""
import argparse
import os
import numpy as np
import pandas as pd

import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
OUT = os.path.join(HERE, "results", "coherence")
HUMAN = np.load(os.path.join(HERE, "data", "human", "leuven_similarity.npy"))

LINEAGES = {
    "olmo": [("olmo2-7b-base", "base"), ("olmo2-7b-sft", "SFT"),
             ("olmo2-7b-dpo", "DPO"), ("olmo2-7b-instruct", "RLVR")],
    "tulu": [("tulu3-8b-base", "base"), ("tulu3-8b-sft", "SFT"),
             ("tulu3-8b-dpo", "DPO"), ("tulu3-8b-final", "RLVR")],
}


def rdm(model, method, suffix):
    path = os.path.join(RAW, model, f"{method}{suffix}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    tmp = os.path.join(RAW, model, f"{method}.csv")
    bak = tmp + ".dualbak"
    had = os.path.exists(tmp)
    if suffix:  # swap the _lp file into <method>.csv for analysis, then restore
        if had:
            os.rename(tmp, bak)
        pd.read_csv(path).to_csv(tmp, index=False)
    try:
        fn = A.triplet_similarity if method == "triplet" else A.pairwise_similarity
        sim = fn(model, A.load_concepts())
    finally:
        if suffix:
            os.remove(tmp)
            if had:
                os.rename(bak, tmp)
    return sim


def coherence(model, suffix):
    t = rdm(model, "triplet", suffix)
    p = rdm(model, "pairwise", suffix)
    if t is None or p is None:
        return np.nan, np.nan
    return A.rsa(t, p), A.rsa(t, HUMAN)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineage", choices=list(LINEAGES), required=True)
    args = ap.parse_args()
    rows = []
    for model, label in LINEAGES[args.lineage]:
        gen_c, gen_h = coherence(model, "")       # generation
        lp_c, lp_h = coherence(model, "_lp")      # logprob
        rows.append({"stage": label,
                     "gen_coherence": gen_c, "gen_human": gen_h,
                     "lp_coherence": lp_c, "lp_human": lp_h})
    df = pd.DataFrame(rows)
    csv = os.path.join(OUT, f"{args.lineage}_dual.csv")
    df.to_csv(csv, index=False)
    print(f"=== {args.lineage.upper()} lineage: generation vs logprob ===\n")
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {csv}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        x = range(len(df))
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        ax.plot(x, df["gen_coherence"], "o-", lw=2, ms=9, color="#C44E52",
                label="generation (instruction)")
        ax.plot(x, df["lp_coherence"], "s-", lw=2, ms=9, color="#4C72B0",
                label="logprob (representation)")
        for i, v in enumerate(df["gen_coherence"]):
            if pd.notna(v):
                ax.text(i, v + 0.02, f"{v:.2f}", ha="center", color="#C44E52", fontsize=9)
        for i, v in enumerate(df["lp_coherence"]):
            if pd.notna(v):
                ax.text(i, v - 0.05, f"{v:.2f}", ha="center", color="#4C72B0", fontsize=9)
        ax.axhline(0.84, color="gray", ls=":", lw=1, label="human ceiling (0.84)")
        ax.set_xticks(list(x)); ax.set_xticklabels(df["stage"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("cross-method coherence (triplet~pairwise)")
        ax.set_title(f"{args.lineage.upper()}: coherence across training stages")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT, f"{args.lineage}_dual.png"), dpi=150)
        print(f"wrote {args.lineage}_dual.png")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
