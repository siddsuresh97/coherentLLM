"""OLMo-2-7B training-stage coherence curve, measured UNIFORMLY via logprob.

Every stage (base/SFT/DPO/RLVR) is scored with the same base-appropriate logprob
method (triplet_lp.csv / pairwise_lp.csv), so base<->SFT<->DPO<->RLVR are directly
comparable (no generation-vs-logprob confound). Reports triplet~pairwise coherence
and human alignment per stage.
"""
import os
import numpy as np
import pandas as pd

import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
HUMAN = np.load(os.path.join(HERE, "data", "human", "leuven_similarity.npy"))
STAGES = [("olmo2-7b-base", "base"), ("olmo2-7b-sft", "SFT"),
          ("olmo2-7b-dpo", "DPO"), ("olmo2-7b-instruct", "RLVR")]


def rdm_from(model, method, suffix):
    """Build a similarity RDM from the _lp file by temporarily reading it."""
    path = os.path.join(RAW, model, f"{method}{suffix}.csv")
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    # analysis reads "<method>.csv"; point it at the _lp file via a shim.
    df = pd.read_csv(path)
    tmp = os.path.join(RAW, model, f"{method}.csv")
    bak = tmp + ".curvebak"
    had = os.path.exists(tmp)
    if had:
        os.rename(tmp, bak)
    df.to_csv(tmp, index=False)
    try:
        sim = (A.triplet_similarity(model, A.load_concepts()) if method == "triplet"
               else A.pairwise_similarity(model, A.load_concepts()))
    finally:
        os.remove(tmp)
        if had:
            os.rename(bak, tmp)
    return sim


def main():
    rows = []
    for model, label in STAGES:
        t = rdm_from(model, "triplet", "_lp")
        p = rdm_from(model, "pairwise", "_lp")
        if t is None or p is None:
            print(f"[skip] {label}: missing _lp files")
            continue
        rows.append({
            "stage": label,
            "triplet~pairwise": A.rsa(t, p),
            "human_rsa_triplet": A.rsa(t, HUMAN),
            "human_rsa_pairwise": A.rsa(p, HUMAN),
        })
    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "results", "coherence", "olmo_curve_logprob.csv")
    df.to_csv(out, index=False)
    print("Uniform (logprob) OLMo-2-7B training-stage curve:\n")
    print(df.round(3).to_string(index=False))
    print(f"\nwrote {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(df["stage"], df["triplet~pairwise"], "o-", lw=2, ms=9,
                color="#4C72B0", label="cross-method coherence (triplet~pairwise)")
        ax.plot(df["stage"], df["human_rsa_triplet"], "s--", lw=1.5, ms=7,
                color="#55A868", label="human alignment (triplet RSA)")
        for i, v in enumerate(df["triplet~pairwise"]):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_ylabel("coherence / alignment")
        ax.set_title("OLMo-2-7B: coherence across training stages (uniform logprob)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(HERE, "results", "coherence",
                                 "olmo_curve_logprob.png"), dpi=150)
        print("wrote olmo_curve_logprob.png")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    main()
