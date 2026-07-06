"""Render the coherence leaderboard + OLMo training-stage curve.

- coherence_leaderboard.png : bar chart of cross-method coherence per model,
  with the human ceiling (0.84) as a reference line.
- olmo_lineage_curve.png    : coherence vs training stage (base->SFT->DPO->RLVR).
Reads results/coherence/coherence_matrix.csv (run compute_coherence.py first).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "coherence")
HUMAN = 0.84  # human cross-method coherence mean (0.96, 0.84, 0.72)

OLMO_ORDER = ["olmo2-7b-base", "olmo2-7b-sft", "olmo2-7b-dpo", "olmo2-7b-instruct"]
OLMO_LABEL = {"olmo2-7b-base": "base\n(pretrain)", "olmo2-7b-sft": "SFT",
              "olmo2-7b-dpo": "DPO", "olmo2-7b-instruct": "RLVR\n(instruct)"}


def load():
    df = pd.read_csv(os.path.join(OUT, "coherence_matrix.csv")).set_index("model")
    if "cross_mean" not in df:
        cols = ["cross_triplet~pairwise", "cross_triplet~feature", "cross_pairwise~feature"]
        df["cross_mean"] = df[[c for c in cols if c in df]].mean(axis=1)
    return df


def leaderboard(df):
    d = df[df["n_methods"] == 3].sort_values("cross_mean", ascending=False)
    d = d[~d.index.isin(OLMO_ORDER)]  # OLMo shown in its own curve
    fig, ax = plt.subplots(figsize=(9, 0.45 * len(d) + 1.5))
    ax.barh(range(len(d)), d["cross_mean"], color="#4C72B0")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d.index)
    ax.invert_yaxis()
    ax.axvline(HUMAN, color="crimson", ls="--", lw=1.5, label=f"human ceiling ({HUMAN})")
    for i, v in enumerate(d["cross_mean"]):
        ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    ax.set_xlabel("cross-method coherence (mean RSA across triplet/pairwise/feature)")
    ax.set_xlim(0, 1)
    ax.set_title("Semantic coherence across models")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "coherence_leaderboard.png"), dpi=150)
    print("wrote coherence_leaderboard.png")


def olmo_curve(df):
    present = [m for m in OLMO_ORDER if m in df.index]
    if len(present) < 2:
        print("OLMo curve skipped (need >=2 stages)")
        return
    ys = [df.loc[m, "cross_mean"] for m in present]
    hf = [df.loc[m, "human_rsa_feature"] if "human_rsa_feature" in df else None
          for m in present]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(range(len(present)), ys, "o-", lw=2, ms=9, color="#4C72B0",
            label="cross-method coherence")
    for i, v in enumerate(ys):
        ax.text(i, v + 0.015, f"{v:.2f}", ha="center", fontsize=10)
    ax.axhline(HUMAN, color="crimson", ls="--", lw=1.3, label=f"human ceiling ({HUMAN})")
    ax.set_xticks(range(len(present)))
    ax.set_xticklabels([OLMO_LABEL[m] for m in present])
    ax.set_ylabel("cross-method coherence (mean RSA)")
    ax.set_ylim(0, 1)
    ax.set_title("OLMo-2-7B: coherence increases with post-training")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "olmo_lineage_curve.png"), dpi=150)
    print("wrote olmo_lineage_curve.png")


if __name__ == "__main__":
    df = load()
    leaderboard(df)
    olmo_curve(df)
