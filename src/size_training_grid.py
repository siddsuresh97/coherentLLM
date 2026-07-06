"""OLMo-2 size x post-training grid: 7B/13B/32B x base/SFT/DPO/RLVR.
Two panels: human alignment and cross-method coherence, one line per size.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "coherence")
HUMAN = np.load(os.path.join(HERE, "data", "human", "leuven_similarity.npy"))
STAGES = ["base", "SFT", "DPO", "RLVR"]
SIZES = {
    "7B": ["olmo2-7b-base", "olmo2-7b-sft", "olmo2-7b-dpo", "olmo2-7b-instruct"],
    "13B": ["olmo2-13b-base", "olmo2-13b-sft", "olmo2-13b-dpo", "olmo2-13b-instruct"],
    "32B": ["olmo2-32b-base", "olmo2-32b-sft", "olmo2-32b-dpo", "olmo2-32b-instruct"],
}


def stats(m):
    c = A.load_concepts()
    r = {me: A.METHOD_FN[me](m, c) for me in ("triplet", "pairwise", "feature")}
    r = {k: v for k, v in r.items() if v is not None}
    tp = A.rsa(r["triplet"], r["pairwise"]) if "triplet" in r and "pairwise" in r else np.nan
    h = A.rsa(r["triplet"], HUMAN) if "triplet" in r else np.nan
    return tp, h


def main():
    data = {sz: [stats(m) for m in models] for sz, models in SIZES.items()}
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    colors = {"7B": "#4C72B0", "13B": "#DD8452", "32B": "#C44E52"}
    for idx, (title, gi) in enumerate([("human alignment (triplet RSA)", 1),
                                       ("cross-method coherence (triplet~pairwise)", 0)]):
        ax = axes[idx]
        for sz in SIZES:
            ys = [data[sz][i][gi] for i in range(4)]
            ax.plot(STAGES, ys, "o-", lw=2, ms=8, color=colors[sz], label=f"OLMo-2-{sz}")
        ax.axhline(0.84, color="gray", ls=":", lw=1)
        ax.set_title(title, fontsize=11); ax.set_ylim(0, 1); ax.grid(alpha=0.3)
    axes[0].set_ylabel("RSA")
    axes[0].legend(loc="upper left")
    fig.suptitle("OLMo-2: coherence & human alignment across SIZE x TRAINING STAGE "
                 "(dotted = human 0.84)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "size_training_grid.png"), dpi=150)
    print("wrote size_training_grid.png")


if __name__ == "__main__":
    main()
