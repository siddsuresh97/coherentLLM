"""Compare OLMo-2 and Tulu-3 lineages side by side.

Reads olmo_dual.csv and tulu_dual.csv (produced by lineage_dual.py) and renders a
2-panel figure: generation coherence and logprob coherence, both lineages overlaid,
so replication (or not) of the "generation rises / representation flat" pattern is
visible at a glance. Also prints the human-alignment (generation) rise for each.
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(HERE, "results", "coherence")


def main():
    olmo = pd.read_csv(os.path.join(OUT, "olmo_dual.csv"))
    tulu = pd.read_csv(os.path.join(OUT, "tulu_dual.csv"))
    stages = ["base", "SFT", "DPO", "RLVR"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    panels = [("gen_coherence", "generation coherence (triplet~pairwise)"),
              ("lp_coherence", "logprob coherence (representation)"),
              ("gen_human", "human alignment (generation, triplet RSA)")]
    for ax, (col, title) in zip(axes, panels):
        for df, name, c in [(olmo, "OLMo-2-7B", "#4C72B0"), (tulu, "Tulu-3-8B", "#C44E52")]:
            d = df.set_index("stage").reindex(stages)
            ax.plot(stages, d[col], "o-", lw=2, ms=8, color=c, label=name)
        ax.axhline(0.84, color="gray", ls=":", lw=1)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(-0.1, 1.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coherence / alignment")
    axes[0].legend(loc="upper left")
    fig.suptitle("OLMo-2 vs Tulu-3: coherence across training stages "
                 "(dotted = human 0.84)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(OUT, "lineage_comparison.png"), dpi=150)
    print("wrote lineage_comparison.png")

    for name, df in [("OLMo-2", olmo), ("Tulu-3", tulu)]:
        d = df.set_index("stage")
        print(f"\n{name}: human alignment (generation) by stage:")
        print("  " + "  ".join(f"{s}={d.loc[s,'gen_human']:.2f}"
                               for s in stages if s in d.index and pd.notna(d.loc[s, 'gen_human'])))


if __name__ == "__main__":
    main()
