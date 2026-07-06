"""What drives coherence? Three factors: model size, instruction-tuning, RL.

Uses cross-method coherence (mean of the 3 method-pair RSAs, generation) and human
alignment, computed from existing raw responses.

- SIZE:            Qwen2.5-Instruct ladder (params_b), plot coherence vs size.
- INSTRUCTION-TUNE: base -> SFT delta within each lineage (OLMo-2-7B/13B, Tulu-8B).
- RL:              SFT -> DPO -> RLVR deltas within each lineage.

Writes results/coherence/factor_*.csv and factor_*.png.
"""
import os
import numpy as np
import pandas as pd
import yaml

import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
OUT = os.path.join(HERE, "results", "coherence")
HUMAN = np.load(os.path.join(HERE, "data", "human", "leuven_similarity.npy"))


def registry():
    with open(os.path.join(HERE, "configs", "models.yaml")) as f:
        return yaml.safe_load(f)["local"]


def cross_and_human(model):
    """Return (cross-method mean coherence, human triplet RSA) from generation data."""
    concepts = A.load_concepts()
    rdms = {}
    for m in ("triplet", "pairwise", "feature"):
        sim = A.METHOD_FN[m](model, concepts)
        if sim is not None:
            rdms[m] = sim
    keys = list(rdms)
    if len(keys) >= 2:
        pairs = [A.rsa(rdms[keys[i]], rdms[keys[j]])
                 for i in range(len(keys)) for j in range(i + 1, len(keys))]
        cross = float(np.mean(pairs))
    else:
        cross = np.nan
    human = A.rsa(rdms["triplet"], HUMAN) if "triplet" in rdms else np.nan
    return cross, human, len(keys)


def size_sweep():
    reg = registry()
    rows = []
    for name, spec in reg.items():
        if "params_b" not in spec:
            continue
        if not os.path.exists(os.path.join(RAW, name, "triplet.csv")):
            continue
        c, h, n = cross_and_human(name)
        rows.append({"model": name, "params_b": spec["params_b"],
                     "coherence": c, "human": h, "n_methods": n})
    df = pd.DataFrame(rows).sort_values("params_b")
    df.to_csv(os.path.join(OUT, "factor_size.csv"), index=False)
    print("=== SIZE (Qwen2.5-Instruct) ===")
    print(df.round(3).to_string(index=False) if len(df) else "  (no size data yet)")
    return df


def lineage_deltas():
    LINS = {
        "OLMo-2-7B": ["olmo2-7b-base", "olmo2-7b-sft", "olmo2-7b-dpo", "olmo2-7b-instruct"],
        "OLMo-2-13B": ["olmo2-13b-base", "olmo2-13b-sft", "olmo2-13b-dpo", "olmo2-13b-instruct"],
        "Tulu-3-8B": ["tulu3-8b-base", "tulu3-8b-sft", "tulu3-8b-dpo", "tulu3-8b-final"],
    }
    labels = ["base", "SFT", "DPO", "RLVR"]
    rows = []
    for lin, models in LINS.items():
        for lab, model in zip(labels, models):
            if not os.path.exists(os.path.join(RAW, model, "triplet.csv")):
                continue
            c, h, n = cross_and_human(model)
            rows.append({"lineage": lin, "stage": lab, "coherence": c, "human": h})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "factor_tuning_rl.csv"), index=False)
    print("\n=== INSTRUCTION-TUNING + RL (by lineage stage, generation) ===")
    print(df.round(3).to_string(index=False) if len(df) else "  (no lineage data)")
    # explicit deltas
    print("\n  deltas (human alignment):")
    for lin in df["lineage"].unique():
        d = df[df.lineage == lin].set_index("stage")
        def g(s): return d.loc[s, "human"] if s in d.index else np.nan
        print(f"   {lin}: base->SFT (instruction) = {g('SFT')-g('base'):+.3f}; "
              f"SFT->RLVR (RL) = {g('RLVR')-g('SFT'):+.3f}")
    return df


def plots(size_df, lin_df):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        # size
        if len(size_df):
            fig, ax = plt.subplots(figsize=(7, 4.5))
            ax.plot(size_df["params_b"], size_df["coherence"], "o-", lw=2, ms=9,
                    color="#4C72B0", label="cross-method coherence")
            ax.plot(size_df["params_b"], size_df["human"], "s--", lw=1.5, ms=7,
                    color="#55A868", label="human alignment")
            ax.set_xscale("log")
            ax.axhline(0.84, color="crimson", ls=":", lw=1, label="human ceiling")
            ax.set_xlabel("parameters (B, log)"); ax.set_ylim(0, 1)
            ax.set_ylabel("coherence / alignment")
            ax.set_title("Coherence vs model size (Qwen2.5-Instruct)")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "factor_size.png"), dpi=150)
            print("\nwrote factor_size.png")
        # tuning/RL
        if len(lin_df):
            fig, ax = plt.subplots(figsize=(7.5, 4.5))
            for lin, c in zip(lin_df["lineage"].unique(),
                              ["#4C72B0", "#DD8452", "#C44E52"]):
                d = lin_df[lin_df.lineage == lin]
                ax.plot(d["stage"], d["human"], "o-", lw=2, ms=8, label=lin, color=c)
            ax.set_ylim(0, 1); ax.axhline(0.84, color="gray", ls=":", lw=1)
            ax.set_ylabel("human alignment (triplet RSA)")
            ax.set_title("Instruction-tuning + RL: human alignment by stage")
            ax.legend()
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "factor_tuning_rl.png"), dpi=150)
            print("wrote factor_tuning_rl.png")
    except Exception as e:
        print(f"[plot skipped] {e}")


if __name__ == "__main__":
    s = size_sweep()
    l = lineage_deltas()
    plots(s, l)
