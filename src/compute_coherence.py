"""Build the model x method coherence matrix and human-alignment table.

Discovers every model under results/raw/, computes per-method RDMs, cross-method
coherence, and (if a human reference is available) human alignment (RSA + Procrustes
R^2). Writes:
  results/coherence/coherence_matrix.csv   (models x metrics)
  results/coherence/coherence_heatmap.png
  results/coherence/rdms.npz               (all similarity matrices)

Human reference: pass --human_rdm <npy 30x30 similarity> or --human_embedding
<npy 30xk>. If neither is given, only cross-method coherence is produced and the
human columns are left blank.
"""
import argparse
import os
import numpy as np
import pandas as pd

import analysis as A

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(HERE, "results", "raw")
OUT = os.path.join(HERE, "results", "coherence")


def discover_models():
    if not os.path.isdir(RAW):
        return []
    return sorted(d for d in os.listdir(RAW)
                  if os.path.isdir(os.path.join(RAW, d)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None,
                    help="subset; default = all under results/raw/")
    ap.add_argument("--human_rdm", default=None, help="path to 30x30 human similarity .npy")
    ap.add_argument("--human_embedding", default=None, help="path to 30xk human embedding .npy")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    concepts = A.load_concepts()
    models = args.models or discover_models()
    if not models:
        print("No models found under results/raw/. Run a runner first.")
        return

    human_rdm = np.load(args.human_rdm) if args.human_rdm else None
    human_emb = np.load(args.human_embedding) if args.human_embedding else None

    rows = []
    store = {}
    for model in models:
        rdms = A.model_rdms(model, concepts)
        if not rdms:
            print(f"[warn] no parseable responses for {model}")
            continue
        for m, sim in rdms.items():
            store[f"{model}__{m}"] = sim

        cm = A.cross_method_coherence(rdms)
        row = {"model": model}
        row.update({f"cross_{k}": v for k, v in cm.items()})
        row["n_methods"] = len(rdms)

        # human alignment per method (RSA) + Procrustes on triplet embedding
        if human_rdm is not None:
            for m, sim in rdms.items():
                row[f"human_rsa_{m}"] = A.rsa(sim, human_rdm)
        if human_emb is not None and "triplet" in rdms:
            emb = A.embed_30d(rdms["triplet"], dim=human_emb.shape[1])
            row["human_procrustes_r2_triplet"] = A.procrustes_r2(emb, human_emb)
        rows.append(row)
        print(f"[ok] {model}: methods={list(rdms)} cross_mean={cm.get('mean'):.3f}")

    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("cross_mean", ascending=False) if "cross_mean" in df else df
    csv_path = os.path.join(OUT, "coherence_matrix.csv")
    df.to_csv(csv_path)
    np.savez_compressed(os.path.join(OUT, "rdms.npz"), **store)
    print(f"\nWrote {csv_path}\n")
    print(df.round(3).to_string())

    # heatmap of cross-method RSA per model
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        cols = [c for c in df.columns if c.startswith("cross_") and c != "cross_mean"]
        if cols:
            fig, ax = plt.subplots(figsize=(1.6 + 1.2 * len(cols), 0.5 * len(df) + 1.5))
            data = df[cols].values.astype(float)
            im = ax.imshow(data, cmap="viridis", aspect="auto", vmin=0, vmax=1)
            ax.set_xticks(range(len(cols)))
            ax.set_xticklabels([c.replace("cross_", "") for c in cols], rotation=45, ha="right")
            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df.index)
            for (i, j), v in np.ndenumerate(data):
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            color="w" if v < 0.6 else "k", fontsize=8)
            fig.colorbar(im, ax=ax, label="Spearman RSA")
            ax.set_title("Cross-method coherence")
            fig.tight_layout()
            fig.savefig(os.path.join(OUT, "coherence_heatmap.png"), dpi=150)
            print("Wrote coherence_heatmap.png")
    except Exception as e:
        print(f"[heatmap skipped] {e}")


if __name__ == "__main__":
    main()
