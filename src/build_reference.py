"""Build a reference similarity matrix / embedding over the 30 concepts.

Two sources supported:

  fasttext : cosine similarity of FastText cc.en.300 embeddings of the concept
             names. This is a SEMANTIC PROXY, not human judgments, used to sanity
             check the pipeline until real human triplet/pairwise data on these 30
             concepts is supplied.

  human    : (placeholder) load a human similarity matrix the user provides at
             data/human/human_similarity.npy with rows/cols in concepts.csv order.

Writes data/human/<name>_similarity.npy (30x30) and <name>_embedding.npy (30xk).
"""
import argparse
import os
import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FASTTEXT = "/mnt/dv/wid/projects3/Rogers-muri-human-ai/mia/cc.en.300.bin"


def load_concepts():
    with open(os.path.join(HERE, "data", "stimuli", "concepts.csv")) as f:
        return [ln.strip() for ln in f if ln.strip()]


def fasttext_reference(concepts, dim=30):
    import fasttext
    from sklearn.preprocessing import normalize
    from sklearn.manifold import MDS

    m = fasttext.load_model(FASTTEXT)

    def vec(word):
        parts = word.split()
        return np.mean([m.get_word_vector(p) for p in parts], axis=0)

    E = normalize(np.array([vec(c) for c in concepts]))
    sim = E @ E.T
    np.fill_diagonal(sim, 1.0)
    d = 1.0 - sim
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    emb = MDS(n_components=min(dim, len(concepts) - 1),
              dissimilarity="precomputed", random_state=0,
              normalized_stress="auto").fit_transform(d)
    return sim, emb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["fasttext", "human"], default="fasttext")
    ap.add_argument("--dim", type=int, default=30)
    args = ap.parse_args()

    concepts = load_concepts()
    outdir = os.path.join(HERE, "data", "human")
    os.makedirs(outdir, exist_ok=True)

    if args.source == "fasttext":
        sim, emb = fasttext_reference(concepts, dim=args.dim)
        np.save(os.path.join(outdir, "fasttext_similarity.npy"), sim)
        np.save(os.path.join(outdir, "fasttext_embedding.npy"), emb)
        print(f"Wrote fasttext reference: sim {sim.shape}, emb {emb.shape}")
        # quick category sanity check
        reptiles = {"Alligator", "Blindworm", "Boa python", "Caiman", "Chameleon",
                    "Cobra", "Crocodile", "Dinosaur", "Gecko", "Lizard", "Salamander",
                    "Snake", "Toad", "Tortoise", "Turtle"}
        idx = [i for i, c in enumerate(concepts) if c in reptiles]
        jdx = [i for i, c in enumerate(concepts) if c not in reptiles]
        wr = sim[np.ix_(idx, idx)][np.triu_indices(len(idx), 1)].mean()
        wt = sim[np.ix_(jdx, jdx)][np.triu_indices(len(jdx), 1)].mean()
        bt = sim[np.ix_(idx, jdx)].mean()
        print(f"within-reptile={wr:.3f} within-tool={wt:.3f} between={bt:.3f} "
              f"structure={'YES' if wr > bt and wt > bt else 'no'}")
    else:
        path = os.path.join(outdir, "human_similarity.npy")
        if not os.path.exists(path):
            raise SystemExit(f"Provide human similarity at {path} (30x30, concepts.csv order)")
        print("human reference already present:", path)


if __name__ == "__main__":
    main()
