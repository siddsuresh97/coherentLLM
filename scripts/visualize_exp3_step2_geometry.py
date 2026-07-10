#!/usr/bin/env python3
"""Visualize Experiment 3 Step 2 geometry backends.

This script uses only existing Step 2 triplet CSVs. It does not call any model
or generate new triplet judgments.
"""

from __future__ import annotations

import csv
import itertools
import json
import runpy
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp3_directional_confusions"
STEP2_DIR = EXP_DIR / "step2_safety"
ARTIFACT_DIR = STEP2_DIR / "artifacts"
VIS_DIR = ARTIFACT_DIR / "visuals"
EMBED_DIR = ARTIFACT_DIR / "embeddings"
RDM_DIR = ARTIFACT_DIR / "rdms"


def write_csv(path: Path, rows: list[list[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open() as handle:
        return json.load(handle)


def clean_method(value: str) -> str:
    return value.replace(" ", "_").replace("/", "_").replace("=", "")


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[order[i:j]] = (i + j - 1) / 2.0
        i = j
    return ranks


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 2:
        return float("nan")
    a = a[ok]
    b = b[ok]
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom else float("nan")


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return pearson(rankdata(a), rankdata(b))


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def cosine_rdm(embedding: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    z = embedding / np.clip(norms, 1e-12, None)
    rdm = 1.0 - z @ z.T
    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def rdm_to_similarity(rdm: np.ndarray) -> np.ndarray:
    rdm = np.asarray(rdm, dtype=float)
    off = rdm[np.triu_indices_from(rdm, k=1)]
    max_dist = float(np.nanmax(off)) if len(off) else 1.0
    if not np.isfinite(max_dist) or max_dist <= 0:
        max_dist = 1.0
    sim = 1.0 - (rdm / max_dist)
    sim = np.clip((sim + sim.T) / 2.0, 0.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def similarity_to_rdm(sim: np.ndarray) -> np.ndarray:
    sim = np.asarray(sim, dtype=float)
    sim = np.clip((sim + sim.T) / 2.0, 0.0, None)
    off = sim[np.triu_indices_from(sim, k=1)]
    max_sim = float(np.nanmax(off)) if len(off) else 1.0
    if not np.isfinite(max_sim) or max_sim <= 0:
        max_sim = 1.0
    rdm = 1.0 - np.clip(sim / max_sim, 0.0, 1.0)
    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def hoyer_sparsity(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    n = values.size
    if n <= 1:
        return 0.0
    l2 = float(np.linalg.norm(values))
    if l2 == 0:
        return 1.0
    return float((np.sqrt(n) - (np.abs(values).sum() / l2)) / (np.sqrt(n) - 1.0))


def fit_srf_snmf(
    similarity: np.ndarray,
    *,
    ranks: list[int],
    l1: float,
    seed: int,
    epochs: int,
    lr: float,
    restarts: int,
) -> tuple[np.ndarray, dict]:
    """Small SRF-compatible SNMF fallback for dense similarity matrices.

    SRF factorizes a non-negative similarity matrix with a non-negative embedding
    whose dot products reconstruct similarities. This local implementation keeps
    the same object-level contract without requiring the external pysrf package.
    """

    similarity = np.asarray(similarity, dtype=float)
    n_items = similarity.shape[0]
    rng = np.random.default_rng(seed)
    pair_i, pair_j = np.triu_indices(n_items, k=1)
    pair_idx = np.arange(len(pair_i))
    rng.shuffle(pair_idx)
    n_test = max(1, int(round(0.2 * len(pair_idx))))
    test_idx = pair_idx[:n_test]
    train_idx = pair_idx[n_test:]
    target = torch.tensor(similarity, dtype=torch.float32)
    train_i = torch.tensor(pair_i[train_idx], dtype=torch.long)
    train_j = torch.tensor(pair_j[train_idx], dtype=torch.long)
    test_i = torch.tensor(pair_i[test_idx], dtype=torch.long)
    test_j = torch.tensor(pair_j[test_idx], dtype=torch.long)
    rank_summaries = []
    all_payloads: list[dict] = []

    for rank in ranks:
        rank_best: dict | None = None
        init_scale = float(np.sqrt(np.clip(np.mean(similarity[pair_i, pair_j]), 1e-6, None) / max(rank, 1)))
        for restart in range(restarts):
            torch.manual_seed(seed + rank * 1000 + restart)
            weights = torch.nn.Parameter(torch.rand(n_items, rank) * init_scale + 0.01)
            opt = torch.optim.Adam([weights], lr=lr)
            for _ in range(epochs):
                opt.zero_grad(set_to_none=True)
                recon_pairs = (weights[train_i] * weights[train_j]).sum(1)
                train_target = target[train_i, train_j]
                loss = torch.mean((recon_pairs - train_target) ** 2) + l1 * weights.mean()
                loss.backward()
                opt.step()
                with torch.no_grad():
                    weights.clamp_(min=0.0)
            with torch.no_grad():
                train_pred = (weights[train_i] * weights[train_j]).sum(1)
                test_pred = (weights[test_i] * weights[test_j]).sum(1)
                train_target = target[train_i, train_j]
                test_target = target[test_i, test_j]
                train_mse = float(torch.mean((train_pred - train_target) ** 2).item())
                test_mse = float(torch.mean((test_pred - test_target) ** 2).item())
                var_test = float(torch.var(test_target, unbiased=False).item())
                test_r2 = float(1.0 - test_mse / var_test) if var_test > 0 else float("nan")
                embedding = weights.detach().cpu().numpy()
            payload = {
                "rank": rank,
                "restart": restart,
                "train_mse": train_mse,
                "test_mse": test_mse,
                "test_r2": test_r2,
                "embedding": embedding,
                "active_dims": int((embedding.max(axis=0) > 1e-4).sum()),
                "active_dims_gt_0p1": int((embedding.max(axis=0) > 0.1).sum()),
                "mean_hoyer_sparsity_by_dim": float(np.mean([hoyer_sparsity(embedding[:, k]) for k in range(rank)])),
            }
            if rank_best is None or payload["test_mse"] < rank_best["test_mse"]:
                rank_best = payload
        assert rank_best is not None
        rank_summaries.append({key: value for key, value in rank_best.items() if key != "embedding"})
        all_payloads.append(rank_best)

    active_rank_payloads = [payload for payload in all_payloads if payload["active_dims_gt_0p1"] == payload["rank"]]
    candidate_payloads = active_rank_payloads or all_payloads
    best_payload = min(candidate_payloads, key=lambda payload: payload["test_mse"])
    best_embedding = best_payload["embedding"]
    best_fit = {
        key: value for key, value in best_payload.items() if key != "embedding"
    }
    best_fit.update(
        {
            "source": "local_srf_compatible_snmf",
            "similarity_source": "normalized_spose_official_rdm_similarity",
            "ranks_tested": ranks,
            "rank_summaries": rank_summaries,
            "l1": l1,
            "epochs": epochs,
            "lr": lr,
            "restarts": restarts,
            "seed": seed,
            "heldout_fraction": 0.2,
            "selection_rule": "minimum_validation_mse_among_ranks_with_all_dimensions_max_loading_gt_0p1",
            "used_fallback_selection": not bool(active_rank_payloads),
        }
    )
    return best_embedding, best_fit


def fit_spose_official_like(
    triplets: np.ndarray,
    *,
    n_items: int,
    dim: int,
    lmbda: float,
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    train, test = train_test_split(triplets, random_state=seed, test_size=0.2)
    train_t = torch.tensor(train, dtype=torch.long)
    test_t = torch.tensor(test, dtype=torch.long)
    weights = torch.nn.Parameter(torch.randn(n_items, dim) * 0.01 + 0.1)
    opt = torch.optim.Adam([weights], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        head, winner, loser = train_t[:, 0], train_t[:, 1], train_t[:, 2]
        winner_sim = (weights[head] * weights[winner]).sum(1)
        loser_sim = (weights[head] * weights[loser]).sum(1)
        logits = torch.stack([winner_sim, loser_sim], dim=1)
        labels = torch.zeros(len(train_t), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss = loss + 0.01 * torch.relu(-weights).sum() + (lmbda / n_items) * torch.norm(weights, p=1)
        loss.backward()
        opt.step()
    with torch.no_grad():
        embedding = np.maximum(weights.detach().cpu().numpy(), 0.0)

    def acc(arr: np.ndarray) -> float:
        head, winner, loser = arr[:, 0], arr[:, 1], arr[:, 2]
        winner_sim = (embedding[head] * embedding[winner]).sum(1)
        loser_sim = (embedding[head] * embedding[loser]).sum(1)
        return float((winner_sim > loser_sim).mean())

    return embedding, {
        "train_acc": acc(train),
        "test_acc": acc(test),
        "active_dims_gt_0p1": int((embedding.max(0) > 0.1).sum()),
        "mean_weight": float(embedding.mean()),
        "dim": dim,
        "lambda": lmbda,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
    }


def fit_spose_softplus(
    triplets: np.ndarray,
    *,
    n_items: int,
    dim: int,
    l1: float,
    seed: int,
    epochs: int,
    lr: float,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    train, test = train_test_split(triplets, random_state=seed, test_size=0.2)
    train_t = torch.tensor(train, dtype=torch.long)
    theta = torch.nn.Parameter(torch.randn(n_items, dim) * 0.02 - 2.0)
    opt = torch.optim.Adam([theta], lr=lr)
    for _ in range(epochs):
        opt.zero_grad(set_to_none=True)
        embedding_t = torch.nn.functional.softplus(theta)
        head, winner, loser = train_t[:, 0], train_t[:, 1], train_t[:, 2]
        winner_sim = (embedding_t[head] * embedding_t[winner]).sum(1)
        loser_sim = (embedding_t[head] * embedding_t[loser]).sum(1)
        logits = torch.stack([winner_sim, loser_sim], dim=1)
        labels = torch.zeros(len(train_t), dtype=torch.long)
        loss = torch.nn.functional.cross_entropy(logits, labels) + l1 * embedding_t.mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        embedding = torch.nn.functional.softplus(theta).detach().cpu().numpy()

    def acc(arr: np.ndarray) -> float:
        head, winner, loser = arr[:, 0], arr[:, 1], arr[:, 2]
        winner_sim = (embedding[head] * embedding[winner]).sum(1)
        loser_sim = (embedding[head] * embedding[loser]).sum(1)
        return float((winner_sim > loser_sim).mean())

    return embedding, {
        "train_acc": acc(train),
        "test_acc": acc(test),
        "active_dims_gt_1e_minus_3": int((embedding.mean(0) > 1e-3).sum()),
        "mean_weight": float(embedding.mean()),
        "dim": dim,
        "l1": l1,
        "epochs": epochs,
        "lr": lr,
        "seed": seed,
    }


def method_order(rdm: np.ndarray) -> np.ndarray:
    condensed = squareform((rdm + rdm.T) / 2.0, checks=False)
    return leaves_list(linkage(condensed, method="average"))


def plot_heatmap(method: str, rdm: np.ndarray, concepts: list[str], meta: dict[str, dict]) -> Path:
    order = method_order(rdm)
    ordered = rdm[np.ix_(order, order)]
    labels = [concepts[i] for i in order]
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(ordered, cmap="viridis_r", interpolation="nearest")
    ax.set_title(f"{method} clustered RDM")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for pos, concept in enumerate(labels):
        side = meta[concept].get("side")
        color = "#3a8f3a" if side == "allowed" else "#b33a3a"
        ax.get_yticklabels()[pos].set_color(color)
        ax.get_xticklabels()[pos].set_color(color)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="distance")
    fig.tight_layout()
    out = VIS_DIR / f"{clean_method(method)}_clustered_rdm.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_mds(method: str, rdm: np.ndarray, concepts: list[str], meta: dict[str, dict]) -> Path:
    mds = MDS(n_components=2, dissimilarity="precomputed", random_state=7303, n_init=8, max_iter=300)
    xy = mds.fit_transform(rdm)
    clusters = sorted({meta[c]["cluster"] for c in concepts})
    palette = dict(zip(clusters, plt.cm.tab10(np.linspace(0, 1, len(clusters)))))
    fig, ax = plt.subplots(figsize=(10, 7))
    for concept, (x, y) in zip(concepts, xy):
        cluster = meta[concept]["cluster"]
        side = meta[concept]["side"]
        marker = "o" if side == "allowed" else "^"
        ax.scatter(x, y, s=85, marker=marker, color=palette[cluster], edgecolor="black", linewidth=0.7)
        ax.text(x + 0.01, y + 0.01, concept, fontsize=7)
    ax.axhline(0, color="#cccccc", linewidth=0.8)
    ax.axvline(0, color="#cccccc", linewidth=0.8)
    ax.set_title(f"{method} 2D MDS from RDM")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=cluster, markerfacecolor=palette[cluster], markeredgecolor="black", markersize=8)
        for cluster in clusters
    ]
    ax.legend(handles=handles, loc="best", fontsize=7)
    fig.tight_layout()
    out = VIS_DIR / f"{clean_method(method)}_mds.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def plot_srf_loadings(method: str, embedding: np.ndarray, concepts: list[str], meta: dict[str, dict]) -> Path:
    order = np.lexsort(
        (
            np.array([concepts.index(c) for c in concepts]),
            -np.max(embedding, axis=1),
            np.argmax(embedding, axis=1),
        )
    )
    ordered = embedding[order]
    labels = [concepts[i] for i in order]
    fig, ax = plt.subplots(figsize=(max(8, embedding.shape[1] * 0.8), 9))
    im = ax.imshow(ordered, cmap="magma", interpolation="nearest", aspect="auto")
    ax.set_title(f"{method} non-negative SRF loadings")
    ax.set_xticks(range(embedding.shape[1]))
    ax.set_xticklabels([f"dim {i}" for i in range(embedding.shape[1])], rotation=45, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    for pos, concept in enumerate(labels):
        side = meta[concept].get("side")
        ax.get_yticklabels()[pos].set_color("#3a8f3a" if side == "allowed" else "#b33a3a")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="loading")
    fig.tight_layout()
    out = VIS_DIR / f"{clean_method(method)}_loadings.png"
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return out


def write_srf_artifacts(method: str, embedding: np.ndarray, concepts: list[str], meta: dict[str, dict]) -> dict:
    clean = clean_method(method)
    load_rows = [["concept", "cluster", "side", *[f"dim_{k:02d}" for k in range(embedding.shape[1])]]]
    for concept, weights in zip(concepts, embedding):
        load_rows.append([concept, meta[concept]["cluster"], meta[concept]["side"], *[float(v) for v in weights]])
    load_path = VIS_DIR / f"{clean}_loadings.csv"
    write_csv(load_path, load_rows)

    dim_rows = [["dimension", "top_concepts", "top_sides", "top_clusters", "max_loading", "active_concepts", "hoyer_sparsity"]]
    dimension_summaries = []
    for dim in range(embedding.shape[1]):
        weights = embedding[:, dim]
        order = np.argsort(-weights)
        top = [concepts[int(i)] for i in order[:5]]
        top_sides = [meta[c]["side"] for c in top]
        top_clusters = [meta[c]["cluster"] for c in top]
        max_loading = float(weights[order[0]])
        threshold = 0.25 * max_loading if max_loading > 0 else np.inf
        active = [concepts[int(i)] for i in order if weights[int(i)] >= threshold]
        sparsity = hoyer_sparsity(weights)
        dim_rows.append(
            [
                dim,
                " | ".join(top),
                " | ".join(top_sides),
                " | ".join(top_clusters),
                max_loading,
                " | ".join(active),
                sparsity,
            ]
        )
        dimension_summaries.append(
            {
                "dimension": dim,
                "top_concepts": top,
                "top_sides": top_sides,
                "top_clusters": top_clusters,
                "max_loading": max_loading,
                "active_concepts": active,
                "hoyer_sparsity": sparsity,
            }
        )
    dim_path = VIS_DIR / f"{clean}_dimensions.csv"
    write_csv(dim_path, dim_rows)
    loadings_plot = plot_srf_loadings(method, embedding, concepts, meta)
    return {
        "loadings_csv": str(load_path.relative_to(ROOT)),
        "dimensions_csv": str(dim_path.relative_to(ROOT)),
        "loadings_plot": str(loadings_plot.relative_to(ROOT)),
        "dimension_summaries": dimension_summaries,
    }


def summarize_method(method: str, rdm: np.ndarray, concepts: list[str], meta: dict[str, dict]) -> dict:
    clusters = [meta[c]["cluster"] for c in concepts]
    sides = [meta[c]["side"] for c in concepts]
    same_cluster = []
    diff_cluster = []
    same_side = []
    diff_side = []
    for i, j in itertools.combinations(range(len(concepts)), 2):
        value = float(rdm[i, j])
        if clusters[i] == clusters[j]:
            same_cluster.append(value)
        else:
            diff_cluster.append(value)
        if sides[i] == sides[j]:
            same_side.append(value)
        else:
            diff_side.append(value)
    try:
        cluster_silhouette = float(silhouette_score(rdm, clusters, metric="precomputed"))
    except Exception:
        cluster_silhouette = float("nan")
    try:
        side_silhouette = float(silhouette_score(rdm, sides, metric="precomputed"))
    except Exception:
        side_silhouette = float("nan")
    nn_same_cluster = 0
    nn_cross_side = 0
    for i, _concept in enumerate(concepts):
        row = rdm[i].copy()
        row[i] = np.inf
        j = int(np.argmin(row))
        nn_same_cluster += int(clusters[i] == clusters[j])
        nn_cross_side += int(sides[i] != sides[j])
    return {
        "method": method,
        "mean_same_cluster_distance": float(np.mean(same_cluster)),
        "mean_diff_cluster_distance": float(np.mean(diff_cluster)),
        "same_minus_diff_cluster_distance": float(np.mean(same_cluster) - np.mean(diff_cluster)),
        "mean_same_side_distance": float(np.mean(same_side)),
        "mean_cross_side_distance": float(np.mean(diff_side)),
        "same_minus_cross_side_distance": float(np.mean(same_side) - np.mean(diff_side)),
        "cluster_silhouette": cluster_silhouette,
        "side_silhouette": side_silhouette,
        "nn_same_cluster": nn_same_cluster,
        "nn_cross_side": nn_cross_side,
    }


def nearest_rows(method: str, rdm: np.ndarray, concepts: list[str], meta: dict[str, dict]) -> list[list[object]]:
    rows = []
    for i, concept in enumerate(concepts):
        row = rdm[i].copy()
        row[i] = np.inf
        j = int(np.argmin(row))
        near = concepts[j]
        rows.append(
            [
                method,
                concept,
                meta[concept]["cluster"],
                meta[concept]["side"],
                near,
                meta[near]["cluster"],
                meta[near]["side"],
                float(row[j]),
                meta[concept]["cluster"] == meta[near]["cluster"],
                meta[concept]["side"] == meta[near]["side"],
            ]
        )
    return rows


def main() -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    RDM_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_DIR.mkdir(parents=True, exist_ok=True)
    ns = runpy.run_path(str(ROOT / "scripts" / "run_experiment3.py"))
    config = ns["load_config"]()
    concepts = ns["step2_concepts"]()
    meta = ns["step2_concept_meta"]()
    runs = ["base_seed_a_canonical_prompt", "base_seed_b_canonical_prompt", "base_seed_a_matched_paraphrase_prompt"]
    triplets_by_run = {}
    pooled_rows = []
    for run in runs:
        rows = ns["parse_triplet_raw"](ns["STEP2_RAW_DIR"] / run / "triplet.csv")
        pooled_rows.extend(rows)
        triplets, _parsed = ns["triplet_array_from_rows"](rows, concepts)
        triplets_by_run[run] = triplets
    pooled_triplets = np.concatenate([triplets_by_run[run] for run in runs], axis=0)

    methods: dict[str, dict] = {}
    previous_summary = read_json(VIS_DIR / "visual_summary.json")
    previous_methods = previous_summary.get("methods") or {}
    count_rdm = ns["rdm_from_triplet_rows"](pooled_rows, concepts)
    np.save(RDM_DIR / "pooled_count_rdm.npy", count_rdm)
    methods["count_rdm"] = {"rdm": count_rdm, "fit": {"source": "direct_choice_rate"}}

    salmon_d5_embedding = np.load(ARTIFACT_DIR / "embeddings" / "pooled_salmon_d5.npy")
    salmon_d5_rdm = ns["cosine_rdm_from_embedding"](salmon_d5_embedding)
    methods["salmon_d5"] = {"rdm": salmon_d5_rdm, "embedding": salmon_d5_embedding, "fit": {"source": "existing_pooled_salmon_d5"}}

    salmon_d15_embedding_path = EMBED_DIR / "pooled_salmon_d15_visual.npy"
    salmon_d15_rdm_path = RDM_DIR / "pooled_salmon_d15_visual.npy"
    if salmon_d15_embedding_path.exists():
        salmon_d15_embedding = np.load(salmon_d15_embedding_path)
        salmon_d15_rdm = np.load(salmon_d15_rdm_path) if salmon_d15_rdm_path.exists() else ns["cosine_rdm_from_embedding"](salmon_d15_embedding)
        salmon_d15_fit = (previous_methods.get("salmon_d15") or {}).get("fit", {"source": "existing_pooled_salmon_d15_visual"})
    else:
        salmon_d15_embedding, salmon_d15_fit = ns["fit_salmon_embedding"](
            pooled_triplets,
            n_concepts=len(concepts),
            dim=15,
            max_epochs=1500,
            seed=ns["stable_seed"](int(config["seed"]), "step2-visual|pooled|salmon|d15"),
            test_fraction=float(config["salmon_test_fraction"]),
            verbose=1000000,
            ident="step2_visual_pooled_salmon_d15",
        )
        salmon_d15_rdm = ns["cosine_rdm_from_embedding"](salmon_d15_embedding)
        np.save(salmon_d15_embedding_path, salmon_d15_embedding)
        np.save(salmon_d15_rdm_path, salmon_d15_rdm)
    methods["salmon_d15"] = {"rdm": salmon_d15_rdm, "embedding": salmon_d15_embedding, "fit": salmon_d15_fit}

    spose_official_embedding_path = EMBED_DIR / "pooled_spose_official_d40_lambda0p008.npy"
    spose_official_rdm_path = RDM_DIR / "pooled_spose_official_d40_lambda0p008.npy"
    if spose_official_embedding_path.exists() and spose_official_rdm_path.exists():
        spose_official_embedding = np.load(spose_official_embedding_path)
        spose_official_rdm = np.load(spose_official_rdm_path)
        spose_official_fit = (previous_methods.get("spose_official_d40_lam0p008") or {}).get("fit", {"source": "existing_pooled_spose_official_d40_lambda0p008"})
    else:
        spose_official_embedding, spose_official_fit = fit_spose_official_like(
            pooled_triplets,
            n_items=len(concepts),
            dim=40,
            lmbda=0.008,
            seed=ns["stable_seed"](int(config["seed"]), "step2-visual|pooled|spose-official|d40|lambda0.008"),
            epochs=1200,
            lr=0.01,
        )
        spose_official_rdm = cosine_rdm(spose_official_embedding)
        np.save(spose_official_embedding_path, spose_official_embedding)
        np.save(spose_official_rdm_path, spose_official_rdm)
    methods["spose_official_d40_lam0p008"] = {
        "rdm": spose_official_rdm,
        "embedding": spose_official_embedding,
        "fit": spose_official_fit,
    }

    spose_softplus_embedding_path = EMBED_DIR / "pooled_spose_softplus_d40_l1_0p01.npy"
    spose_softplus_rdm_path = RDM_DIR / "pooled_spose_softplus_d40_l1_0p01.npy"
    if spose_softplus_embedding_path.exists() and spose_softplus_rdm_path.exists():
        spose_softplus_embedding = np.load(spose_softplus_embedding_path)
        spose_softplus_rdm = np.load(spose_softplus_rdm_path)
        spose_softplus_fit = (previous_methods.get("spose_softplus_d40_l1_0p01") or {}).get("fit", {"source": "existing_pooled_spose_softplus_d40_l1_0p01"})
    else:
        spose_softplus_embedding, spose_softplus_fit = fit_spose_softplus(
            pooled_triplets,
            n_items=len(concepts),
            dim=40,
            l1=0.01,
            seed=ns["stable_seed"](int(config["seed"]), "step2-visual|pooled|spose-softplus|d40|l1=0.01"),
            epochs=1800,
            lr=0.05,
        )
        spose_softplus_rdm = cosine_rdm(spose_softplus_embedding)
        np.save(spose_softplus_embedding_path, spose_softplus_embedding)
        np.save(spose_softplus_rdm_path, spose_softplus_rdm)
    methods["spose_softplus_d40_l1_0p01"] = {
        "rdm": spose_softplus_rdm,
        "embedding": spose_softplus_embedding,
        "fit": spose_softplus_fit,
    }

    srf_similarity = rdm_to_similarity(spose_official_rdm)
    np.save(RDM_DIR / "pooled_spose_official_similarity_for_srf.npy", srf_similarity)
    srf_embedding, srf_fit = fit_srf_snmf(
        srf_similarity,
        ranks=list(range(2, 9)),
        l1=0.002,
        seed=ns["stable_seed"](int(config["seed"]), "step2-visual|pooled|srf-from-spose-official"),
        epochs=2500,
        lr=0.03,
        restarts=4,
    )
    srf_recon_similarity = srf_embedding @ srf_embedding.T
    srf_rdm = similarity_to_rdm(srf_recon_similarity)
    np.save(EMBED_DIR / f"pooled_srf_from_spose_official_rank{srf_embedding.shape[1]}.npy", srf_embedding)
    np.save(RDM_DIR / f"pooled_srf_from_spose_official_rank{srf_embedding.shape[1]}.npy", srf_rdm)
    np.save(RDM_DIR / f"pooled_srf_from_spose_official_rank{srf_embedding.shape[1]}_reconstructed_similarity.npy", srf_recon_similarity)
    methods["srf_from_spose_official"] = {
        "rdm": srf_rdm,
        "embedding": srf_embedding,
        "fit": srf_fit,
        "factor_artifacts": write_srf_artifacts("srf_from_spose_official", srf_embedding, concepts, meta),
    }

    nearest_table = [["method", "target", "target_cluster", "target_side", "nearest", "nearest_cluster", "nearest_side", "distance", "same_cluster", "same_side"]]
    summary_rows = [[
        "method",
        "mean_same_cluster_distance",
        "mean_diff_cluster_distance",
        "same_minus_diff_cluster_distance",
        "mean_same_side_distance",
        "mean_cross_side_distance",
        "same_minus_cross_side_distance",
        "cluster_silhouette",
        "side_silhouette",
        "nn_same_cluster",
        "nn_cross_side",
    ]]
    order_rows = [["method", "leaf_rank", "concept", "cluster", "side"]]
    visual_paths = {}
    summary = {"methods": {}, "pairwise_rdm_correlations": []}
    for method, payload in methods.items():
        rdm = payload["rdm"]
        visual_paths[method] = {
            "heatmap": str(plot_heatmap(method, rdm, concepts, meta).relative_to(ROOT)),
            "mds": str(plot_mds(method, rdm, concepts, meta).relative_to(ROOT)),
        }
        if payload.get("factor_artifacts"):
            visual_paths[method].update(
                {
                    "loadings": payload["factor_artifacts"]["loadings_plot"],
                    "loadings_csv": payload["factor_artifacts"]["loadings_csv"],
                    "dimensions_csv": payload["factor_artifacts"]["dimensions_csv"],
                }
            )
        method_summary = summarize_method(method, rdm, concepts, meta)
        extra = {"fit": payload.get("fit", {})}
        if payload.get("factor_artifacts"):
            extra["factor_artifacts"] = payload["factor_artifacts"]
        summary["methods"][method] = {**method_summary, **extra}
        summary_rows.append([method_summary[col] for col in summary_rows[0]])
        nearest_table.extend(nearest_rows(method, rdm, concepts, meta))
        order = method_order(rdm)
        for rank, idx in enumerate(order):
            concept = concepts[int(idx)]
            order_rows.append([method, rank, concept, meta[concept]["cluster"], meta[concept]["side"]])

    for method_a, method_b in itertools.combinations(methods, 2):
        a = upper_values(methods[method_a]["rdm"])
        b = upper_values(methods[method_b]["rdm"])
        summary["pairwise_rdm_correlations"].append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "pearson": pearson(a, b),
                "spearman": spearman(a, b),
            }
        )

    write_csv(VIS_DIR / "nearest_neighbors_by_method.csv", nearest_table)
    write_csv(VIS_DIR / "cluster_summary_by_method.csv", summary_rows)
    write_csv(VIS_DIR / "cluster_order_by_method.csv", order_rows)
    summary["visual_paths"] = visual_paths
    summary["notes"] = "Generated from existing Step 2 triplet CSVs only; no new model triplets were run."
    write_json(VIS_DIR / "visual_summary.json", summary)
    print(f"[visuals] wrote {VIS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
