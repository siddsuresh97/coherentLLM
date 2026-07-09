#!/usr/bin/env python3
"""Experiment 1 scaffold: concept-move detection from triplet judgments.

This script owns the CPU-reproducible part of Experiment 1:

* compute the NOVA feature-space base RDM over the 128 held-out concepts,
* select the 30-item experiment set,
* freeze the 30-item triplet protocol/stimuli,
* build a provisional behavioral base RDM from the archived Llama triplet
  embedding,
* estimate an archived-data split-half floor, and
* optionally write feature-only candidate edit specs.

It deliberately marks the behavioral gate red until independent model runs under
the frozen 30-item protocol, including prompt paraphrases, exist. That prevents a
single archived triplet file from being mistaken for the full baseline required
by the experiment spec.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
ARTIFACT_DIR = EXP_DIR / "artifacts"
RDM_DIR = ARTIFACT_DIR / "rdms"
FIG_DIR = EXP_DIR / "figs"
STIM_DIR = EXP_DIR / "stimuli"
CANDIDATE_EDIT_DIR = EXP_DIR / "candidate_edits"

DEFAULT_CONFIG = {
    "seed": 1729,
    "base_model": "llama-3.1-8b-instruct",
    "source_concepts_csv": "data/scale128/concepts.csv",
    "nova_feature_matrix": "data/nova/verified_matrix_cogsci2025.parquet",
    "archived_triplet_embedding": "data/scale128/llama-3.1-8b-instruct_triplet_d5.npy",
    "archived_triplet_raw": "results/raw_128/llama-3.1-8b-instruct/triplet.csv",
    "target_concept": "antelope",
    "target_neighbor": "bison",
    "n_items": 30,
    "cluster_items": 12,
    "bootstrap_splits": 64,
    "candidate_edit_fraction_grid": [0.15, 0.35, 0.65, 1.0],
    "triplet_protocol": {
        "scheme": "full_anchor_candidate_enumeration",
        "description": (
            "For each anchor, enumerate every unordered candidate pair among the "
            "other 29 concepts. This gives 30 * C(29, 2) = 12180 judgments per "
            "model run."
        ),
        "prompt_template": (
            "Answer using only one word - {concept1} or {concept2} and not "
            "{anchor}. Which is more similar in semantic meaning to {anchor}?"
        ),
        "paraphrase_template": (
            "Reply with only {concept1} or {concept2}. Compared with {anchor}, "
            "which option is closer in meaning?"
        ),
        "temperature": 0.0,
        "aggregation": (
            "For each anchor i and candidate j, count the fraction of triplets "
            "where j is chosen over the alternative candidate. Symmetrize by "
            "averaging i->j and j->i rates, then convert similarity to RDM as "
            "1 - similarity."
        ),
        "required_baseline_runs": [
            "base_seed_a_canonical_prompt",
            "base_seed_b_canonical_prompt",
            "base_seed_a_paraphrase_prompt",
        ],
    },
}


@dataclass(frozen=True)
class FeatureData:
    concepts: list[str]
    feature_names: list[str]
    matrix: np.ndarray
    similarity: np.ndarray
    rdm: np.ndarray


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def norm_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def ensure_dirs() -> None:
    for path in (EXP_DIR, ARTIFACT_DIR, RDM_DIR, FIG_DIR, STIM_DIR, CANDIDATE_EDIT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_lines(path: Path, rows: Iterable[Iterable[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def load_config(path: Path) -> dict:
    if not path.exists():
        write_json(path, DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    with path.open() as handle:
        config = json.load(handle)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in config.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def cosine_similarity(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.divide(matrix, norms, out=np.zeros_like(matrix), where=norms > 0)
    sim = safe @ safe.T
    sim = np.clip(sim, -1.0, 1.0)
    np.fill_diagonal(sim, 1.0)
    return sim


def load_feature_data(config: dict) -> FeatureData:
    concepts_path = ROOT / config["source_concepts_csv"]
    concepts = [clean_text(line) for line in concepts_path.read_text().splitlines() if line.strip()]

    nova = pd.read_parquet(ROOT / config["nova_feature_matrix"])
    nova.index = [clean_text(index) for index in nova.index]
    feature_names = [clean_text(col) for col in nova.columns]
    nova.columns = feature_names
    aligned = nova.reindex(concepts)
    if aligned.isna().any().any():
        missing = [concept for concept in concepts if concept not in set(nova.index)]
        raise ValueError(f"NOVA matrix is missing selected concepts: {missing[:20]}")

    matrix = aligned.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("NOVA feature matrix contains non-finite values after alignment")
    sim = cosine_similarity(matrix)
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return FeatureData(concepts=concepts, feature_names=feature_names, matrix=matrix, similarity=sim, rdm=rdm)


def top_neighbors(sim: np.ndarray, concepts: list[str], concept: str, n: int) -> list[dict]:
    idx = concepts.index(concept)
    order = np.argsort(sim[idx])[::-1]
    out = []
    for j in order:
        if j == idx:
            continue
        out.append({"concept": concepts[j], "cosine": float(sim[idx, j])})
        if len(out) >= n:
            break
    return out


def pick_items(feature_data: FeatureData, config: dict) -> dict:
    concepts = feature_data.concepts
    sim = feature_data.similarity
    rdm = feature_data.rdm
    target = clean_text(config["target_concept"])
    neighbor = clean_text(config["target_neighbor"])
    if target not in concepts:
        raise ValueError(f"target_concept={target!r} not present in source concepts")
    if neighbor not in concepts:
        raise ValueError(f"target_neighbor={neighbor!r} not present in source concepts")

    target_neighbors = top_neighbors(sim, concepts, target, 20)
    neighbor_neighbors = top_neighbors(sim, concepts, neighbor, 12)
    cluster: list[str] = []
    for concept in [target, neighbor] + [row["concept"] for row in target_neighbors + neighbor_neighbors]:
        if concept not in cluster:
            cluster.append(concept)
        if len(cluster) >= int(config["cluster_items"]):
            break

    selected = cluster[:]
    selected_set = set(selected)
    n_items = int(config["n_items"])

    # Farthest-point fill gives a broad similarity range after the local cluster.
    while len(selected) < n_items:
        best_concept = None
        best_score = -math.inf
        selected_idx = [concepts.index(c) for c in selected]
        for concept in concepts:
            if concept in selected_set:
                continue
            idx = concepts.index(concept)
            min_dist = float(np.min(rdm[idx, selected_idx]))
            if min_dist > best_score:
                best_score = min_dist
                best_concept = concept
        if best_concept is None:
            raise RuntimeError("Could not fill selected item set")
        selected.append(best_concept)
        selected_set.add(best_concept)

    idx = [concepts.index(c) for c in selected]
    selected_rdm = rdm[np.ix_(idx, idx)]
    upper = selected_rdm[np.triu_indices_from(selected_rdm, k=1)]
    candidate_scores = []
    for concept in concepts:
        i = concepts.index(concept)
        top5 = np.sort(np.delete(sim[i], i))[::-1][:5]
        candidate_scores.append(
            {
                "concept": concept,
                "nearest_cosine": float(top5[0]),
                "top5_mean_cosine": float(np.mean(top5)),
                "top5_neighbors": [row["concept"] for row in top_neighbors(sim, concepts, concept, 5)],
            }
        )
    candidate_scores = sorted(candidate_scores, key=lambda row: row["top5_mean_cosine"], reverse=True)

    return {
        "source": {
            "concepts_csv": config["source_concepts_csv"],
            "nova_feature_matrix": config["nova_feature_matrix"],
            "n_source_concepts": len(concepts),
        },
        "target_concept": target,
        "target_neighbor": neighbor,
        "concepts": selected,
        "selection_algorithm": (
            "Seed with target, its designated neighbor, and their nearest feature-space "
            "neighbors; fill the remaining slots by farthest-point sampling in the "
            "full 128-concept feature RDM."
        ),
        "rationale": (
            f"Chose {target} because it sits in a tight animal cluster while still "
            f"having far items available for contrast. The concentrated move is "
            f"defined against {neighbor}, its nearest selected neighbor in the "
            "NOVA feature-space map."
        ),
        "counterfactuals_considered": {
            "highly_clustered_food_items": (
                "Food concepts such as banana, blueberry, broccoli, and cabbage have "
                "even tighter feature clusters, but a move within that cluster is less "
                "diagnostic of an animate-object relation and gives less clean "
                "spillover language for the first edit-attribution experiment."
            ),
            "more_isolated_items": (
                "Items with no tight local neighborhood were rejected because the "
                "concentrated locality manipulation needs a near neighbor to push "
                "against."
            ),
        },
        "target_top_neighbors": target_neighbors[:10],
        "neighbor_top_neighbors": neighbor_neighbors[:10],
        "top_cluster_candidates_by_feature_density": candidate_scores[:12],
        "selected_feature_rdm_summary": {
            "min_distance": float(np.min(upper)),
            "median_distance": float(np.median(upper)),
            "max_distance": float(np.max(upper)),
            "target_neighbor_distance": float(rdm[concepts.index(target), concepts.index(neighbor)]),
        },
    }


def save_selected_feature_rdms(feature_data: FeatureData, items: dict) -> None:
    idx = [feature_data.concepts.index(c) for c in items["concepts"]]
    sim = feature_data.similarity[np.ix_(idx, idx)]
    rdm = feature_data.rdm[np.ix_(idx, idx)]
    np.save(RDM_DIR / "feature_similarity_full128.npy", feature_data.similarity)
    np.save(RDM_DIR / "feature_rdm_full128.npy", feature_data.rdm)
    np.save(RDM_DIR / "feature_similarity_base.npy", sim)
    np.save(RDM_DIR / "feature_rdm_base.npy", rdm)


def plot_feature_maps(feature_data: FeatureData, items: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        from sklearn.manifold import MDS
    except Exception as exc:  # noqa: BLE001
        write_json(ARTIFACT_DIR / "plot_warning.json", {"warning": str(exc)})
        return

    concepts = items["concepts"]
    idx = [feature_data.concepts.index(c) for c in concepts]
    rdm = feature_data.rdm[np.ix_(idx, idx)]

    emb = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto").fit_transform(rdm)
    fig, ax = plt.subplots(figsize=(9, 7))
    target = items["target_concept"]
    neighbor = items["target_neighbor"]
    for concept, (x, y) in zip(concepts, emb):
        color = "#b00020" if concept == target else "#005f73" if concept == neighbor else "#303030"
        ax.scatter(x, y, s=48 if concept in {target, neighbor} else 26, color=color)
        ax.text(x + 0.01, y + 0.01, concept, fontsize=8, color=color)
    ax.set_title("Experiment 1 selected concepts: NOVA feature-space MDS")
    ax.set_xlabel("MDS 1")
    ax.set_ylabel("MDS 2")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_mds.png", dpi=180)
    plt.close(fig)

    condensed = squareform((rdm + rdm.T) / 2.0, checks=False)
    Z = linkage(condensed, method="average")
    fig, ax = plt.subplots(figsize=(10, 6))
    dendrogram(Z, labels=concepts, leaf_rotation=90, ax=ax)
    ax.set_title("Experiment 1 selected concepts: feature-RDM dendrogram")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "feature_dendrogram.png", dpi=180)
    plt.close(fig)


def write_triplet_stimuli(items: dict, config: dict) -> dict:
    concepts = items["concepts"]
    write_lines(STIM_DIR / "concepts.csv", [[concept] for concept in concepts])
    triplets = []
    for anchor in concepts:
        others = [concept for concept in concepts if concept != anchor]
        for i, concept1 in enumerate(others):
            for concept2 in others[i + 1 :]:
                triplets.append([anchor, concept1, concept2])
    write_lines(STIM_DIR / "triplets.csv", triplets)

    pairs = []
    for i, concept1 in enumerate(concepts):
        for concept2 in concepts[i + 1 :]:
            pairs.append([concept1, concept2])
    write_lines(STIM_DIR / "pairs.csv", pairs)

    protocol = dict(config["triplet_protocol"])
    protocol.update(
        {
            "base_model": config["base_model"],
            "n_concepts": len(concepts),
            "n_triplets_per_run": len(triplets),
            "n_pairwise_pairs": len(pairs),
            "stimuli_dir": rel(STIM_DIR),
            "concepts_sha256": sha256_file(STIM_DIR / "concepts.csv"),
            "triplets_sha256": sha256_file(STIM_DIR / "triplets.csv"),
            "pairs_sha256": sha256_file(STIM_DIR / "pairs.csv"),
            "runner_command_canonical": (
                f"COHERENCE_STIM_DIR={rel(STIM_DIR)} "
                f"COHERENCE_RAW_DIR={rel(EXP_DIR / 'raw')} "
                f"python scripts/run_experiment1_triplets.py --model {config['base_model']} "
                "--out-run base_seed_a_canonical_prompt --prompt-variant canonical --overwrite"
            ),
            "runner_command_paraphrase": (
                f"python scripts/run_experiment1_triplets.py --model {config['base_model']} "
                "--out-run base_seed_a_paraphrase_prompt --prompt-variant paraphrase --overwrite"
            ),
            "status": (
                "frozen_stimuli_written; independent canonical/paraphrase model runs "
                "still required before behavioral gate is green"
            ),
        }
    )
    write_json(EXP_DIR / "triplet_protocol.json", protocol)
    return protocol


def behavioral_rdm_from_embedding(feature_data: FeatureData, items: dict, config: dict) -> dict:
    emb_path = ROOT / config["archived_triplet_embedding"]
    emb = np.load(emb_path)
    if emb.shape[0] != len(feature_data.concepts):
        raise ValueError(
            f"Archived embedding shape {emb.shape} does not match {len(feature_data.concepts)} concepts"
        )
    idx = [feature_data.concepts.index(c) for c in items["concepts"]]
    selected = emb[idx]
    diffs = selected[:, None, :] - selected[None, :, :]
    rdm = np.linalg.norm(diffs, axis=2)
    upper = rdm[np.triu_indices_from(rdm, k=1)]
    if upper.max() > 0:
        rdm = rdm / float(upper.max())
    np.fill_diagonal(rdm, 0.0)
    np.save(RDM_DIR / "rdm_base.npy", rdm)
    np.save(RDM_DIR / "behavior_embedding_base.npy", selected)
    return {
        "source": "archived_triplet_embedding",
        "path": config["archived_triplet_embedding"],
        "embedding_shape": list(emb.shape),
        "rdm_shape": list(rdm.shape),
        "distance_normalization": "divided by selected-set max Euclidean distance",
    }


def parse_triplet_raw(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                anchor, concept1, concept2 = str(row["input"]).split("|")
            except ValueError:
                continue
            rows.append((clean_text(anchor), clean_text(concept1), clean_text(concept2), str(row["response"])))
    return rows


def rdm_from_triplet_rows(rows: list[tuple[str, str, str, str]], concepts: list[str]) -> np.ndarray:
    index = {norm_key(concept): i for i, concept in enumerate(concepts)}
    n = len(concepts)
    close = np.zeros((n, n), dtype=float)
    total = np.zeros((n, n), dtype=float)
    for anchor, concept1, concept2, response in rows:
        ai = index.get(norm_key(anchor))
        i1 = index.get(norm_key(concept1))
        i2 = index.get(norm_key(concept2))
        if ai is None or i1 is None or i2 is None:
            continue
        resp = norm_key(response)
        chosen = None
        if norm_key(concept1) and norm_key(concept1) in resp:
            chosen = i1
        elif norm_key(concept2) and norm_key(concept2) in resp:
            chosen = i2
        total[ai, i1] += 1
        total[ai, i2] += 1
        if chosen is not None:
            close[ai, chosen] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total > 0, close / total, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim = np.nanmean(np.dstack([rate, rate.T]), axis=2)
    np.fill_diagonal(sim, 1.0)
    mean = np.nanmean(sim)
    if not np.isfinite(mean):
        mean = 0.5
    sim = np.where(np.isnan(sim), mean, sim)
    sim = np.clip(sim, 0.0, 1.0)
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def estimate_archived_floor(feature_data: FeatureData, items: dict, config: dict) -> dict:
    raw_path = ROOT / config["archived_triplet_raw"]
    rows = parse_triplet_raw(raw_path)
    rng = np.random.default_rng(int(config["seed"]))
    concepts = feature_data.concepts
    selected_idx = [concepts.index(c) for c in items["concepts"]]
    n_splits = int(config["bootstrap_splits"])
    pair_abs = []
    overall_rms = []
    split_corrs = []
    for _ in range(n_splits):
        order = rng.permutation(len(rows))
        half = len(order) // 2
        rows_a = [rows[i] for i in order[:half]]
        rows_b = [rows[i] for i in order[half:]]
        rdm_a = rdm_from_triplet_rows(rows_a, concepts)[np.ix_(selected_idx, selected_idx)]
        rdm_b = rdm_from_triplet_rows(rows_b, concepts)[np.ix_(selected_idx, selected_idx)]
        delta = upper_values(rdm_a - rdm_b)
        pair_abs.extend(np.abs(delta).tolist())
        overall_rms.append(float(np.sqrt(np.mean(delta**2))))
        split_corrs.append(pearson_corr(upper_values(rdm_a), upper_values(rdm_b)))

    mean_corr = float(np.nanmean(split_corrs))
    spearman_brown_like = float((2 * mean_corr) / (1 + mean_corr)) if mean_corr > -1 else float("nan")
    custom_raw_dir = EXP_DIR / "raw"
    required_runs = config["triplet_protocol"]["required_baseline_runs"]
    present_runs = [
        run_name
        for run_name in required_runs
        if (custom_raw_dir / run_name / "triplet.csv").exists()
    ]
    gate_passed = len(present_runs) == len(required_runs) and spearman_brown_like >= 0.8
    floor = {
        "status": "red" if not gate_passed else "green",
        "gate_passed": gate_passed,
        "why_not_green": (
            "Independent model runs under the frozen 30-item protocol are missing. "
            "The numbers below are archived 128-run split-half diagnostics only."
            if not gate_passed
            else ""
        ),
        "archived_raw_source": config["archived_triplet_raw"],
        "archived_raw_rows": len(rows),
        "bootstrap_splits": n_splits,
        "present_required_protocol_runs": present_runs,
        "missing_required_protocol_runs": [run for run in required_runs if run not in present_runs],
        "archived_split_half": {
            "mean_upper_triangle_pearson": mean_corr,
            "spearman_brown_like_reliability": spearman_brown_like,
            "overall_rms_floor_median": float(np.median(overall_rms)),
            "overall_rms_floor_p90": float(np.quantile(overall_rms, 0.90)),
            "pair_abs_floor_median": float(np.median(pair_abs)),
            "pair_abs_floor_p90": float(np.quantile(pair_abs, 0.90)),
        },
        "floor_interpretation": (
            "Use these archived split-half numbers only for engineering triage. The "
            "experiment's reliability floor must come from fresh canonical and "
            "paraphrase runs over the selected 30 concepts."
        ),
    }
    write_json(EXP_DIR / "floor_stats.json", floor)
    return floor


def estimate_protocol_floor(items: dict, config: dict) -> dict | None:
    raw_dir = EXP_DIR / "raw"
    required_runs = config["triplet_protocol"]["required_baseline_runs"]
    present_runs = [
        run_name
        for run_name in required_runs
        if (raw_dir / run_name / "triplet.csv").exists()
    ]
    if len(present_runs) != len(required_runs):
        return None

    concepts = items["concepts"]
    rdms = {}
    for run_name in required_runs:
        rdms[run_name] = rdm_from_triplet_rows(parse_triplet_raw(raw_dir / run_name / "triplet.csv"), concepts)

    base_run = "base_seed_a_canonical_prompt"
    base_rdm = rdms[base_run]
    np.save(RDM_DIR / "rdm_base.npy", base_rdm)
    comparisons = []
    for run_name, rdm in rdms.items():
        if run_name == base_run:
            continue
        delta = upper_values(rdm - base_rdm)
        comparisons.append(
            {
                "run": run_name,
                "upper_triangle_pearson_vs_base": pearson_corr(upper_values(base_rdm), upper_values(rdm)),
                "overall_rms_vs_base": float(np.sqrt(np.mean(delta**2))),
                "pair_abs_median_vs_base": float(np.median(np.abs(delta))),
                "pair_abs_p90_vs_base": float(np.quantile(np.abs(delta), 0.90)),
            }
        )
    mean_corr = float(np.nanmean([row["upper_triangle_pearson_vs_base"] for row in comparisons]))
    mean_rms = float(np.nanmean([row["overall_rms_vs_base"] for row in comparisons]))
    gate_passed = bool(mean_corr >= 0.80 and np.isfinite(mean_rms))
    floor = {
        "status": "green" if gate_passed else "red",
        "gate_passed": gate_passed,
        "why_not_green": "" if gate_passed else "Frozen-protocol runs exist, but reliability is below threshold.",
        "source": "frozen_30_item_protocol_runs",
        "base_run": base_run,
        "required_protocol_runs": required_runs,
        "present_required_protocol_runs": present_runs,
        "missing_required_protocol_runs": [],
        "protocol_comparisons": comparisons,
        "protocol_floor": {
            "mean_upper_triangle_pearson": mean_corr,
            "overall_rms_floor_mean": mean_rms,
            "overall_rms_floor_max": float(np.max([row["overall_rms_vs_base"] for row in comparisons])),
            "pair_abs_floor_median_max": float(np.max([row["pair_abs_median_vs_base"] for row in comparisons])),
            "pair_abs_floor_p90_max": float(np.max([row["pair_abs_p90_vs_base"] for row in comparisons])),
        },
        "floor_interpretation": (
            "This is the experiment floor: canonical rerun plus paraphrase variation "
            "under the frozen 30-item protocol."
        ),
    }
    write_json(EXP_DIR / "floor_stats.json", floor)
    return floor


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return 0.0
    values = np.abs(values)
    if np.allclose(values.sum(), 0):
        return 0.0
    values = np.sort(values)
    n = values.size
    return float((2 * np.sum((np.arange(1, n + 1) * values)) / (n * values.sum())) - ((n + 1) / n))


def normalized_entropy(values: np.ndarray) -> float:
    values = np.abs(np.asarray(values, dtype=float))
    total = values.sum()
    if total <= 0 or values.size <= 1:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(values.size))


def selected_feature_rdm_from_matrix(matrix: np.ndarray, feature_data: FeatureData, items: dict) -> np.ndarray:
    sim = cosine_similarity(matrix)
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    idx = [feature_data.concepts.index(c) for c in items["concepts"]]
    return rdm[np.ix_(idx, idx)]


def edit_metrics(base_rdm: np.ndarray, edited_rdm: np.ndarray, target_pos: int) -> dict:
    delta_row = edited_rdm[target_pos] - base_rdm[target_pos]
    mask = np.ones(delta_row.shape[0], dtype=bool)
    mask[target_pos] = False
    row = delta_row[mask]
    abs_row = np.abs(row)
    return {
        "row_l2_rms": float(np.sqrt(np.mean(row**2))),
        "row_mean_abs": float(np.mean(abs_row)),
        "row_max_abs": float(np.max(abs_row)) if abs_row.size else 0.0,
        "locality_gini_abs_delta": gini(abs_row),
        "locality_one_minus_entropy": float(1.0 - normalized_entropy(abs_row)),
    }


def design_candidate_edits(feature_data: FeatureData, items: dict, config: dict) -> list[dict]:
    concepts = feature_data.concepts
    selected = items["concepts"]
    target = items["target_concept"]
    neighbor = items["target_neighbor"]
    target_i = concepts.index(target)
    neighbor_i = concepts.index(neighbor)
    target_pos = selected.index(target)
    base_selected_rdm = feature_data.rdm[np.ix_([concepts.index(c) for c in selected], [concepts.index(c) for c in selected])]

    matrix = feature_data.matrix
    active_target = matrix[target_i] > 0
    active_neighbor = matrix[neighbor_i] > 0
    shared = np.flatnonzero(active_target & active_neighbor)
    target_active = np.flatnonzero(active_target)
    feature_freq = matrix.mean(axis=0)
    shared_order = sorted(shared.tolist(), key=lambda col: (feature_freq[col], feature_data.feature_names[col]))
    diffuse_order = sorted(target_active.tolist(), key=lambda col: (abs(feature_freq[col] - 0.5), feature_data.feature_names[col]))

    specs = []
    for mode, ordered_features in (("concentrated", shared_order), ("diffuse", diffuse_order)):
        for frac in config["candidate_edit_fraction_grid"]:
            n_remove = max(1, int(round(float(frac) * len(ordered_features))))
            n_remove = min(n_remove, len(ordered_features))
            remove_cols = ordered_features[:n_remove]
            edited = matrix.copy()
            edited[target_i, remove_cols] = 0.0
            edited_rdm = selected_feature_rdm_from_matrix(edited, feature_data, items)
            metrics = edit_metrics(base_selected_rdm, edited_rdm, target_pos)
            spec = {
                "status": "draft_do_not_train_until_behavioral_gate_green",
                "edit_id": f"{mode}_drop_{int(round(float(frac) * 100)):03d}",
                "mode": mode,
                "target_concept": target,
                "target_neighbor": neighbor if mode == "concentrated" else None,
                "feature_operation": "set_target_features_false",
                "fraction_of_ordered_feature_pool": float(frac),
                "n_features_removed": len(remove_cols),
                "removed_features": [feature_data.feature_names[col] for col in remove_cols],
                "feature_pool_definition": (
                    f"features true for both {target} and {neighbor}, ordered by rarity"
                    if mode == "concentrated"
                    else f"features true for {target}, ordered to spread perturbation across common/specific features"
                ),
                "intended_displacement_feature_rdm": metrics,
                "config_hash": "",
            }
            spec["config_hash"] = sha256_json(spec)
            specs.append(spec)
            write_json(CANDIDATE_EDIT_DIR / f"{spec['edit_id']}.json", spec)
    write_json(
        CANDIDATE_EDIT_DIR / "candidate_edit_summary.json",
        {
            "warning": (
                "These are feature-space draft edits only. Do not train LoRAs from "
                "them until floor_stats.json reports the behavioral gate green."
            ),
            "n_specs": len(specs),
            "specs": [
                {
                    "edit_id": spec["edit_id"],
                    "mode": spec["mode"],
                    **spec["intended_displacement_feature_rdm"],
                }
                for spec in specs
            ],
        },
    )
    return specs


def update_report(items: dict, protocol: dict, behavior_meta: dict, floor: dict, specs: list[dict] | None) -> None:
    specs = specs or []
    table_rows = [
        "| Edit | Intended move (row L2 RMS) | Locality (Gini) | Did it fire? | SNR |",
        "|---|---:|---:|---|---:|",
    ]
    if specs:
        for spec in specs:
            metrics = spec["intended_displacement_feature_rdm"]
            table_rows.append(
                f"| `{spec['edit_id']}` | {metrics['row_l2_rms']:.4f} | "
                f"{metrics['locality_gini_abs_delta']:.3f} | not trained | n/a |"
            )
    else:
        table_rows.append("| n/a | n/a | n/a | behavioral gate not green | n/a |")

    pipeline = {
        "0a feature-space base RDM": "green",
        "0b behavioral triplet base RDM": "red" if not floor["gate_passed"] else "green",
        "1 item selection": "green",
        "2 edit operator pre-check": "draft only" if specs else "blocked by gate",
        "3 two-LoRA training": "not started",
        "4 detection/localization": "not started",
        "5 resolution map": "not started",
    }
    pipeline_lines = "\n".join(f"- {stage}: {status}" for stage, status in pipeline.items())

    report = f"""# Experiment 1 Report - Triplet Detection of Concept Moves

Current status: the experiment branch and CPU-reproducible baseline scaffold are in place; the behavioral gate is still red because fresh 30-item canonical/paraphrase model runs have not been collected.

## Pipeline State

{pipeline_lines}

## Item Choice

Target: `{items['target_concept']}`. Neighbor for concentrated move: `{items['target_neighbor']}`.

Rationale: {items['rationale']}

Artifacts:
- Items: `{rel(EXP_DIR / 'items.json')}`
- Feature RDM: `{rel(RDM_DIR / 'feature_rdm_base.npy')}`
- Provisional behavioral RDM: `{rel(RDM_DIR / 'rdm_base.npy')}`
- Protocol: `{rel(EXP_DIR / 'triplet_protocol.json')}`
- Floor stats: `{rel(EXP_DIR / 'floor_stats.json')}`
- Feature MDS: `{rel(FIG_DIR / 'feature_mds.png')}`
- Feature dendrogram: `{rel(FIG_DIR / 'feature_dendrogram.png')}`

## Perturbations Tried

{chr(10).join(table_rows)}

## Current Course-Correction Reasoning

The archived 128-item Llama triplet embedding is useful as a provisional behavioral map, but it does not satisfy the hard gate. The next action is to run the frozen 30-item protocol for the base model at least twice with the canonical prompt and once with the paraphrase prompt, then replace the provisional floor with the real split/paraphrase reliability estimate. Training any LoRA before that would contaminate the experiment because detection would have no accepted noise denominator.

## Current Findings

- Feature-space map is established over all 128 held-out concepts and the selected 30-item subset.
- `antelope` has a clear local feature-space neighbor (`bison`) plus a broader animal cluster, so a concentrated push has a concrete target pair.
- The frozen triplet protocol has {protocol['n_triplets_per_run']} judgments per run and hashes the exact stimuli files.
- Behavioral base is provisional: {behavior_meta['source']} from `{behavior_meta['path']}`.

## Live Risks

- Missing true behavioral floor: collect required runs listed in `triplet_protocol.json`.
- No local GPU in this environment: LoRA training and vLLM triplet recovery need a GPU host or CHTC submission.
- Candidate edits are feature-space drafts only: promote them to `edits/` only after the behavioral gate is green.
"""
    (EXP_DIR / "REPORT.md").write_text(report)


def append_research_log(items: dict, floor: dict, specs: list[dict] | None) -> None:
    specs = specs or []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    if "archived_split_half" in floor:
        reliability = floor["archived_split_half"]["spearman_brown_like_reliability"]
        reliability_label = "archived split-half reliability proxy"
    else:
        reliability = floor["protocol_floor"]["mean_upper_triangle_pearson"]
        reliability_label = "frozen-protocol mean reliability"
    if specs:
        spec_summary = ", ".join(
            f"{spec['edit_id']} row_l2={spec['intended_displacement_feature_rdm']['row_l2_rms']:.4f}"
            for spec in specs[:4]
        )
    else:
        spec_summary = "none; held because behavioral gate is red"
    text = f"""
## {now} Branch scaffold and baseline gate
- Goal this session: create the Experiment 1 branch and establish the CPU-reproducible baseline scaffold without skipping the behavioral reliability gate.
- What I ran / built: wrote experiment-local artifacts under `{rel(EXP_DIR)}`, computed feature RDMs from NOVA over the 128 held-out concepts and selected 30 items, froze the 30-item triplet stimuli/protocol, and built a provisional behavioral RDM from the archived Llama triplet embedding.
- Result (numbers; plots saved to /figs with filenames): selected target `{items['target_concept']}` against `{items['target_neighbor']}`; protocol has 12180 triplets per run; {reliability_label} is {reliability:.3f}; plots saved as `figs/feature_mds.png` and `figs/feature_dendrogram.png`.
- Interpretation (what the result means, not just restating it): feature-space design is available, but the behavioral gate is not green because the archived 128-item run is not an independent 30-item canonical/paraphrase floor under the frozen protocol.
- Lit found + how it changes the plan: pending; initial local work prioritized the hard gate and artifact layout. Literature notes must be appended before paper claims are made.
- Decision / next step + WHY this over the alternatives I considered: next collect fresh base triplet runs under `triplet_protocol.json`; this is required before LoRA training because detection SNR needs a valid floor. I rejected using the archived 128-item run as the final floor because it would blur a legacy sampling scheme with the experiment's frozen 30-item scheme.
- Open risks: no local GPU is visible; the base/control/edit model calls likely need CHTC or another GPU host.

### DECISION {now} - Item selection / concept to perturb
- Choice: perturb `{items['target_concept']}` with `{items['target_neighbor']}` as the concentrated neighbor.
- Alternatives considered: food clusters are tighter in NOVA feature space, and isolated items would simplify far-item contrast. I did not choose the food clusters because an animal-neighbor move is easier to interpret as a relation-level concept geometry change; I did not choose isolated items because locality needs a near neighbor to push against.
- Position in feature-space RDM: target-neighbor distance is {items['selected_feature_rdm_summary']['target_neighbor_distance']:.4f}; selected-set median distance is {items['selected_feature_rdm_summary']['median_distance']:.4f}; max distance is {items['selected_feature_rdm_summary']['max_distance']:.4f}.
- What would have made a different concept better: a held-out concept with an even clearer named pair, such as zebra/horse, would be preferable if present in the 128-item set and covered by the same feature data.

### DECISION {now} - Gate before training
- Choice: keep the behavioral gate red and do not train LoRAs yet.
- Why this and not the alternative: the alternative was to treat the archived 128-item triplet embedding as `RDM_base`; that would create a base map but not the required sampling/paraphrase noise floor. Because SNR is denominated in floor units, using the wrong floor would make every later detection number uninterpretable.
- Course-correction criterion: once all required protocol runs exist and split/paraphrase reliability is acceptable, promote feature-space draft edits into trainable `edits/` specs and start the two-LoRA null.
- Draft edit prechecks: {spec_summary}.
"""
    path = EXP_DIR / "RESEARCH_LOG.md"
    with path.open("a") as handle:
        handle.write(text)


def run(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config(EXP_DIR / "config.json")
    feature_data = load_feature_data(config)
    items = pick_items(feature_data, config)
    write_json(EXP_DIR / "items.json", items)
    save_selected_feature_rdms(feature_data, items)
    plot_feature_maps(feature_data, items)
    protocol = write_triplet_stimuli(items, config)
    behavior_meta = behavioral_rdm_from_embedding(feature_data, items, config)
    floor = estimate_protocol_floor(items, config)
    if floor is not None:
        behavior_meta = {
            "source": "frozen_30_item_protocol_runs",
            "path": str((EXP_DIR / "raw" / floor["base_run"] / "triplet.csv").relative_to(ROOT)),
            "rdm_shape": [len(items["concepts"]), len(items["concepts"])],
            "distance_normalization": "triplet choice-rate RDM, 1 - symmetrized choice similarity",
        }
    else:
        floor = estimate_archived_floor(feature_data, items, config)

    specs = []
    if args.allow_provisional_edits:
        specs = design_candidate_edits(feature_data, items, config)

    update_report(items, protocol, behavior_meta, floor, specs)
    append_research_log(items, floor, specs)
    print(f"[experiment1] wrote {rel(EXP_DIR / 'REPORT.md')}")
    print(f"[experiment1] behavioral gate: {'green' if floor['gate_passed'] else 'red'}")
    if not floor["gate_passed"]:
        print(f"[experiment1] missing runs: {', '.join(floor['missing_required_protocol_runs'])}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--allow-provisional-edits",
        action="store_true",
        help="Write draft feature-space candidate edit specs even though LoRA training remains gated.",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
