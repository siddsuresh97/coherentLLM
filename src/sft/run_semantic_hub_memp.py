"""Run a paper-style semantic hub / MEMP control harness.

This script adapts the relative-similarity test from Wu et al. (arXiv:2411.04986)
to this repo's text-only MEMP setting: the same concept shown through different
prompt spokes should be closer than controlled wrong concepts, especially in
middle layers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIDDEN = ROOT / "results" / "sft_semantic_hub" / "hidden_states"
DEFAULT_OUT = ROOT / "results" / "sft_semantic_hub" / "memp_paper_harness"
DEFAULT_CONFIG = ROOT / "results" / "sft_semantic_hub" / "memp_paper_config.json"
DEFAULT_ARMS = [
    "base",
    "scrambled",
    "lowLR",
    "lowrank",
    "taskvec_a0p25",
    "taskvec_a0p5",
    "taskvec_a1p0",
]
DEFAULT_CONTROLS = [
    "random",
    "close_neighbor",
    "far_neighbor",
    "lexical",
    "category_proxy",
]
CONTROLS_BY_LAYER_FIELDS = [
    "arm",
    "layer",
    "layer_name",
    "format_a",
    "format_b",
    "control_type",
    "n_concepts",
    "matched_score",
    "matched_ci_low",
    "matched_ci_high",
    "control_score",
    "control_ci_low",
    "control_ci_high",
    "paired_delta",
    "ci_low",
    "ci_high",
    "p_perm_greater",
    "retrieval_top1",
    "retrieval_top5",
    "retrieval_margin",
]
CONTROL_SUMMARY_FIELDS = [
    "arm",
    "control_type",
    "mid_layer_start",
    "mid_layer_end",
    "mid_paired_delta",
    "mid_ci_low",
    "mid_ci_high",
    "mid_p_perm_greater",
    "mid_matched_score",
    "mid_control_score",
    "mid_retrieval_top1",
    "mid_retrieval_top5",
    "best_layer",
    "best_layer_name",
    "best_paired_delta",
    "n_layers_scored",
    "n_concepts",
]


@dataclass(frozen=True)
class RunOptions:
    hidden_dir: Path
    out_dir: Path
    config_path: Path
    arms: list[str]
    controls: list[str]
    mid_layers: str
    layers: list[int] | None
    max_concepts: int | None
    n_boot: int
    n_perm: int
    seed: int
    category_table: Path | None
    category_concept_column: str
    category_column: str
    category_clusters: int
    dry_run: bool
    overwrite: bool


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--write-default-config", action="store_true")
    ap.add_argument("--hidden-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--arms", nargs="+", default=None)
    ap.add_argument("--controls", nargs="+", default=None)
    ap.add_argument("--mid-layers", default=None, help="Inclusive layer band, e.g. 10:20")
    ap.add_argument("--layers", default=None, help="Layer list/ranges, e.g. 8,10:20,24")
    ap.add_argument("--max-concepts", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument("--n-perm", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--category-table", type=Path, default=None)
    ap.add_argument("--category-concept-column", default=None)
    ap.add_argument("--category-column", default=None)
    ap.add_argument("--category-clusters", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def default_config() -> dict[str, Any]:
    return {
        "experiment": "semantic_hub_memp_paper_style",
        "paper_source": "https://arxiv.org/abs/2411.04986",
        "repo_analogue": "same held-out concept across prompt-format spokes",
        "hidden_dir": "results/sft_semantic_hub/hidden_states",
        "out_dir": "results/sft_semantic_hub/memp_paper_harness",
        "arms": DEFAULT_ARMS,
        "primary_arms": ["base", "taskvec_a0p25", "scrambled"],
        "mid_layers": "10:20",
        "layers": "all",
        "controls": DEFAULT_CONTROLS,
        "statistics": {
            "n_boot": 300,
            "n_perm": 1000,
            "seed": 17,
        },
        "category_control": {
            "source": "auto_sstar_kmeans_proxy",
            "n_clusters": 16,
            "note": (
                "Proxy category labels are deterministic clusters over the S* "
                "concept-similarity matrix. Replace with an explicit category "
                "table when one is available."
            ),
        },
        "paper_method_pieces": {
            "implemented": [
                "Eq. 1 relative similarity: matched cross-spoke states vs nonmatches",
                "middle-layer band fixed before scoring",
                "random, S*-close, S*-far, lexical, and category-proxy controls",
                "bootstrap confidence intervals over concepts",
                "sign-flip permutation p-values over paired deltas",
                "machine-readable manifest and metric schema",
            ],
            "planned": [
                "logit-lens dominant-token anchoring",
                "symbolic S* spoke extraction",
                "causal cross-spoke activation patching / addition",
                "cluster-level layer statistics and FDR correction",
            ],
        },
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def resolve_options(args: argparse.Namespace, config: dict[str, Any]) -> RunOptions:
    stats = config.get("statistics", {})
    category_cfg = config.get("category_control", {})
    hidden_dir = args.hidden_dir or resolve_path(config.get("hidden_dir", DEFAULT_HIDDEN))
    out_dir = args.out_dir or resolve_path(config.get("out_dir", DEFAULT_OUT))
    controls = args.controls or list(config.get("controls", DEFAULT_CONTROLS))
    unknown = [control for control in controls if control not in DEFAULT_CONTROLS]
    if unknown:
        raise KeyError(f"unknown controls: {unknown}; available: {DEFAULT_CONTROLS}")
    return RunOptions(
        hidden_dir=hidden_dir,
        out_dir=out_dir,
        config_path=args.config,
        arms=args.arms or list(config.get("arms", DEFAULT_ARMS)),
        controls=controls,
        mid_layers=args.mid_layers or str(config.get("mid_layers", "10:20")),
        layers=parse_layers(args.layers or str(config.get("layers", "all"))),
        max_concepts=args.max_concepts,
        n_boot=int(args.n_boot if args.n_boot is not None else stats.get("n_boot", 300)),
        n_perm=int(args.n_perm if args.n_perm is not None else stats.get("n_perm", 1000)),
        seed=int(args.seed if args.seed is not None else stats.get("seed", 17)),
        category_table=args.category_table,
        category_concept_column=args.category_concept_column
        or str(category_cfg.get("concept_column", "concept")),
        category_column=args.category_column or str(category_cfg.get("category_column", "category")),
        category_clusters=int(
            args.category_clusters if args.category_clusters is not None else category_cfg.get("n_clusters", 16)
        ),
        dry_run=bool(args.dry_run),
        overwrite=bool(args.overwrite),
    )


def parse_layer_band(text: str) -> tuple[int, int]:
    left, right = text.split(":", 1)
    start = int(left)
    end = int(right)
    if start > end:
        raise ValueError(text)
    return start, end


def parse_layers(text: str) -> list[int] | None:
    if text.strip().lower() == "all":
        return None
    layers: set[int] = set()
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        if ":" in item:
            start, end = parse_layer_band(item)
            layers.update(range(start, end + 1))
        else:
            layers.add(int(item))
    return sorted(layers)


def read_hidden_metadata(path: Path) -> tuple[list[str], list[str], list[str], tuple[int, ...]]:
    data = np.load(path, allow_pickle=False)
    concepts = [str(x) for x in data["concepts"]]
    formats = [str(x) for x in data["formats"]]
    layers = [str(x) for x in data["layer_names"]]
    shape = tuple(int(x) for x in data["hidden"].shape)
    if shape[:3] != (len(formats), len(concepts), len(layers)):
        raise ValueError(f"{path} shape/metadata mismatch: {shape}")
    return concepts, formats, layers, shape


def read_hidden(path: Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    data = np.load(path, allow_pickle=False)
    hidden = data["hidden"].astype(np.float32, copy=False)
    concepts = [str(x) for x in data["concepts"]]
    formats = [str(x) for x in data["formats"]]
    layers = [str(x) for x in data["layer_names"]]
    if hidden.shape[:3] != (len(formats), len(concepts), len(layers)):
        raise ValueError(f"{path} shape/metadata mismatch: {hidden.shape}")
    return hidden, concepts, formats, layers


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def read_neighbor_indices(path: Path, concepts: list[str]) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, str]]]:
    concept_to_idx = {concept: i for i, concept in enumerate(concepts)}
    rows = read_csv_rows(path)
    row_by_concept = {row["concept"]: row for row in rows}
    close = np.empty(len(concepts), dtype=int)
    far = np.empty(len(concepts), dtype=int)
    missing = [concept for concept in concepts if concept not in row_by_concept]
    if missing:
        raise KeyError(f"neighbor table is missing concepts: {missing[:10]}")
    for concept in concepts:
        row = row_by_concept[concept]
        close[concept_to_idx[concept]] = concept_to_idx[row["close_neighbor"]]
        far[concept_to_idx[concept]] = concept_to_idx[row["far_neighbor"]]
    return close, far, row_by_concept


def read_sstar(concepts: list[str]) -> np.ndarray:
    s_path = ROOT / "data" / "sft" / "S_star.npy"
    c_path = ROOT / "data" / "sft" / "S_star_concepts.csv"
    s_star = np.load(s_path).astype(np.float32, copy=False)
    s_concepts = [line.strip() for line in c_path.read_text().splitlines() if line.strip()]
    index = {concept: i for i, concept in enumerate(s_concepts)}
    missing = [concept for concept in concepts if concept not in index]
    if missing:
        raise KeyError(f"concepts missing from S*: {missing[:10]}")
    idx = np.array([index[concept] for concept in concepts], dtype=int)
    return s_star[np.ix_(idx, idx)]


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def tokenize_concept(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def char_ngrams(text: str, n: int = 3) -> set[str]:
    compact = re.sub(r"[^a-z0-9]+", "", text.lower())
    if len(compact) <= n:
        return {compact} if compact else set()
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def lexical_similarity(a: str, b: str) -> float:
    ta = set(tokenize_concept(a))
    tb = set(tokenize_concept(b))
    ca = char_ngrams(a)
    cb = char_ngrams(b)
    len_a = max(len(re.sub(r"\s+", "", a)), 1)
    len_b = max(len(re.sub(r"\s+", "", b)), 1)
    length_score = 1.0 - min(abs(len_a - len_b) / max(len_a, len_b), 1.0)
    token_count_score = 1.0 - min(abs(len(ta) - len(tb)) / max(len(ta), len(tb), 1), 1.0)
    return 0.40 * jaccard(ta, tb) + 0.35 * jaccard(ca, cb) + 0.15 * length_score + 0.10 * token_count_score


def kmeans_labels(features: np.ndarray, n_clusters: int, seed: int) -> np.ndarray:
    n = features.shape[0]
    k = max(2, min(int(n_clusters), n))
    rng = np.random.default_rng(seed)
    x = normalize_rows(features.astype(np.float32, copy=False))
    centroid_idx = rng.choice(n, size=k, replace=False)
    centroids = x[centroid_idx]
    labels = np.zeros(n, dtype=int)
    for _ in range(50):
        sim = x @ normalize_rows(centroids).T
        new_labels = np.argmax(sim, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(k):
            members = x[labels == cluster]
            if members.size == 0:
                centroids[cluster] = x[int(rng.integers(0, n))]
            else:
                centroids[cluster] = members.mean(axis=0)
    return labels


def read_category_labels(opts: RunOptions, concepts: list[str], sstar: np.ndarray) -> tuple[list[str], str]:
    if opts.category_table is not None:
        rows = read_csv_rows(opts.category_table)
        by_concept = {row[opts.category_concept_column]: row for row in rows}
        labels = []
        missing = []
        for concept in concepts:
            row = by_concept.get(concept)
            if row is None:
                missing.append(concept)
                labels.append("__missing__")
            else:
                labels.append(row[opts.category_column])
        if missing:
            raise KeyError(f"category table missing concepts: {missing[:10]}")
        return labels, f"category_table:{opts.category_table}"
    labels = kmeans_labels(sstar, opts.category_clusters, opts.seed)
    return [f"sstar_cluster_{label:02d}" for label in labels], "auto_sstar_kmeans_proxy"


def pick_best_candidate(scores: np.ndarray, blocked: set[int]) -> int:
    masked = scores.astype(np.float64, copy=True)
    if len(blocked) >= masked.size:
        raise ValueError("all candidates blocked")
    for idx in blocked:
        if 0 <= idx < masked.size:
            masked[idx] = -np.inf
    if np.all(~np.isfinite(masked)):
        allowed = [idx for idx in range(masked.size) if idx not in blocked]
        return allowed[0]
    return int(np.nanargmax(masked))


def build_control_indices(
    concepts: list[str],
    close_idx: np.ndarray,
    far_idx: np.ndarray,
    category_labels: list[str],
    sstar: np.ndarray,
    opts: RunOptions,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    rng = np.random.default_rng(opts.seed)
    n = len(concepts)
    controls: dict[str, np.ndarray] = {}

    random_idx = np.empty(n, dtype=int)
    lexical_idx = np.empty(n, dtype=int)
    category_idx = np.empty(n, dtype=int)
    lexical_matrix = np.zeros((n, n), dtype=np.float32)
    for i, a in enumerate(concepts):
        for j, b in enumerate(concepts):
            if i != j:
                lexical_matrix[i, j] = lexical_similarity(a, b)

    for i in range(n):
        blocked = {i}
        candidates = [j for j in range(n) if j not in blocked]
        random_idx[i] = int(rng.choice(candidates))

        lexical_blocked = {i, int(close_idx[i]), int(far_idx[i])}
        lexical_idx[i] = pick_best_candidate(lexical_matrix[i], lexical_blocked)

        same_category = np.array(
            [j for j, label in enumerate(category_labels) if label == category_labels[i] and j != i],
            dtype=int,
        )
        category_blocked = {i, int(close_idx[i]), int(far_idx[i])}
        category_scores = np.full(n, -np.inf, dtype=np.float64)
        if same_category.size:
            category_scores[same_category] = sstar[i, same_category]
        category_idx[i] = pick_best_candidate(category_scores, category_blocked)

    controls["random"] = random_idx
    controls["close_neighbor"] = close_idx.copy()
    controls["far_neighbor"] = far_idx.copy()
    controls["lexical"] = lexical_idx
    controls["category_proxy"] = category_idx

    assignments = []
    for i, concept in enumerate(concepts):
        for control_type, indices in controls.items():
            j = int(indices[i])
            assignments.append(
                {
                    "concept_index": i,
                    "concept": concept,
                    "control_type": control_type,
                    "control_index": j,
                    "control_concept": concepts[j],
                    "sstar_similarity": float(sstar[i, j]),
                    "lexical_similarity": float(lexical_matrix[i, j]),
                    "category_label": category_labels[i],
                    "control_category_label": category_labels[j],
                }
            )
    return controls, assignments


def retrieval_metrics(sim: np.ndarray, query_indices: np.ndarray) -> dict[str, float]:
    order = np.argsort(-sim, axis=1)
    ranks = np.empty(len(query_indices), dtype=int)
    for row_idx, concept_idx in enumerate(query_indices):
        ranks[row_idx] = int(np.where(order[row_idx] == concept_idx)[0][0]) + 1
    off = sim.copy()
    off[np.arange(len(query_indices)), query_indices] = -np.inf
    margin = sim[np.arange(len(query_indices)), query_indices] - np.max(off, axis=1)
    return {
        "retrieval_top1": float(np.mean(ranks == 1)),
        "retrieval_top5": float(np.mean(ranks <= 5)),
        "retrieval_margin": float(np.mean(margin)),
    }


def mean_ci(values: np.ndarray, rng: np.random.Generator, n_boot: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    mean = float(values.mean())
    if n_boot <= 0 or values.size == 1:
        return mean, math.nan, math.nan
    idx = rng.integers(0, values.size, size=(n_boot, values.size))
    boot = values[idx].mean(axis=1)
    low, high = np.percentile(boot, [2.5, 97.5])
    return mean, float(low), float(high)


def signflip_p_greater(values: np.ndarray, rng: np.random.Generator, n_perm: int) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    observed = float(values.mean())
    if n_perm <= 0:
        return math.nan
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, values.size))
    perm = (signs * values).mean(axis=1)
    return float((1 + np.sum(perm >= observed)) / (n_perm + 1))


def summarize_values(values: np.ndarray, rng: np.random.Generator, opts: RunOptions) -> dict[str, float]:
    mean, low, high = mean_ci(values, rng, opts.n_boot)
    return {
        "paired_delta": mean,
        "ci_low": low,
        "ci_high": high,
        "p_perm_greater": signflip_p_greater(values, rng, opts.n_perm),
    }


def mean_or_nan(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return math.nan
    return float(np.nanmean(arr))


def compute_arm(
    path: Path,
    selected_controls: dict[str, np.ndarray],
    opts: RunOptions,
    query_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arm = path.stem
    hidden, concepts, formats, layers = read_hidden(path)
    selected_layers = opts.layers or list(range(len(layers)))
    rng = np.random.default_rng(opts.seed + stable_int(arm))
    rows: list[dict[str, Any]] = []

    for layer_idx in selected_layers:
        pair_rows_by_control: dict[str, list[dict[str, Any]]] = {control: [] for control in opts.controls}
        for fmt_a, fmt_b in combinations(range(len(formats)), 2):
            x = normalize_rows(hidden[fmt_a, :, layer_idx, :])
            y = normalize_rows(hidden[fmt_b, :, layer_idx, :])
            sim_ab = x[query_indices] @ y.T
            sim_ba = y[query_indices] @ x.T
            same = 0.5 * (
                sim_ab[np.arange(len(query_indices)), query_indices]
                + sim_ba[np.arange(len(query_indices)), query_indices]
            )
            ret_ab = retrieval_metrics(sim_ab, query_indices)
            ret_ba = retrieval_metrics(sim_ba, query_indices)
            retrieval = {
                key: 0.5 * (ret_ab[key] + ret_ba[key])
                for key in ("retrieval_top1", "retrieval_top5", "retrieval_margin")
            }
            for control_type in opts.controls:
                control_idx = selected_controls[control_type][query_indices]
                control = 0.5 * (
                    sim_ab[np.arange(len(query_indices)), control_idx]
                    + sim_ba[np.arange(len(query_indices)), control_idx]
                )
                delta = same - control
                delta_stats = summarize_values(delta, rng, opts)
                same_mean, same_low, same_high = mean_ci(same, rng, opts.n_boot)
                control_mean, control_low, control_high = mean_ci(control, rng, opts.n_boot)
                row = {
                    "arm": arm,
                    "layer": layer_idx,
                    "layer_name": layers[layer_idx],
                    "format_a": formats[fmt_a],
                    "format_b": formats[fmt_b],
                    "control_type": control_type,
                    "n_concepts": int(len(query_indices)),
                    "matched_score": same_mean,
                    "matched_ci_low": same_low,
                    "matched_ci_high": same_high,
                    "control_score": control_mean,
                    "control_ci_low": control_low,
                    "control_ci_high": control_high,
                    **delta_stats,
                    **retrieval,
                }
                rows.append(row)
                pair_rows_by_control[control_type].append(row)

        for control_type, pair_rows in pair_rows_by_control.items():
            if not pair_rows:
                continue
            mean_row = {
                "arm": arm,
                "layer": layer_idx,
                "layer_name": layers[layer_idx],
                "format_a": "__mean__",
                "format_b": "__mean__",
                "control_type": control_type,
                "n_concepts": int(len(query_indices)),
            }
            for key in [
                "matched_score",
                "matched_ci_low",
                "matched_ci_high",
                "control_score",
                "control_ci_low",
                "control_ci_high",
                "paired_delta",
                "ci_low",
                "ci_high",
                "p_perm_greater",
                "retrieval_top1",
                "retrieval_top5",
                "retrieval_margin",
            ]:
                mean_row[key] = mean_or_nan([float(row[key]) for row in pair_rows])
            rows.append(mean_row)

    summary = summarize_arm(rows, opts)
    return rows, summary


def summarize_arm(rows: list[dict[str, Any]], opts: RunOptions) -> list[dict[str, Any]]:
    start, end = parse_layer_band(opts.mid_layers)
    summaries = []
    for control_type in opts.controls:
        mean_rows = [
            row
            for row in rows
            if row["control_type"] == control_type
            and row["format_a"] == "__mean__"
            and math.isfinite(float(row["paired_delta"]))
        ]
        if not mean_rows:
            continue
        mid_rows = [row for row in mean_rows if start <= int(row["layer"]) <= end]
        best = max(mean_rows, key=lambda row: float(row["paired_delta"]))
        arm = str(mean_rows[0]["arm"])
        summaries.append(
            {
                "arm": arm,
                "control_type": control_type,
                "mid_layer_start": start,
                "mid_layer_end": end,
                "mid_paired_delta": mean_or_nan([float(row["paired_delta"]) for row in mid_rows]),
                "mid_ci_low": mean_or_nan([float(row["ci_low"]) for row in mid_rows]),
                "mid_ci_high": mean_or_nan([float(row["ci_high"]) for row in mid_rows]),
                "mid_p_perm_greater": mean_or_nan([float(row["p_perm_greater"]) for row in mid_rows]),
                "mid_matched_score": mean_or_nan([float(row["matched_score"]) for row in mid_rows]),
                "mid_control_score": mean_or_nan([float(row["control_score"]) for row in mid_rows]),
                "mid_retrieval_top1": mean_or_nan([float(row["retrieval_top1"]) for row in mid_rows]),
                "mid_retrieval_top5": mean_or_nan([float(row["retrieval_top5"]) for row in mid_rows]),
                "best_layer": best["layer"],
                "best_layer_name": best["layer_name"],
                "best_paired_delta": best["paired_delta"],
                "n_layers_scored": len(mean_rows),
                "n_concepts": int(mean_rows[0]["n_concepts"]),
            }
        )
    return summaries


def stable_int(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 1_000_003
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def metric_schema() -> dict[str, Any]:
    return {
        "controls_by_layer.csv": {
            "arm": "model arm / adapter condition",
            "layer": "integer layer index; 0 is embedding for current hidden-state files",
            "layer_name": "stored layer label",
            "format_a": "source prompt spoke; __mean__ rows average format pairs",
            "format_b": "target prompt spoke; __mean__ rows average format pairs",
            "control_type": "random, close_neighbor, far_neighbor, lexical, or category_proxy",
            "n_concepts": "number of query concepts scored",
            "matched_score": "mean cosine similarity for same concept across spokes",
            "control_score": "mean cosine similarity for matched wrong-control concept",
            "paired_delta": "mean matched_score - control_score over concepts",
            "ci_low": "bootstrap 2.5% bound over concept-level paired deltas",
            "ci_high": "bootstrap 97.5% bound over concept-level paired deltas",
            "p_perm_greater": "one-sided paired sign-flip p-value for positive delta",
            "retrieval_top1": "same-concept retrieval top-1 among all concepts",
            "retrieval_top5": "same-concept retrieval top-5 among all concepts",
            "retrieval_margin": "same-concept similarity minus best nonself similarity",
        },
        "control_summary.csv": {
            "mid_paired_delta": "mean paired_delta over the fixed mid-layer band",
            "mid_p_perm_greater": "mean layer-level sign-flip p-value over the mid-layer band",
            "best_layer_name": "descriptive best layer by paired_delta; not a primary selector",
        },
        "control_assignments.csv": {
            "control_type": "control family assigned before scoring hidden-state similarities",
            "control_concept": "wrong concept used for the control",
            "sstar_similarity": "S* similarity between query concept and control concept",
            "lexical_similarity": "surface-form lexical score between query and control",
        },
    }


def build_manifest(
    opts: RunOptions,
    concepts: list[str],
    formats: list[str],
    layer_names: list[str],
    query_indices: np.ndarray,
    available_arms: list[dict[str, Any]],
    category_source: str,
    outputs: dict[str, str],
) -> dict[str, Any]:
    start, end = parse_layer_band(opts.mid_layers)
    selected_layers = opts.layers or list(range(len(layer_names)))
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "script": "src/sft/run_semantic_hub_memp.py",
        "paper_source": "https://arxiv.org/abs/2411.04986",
        "method_note": (
            "Text-only MEMP analogue of semantic-hub Eq. 1: same concept across "
            "prompt spokes vs controlled wrong concepts."
        ),
        "dry_run": opts.dry_run,
        "inputs": {
            "hidden_dir": str(opts.hidden_dir),
            "neighbor_table": str(opts.hidden_dir / "neighbor_table.csv"),
            "config": str(opts.config_path),
        },
        "arms": available_arms,
        "concepts": {
            "n_available": len(concepts),
            "n_scored": int(len(query_indices)),
            "subset_policy": "first_n_from_hidden_state_order" if opts.max_concepts else "all",
            "first_scored": [concepts[int(i)] for i in query_indices[:10]],
        },
        "formats": formats,
        "layers": {
            "n_available": len(layer_names),
            "selected_layers": selected_layers,
            "mid_layer_start": start,
            "mid_layer_end": end,
            "mid_layer_names": [
                layer_names[i] for i in selected_layers if start <= i <= end and i < len(layer_names)
            ],
        },
        "controls": {
            "selected": opts.controls,
            "category_source": category_source,
            "category_clusters": opts.category_clusters,
        },
        "statistics": {
            "n_boot": opts.n_boot,
            "n_perm": opts.n_perm,
            "seed": opts.seed,
        },
        "outputs": outputs,
        "paper_method_pieces": {
            "implemented": [
                "relative matched-vs-control similarity",
                "middle-layer primary readout",
                "random, S*-close, S*-far, lexical, and category-proxy controls",
                "bootstrap CIs and paired sign-flip p-values",
                "machine-readable config, manifest, and schema",
            ],
            "still_planned": [
                "logit-lens semantic anchoring",
                "symbolic S* spoke extraction",
                "causal activation interventions",
                "cluster-corrected layer statistics",
            ],
        },
    }


def write_report(out_dir: Path, summaries: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    lines = [
        "# Semantic Hub MEMP Harness Report",
        "",
        f"Created UTC: {manifest['created_utc']}",
        "",
        "## Status",
        "",
    ]
    if manifest["dry_run"]:
        lines.append("Dry run completed: manifest, control assignments, and schemas were validated without scoring hidden states.")
    else:
        lines.append("Computed paper-style matched-vs-control similarity from existing semantic-hub hidden states.")
    lines.extend(
        [
            "",
            "## Method",
            "",
            "- Paper basis: Wu et al. semantic-hub relative similarity, adapted to same-concept prompt spokes.",
            "- Prompt spokes: `" + "`, `".join(manifest["formats"]) + "`.",
            "- Primary layer band: `{}`-`{}`.".format(
                manifest["layers"]["mid_layer_start"],
                manifest["layers"]["mid_layer_end"],
            ),
            "- Controls: `" + "`, `".join(manifest["controls"]["selected"]) + "`.",
            "- Category control source: `{}`.".format(manifest["controls"]["category_source"]),
            "",
            "## Mid-Layer Summary",
            "",
            "| Arm | Control | Delta | CI Low | CI High | p | Top-1 | Top-5 |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    if summaries:
        for row in sorted(summaries, key=lambda item: (item["control_type"], -float(item["mid_paired_delta"]))):
            lines.append(
                "| `{arm}` | `{control}` | {delta:.4f} | {low:.4f} | {high:.4f} | {p:.4f} | {top1:.4f} | {top5:.4f} |".format(
                    arm=row["arm"],
                    control=row["control_type"],
                    delta=float(row["mid_paired_delta"]),
                    low=float(row["mid_ci_low"]),
                    high=float(row["mid_ci_high"]),
                    p=float(row["mid_p_perm_greater"]),
                    top1=float(row["mid_retrieval_top1"]),
                    top5=float(row["mid_retrieval_top5"]),
                )
            )
    else:
        lines.append("| n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines.extend(
        [
            "",
            "## Readout",
            "",
            "- `paired_delta > 0` means same-concept cross-spoke states beat that control.",
            "- `close_neighbor` and `category_proxy` are stricter than random and far controls.",
            "- `category_proxy` is an S*-cluster proxy until an explicit category table is supplied.",
            "- Best-layer rows are descriptive; the primary scientific gate is the fixed mid-layer band.",
            "",
            "## Files",
            "",
        ]
    )
    for label, path in manifest["outputs"].items():
        lines.append(f"- {label}: `{path}`")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def validate_output_target(out_dir: Path, overwrite: bool) -> None:
    if out_dir.exists() and any(out_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"{out_dir} is not empty; pass --overwrite or choose another --out-dir")
    out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    if args.write_default_config:
        args.config.parent.mkdir(parents=True, exist_ok=True)
        args.config.write_text(json.dumps(default_config(), indent=2, sort_keys=True) + "\n")
        print(f"[write] {args.config}")
        return

    config = load_json(args.config)
    if not config:
        config = default_config()
    opts = resolve_options(args, config)
    validate_output_target(opts.out_dir, opts.overwrite)

    neighbor_table = opts.hidden_dir / "neighbor_table.csv"
    if not neighbor_table.exists():
        raise FileNotFoundError(neighbor_table)

    available_arms: list[dict[str, Any]] = []
    first_metadata: tuple[list[str], list[str], list[str], tuple[int, ...]] | None = None
    for arm in opts.arms:
        path = opts.hidden_dir / f"{arm}.npz"
        status = "available" if path.exists() else "missing_placeholder"
        entry: dict[str, Any] = {"arm": arm, "status": status, "path": str(path)}
        if path.exists():
            metadata = read_hidden_metadata(path)
            concepts, formats, layer_names, shape = metadata
            entry["shape"] = list(shape)
            if first_metadata is None:
                first_metadata = metadata
        available_arms.append(entry)

    if first_metadata is None:
        raise FileNotFoundError(f"no requested arms found in {opts.hidden_dir}")

    concepts, formats, layer_names, _shape = first_metadata
    selected_layers = opts.layers or list(range(len(layer_names)))
    bad_layers = [layer for layer in selected_layers if layer < 0 or layer >= len(layer_names)]
    if bad_layers:
        raise ValueError(f"selected layers out of range: {bad_layers}")
    n_scored = len(concepts) if opts.max_concepts is None else min(opts.max_concepts, len(concepts))
    query_indices = np.arange(n_scored, dtype=int)

    close_idx, far_idx, _neighbor_rows = read_neighbor_indices(neighbor_table, concepts)
    sstar = read_sstar(concepts)
    category_labels, category_source = read_category_labels(opts, concepts, sstar)
    all_controls, assignments = build_control_indices(
        concepts,
        close_idx,
        far_idx,
        category_labels,
        sstar,
        opts,
    )
    selected_controls = {name: all_controls[name] for name in opts.controls}

    outputs = {
        "manifest": str(opts.out_dir / "manifest.json"),
        "run_config": str(opts.out_dir / "run_config.json"),
        "metric_schema": str(opts.out_dir / "metric_schema.json"),
        "control_assignments": str(opts.out_dir / "control_assignments.csv"),
        "controls_by_layer": str(opts.out_dir / "controls_by_layer.csv"),
        "control_summary": str(opts.out_dir / "control_summary.csv"),
        "report": str(opts.out_dir / "REPORT.md"),
    }

    write_csv(opts.out_dir / "control_assignments.csv", assignments)
    (opts.out_dir / "metric_schema.json").write_text(json.dumps(metric_schema(), indent=2, sort_keys=True) + "\n")
    (opts.out_dir / "run_config.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")

    all_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    if not opts.dry_run:
        for arm in opts.arms:
            path = opts.hidden_dir / f"{arm}.npz"
            if not path.exists():
                print(f"[skip] missing {path}", flush=True)
                continue
            print(f"[arm] {arm}", flush=True)
            rows, summary = compute_arm(path, selected_controls, opts, query_indices)
            all_rows.extend(rows)
            summaries.extend(summary)
        write_csv(opts.out_dir / "controls_by_layer.csv", all_rows)
        write_csv(opts.out_dir / "control_summary.csv", summaries)
    else:
        write_csv(opts.out_dir / "controls_by_layer.csv", [], CONTROLS_BY_LAYER_FIELDS)
        write_csv(opts.out_dir / "control_summary.csv", [], CONTROL_SUMMARY_FIELDS)

    manifest = build_manifest(
        opts=opts,
        concepts=concepts,
        formats=formats,
        layer_names=layer_names,
        query_indices=query_indices,
        available_arms=available_arms,
        category_source=category_source,
        outputs=outputs,
    )
    (opts.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    write_report(opts.out_dir, summaries, manifest)
    print(f"[done] wrote {opts.out_dir}", flush=True)


if __name__ == "__main__":
    main()
