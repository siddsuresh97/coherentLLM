"""Human-behavior evaluation for Task 6.

The THINGS odd-one-out score compares each model's triplet embedding choices to
the SPoSE-derived human majority choice for the same triplets.
"""
from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODELS = {
    "base": ROOT / "results" / "sft_eval" / "llama-3.1-8b-instruct_triplet_d5.npy",
    "real": ROOT / "results" / "sft_eval" / "llama31-sft-real_triplet_d5.npy",
    "lowLR": ROOT / "results" / "sft_eval" / "llama31-mit-lowLR_triplet_d5.npy",
    "lowrank": ROOT / "results" / "sft_eval" / "llama31-mit-lowrank_triplet_d5.npy",
}


@dataclass(frozen=True)
class Triplet:
    anchor: str
    left: str
    right: str


def read_concepts(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_triplets(path: Path) -> list[Triplet]:
    rows = []
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3 or not all(parts):
            continue
        rows.append(Triplet(*parts))
    return rows


def cosine_sim_matrix(embedding: np.ndarray) -> np.ndarray:
    emb = np.asarray(embedding, dtype=np.float64)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.clip(norms, 1e-12, None)
    sim = emb @ emb.T
    sim = (sim + sim.T) / 2.0
    np.fill_diagonal(sim, 1.0)
    return sim


def odd_one_out_agreement(
    model_sim: np.ndarray,
    human_sim: np.ndarray,
    concepts: list[str],
    triplets: list[Triplet],
) -> tuple[float, int, int]:
    idx = {concept: i for i, concept in enumerate(concepts)}
    agree = 0
    total = 0
    ties = 0
    for triplet in triplets:
        if triplet.anchor not in idx or triplet.left not in idx or triplet.right not in idx:
            continue
        a = idx[triplet.anchor]
        l = idx[triplet.left]
        r = idx[triplet.right]
        human_delta = float(human_sim[a, l] - human_sim[a, r])
        model_delta = float(model_sim[a, l] - model_sim[a, r])
        if human_delta == 0.0 or model_delta == 0.0:
            ties += 1
            continue
        agree += int((human_delta > 0.0) == (model_delta > 0.0))
        total += 1
    return (agree / total if total else math.nan), total, ties


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    upper = np.triu_indices(matrix.shape[0], k=1)
    return matrix[upper]


def similarity_rho(model_sim: np.ndarray, human_sim: np.ndarray) -> float:
    rho, _ = spearmanr(upper_triangle_values(human_sim), upper_triangle_values(model_sim))
    return float(rho)


def verdict(delta: float) -> str:
    if math.isnan(delta):
        return "NA"
    if delta > 0.02:
        return "GAIN"
    if delta < -0.02:
        return "DROP"
    return "FLAT"


def maybe_external_rows(external_summary: Path, base_values: dict[str, float]) -> list[dict]:
    if not external_summary.exists():
        return []
    rows = []
    with external_summary.open(newline="") as f:
        for row in csv.DictReader(f):
            benchmark = row.get("benchmark", "")
            model = row.get("model", "")
            if benchmark not in {
                "SimLex-999",
                "SimVerb-3500",
                "MEN",
                "WordSim-353",
                "RG-65",
                "MTurk-771",
            }:
                continue
            value = float(row["rho"])
            if model == "base":
                base_values[f"human_similarity::{benchmark}"] = value
    with external_summary.open(newline="") as f:
        for row in csv.DictReader(f):
            benchmark = row.get("benchmark", "")
            model = row.get("model", "")
            if benchmark not in {
                "SimLex-999",
                "SimVerb-3500",
                "MEN",
                "WordSim-353",
                "RG-65",
                "MTurk-771",
            }:
                continue
            value = float(row["rho"])
            key = f"human_similarity::{benchmark}"
            base = base_values.get(key, math.nan)
            rows.append(
                {
                    "group": "human-behavior",
                    "task": benchmark,
                    "model": model,
                    "metric": "spearman_rho",
                    "value": value,
                    "n": row.get("n_pairs", ""),
                    "base_value": base,
                    "delta_vs_base": value - base if not math.isnan(base) else math.nan,
                    "verdict": verdict(value - base if not math.isnan(base) else math.nan),
                    "source": "external_similarity",
                    "note": "human similarity judgment dataset",
                }
            )
    return rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out_dir", type=Path, default=ROOT / "results" / "sft_eval" / "human_tasks")
    ap.add_argument("--external_summary", type=Path, default=ROOT / "results" / "sft_eval" / "external" / "summary.csv")
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    concepts = read_concepts(ROOT / "data" / "scale128" / "concepts.csv")
    triplets = read_triplets(ROOT / "data" / "scale128" / "stimuli" / "triplets.csv")
    human_sim = np.load(ROOT / "data" / "scale128" / "human_spose_triplet_sim.npy")

    rows = []
    base_values: dict[str, float] = {}
    per_model_sims = {}
    for model_name in args.models:
        path = DEFAULT_MODELS[model_name]
        if not path.exists():
            rows.append(
                {
                    "group": "human-behavior",
                    "task": "THINGS odd-one-out",
                    "model": model_name,
                    "metric": "agreement",
                    "value": math.nan,
                    "n": 0,
                    "base_value": math.nan,
                    "delta_vs_base": math.nan,
                    "verdict": "NA",
                    "source": str(path),
                    "note": "skipped: model triplet embedding not found",
                }
            )
            continue
        sim = cosine_sim_matrix(np.load(path))
        per_model_sims[model_name] = sim
        agreement, n, ties = odd_one_out_agreement(sim, human_sim, concepts, triplets)
        rho = similarity_rho(sim, human_sim)
        if model_name == "base":
            base_values["THINGS odd-one-out::agreement"] = agreement
            base_values["THINGS triplet similarity::spearman_rho"] = rho
        for task, metric, value in [
            ("THINGS odd-one-out", "agreement", agreement),
            ("THINGS triplet similarity", "spearman_rho", rho),
        ]:
            key = f"{task}::{metric}"
            base = base_values.get(key, math.nan)
            delta = value - base if not math.isnan(base) else math.nan
            rows.append(
                {
                    "group": "human-behavior",
                    "task": task,
                    "model": model_name,
                    "metric": metric,
                    "value": value,
                    "n": n if task == "THINGS odd-one-out" else human_sim.shape[0] * (human_sim.shape[0] - 1) // 2,
                    "base_value": base,
                    "delta_vs_base": delta,
                    "verdict": verdict(delta),
                    "source": str(path),
                    "note": f"ties_skipped={ties}; human target=data/scale128/human_spose_triplet_sim.npy",
                }
            )

    rows.extend(maybe_external_rows(args.external_summary, base_values))
    rows.append(
        {
            "group": "human-behavior",
            "task": "Typicality/category norms",
            "model": "all",
            "metric": "not_run",
            "value": math.nan,
            "n": 0,
            "base_value": math.nan,
            "delta_vs_base": math.nan,
            "verdict": "NA",
            "source": "",
            "note": "skipped: no ready typicality/ranked category norm dataset found in repo",
        }
    )

    summary_path = args.out_dir / "summary.csv"
    fieldnames = [
        "group",
        "task",
        "model",
        "metric",
        "value",
        "n",
        "base_value",
        "delta_vs_base",
        "verdict",
        "source",
        "note",
    ]
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
