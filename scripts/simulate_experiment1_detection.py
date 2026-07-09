#!/usr/bin/env python3
"""Simulate Experiment 1 end-to-end through the triplet task.

This is a pipeline smoke test, not a substitute for model evidence. It takes the
feature-space candidate edits, treats feature-RDM displacement as a controllable
behavioral displacement, samples odd-one-out choices, rebuilds RDMs from those
choices, and scores detection SNR. Its purpose is to verify that the frozen
triplet protocol, aggregation, localization scorer, and heatmap machinery can
recover a known injected move before GPU LoRA runs are launched.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
SIM_DIR = EXP_DIR / "simulation"
SRC_SCRIPTS = ROOT / "scripts"
if str(SRC_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SRC_SCRIPTS))

from run_experiment1 import (  # noqa: E402
    clean_text,
    load_feature_data,
    rdm_from_triplet_rows,
    selected_feature_rdm_from_matrix,
)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_triplets(path: Path) -> list[tuple[str, str, str]]:
    rows = []
    with path.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3:
                rows.append((clean_text(row[0]), clean_text(row[1]), clean_text(row[2])))
    return rows


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def sample_triplet_rows(
    rdm: np.ndarray,
    concepts: list[str],
    triplets: list[tuple[str, str, str]],
    seed: int,
    choice_temperature: float,
    lapse: float,
) -> list[tuple[str, str, str, str]]:
    rng = np.random.default_rng(seed)
    index = {concept: i for i, concept in enumerate(concepts)}
    rows = []
    temp = max(choice_temperature, 1e-6)
    for anchor, concept1, concept2 in triplets:
        ai = index[anchor]
        i1 = index[concept1]
        i2 = index[concept2]
        # Smaller RDM distance means more similar. Positive margin favors concept1.
        margin = (rdm[ai, i2] - rdm[ai, i1]) / temp
        p_concept1 = sigmoid(float(margin))
        p_concept1 = lapse * 0.5 + (1.0 - lapse) * p_concept1
        response = concept1 if rng.random() < p_concept1 else concept2
        rows.append((anchor, concept1, concept2, response))
    return rows


def write_triplet_csv(path: Path, rows: list[tuple[str, str, str, str]], prompt_variant: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "prompt", "response", "prompt_variant"])
        for anchor, concept1, concept2, response in rows:
            prompt = (
                f"[simulated] Which is more similar in semantic meaning to {anchor}: "
                f"{concept1} or {concept2}?"
            )
            writer.writerow([f"{anchor}|{concept1}|{concept2}", prompt, response, prompt_variant])


def sym_jitter(rdm: np.ndarray, rng: np.random.Generator, scale: float) -> np.ndarray:
    if scale <= 0:
        return rdm.copy()
    noise = rng.normal(loc=0.0, scale=scale, size=rdm.shape)
    noise = (noise + noise.T) / 2.0
    np.fill_diagonal(noise, 0.0)
    out = np.clip(rdm + noise, 0.0, 1.0)
    out = (out + out.T) / 2.0
    np.fill_diagonal(out, 0.0)
    return out


def edited_feature_rdm(edit: dict, config: dict, items: dict) -> np.ndarray:
    feature_data = load_feature_data(config)
    concepts = feature_data.concepts
    target = clean_text(edit["target_concept"])
    target_i = concepts.index(target)
    feature_lookup = {clean_text(name): idx for idx, name in enumerate(feature_data.feature_names)}
    matrix = feature_data.matrix.copy()
    for feature in edit["removed_features"]:
        idx = feature_lookup.get(clean_text(feature))
        if idx is not None:
            matrix[target_i, idx] = 0.0
    return selected_feature_rdm_from_matrix(matrix, feature_data, items)


def row_rms(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out[i] = float(np.sqrt(np.mean(matrix[i, mask] ** 2)))
    return out


def score_detection(
    concepts: list[str],
    target: str,
    neighbor: str | None,
    rdm_base: np.ndarray,
    rdm_control: np.ndarray,
    rdm_edit: np.ndarray,
    floor_epsilon: float,
) -> dict:
    delta_edit = rdm_edit - rdm_base
    delta_null = rdm_control - rdm_base
    edit_row = row_rms(delta_edit)
    null_row = row_rms(delta_null)
    empirical_floor = max(float(np.median(null_row)), floor_epsilon)
    denom = np.maximum(null_row, empirical_floor)
    snr = edit_row / denom
    ranking = sorted(
        [
            {
                "rank": 0,
                "concept": concept,
                "edit_row_rms": float(edit_row[i]),
                "null_row_rms": float(null_row[i]),
                "snr": float(snr[i]),
            }
            for i, concept in enumerate(concepts)
        ],
        key=lambda row: row["snr"],
        reverse=True,
    )
    for rank, row in enumerate(ranking, 1):
        row["rank"] = rank

    target_i = concepts.index(target)
    target_rank = next(row["rank"] for row in ranking if row["concept"] == target)
    pair_rows = []
    for j, concept in enumerate(concepts):
        if j == target_i:
            continue
        pair_rows.append(
            {
                "rank": 0,
                "concept": concept,
                "signed_target_delta": float(delta_edit[target_i, j]),
                "abs_target_delta": float(abs(delta_edit[target_i, j])),
                "abs_null_delta": float(abs(delta_null[target_i, j])),
            }
        )
    pair_rows.sort(key=lambda row: row["abs_target_delta"], reverse=True)
    for rank, row in enumerate(pair_rows, 1):
        row["rank"] = rank
    neighbor_pair_rank = None
    if neighbor:
        neighbor_pair_rank = next((row["rank"] for row in pair_rows if row["concept"] == neighbor), None)
    return {
        "target_rank": int(target_rank),
        "target_top_ranked": bool(target_rank == 1),
        "target_snr": float(snr[target_i]),
        "target_edit_row_rms": float(edit_row[target_i]),
        "target_null_row_rms": float(null_row[target_i]),
        "empirical_floor": empirical_floor,
        "neighbor_pair_rank": neighbor_pair_rank,
        "ranking": ranking,
        "top_target_pairs": pair_rows[:10],
    }


def simulate_one(
    edit: dict,
    config: dict,
    items: dict,
    triplets: list[tuple[str, str, str]],
    seed: int,
    transfer_gain: float,
    control_drift: float,
    edit_drift: float,
    choice_temperature: float,
    lapse: float,
    floor_epsilon: float,
    write_raw: bool,
    write_rdms: bool,
    write_detection_json: bool,
) -> dict:
    concepts = [clean_text(concept) for concept in items["concepts"]]
    base_feature_rdm = np.load(EXP_DIR / "artifacts" / "rdms" / "feature_rdm_base.npy")
    edit_feature_rdm = edited_feature_rdm(edit, config, items)
    true_delta = edit_feature_rdm - base_feature_rdm
    rng = np.random.default_rng(seed)
    true_base = base_feature_rdm.copy()
    true_control = sym_jitter(true_base, rng, control_drift)
    true_edit = sym_jitter(np.clip(true_base + transfer_gain * true_delta, 0.0, 1.0), rng, edit_drift)

    base_rows = sample_triplet_rows(true_base, concepts, triplets, seed * 1000 + 11, choice_temperature, lapse)
    control_rows = sample_triplet_rows(true_control, concepts, triplets, seed * 1000 + 23, choice_temperature, lapse)
    edit_rows = sample_triplet_rows(true_edit, concepts, triplets, seed * 1000 + 37, choice_temperature, lapse)
    raw_artifacts = {}
    if write_raw:
        raw_seed_dir = SIM_DIR / "raw" / f"seed_{seed:03d}"
        base_raw = raw_seed_dir / "base" / "triplet.csv"
        control_raw = raw_seed_dir / "control" / "triplet.csv"
        edit_raw = raw_seed_dir / edit["edit_id"] / "triplet.csv"
        write_triplet_csv(base_raw, base_rows, "simulated_base")
        write_triplet_csv(control_raw, control_rows, "simulated_control")
        write_triplet_csv(edit_raw, edit_rows, "simulated_edit")
        raw_artifacts = {
            "raw_base": str(base_raw.relative_to(ROOT)),
            "raw_control": str(control_raw.relative_to(ROOT)),
            "raw_edit": str(edit_raw.relative_to(ROOT)),
        }

    rdm_base = rdm_from_triplet_rows(base_rows, concepts)
    rdm_control = rdm_from_triplet_rows(control_rows, concepts)
    rdm_edit = rdm_from_triplet_rows(edit_rows, concepts)
    rdm_artifacts = {}
    if write_rdms:
        rdm_seed_dir = SIM_DIR / "rdms" / f"seed_{seed:03d}" / edit["edit_id"]
        rdm_seed_dir.mkdir(parents=True, exist_ok=True)
        base_rdm_path = rdm_seed_dir / "rdm_base.npy"
        control_rdm_path = rdm_seed_dir / "rdm_control.npy"
        edit_rdm_path = rdm_seed_dir / "rdm_edit.npy"
        np.save(base_rdm_path, rdm_base)
        np.save(control_rdm_path, rdm_control)
        np.save(edit_rdm_path, rdm_edit)
        rdm_artifacts = {
            "rdm_base": str(base_rdm_path.relative_to(ROOT)),
            "rdm_control": str(control_rdm_path.relative_to(ROOT)),
            "rdm_edit": str(edit_rdm_path.relative_to(ROOT)),
        }

    score = score_detection(
        concepts=concepts,
        target=edit["target_concept"],
        neighbor=edit.get("target_neighbor"),
        rdm_base=rdm_base,
        rdm_control=rdm_control,
        rdm_edit=rdm_edit,
        floor_epsilon=floor_epsilon,
    )
    result = {
        "edit_id": edit["edit_id"],
        "mode": edit["mode"],
        "seed": seed,
        "simulator": {
            "transfer_gain": transfer_gain,
            "control_drift": control_drift,
            "edit_drift": edit_drift,
            "choice_temperature": choice_temperature,
            "lapse": lapse,
            "floor_epsilon": floor_epsilon,
        },
        "intended_displacement_feature_rdm": edit["intended_displacement_feature_rdm"],
        "detection": score,
        "artifacts": {**rdm_artifacts, **raw_artifacts},
    }
    if write_detection_json:
        write_json(SIM_DIR / "detection" / f"{edit['edit_id']}_seed_{seed:03d}.json", result)
    return result


def plot_heatmap(summary: pd.DataFrame, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = summary.pivot(index="mode", columns="fraction", values="median_snr")
    pivot = pivot.reindex(index=["concentrated", "diffuse"])
    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{float(col):.2g}" for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Feature pool fraction removed")
    ax.set_title("Simulated detection SNR from triplet choices")
    for (i, j), value in np.ndenumerate(data):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 1.8 else "black")
    try:
        ax.contour(data, levels=[1.0], colors="white", linewidths=1.5)
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Median target SNR over seeds")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--transfer-gain", type=float, default=0.75)
    parser.add_argument("--control-drift", type=float, default=0.018)
    parser.add_argument("--edit-drift", type=float, default=0.018)
    parser.add_argument("--choice-temperature", type=float, default=0.12)
    parser.add_argument("--lapse", type=float, default=0.03)
    parser.add_argument("--floor-epsilon", type=float, default=0.015)
    parser.add_argument(
        "--write-raw",
        action="store_true",
        help="Also write simulated triplet CSVs. Off by default because they are large and regenerable.",
    )
    parser.add_argument(
        "--write-rdms",
        action="store_true",
        help="Also write per-seed simulated RDM .npy files. Off by default because they are regenerable.",
    )
    parser.add_argument(
        "--write-detection-json",
        action="store_true",
        help="Also write per-seed simulated detection JSONs. Off by default because the CSV contains the sweep.",
    )
    args = parser.parse_args()

    config = load_json(EXP_DIR / "config.json")
    items = load_json(EXP_DIR / "items.json")
    triplets = read_triplets(EXP_DIR / "stimuli" / "triplets.csv")
    edit_paths = sorted((EXP_DIR / "candidate_edits").glob("*_drop_*.json"))
    edits = [load_json(path) for path in edit_paths]
    all_results = []
    for edit in edits:
        for seed in args.seeds:
            all_results.append(
                simulate_one(
                    edit=edit,
                    config=config,
                    items=items,
                    triplets=triplets,
                    seed=seed,
                    transfer_gain=args.transfer_gain,
                    control_drift=args.control_drift,
                    edit_drift=args.edit_drift,
                    choice_temperature=args.choice_temperature,
                    lapse=args.lapse,
                    floor_epsilon=args.floor_epsilon,
                    write_raw=args.write_raw,
                    write_rdms=args.write_rdms,
                    write_detection_json=args.write_detection_json,
                )
            )

    rows = []
    for result in all_results:
        edit_id = result["edit_id"]
        fraction = float(edit_id.rsplit("_", 1)[-1]) / 100.0
        rows.append(
            {
                "edit_id": edit_id,
                "mode": result["mode"],
                "fraction": fraction,
                "seed": result["seed"],
                "target_snr": result["detection"]["target_snr"],
                "target_rank": result["detection"]["target_rank"],
                "target_top_ranked": result["detection"]["target_top_ranked"],
                "feature_row_l2_rms": result["intended_displacement_feature_rdm"]["row_l2_rms"],
                "feature_locality_gini": result["intended_displacement_feature_rdm"]["locality_gini_abs_delta"],
            }
        )
    df = pd.DataFrame(rows)
    detail_path = SIM_DIR / "simulated_detection_rows.csv"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(detail_path, index=False)
    summary = (
        df.groupby(["edit_id", "mode", "fraction"], as_index=False)
        .agg(
            median_snr=("target_snr", "median"),
            min_snr=("target_snr", "min"),
            max_snr=("target_snr", "max"),
            top_rank_rate=("target_top_ranked", "mean"),
            median_rank=("target_rank", "median"),
            feature_row_l2_rms=("feature_row_l2_rms", "first"),
            feature_locality_gini=("feature_locality_gini", "first"),
        )
        .sort_values(["mode", "fraction"])
    )
    summary_path = SIM_DIR / "resolution_heatmap_simulated.csv"
    summary.to_csv(summary_path, index=False)
    plot_heatmap(summary, SIM_DIR / "resolution_heatmap_simulated.png")
    write_json(
        SIM_DIR / "simulation_summary.json",
        {
            "status": "simulated_pipeline_smoke_only",
            "warning": (
                "These results use an oracle mapping from feature-RDM edits to "
                "behavioral RDM changes. They validate analysis mechanics but do "
                "not satisfy Experiment 1's model-edit success criterion."
            ),
            "n_edits": len(edits),
            "seeds": args.seeds,
            "parameters": vars(args),
            "detail_csv": str(detail_path.relative_to(ROOT)),
            "heatmap_csv": str(summary_path.relative_to(ROOT)),
            "heatmap_png": str((SIM_DIR / "resolution_heatmap_simulated.png").relative_to(ROOT)),
            "best_cells": summary.sort_values("median_snr", ascending=False).head(5).to_dict(orient="records"),
        },
    )
    print(f"[simulate-exp1] wrote {detail_path.relative_to(ROOT)}")
    print(f"[simulate-exp1] wrote {summary_path.relative_to(ROOT)}")
    print(f"[simulate-exp1] wrote {(SIM_DIR / 'resolution_heatmap_simulated.png').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
