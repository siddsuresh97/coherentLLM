#!/usr/bin/env python3
"""Score Experiment 1 detection/localization from base/control/edit RDMs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def row_rms(matrix: np.ndarray) -> np.ndarray:
    n = matrix.shape[0]
    out = np.zeros(n, dtype=float)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        out[i] = float(np.sqrt(np.mean(matrix[i, mask] ** 2)))
    return out


def upper_rms(matrix: np.ndarray) -> float:
    vals = matrix[np.triu_indices_from(matrix, k=1)]
    return float(np.sqrt(np.mean(vals**2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-spec", required=True)
    parser.add_argument("--rdm-base", default=str(EXP_DIR / "artifacts" / "rdms" / "rdm_base.npy"))
    parser.add_argument("--rdm-control", required=True)
    parser.add_argument("--rdm-edit", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    args = parser.parse_args()

    edit_spec = Path(args.edit_spec)
    if not edit_spec.is_absolute():
        edit_spec = ROOT / edit_spec
    edit = load_json(edit_spec)
    items = load_json(EXP_DIR / "items.json")
    floor = load_json(EXP_DIR / "floor_stats.json")
    concepts = items["concepts"]
    target = edit["target_concept"]
    neighbor = edit.get("target_neighbor")
    if target not in concepts:
        raise ValueError(f"target {target!r} not in items.json")

    rdm_base = np.load(args.rdm_base)
    rdm_control = np.load(args.rdm_control)
    rdm_edit = np.load(args.rdm_edit)
    if rdm_base.shape != rdm_control.shape or rdm_base.shape != rdm_edit.shape:
        raise ValueError(
            f"RDM shapes differ: base={rdm_base.shape}, control={rdm_control.shape}, edit={rdm_edit.shape}"
        )

    delta_edit = rdm_edit - rdm_base
    delta_null = rdm_control - rdm_base
    delta_residual = rdm_edit - rdm_control
    edit_row = row_rms(delta_edit)
    null_row = row_rms(delta_null)
    residual_row = row_rms(delta_residual)
    if "protocol_floor" in floor:
        floor_rms = float(floor["protocol_floor"].get("overall_rms_floor_mean", 0.0))
        floor_source = "protocol_floor.overall_rms_floor_mean"
    else:
        floor_rms = float(floor.get("archived_split_half", {}).get("overall_rms_floor_median", 0.0))
        floor_source = "archived_split_half.overall_rms_floor_median"
    denom_control = np.maximum(null_row, args.epsilon)
    denom_floor = np.maximum(np.maximum(null_row, floor_rms), args.epsilon)
    snr_control_only = edit_row / denom_control
    snr_floor_adjusted = edit_row / denom_floor
    residual_snr_floor = residual_row / np.maximum(floor_rms, args.epsilon)

    ranking = sorted(
        [
            {
                "rank": 0,
                "concept": concept,
                "edit_row_rms": float(edit_row[i]),
                "null_row_rms": float(null_row[i]),
                "residual_row_rms": float(residual_row[i]),
                "snr_control_only": float(snr_control_only[i]),
                "snr_floor_adjusted": float(snr_floor_adjusted[i]),
                "residual_snr_floor": float(residual_snr_floor[i]),
            }
            for i, concept in enumerate(concepts)
        ],
        key=lambda row: row["snr_floor_adjusted"],
        reverse=True,
    )
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank

    residual_ranking = sorted(
        [
            {
                "rank": 0,
                "concept": concept,
                "residual_row_rms": float(residual_row[i]),
                "residual_snr_floor": float(residual_snr_floor[i]),
                "edit_row_rms": float(edit_row[i]),
                "null_row_rms": float(null_row[i]),
            }
            for i, concept in enumerate(concepts)
        ],
        key=lambda row: row["residual_snr_floor"],
        reverse=True,
    )
    for rank, row in enumerate(residual_ranking, start=1):
        row["rank"] = rank

    target_idx = concepts.index(target)
    pair_rows = []
    for j, concept in enumerate(concepts):
        if j == target_idx:
            continue
        pair_rows.append(
            {
                "concept": concept,
                "abs_target_delta": float(abs(delta_edit[target_idx, j])),
                "signed_target_delta": float(delta_edit[target_idx, j]),
                "abs_null_delta": float(abs(delta_null[target_idx, j])),
                "abs_residual_delta": float(abs(delta_residual[target_idx, j])),
                "signed_residual_delta": float(delta_residual[target_idx, j]),
            }
        )
    pair_rows = sorted(pair_rows, key=lambda row: row["abs_target_delta"], reverse=True)
    for rank, row in enumerate(pair_rows, start=1):
        row["rank"] = rank

    residual_pair_rows = sorted(pair_rows, key=lambda row: row["abs_residual_delta"], reverse=True)
    for rank, row in enumerate(residual_pair_rows, start=1):
        row["residual_rank"] = rank

    target_rank = next(row["rank"] for row in ranking if row["concept"] == target)
    target_residual_rank = next(row["rank"] for row in residual_ranking if row["concept"] == target)
    neighbor_pair_rank = None
    neighbor_residual_pair_rank = None
    if neighbor:
        neighbor_pair_rank = next((row["rank"] for row in pair_rows if row["concept"] == neighbor), None)
        neighbor_residual_pair_rank = next(
            (row["residual_rank"] for row in residual_pair_rows if row["concept"] == neighbor),
            None,
        )

    result = {
        "edit_id": edit["edit_id"],
        "target_concept": target,
        "target_neighbor": neighbor,
        "detection": {
            "target_rank": target_rank,
            "target_top_ranked": target_rank == 1,
            "target_snr_control_only": float(snr_control_only[target_idx]),
            "target_snr_floor_adjusted": float(snr_floor_adjusted[target_idx]),
            "boundary_crossed_control_only": bool(snr_control_only[target_idx] > 1.0),
            "boundary_crossed_floor_adjusted": bool(snr_floor_adjusted[target_idx] > 1.0),
        },
        "residual_detection": {
            "target_rank": target_residual_rank,
            "target_top_ranked": target_residual_rank == 1,
            "target_residual_row_rms": float(residual_row[target_idx]),
            "target_residual_snr_floor": float(residual_snr_floor[target_idx]),
            "boundary_crossed_floor": bool(residual_snr_floor[target_idx] > 1.0),
        },
        "localization": {
            "neighbor_pair_rank": neighbor_pair_rank,
            "neighbor_residual_pair_rank": neighbor_residual_pair_rank,
            "top_target_pairs": pair_rows[:10],
            "top_residual_target_pairs": residual_pair_rows[:10],
        },
        "global": {
            "upper_rms_edit": upper_rms(delta_edit),
            "upper_rms_control_null": upper_rms(delta_null),
            "upper_rms_residual": upper_rms(delta_residual),
            "floor_rms_used": floor_rms,
            "floor_source": floor_source,
        },
        "ranking": ranking,
        "residual_ranking": residual_ranking,
        "inputs": {
            "rdm_base": args.rdm_base,
            "rdm_control": args.rdm_control,
            "rdm_edit": args.rdm_edit,
            "edit_spec": str(edit_spec.relative_to(ROOT)),
        },
    }

    out = Path(args.out) if args.out else EXP_DIR / "detection" / f"{edit['edit_id']}.json"
    if not out.is_absolute():
        out = ROOT / out
    write_json(out, result)
    print(f"[detection] wrote {display_path(out)}")
    print(
        f"[detection] target_rank={target_rank} "
        f"snr={result['detection']['target_snr_floor_adjusted']:.3f}"
    )


if __name__ == "__main__":
    main()
