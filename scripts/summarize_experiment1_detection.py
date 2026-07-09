#!/usr/bin/env python3
"""Summarize real Experiment 1 detection JSONs into a resolution heatmap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def edit_fraction(edit_id: str) -> float:
    return float(edit_id.rsplit("_", 1)[-1]) / 100.0


def plot_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pivot = df.pivot(index="mode", columns="fraction", values="snr")
    pivot = pivot.reindex(index=["concentrated", "diffuse"])
    data = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    im = ax.imshow(data, aspect="auto", cmap="viridis", vmin=0.0)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{float(col):.2g}" for col in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Feature pool fraction removed")
    ax.set_title("Experiment 1 detection SNR")
    for (i, j), value in np.ndenumerate(data):
        if np.isfinite(value):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 1.8 else "black")
    try:
        ax.contour(data, levels=[1.0], colors="white", linewidths=1.5)
    except Exception:
        pass
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Target SNR above control/floor")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--detection-dir", default=str(EXP_DIR / "detection"))
    parser.add_argument("--out-csv", default=str(EXP_DIR / "figs" / "resolution_heatmap.csv"))
    parser.add_argument("--out-png", default=str(EXP_DIR / "figs" / "resolution_heatmap.png"))
    args = parser.parse_args()

    detection_dir = Path(args.detection_dir)
    if not detection_dir.is_absolute():
        detection_dir = ROOT / detection_dir
    paths = sorted(detection_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"No detection JSONs found under {detection_dir.relative_to(ROOT)}")

    rows = []
    for path in paths:
        result = load_json(path)
        edit_id = result["edit_id"]
        edit_spec_path = EXP_DIR / "candidate_edits" / f"{edit_id}.json"
        if not edit_spec_path.exists():
            edit_spec_path = EXP_DIR / "edits" / f"{edit_id}.json"
        edit_spec = load_json(edit_spec_path)
        rows.append(
            {
                "edit_id": edit_id,
                "mode": edit_spec["mode"],
                "fraction": edit_fraction(edit_id),
                "snr": result["detection"]["target_snr_floor_adjusted"],
                "target_rank": result["detection"]["target_rank"],
                "target_top_ranked": result["detection"]["target_top_ranked"],
                "boundary_crossed": result["detection"]["boundary_crossed_floor_adjusted"],
                "feature_row_l2_rms": edit_spec["intended_displacement_feature_rdm"]["row_l2_rms"],
                "feature_locality_gini": edit_spec["intended_displacement_feature_rdm"]["locality_gini_abs_delta"],
            }
        )
    df = pd.DataFrame(rows).sort_values(["mode", "fraction"])
    out_csv = Path(args.out_csv)
    out_png = Path(args.out_png)
    if not out_csv.is_absolute():
        out_csv = ROOT / out_csv
    if not out_png.is_absolute():
        out_png = ROOT / out_png
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    plot_heatmap(df, out_png)

    summary = {
        "status": "real_detection_summary",
        "n_cells": len(df),
        "n_boundary_crossed": int(df["boundary_crossed"].sum()),
        "n_top_ranked": int(df["target_top_ranked"].sum()),
        "best_cell": df.sort_values("snr", ascending=False).head(1).to_dict(orient="records")[0],
        "csv": str(out_csv.relative_to(ROOT)),
        "png": str(out_png.relative_to(ROOT)),
    }
    with (EXP_DIR / "figs" / "resolution_heatmap_summary.json").open("w") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[summary] wrote {out_csv.relative_to(ROOT)}")
    print(f"[summary] wrote {out_png.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
