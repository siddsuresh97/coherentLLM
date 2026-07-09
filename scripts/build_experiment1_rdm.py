#!/usr/bin/env python3
"""Build an Experiment 1 triplet RDM from a raw triplet CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_experiment1 import parse_triplet_raw, rdm_from_triplet_rows  # noqa: E402


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_items() -> dict:
    with (EXP_DIR / "items.json").open() as handle:
        return json.load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default=None, help="Run name under experiments/.../raw/<run>/triplet.csv")
    parser.add_argument("--raw-csv", default=None, help="Explicit raw triplet CSV path")
    parser.add_argument("--out", default=None, help="Output .npy path")
    args = parser.parse_args()
    if not args.run and not args.raw_csv:
        raise SystemExit("Pass --run or --raw-csv")

    if args.raw_csv:
        raw_csv = Path(args.raw_csv)
        if not raw_csv.is_absolute():
            raw_csv = ROOT / raw_csv
        run_name = raw_csv.parent.name
    else:
        run_name = args.run
        raw_csv = EXP_DIR / "raw" / run_name / "triplet.csv"
    if not raw_csv.exists():
        raise FileNotFoundError(raw_csv)

    out = Path(args.out) if args.out else EXP_DIR / "rdms" / run_name / "rdm.npy"
    if not out.is_absolute():
        out = ROOT / out
    items = load_items()
    rdm = rdm_from_triplet_rows(parse_triplet_raw(raw_csv), items["concepts"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, rdm)
    meta = {
        "run": run_name,
        "raw_csv": display_path(raw_csv),
        "out": display_path(out),
        "shape": list(rdm.shape),
        "aggregation": "1 - symmetrized triplet choice-rate similarity",
    }
    with out.with_suffix(".json").open("w") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"[rdm] {run_name}: {display_path(raw_csv)} -> {display_path(out)}")


if __name__ == "__main__":
    main()
