"""Create deterministic stimuli for SFT step-4 evaluation.

This only adds missing files. It does not overwrite existing stimuli or results.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STIM_DIR = ROOT / "data" / "scale128" / "stimuli"
DEFAULT_OUT_DIR = ROOT / "results" / "sft_eval" / "stimuli"


def _read_rows(path: Path):
    with path.open() as f:
        return [[c.strip() for c in row] for row in csv.reader(f) if any(row)]


def _write_rows(path: Path, rows, header=None):
    if path.exists():
        print(f"[skip] {path} exists")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        if header:
            w.writerow(header)
        w.writerows(rows)
    print(f"[write] {path}: {len(rows)} rows")


def make_ordered_pairs(stim_dir: Path):
    concepts = [r[0] for r in _read_rows(stim_dir / "concepts.csv")]
    rows = [(a, b) for a in concepts for b in concepts if a != b]
    _write_rows(stim_dir / "pairs.csv", rows)


def make_paraphrase_triplets(stim_dir: Path, out_dir: Path, rng: np.random.Generator):
    rows = _read_rows(stim_dir / "triplets.csv")
    n = min(1500, len(rows))
    idx = np.sort(rng.choice(len(rows), size=n, replace=False))
    out = [(i, *rows[j][:3]) for i, j in enumerate(idx)]
    _write_rows(out_dir / "paraphrase_triplets.csv", out,
                header=["item_id", "anchor", "c1", "c2"])


def make_paraphrase_pairs(stim_dir: Path, out_dir: Path, rng: np.random.Generator):
    rows = _read_rows(stim_dir / "pairs.csv")
    n = min(1000, len(rows))
    idx = np.sort(rng.choice(len(rows), size=n, replace=False))
    out = [(i, *rows[j][:2]) for i, j in enumerate(idx)]
    _write_rows(out_dir / "paraphrase_pairs.csv", out,
                header=["item_id", "a", "b"])


def make_paraphrase_features(out_dir: Path, base_verify_pairs: Path,
                             rng: np.random.Generator):
    if base_verify_pairs is None or not base_verify_pairs.exists():
        print("[skip] paraphrase_features.csv needs base verify_pairs.csv")
        return
    rows = _read_rows(base_verify_pairs)
    n = min(2000, len(rows))
    idx = np.sort(rng.choice(len(rows), size=n, replace=False))
    out = [(i, *rows[j][:2]) for i, j in enumerate(idx)]
    _write_rows(out_dir / "paraphrase_features.csv", out,
                header=["item_id", "feature", "concept"])


def make_transitivity(stim_dir: Path, out_dir: Path, rng: np.random.Generator):
    concepts = [r[0] for r in _read_rows(stim_dir / "concepts.csv")]
    rows = []
    seen = set()
    while len(rows) < 400:
        anchor, x, y, z = rng.choice(concepts, size=4, replace=False).tolist()
        key = (anchor, x, y, z)
        if key in seen:
            continue
        seen.add(key)
        rows.append((len(rows), anchor, x, y, z))
    _write_rows(out_dir / "transitivity_cycles.csv", rows,
                header=["cycle_id", "anchor", "x", "y", "z"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim_dir", default=str(DEFAULT_STIM_DIR))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--base_verify_pairs", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    stim_dir = Path(args.stim_dir)
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)

    make_ordered_pairs(stim_dir)
    make_paraphrase_triplets(stim_dir, out_dir, rng)
    make_paraphrase_pairs(stim_dir, out_dir, rng)
    make_transitivity(stim_dir, out_dir, rng)
    if args.base_verify_pairs:
        make_paraphrase_features(out_dir, Path(args.base_verify_pairs), rng)


if __name__ == "__main__":
    main()

