"""Paper-style semantic-hub similarity baselines.

This is the first stricter follow-up to ``run_semantic_hub.py``. It mirrors the
relative-similarity logic from Wu et al. (2025): same-concept cross-format
representations should be closer than controlled mismatches, especially in
middle layers.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIDDEN = ROOT / "results" / "sft_semantic_hub" / "hidden_states"
DEFAULT_OUT = ROOT / "results" / "sft_semantic_hub_paper"
DEFAULT_ARMS = [
    "base",
    "scrambled",
    "lowLR",
    "lowrank",
    "taskvec_a0p25",
    "taskvec_a0p5",
    "taskvec_a1p0",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden_dir", type=Path, default=DEFAULT_HIDDEN)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--mid_layers", default="10:20", help="Inclusive layer band, e.g. 10:20")
    ap.add_argument("--n_boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=17)
    return ap.parse_args()


def read_hidden(path: Path) -> tuple[np.ndarray, list[str], list[str], list[str]]:
    data = np.load(path, allow_pickle=False)
    hidden = data["hidden"].astype(np.float32, copy=False)
    concepts = [str(x) for x in data["concepts"]]
    formats = [str(x) for x in data["formats"]]
    layers = [str(x) for x in data["layer_names"]]
    if hidden.shape[:3] != (len(formats), len(concepts), len(layers)):
        raise ValueError(f"{path} shape/metadata mismatch: {hidden.shape}")
    return hidden, concepts, formats, layers


def read_neighbor_table(path: Path, concepts: list[str]) -> tuple[np.ndarray, np.ndarray]:
    concept_to_idx = {concept: i for i, concept in enumerate(concepts)}
    close = np.empty(len(concepts), dtype=int)
    far = np.empty(len(concepts), dtype=int)
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if len(rows) != len(concepts):
        raise ValueError(f"{path} has {len(rows)} rows for {len(concepts)} concepts")
    for row in rows:
        i = concept_to_idx[row["concept"]]
        close[i] = concept_to_idx[row["close_neighbor"]]
        far[i] = concept_to_idx[row["far_neighbor"]]
    return close, far


def parse_layer_band(text: str) -> tuple[int, int]:
    left, right = text.split(":", 1)
    start = int(left)
    end = int(right)
    if start > end:
        raise ValueError(text)
    return start, end


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def retrieval(sim: np.ndarray) -> dict[str, float]:
    def one_direction(mat: np.ndarray) -> tuple[float, float, float]:
        n = mat.shape[0]
        order = np.argsort(-mat, axis=1)
        ranks = np.empty(n, dtype=int)
        for i in range(n):
            ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
        off = mat.copy()
        np.fill_diagonal(off, -np.inf)
        margin = np.diag(mat) - np.max(off, axis=1)
        return float(np.mean(ranks == 1)), float(np.mean(ranks <= 5)), float(np.mean(margin))

    a1, a5, am = one_direction(sim)
    b1, b5, bm = one_direction(sim.T)
    return {
        "retrieval_top1": (a1 + b1) / 2.0,
        "retrieval_top5": (a5 + b5) / 2.0,
        "retrieval_margin": (am + bm) / 2.0,
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


def row_metrics(
    sim: np.ndarray,
    close_idx: np.ndarray,
    far_idx: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
) -> dict[str, float]:
    n = sim.shape[0]
    idx = np.arange(n)
    same = np.diag(sim)
    close = 0.5 * (sim[idx, close_idx] + sim[close_idx, idx])
    far = 0.5 * (sim[idx, far_idx] + sim[far_idx, idx])
    off = sim.copy()
    np.fill_diagonal(off, np.nan)
    random_baseline = np.nanmean(off, axis=1)

    same_mean, same_low, same_high = mean_ci(same, rng, n_boot)
    random_mean, random_low, random_high = mean_ci(random_baseline, rng, n_boot)
    close_mean, close_low, close_high = mean_ci(close, rng, n_boot)
    far_mean, far_low, far_high = mean_ci(far, rng, n_boot)

    same_minus_random, smr_low, smr_high = mean_ci(same - random_baseline, rng, n_boot)
    same_minus_close, smc_low, smc_high = mean_ci(same - close, rng, n_boot)
    same_minus_far, smf_low, smf_high = mean_ci(same - far, rng, n_boot)

    return {
        "same_similarity": same_mean,
        "same_similarity_ci_low": same_low,
        "same_similarity_ci_high": same_high,
        "random_similarity": random_mean,
        "random_similarity_ci_low": random_low,
        "random_similarity_ci_high": random_high,
        "close_neighbor_similarity": close_mean,
        "close_neighbor_similarity_ci_low": close_low,
        "close_neighbor_similarity_ci_high": close_high,
        "far_neighbor_similarity": far_mean,
        "far_neighbor_similarity_ci_low": far_low,
        "far_neighbor_similarity_ci_high": far_high,
        "same_minus_random": same_minus_random,
        "same_minus_random_ci_low": smr_low,
        "same_minus_random_ci_high": smr_high,
        "same_minus_close_neighbor": same_minus_close,
        "same_minus_close_neighbor_ci_low": smc_low,
        "same_minus_close_neighbor_ci_high": smc_high,
        "same_minus_far_neighbor": same_minus_far,
        "same_minus_far_neighbor_ci_low": smf_low,
        "same_minus_far_neighbor_ci_high": smf_high,
        **retrieval(sim),
    }


def mean_or_nan(values: list[float]) -> float:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return math.nan
    return float(np.nanmean(arr))


def compute_arm(
    path: Path,
    neighbor_table: Path,
    mid_band: tuple[int, int],
    rng: np.random.Generator,
    n_boot: int,
) -> tuple[list[dict], dict]:
    arm = path.stem
    hidden, concepts, formats, layers = read_hidden(path)
    close_idx, far_idx = read_neighbor_table(neighbor_table, concepts)
    rows: list[dict] = []
    n_concepts = len(concepts)

    for layer_idx, layer_name in enumerate(layers):
        pair_rows = []
        for fmt_a, fmt_b in combinations(range(len(formats)), 2):
            x = normalize_rows(hidden[fmt_a, :, layer_idx, :])
            y = normalize_rows(hidden[fmt_b, :, layer_idx, :])
            sim = x @ y.T
            metrics = row_metrics(sim, close_idx, far_idx, rng, n_boot)
            row = {
                "arm": arm,
                "layer": layer_idx,
                "layer_name": layer_name,
                "format_a": formats[fmt_a],
                "format_b": formats[fmt_b],
                "n_concepts": n_concepts,
                **metrics,
            }
            rows.append(row)
            pair_rows.append(row)

        mean_row = {
            "arm": arm,
            "layer": layer_idx,
            "layer_name": layer_name,
            "format_a": "__mean__",
            "format_b": "__mean__",
            "n_concepts": n_concepts,
        }
        metric_names = [key for key in pair_rows[0] if key not in mean_row]
        for key in metric_names:
            if isinstance(pair_rows[0].get(key), (float, int)):
                mean_row[key] = mean_or_nan([float(row[key]) for row in pair_rows])
        rows.append(mean_row)

    mean_rows = [row for row in rows if row["format_a"] == "__mean__"]
    start, end = mid_band
    mid = [row for row in mean_rows if start <= int(row["layer"]) <= end]
    best_random = max(mean_rows, key=lambda row: float(row["same_minus_random"]))
    best_close = max(mean_rows, key=lambda row: float(row["same_minus_close_neighbor"]))
    summary = {
        "arm": arm,
        "n_concepts": n_concepts,
        "formats": ",".join(formats),
        "mid_layer_start": start,
        "mid_layer_end": end,
        "mid_same_minus_random": mean_or_nan([float(row["same_minus_random"]) for row in mid]),
        "mid_same_minus_close_neighbor": mean_or_nan([
            float(row["same_minus_close_neighbor"]) for row in mid
        ]),
        "mid_same_minus_far_neighbor": mean_or_nan([float(row["same_minus_far_neighbor"]) for row in mid]),
        "mid_retrieval_top1": mean_or_nan([float(row["retrieval_top1"]) for row in mid]),
        "mid_retrieval_top5": mean_or_nan([float(row["retrieval_top5"]) for row in mid]),
        "best_random_layer": best_random["layer"],
        "best_random_same_minus_random": best_random["same_minus_random"],
        "best_close_layer": best_close["layer"],
        "best_close_same_minus_close_neighbor": best_close["same_minus_close_neighbor"],
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    by_random = sorted(summaries, key=lambda row: float(row["mid_same_minus_random"]), reverse=True)
    by_close = sorted(
        summaries,
        key=lambda row: float(row["mid_same_minus_close_neighbor"]),
        reverse=True,
    )
    top_random = by_random[0]
    top_close = by_close[0]
    lines = [
        "# Paper-Style Semantic Hub Similarity Report",
        "",
        "Created: 2026-07-08",
        "",
        "## Status",
        "",
        "Computed matched-vs-mismatched cross-format similarity baselines from",
        "existing semantic-hub hidden states.",
        "",
        "## Method",
        "",
        "- Same-concept cross-format cosine similarity is compared against random",
        "  nonmatching concepts, S*-close nonmatching concepts, and S*-far",
        "  nonmatching concepts.",
        "- Metrics are layer-resolved and averaged across format pairs.",
        f"- Mid-layer band: `{args.mid_layers}`.",
        f"- Bootstrap resamples per layer/format pair: `{args.n_boot}`.",
        "",
        "## Mid-Layer Summary",
        "",
        "| Arm | Same-Random | Same-Close | Same-Far | Top-1 | Top-5 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_random:
        lines.append(
            "| `{arm}` | {sr:.4f} | {sc:.4f} | {sf:.4f} | {t1:.4f} | {t5:.4f} |".format(
                arm=row["arm"],
                sr=float(row["mid_same_minus_random"]),
                sc=float(row["mid_same_minus_close_neighbor"]),
                sf=float(row["mid_same_minus_far_neighbor"]),
                t1=float(row["mid_retrieval_top1"]),
                t5=float(row["mid_retrieval_top5"]),
            )
        )
    lines.extend(
        [
            "",
            "## Initial Read",
            "",
            "- Best mid-layer same-minus-random arm: `{}` (`{:.4f}`).".format(
                top_random["arm"],
                float(top_random["mid_same_minus_random"]),
            ),
            "- Best mid-layer same-minus-S*-close arm: `{}` (`{:.4f}`).".format(
                top_close["arm"],
                float(top_close["mid_same_minus_close_neighbor"]),
            ),
            "- `same_minus_close_neighbor` is the stricter concept-identity test:",
            "  it asks whether the exact same concept beats a semantically close",
            "  but non-identical concept across prompt formats.",
            "",
            "## Files",
            "",
            f"- Layer metrics: `{out_dir / 'similarity_by_layer.csv'}`",
            f"- Summary: `{out_dir / 'similarity_summary.csv'}`",
            f"- Metadata: `{out_dir / 'similarity_meta.json'}`",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    mid_band = parse_layer_band(args.mid_layers)
    rng = np.random.default_rng(args.seed)
    neighbor_table = args.hidden_dir / "neighbor_table.csv"
    if not neighbor_table.exists():
        raise FileNotFoundError(neighbor_table)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for arm in args.arms:
        path = args.hidden_dir / f"{arm}.npz"
        if not path.exists():
            print(f"[skip] missing {path}")
            continue
        print(f"[arm] {arm}")
        rows, summary = compute_arm(path, neighbor_table, mid_band, rng, args.n_boot)
        all_rows.extend(rows)
        summaries.append(summary)

    write_csv(args.out_dir / "similarity_by_layer.csv", all_rows)
    write_csv(args.out_dir / "similarity_summary.csv", summaries)
    meta = {
        "hidden_dir": str(args.hidden_dir),
        "neighbor_table": str(neighbor_table),
        "arms": [row["arm"] for row in summaries],
        "mid_layers": args.mid_layers,
        "n_boot": args.n_boot,
        "seed": args.seed,
    }
    (args.out_dir / "similarity_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    write_report(args.out_dir, summaries, args)
    print(f"[write] {args.out_dir}")


if __name__ == "__main__":
    main()
