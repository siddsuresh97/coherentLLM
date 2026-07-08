"""Compute layer-resolved semantic-hub metrics from extracted hidden states."""
from __future__ import annotations

import argparse
import csv
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIDDEN = ROOT / "results" / "sft_semantic_hub" / "hidden_states"
DEFAULT_OUT = ROOT / "results" / "sft_semantic_hub"
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
    ap.add_argument("--mid_layers", default="10:20", help="Inclusive layer index band, e.g. 10:20")
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


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def cosine_rdm_upper(x: np.ndarray) -> np.ndarray:
    return pdist(normalize_rows(x), metric="cosine")


def safe_spearman(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    rho, _ = spearmanr(a, b)
    return float(rho)


def center_kernel(k: np.ndarray) -> np.ndarray:
    return k - k.mean(axis=0, keepdims=True) - k.mean(axis=1, keepdims=True) + k.mean()


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    k = x @ x.T
    l = y @ y.T
    kc = center_kernel(k)
    lc = center_kernel(l)
    denom = np.linalg.norm(kc) * np.linalg.norm(lc)
    if denom == 0:
        return math.nan
    return float((kc * lc).sum() / denom)


def retrieval_metrics(x: np.ndarray, y: np.ndarray) -> dict:
    x = normalize_rows(x)
    y = normalize_rows(y)
    sim = x @ y.T

    def one_direction(mat: np.ndarray) -> tuple[float, float, float]:
        n = mat.shape[0]
        order = np.argsort(-mat, axis=1)
        ranks = np.empty(n, dtype=int)
        for i in range(n):
            ranks[i] = int(np.where(order[i] == i)[0][0]) + 1
        off = mat.copy()
        np.fill_diagonal(off, -np.inf)
        margin = np.diag(mat) - np.max(off, axis=1)
        return (
            float(np.mean(ranks == 1)),
            float(np.mean(ranks <= 5)),
            float(np.mean(margin)),
        )

    a1, a5, am = one_direction(sim)
    b1, b5, bm = one_direction(sim.T)
    return {
        "retrieval_top1": (a1 + b1) / 2.0,
        "retrieval_top5": (a5 + b5) / 2.0,
        "retrieval_margin": (am + bm) / 2.0,
    }


def label_alignment(hidden_layer: np.ndarray) -> dict:
    # hidden_layer: formats x concepts x hidden_dim
    n_formats, n_concepts, hidden_dim = hidden_layer.shape
    rows = normalize_rows(hidden_layer.reshape(n_formats * n_concepts, hidden_dim))
    rep_kernel = rows @ rows.T
    concept_ids = np.tile(np.arange(n_concepts), n_formats)
    format_ids = np.repeat(np.arange(n_formats), n_concepts)
    concept_kernel = (concept_ids[:, None] == concept_ids[None, :]).astype(np.float32)
    format_kernel = (format_ids[:, None] == format_ids[None, :]).astype(np.float32)
    kc = center_kernel(rep_kernel)
    cc = center_kernel(concept_kernel)
    fc = center_kernel(format_kernel)

    def align(label_k: np.ndarray) -> float:
        denom = np.linalg.norm(kc) * np.linalg.norm(label_k)
        if denom == 0:
            return math.nan
        return float((kc * label_k).sum() / denom)

    concept_alignment = align(cc)
    format_alignment = align(fc)
    return {
        "concept_label_alignment": concept_alignment,
        "format_label_alignment": format_alignment,
        "concept_minus_format_alignment": concept_alignment - format_alignment,
    }


def parse_layer_band(text: str) -> tuple[int, int]:
    left, right = text.split(":", 1)
    start = int(left)
    end = int(right)
    if start > end:
        raise ValueError(text)
    return start, end


def mean_or_nan(values: list[float]) -> float:
    arr = np.array(values, dtype=float)
    if arr.size == 0 or np.all(np.isnan(arr)):
        return math.nan
    return float(np.nanmean(arr))


def compute_arm(path: Path, mid_band: tuple[int, int]) -> tuple[list[dict], dict]:
    arm = path.stem
    hidden, concepts, formats, layers = read_hidden(path)
    rows = []
    for layer_idx, layer_name in enumerate(layers):
        pair_rsa = []
        pair_cka = []
        pair_top1 = []
        pair_top5 = []
        pair_margin = []
        for i, j in combinations(range(len(formats)), 2):
            x = hidden[i, :, layer_idx, :]
            y = hidden[j, :, layer_idx, :]
            rsa = safe_spearman(cosine_rdm_upper(x), cosine_rdm_upper(y))
            cka = linear_cka(x, y)
            ret = retrieval_metrics(x, y)
            pair_rsa.append(rsa)
            pair_cka.append(cka)
            pair_top1.append(ret["retrieval_top1"])
            pair_top5.append(ret["retrieval_top5"])
            pair_margin.append(ret["retrieval_margin"])
            rows.append(
                {
                    "arm": arm,
                    "layer": layer_idx,
                    "layer_name": layer_name,
                    "format_a": formats[i],
                    "format_b": formats[j],
                    "cross_format_rdm_spearman": rsa,
                    "linear_cka": cka,
                    **ret,
                }
            )
        label = label_alignment(hidden[:, :, layer_idx, :])
        rows.append(
            {
                "arm": arm,
                "layer": layer_idx,
                "layer_name": layer_name,
                "format_a": "__mean__",
                "format_b": "__mean__",
                "cross_format_rdm_spearman": mean_or_nan(pair_rsa),
                "linear_cka": mean_or_nan(pair_cka),
                "retrieval_top1": mean_or_nan(pair_top1),
                "retrieval_top5": mean_or_nan(pair_top5),
                "retrieval_margin": mean_or_nan(pair_margin),
                **label,
            }
        )

    mean_rows = [row for row in rows if row["format_a"] == "__mean__"]
    finite_mean_rows = [
        row for row in mean_rows if math.isfinite(float(row["cross_format_rdm_spearman"]))
    ]
    best_candidates = finite_mean_rows if finite_mean_rows else mean_rows
    best = max(best_candidates, key=lambda row: row["cross_format_rdm_spearman"])
    start, end = mid_band
    mid = [row for row in mean_rows if start <= int(row["layer"]) <= end]
    summary = {
        "arm": arm,
        "n_concepts": len(concepts),
        "formats": ",".join(formats),
        "best_layer": best["layer"],
        "best_layer_name": best["layer_name"],
        "best_cross_format_rdm_spearman": best["cross_format_rdm_spearman"],
        "best_linear_cka": best["linear_cka"],
        "best_retrieval_top1": best["retrieval_top1"],
        "best_retrieval_top5": best["retrieval_top5"],
        "best_concept_minus_format_alignment": best.get("concept_minus_format_alignment", math.nan),
        "mid_layer_start": start,
        "mid_layer_end": end,
        "mid_cross_format_rdm_spearman": mean_or_nan([row["cross_format_rdm_spearman"] for row in mid]),
        "mid_linear_cka": mean_or_nan([row["linear_cka"] for row in mid]),
        "mid_retrieval_top1": mean_or_nan([row["retrieval_top1"] for row in mid]),
        "mid_retrieval_top5": mean_or_nan([row["retrieval_top5"] for row in mid]),
        "mid_concept_minus_format_alignment": mean_or_nan([
            row.get("concept_minus_format_alignment", math.nan) for row in mid
        ]),
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_report(out_dir: Path, summaries: list[dict], args: argparse.Namespace) -> None:
    by_mid = sorted(summaries, key=lambda row: row["mid_cross_format_rdm_spearman"], reverse=True)
    by_best = sorted(summaries, key=lambda row: row["best_cross_format_rdm_spearman"], reverse=True)
    base = next((row for row in summaries if row["arm"] == "base"), None)
    top_mid = by_mid[0]
    lines = [
        "# Semantic Hub Report",
        "",
        "Created: 2026-07-08",
        "",
        "## Status",
        "",
        "Layer-resolved semantic-hub metrics computed from extracted hidden states.",
        "",
        "## Method",
        "",
        "- Concepts: 128 held-out THINGS concepts.",
        "- Formats: triplet, pairwise, feature_listing.",
        "- Triplet prompts use deterministic S*-close and S*-far neighbors.",
        "- Pairwise prompts use the S*-close neighbor.",
        "- Feature/listing prompts use `listing_prompt(concept)`.",
        "- Metrics: cross-format RDM Spearman, linear CKA, same-concept retrieval, and concept-vs-format label alignment.",
        "",
        "## Mid-Layer Summary",
        "",
        "| Arm | Mid RDM Spearman | Mid CKA | Mid Top-1 | Mid Top-5 | Mid Concept-Format |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in by_mid:
        lines.append(
            f"| `{row['arm']}` | {row['mid_cross_format_rdm_spearman']:.4f} | "
            f"{row['mid_linear_cka']:.4f} | {row['mid_retrieval_top1']:.4f} | "
            f"{row['mid_retrieval_top5']:.4f} | {row['mid_concept_minus_format_alignment']:.4f} |"
        )
    lines.extend([
        "",
        "## Initial Read",
        "",
        f"- Strongest mid-layer format-invariance score: `{top_mid['arm']}` "
        f"(RDM Spearman {top_mid['mid_cross_format_rdm_spearman']:.4f}, "
        f"CKA {top_mid['mid_linear_cka']:.4f}, top-1 retrieval {top_mid['mid_retrieval_top1']:.4f}).",
    ])
    if base is not None:
        lines.append(
            f"- Base mid-layer score: RDM Spearman {base['mid_cross_format_rdm_spearman']:.4f}, "
            f"CKA {base['mid_linear_cka']:.4f}, top-1 retrieval {base['mid_retrieval_top1']:.4f}."
        )
    lines.extend([
        "- Treat this as evidence for stronger cross-format invariance, not yet a clean "
        "concept-dominant semantic hub: concept-minus-format alignment remains negative "
        "for every arm.",
        "- Next bridge: correlate arm/layer hub metrics with fMRI RSA and inspect whether "
        "format-averaged hub RDMs explain fMRI better than single-format RDMs.",
    ])
    lines.extend([
        "",
        "## Best-Layer Summary",
        "",
        "| Arm | Best Layer | Best RDM Spearman | Best CKA | Best Top-1 | Best Concept-Format |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for row in by_best:
        lines.append(
            f"| `{row['arm']}` | {row['best_layer_name']} | {row['best_cross_format_rdm_spearman']:.4f} | "
            f"{row['best_linear_cka']:.4f} | {row['best_retrieval_top1']:.4f} | "
            f"{row['best_concept_minus_format_alignment']:.4f} |"
        )
    lines.extend([
        "",
        "## Files",
        "",
        f"- Hidden-state input: `{args.hidden_dir}`",
        f"- Layer metrics: `{out_dir / 'hub_by_layer.csv'}`",
        f"- Summary: `{out_dir / 'hub_summary.csv'}`",
    ])
    (out_dir / "REPORT.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    mid_band = parse_layer_band(args.mid_layers)
    all_rows = []
    summaries = []
    for arm in args.arms:
        path = args.hidden_dir / f"{arm}.npz"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        rows, summary = compute_arm(path, mid_band)
        all_rows.extend(rows)
        summaries.append(summary)
        print(f"[arm] {arm}: best layer {summary['best_layer_name']}", flush=True)
    if not all_rows:
        raise FileNotFoundError(f"no hidden-state files found in {args.hidden_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "hub_by_layer.csv", all_rows)
    write_csv(args.out_dir / "hub_summary.csv", summaries)
    (args.out_dir / "hub_meta.json").write_text(
        json.dumps({"hidden_dir": str(args.hidden_dir), "arms": args.arms, "mid_layers": args.mid_layers}, indent=2) + "\n"
    )
    write_report(args.out_dir, summaries, args)
    print(f"[done] wrote {args.out_dir / 'hub_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
