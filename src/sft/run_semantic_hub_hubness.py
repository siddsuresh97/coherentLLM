"""Measure nearest-neighbor hubness in semantic-hub hidden states."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HIDDEN = ROOT / "results" / "sft_semantic_hub" / "hidden_states"
DEFAULT_OUT = ROOT / "results" / "sft_semantic_hub" / "hubness"
DEFAULT_ARMS = [
    "base",
    "scrambled",
    "lowLR",
    "lowrank",
    "taskvec_a0p25",
    "taskvec_a0p5",
    "taskvec_a1p0",
]

PAIR_FIELDS = [
    "arm",
    "layer",
    "layer_name",
    "query_format",
    "candidate_format",
    "n_concepts",
    "top1_accuracy",
    "top5_accuracy",
    "top1_unique_fraction",
    "top1_max_occurrence",
    "top1_gini",
    "top1_entropy_norm",
    "top5_unique_fraction",
    "top5_max_occurrence",
    "top_hub_concept",
]

LAYER_FIELDS = [
    "arm",
    "layer",
    "layer_name",
    "n_concepts",
    "top1_accuracy",
    "top5_accuracy",
    "top1_unique_fraction",
    "top1_max_occurrence",
    "top1_gini",
    "top1_entropy_norm",
    "top5_unique_fraction",
    "top5_max_occurrence",
]

SUMMARY_FIELDS = [
    "arm",
    "mid_layer_start",
    "mid_layer_end",
    "mid_top1_accuracy",
    "mid_top5_accuracy",
    "mid_top1_unique_fraction",
    "mid_top1_max_occurrence",
    "mid_top1_gini",
    "mid_top1_entropy_norm",
    "mid_top5_unique_fraction",
    "mid_top5_max_occurrence",
    "hubness_read",
]

CONCEPT_FIELDS = [
    "arm",
    "concept_index",
    "concept",
    "mid_layer_start",
    "mid_layer_end",
    "top1_count",
    "top1_share",
    "top5_count",
    "top5_share",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden-dir", type=Path, default=DEFAULT_HIDDEN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--mid-layers", default="10:20", help="Inclusive layer band, e.g. 10:20")
    return ap.parse_args()


def parse_band(spec: str) -> tuple[int, int]:
    start, end = spec.split(":", 1)
    return int(start), int(end)


def normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    denom = np.linalg.norm(x, axis=1, keepdims=True)
    denom[denom == 0] = 1.0
    return x / denom


def gini(counts: np.ndarray) -> float:
    values = np.sort(counts.astype(np.float64))
    total = values.sum()
    if total <= 0:
        return 0.0
    n = values.size
    index = np.arange(1, n + 1, dtype=np.float64)
    return float((2 * np.sum(index * values) / (n * total)) - ((n + 1) / n))


def entropy_norm(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0].astype(np.float64) / total
    return float(-(p * np.log(p)).sum() / np.log(counts.size))


def mean_float(rows: list[dict[str, object]], key: str) -> float:
    vals = [float(row[key]) for row in rows]
    return float(np.mean(vals)) if vals else float("nan")


def score_pair(
    arm: str,
    layer: int,
    layer_name: str,
    query_format: str,
    candidate_format: str,
    query: np.ndarray,
    candidates: np.ndarray,
    concepts: np.ndarray,
) -> dict[str, object]:
    q = normalize(query)
    c = normalize(candidates)
    sims = q @ c.T
    n = sims.shape[0]
    top1 = np.argmax(sims, axis=1)
    top5 = np.argpartition(-sims, kth=min(4, n - 1), axis=1)[:, : min(5, n)]
    target = np.arange(n)

    top1_counts = np.bincount(top1, minlength=n)
    top5_counts = np.bincount(top5.reshape(-1), minlength=n)
    top_hub_idx = int(np.argmax(top1_counts))
    return {
        "arm": arm,
        "layer": layer,
        "layer_name": layer_name,
        "query_format": query_format,
        "candidate_format": candidate_format,
        "n_concepts": n,
        "top1_accuracy": float(np.mean(top1 == target)),
        "top5_accuracy": float(np.mean(np.any(top5 == target[:, None], axis=1))),
        "top1_unique_fraction": float(np.mean(top1_counts > 0)),
        "top1_max_occurrence": int(top1_counts.max()),
        "top1_gini": gini(top1_counts),
        "top1_entropy_norm": entropy_norm(top1_counts),
        "top5_unique_fraction": float(np.mean(top5_counts > 0)),
        "top5_max_occurrence": int(top5_counts.max()),
        "top_hub_concept": str(concepts[top_hub_idx]),
    }


def score_arm(
    path: Path, arm: str, mid_start: int, mid_end: int
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    data = np.load(path, allow_pickle=True)
    hidden = data["hidden"]
    concepts = data["concepts"]
    formats = [str(x) for x in data["formats"]]
    layer_names = [str(x) for x in data["layer_names"]]

    pair_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    mid_top1_counts = np.zeros(hidden.shape[1], dtype=np.int64)
    mid_top5_counts = np.zeros(hidden.shape[1], dtype=np.int64)
    mid_top1_total = 0
    mid_top5_total = 0
    for layer, layer_name in enumerate(layer_names):
        per_layer = []
        for q_idx, q_format in enumerate(formats):
            for c_idx, c_format in enumerate(formats):
                if q_idx == c_idx:
                    continue
                q = normalize(hidden[q_idx, :, layer, :])
                c = normalize(hidden[c_idx, :, layer, :])
                sims = q @ c.T
                n = sims.shape[0]
                top1 = np.argmax(sims, axis=1)
                top5 = np.argpartition(-sims, kth=min(4, n - 1), axis=1)[:, : min(5, n)]
                if mid_start <= layer <= mid_end:
                    mid_top1_counts += np.bincount(top1, minlength=n)
                    mid_top5_counts += np.bincount(top5.reshape(-1), minlength=n)
                    mid_top1_total += n
                    mid_top5_total += n * min(5, n)
                row = score_pair(
                    arm=arm,
                    layer=layer,
                    layer_name=layer_name,
                    query_format=q_format,
                    candidate_format=c_format,
                    query=q,
                    candidates=c,
                    concepts=concepts,
                )
                pair_rows.append(row)
                per_layer.append(row)
        layer_rows.append(
            {
                "arm": arm,
                "layer": layer,
                "layer_name": layer_name,
                "n_concepts": int(hidden.shape[1]),
                "top1_accuracy": mean_float(per_layer, "top1_accuracy"),
                "top5_accuracy": mean_float(per_layer, "top5_accuracy"),
                "top1_unique_fraction": mean_float(per_layer, "top1_unique_fraction"),
                "top1_max_occurrence": mean_float(per_layer, "top1_max_occurrence"),
                "top1_gini": mean_float(per_layer, "top1_gini"),
                "top1_entropy_norm": mean_float(per_layer, "top1_entropy_norm"),
                "top5_unique_fraction": mean_float(per_layer, "top5_unique_fraction"),
                "top5_max_occurrence": mean_float(per_layer, "top5_max_occurrence"),
            }
        )
    concept_rows = []
    for idx, concept in enumerate(concepts):
        top1_count = int(mid_top1_counts[idx])
        top5_count = int(mid_top5_counts[idx])
        concept_rows.append(
            {
                "arm": arm,
                "concept_index": idx,
                "concept": str(concept),
                "mid_layer_start": mid_start,
                "mid_layer_end": mid_end,
                "top1_count": top1_count,
                "top1_share": float(top1_count / mid_top1_total) if mid_top1_total else 0.0,
                "top5_count": top5_count,
                "top5_share": float(top5_count / mid_top5_total) if mid_top5_total else 0.0,
            }
        )
    concept_rows.sort(key=lambda row: (str(row["arm"]), -int(row["top1_count"]), str(row["concept"])))
    return pair_rows, layer_rows, concept_rows


def write_rows(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def hubness_read(row: dict[str, object]) -> str:
    top5 = float(row["mid_top5_accuracy"])
    unique = float(row["mid_top1_unique_fraction"])
    g = float(row["mid_top1_gini"])
    max_occ = float(row["mid_top1_max_occurrence"])
    if top5 >= 0.50 and unique >= 0.28 and max_occ <= 32:
        return "strong retrieval with reduced, not eliminated, hubness"
    if top5 >= 0.45 and unique >= 0.22:
        return "strong retrieval but still hub-skewed"
    if top5 >= 0.25 and (unique < 0.22 or g > 0.75):
        return "retrieval may be inflated by hub concepts"
    if top5 < 0.20:
        return "weak retrieval; hubness is not the main claim"
    return "mixed; inspect pair_by_layer rows"


def summarize(layer_rows: list[dict[str, object]], start: int, end: int) -> list[dict[str, object]]:
    by_arm: dict[str, list[dict[str, object]]] = {}
    for row in layer_rows:
        layer = int(row["layer"])
        if start <= layer <= end:
            by_arm.setdefault(str(row["arm"]), []).append(row)

    rows = []
    for arm in DEFAULT_ARMS:
        arm_rows = by_arm.get(arm, [])
        if not arm_rows:
            continue
        row: dict[str, object] = {
            "arm": arm,
            "mid_layer_start": start,
            "mid_layer_end": end,
            "mid_top1_accuracy": mean_float(arm_rows, "top1_accuracy"),
            "mid_top5_accuracy": mean_float(arm_rows, "top5_accuracy"),
            "mid_top1_unique_fraction": mean_float(arm_rows, "top1_unique_fraction"),
            "mid_top1_max_occurrence": mean_float(arm_rows, "top1_max_occurrence"),
            "mid_top1_gini": mean_float(arm_rows, "top1_gini"),
            "mid_top1_entropy_norm": mean_float(arm_rows, "top1_entropy_norm"),
            "mid_top5_unique_fraction": mean_float(arm_rows, "top5_unique_fraction"),
            "mid_top5_max_occurrence": mean_float(arm_rows, "top5_max_occurrence"),
        }
        row["hubness_read"] = hubness_read(row)
        rows.append(row)
    return rows


def fmt(value: object, digits: int = 4) -> str:
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_report(path: Path, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Semantic Hub Hubness Control",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Question",
        "",
        "Do semantic-hub retrieval gains reflect broad same-concept alignment, or are they inflated",
        "because a few concepts become nearest-neighbor hubs for many unrelated queries?",
        "",
        "## Mid-Layer Summary",
        "",
        "| Arm | Top1 | Top5 | Unique top1 | Max top1 occ | Gini | Entropy | Read |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary_rows:
        lines.append(
            "| {arm} | {top1} | {top5} | {unique} | {maxocc} | {gini} | {entropy} | {read} |".format(
                arm=row["arm"],
                top1=fmt(row["mid_top1_accuracy"]),
                top5=fmt(row["mid_top5_accuracy"]),
                unique=fmt(row["mid_top1_unique_fraction"]),
                maxocc=fmt(row["mid_top1_max_occurrence"]),
                gini=fmt(row["mid_top1_gini"]),
                entropy=fmt(row["mid_top1_entropy_norm"]),
                read=row["hubness_read"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Higher `top5` is useful only if `unique top1` and entropy stay reasonably high.",
            "- A high Gini or high max occurrence means a small number of candidate concepts are",
            "  absorbing many nearest-neighbor queries.",
            "- This control should be read alongside `memp_paper_harness/control_summary.csv`;",
            "  it is a hubness artifact check, not a substitute for matched-vs-control deltas.",
            "",
            "## Files",
            "",
            "- `pair_by_layer.csv`: ordered format-pair hubness rows.",
            "- `layer_summary.csv`: layer means across ordered format pairs.",
            "- `hubness_summary.csv`: mid-layer aggregate used in this report.",
            "- `hubness_concept_counts.csv`: mid-layer attractor counts by arm and concept.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_manifest(path: Path, args: argparse.Namespace, summary_rows: list[dict[str, object]]) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/sft/run_semantic_hub_hubness.py",
        "hidden_dir": str(args.hidden_dir.relative_to(ROOT) if args.hidden_dir.is_relative_to(ROOT) else args.hidden_dir),
        "arms": [row["arm"] for row in summary_rows],
        "mid_layers": args.mid_layers,
        "metric_note": "top1 occurrence statistics are averaged across ordered prompt-format retrieval pairs",
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    start, end = parse_band(args.mid_layers)
    pair_rows: list[dict[str, object]] = []
    layer_rows: list[dict[str, object]] = []
    concept_rows: list[dict[str, object]] = []
    for arm in args.arms:
        path = args.hidden_dir / f"{arm}.npz"
        if not path.exists():
            print(f"[skip] missing {path}", flush=True)
            continue
        arm_pairs, arm_layers, arm_concepts = score_arm(path, arm, start, end)
        pair_rows.extend(arm_pairs)
        layer_rows.extend(arm_layers)
        concept_rows.extend(arm_concepts)

    summary_rows = summarize(layer_rows, start, end)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.out_dir / "pair_by_layer.csv", pair_rows, PAIR_FIELDS)
    write_rows(args.out_dir / "layer_summary.csv", layer_rows, LAYER_FIELDS)
    write_rows(args.out_dir / "hubness_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_rows(args.out_dir / "hubness_concept_counts.csv", concept_rows, CONCEPT_FIELDS)
    write_report(args.out_dir / "REPORT.md", summary_rows)
    write_manifest(args.out_dir / "hubness_meta.json", args, summary_rows)
    print(f"[ok] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
