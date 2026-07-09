"""Audit positive fMRI x semantic-hub signals against same-layer base controls.

This is a post-hoc synthesis over the existing
results/sft_fmri_hub_regression CSVs. It does not rerun model extraction or
ridge/RDM regression. The goal is to separate a defensible positive lead from
single-cell cherry picking by reporting:

- mid-layer mean-representation deltas versus base,
- subject consistency for those mid-layer deltas,
- top best-layer cells with the same model/layer base value.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_IN = ROOT / "results" / "sft_fmri_hub_regression"
DEFAULT_OUT = DEFAULT_IN
REGIONS = ("Ventral Visual", "ATL (Semantic)", "Language")
ARM_ORDER = (
    "base",
    "lowLR",
    "lowrank",
    "taskvec_a0p25",
    "taskvec_a0p5",
    "taskvec_a1p0",
    "scrambled",
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", type=Path, default=DEFAULT_IN)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--mid-layer-start", type=int, default=10)
    ap.add_argument("--mid-layer-end", type=int, default=20)
    ap.add_argument("--top-n", type=int, default=8)
    return ap.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    out = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    out.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(out)


def midlayer_summary(in_dir: Path, mid_start: int, mid_end: int) -> tuple[list[dict], list[dict]]:
    comparison = read_csv(in_dir / "cv_model_comparison.csv")
    comparison = [r for r in comparison if r["region"] in REGIONS]
    base_mid = {
        r["region"]: float(r["mean_repr_pearson"])
        for r in comparison
        if r["arm"] == "base"
    }

    score_rows = read_csv(in_dir / "cv_model_scores.csv")
    subject_vals: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in score_rows:
        if r["region"] not in REGIONS:
            continue
        if r["model"] != "mean_repr":
            continue
        layer = int(r["layer"])
        if not (mid_start <= layer <= mid_end):
            continue
        subject_vals[(r["region"], r["arm"], r["subject"])].append(float(r["test_pearson_r"]))
    subject_means = {
        key: sum(vals) / len(vals)
        for key, vals in subject_vals.items()
        if vals
    }
    subjects = sorted({key[2] for key in subject_means})

    mid_rows = []
    subject_rows = []
    for region in REGIONS:
        for arm in ARM_ORDER:
            row = next((r for r in comparison if r["region"] == region and r["arm"] == arm), None)
            if row is None:
                continue
            value = float(row["mean_repr_pearson"])
            delta = value - base_mid[region]
            subject_deltas = []
            for subject in subjects:
                key = (region, arm, subject)
                base_key = (region, "base", subject)
                if key in subject_means and base_key in subject_means:
                    subject_deltas.append(subject_means[key] - subject_means[base_key])
            n_pos = sum(d > 0 for d in subject_deltas)
            mid_rows.append(
                {
                    "region": region,
                    "arm": arm,
                    "mean_repr_pearson": value,
                    "base_mean_repr_pearson": base_mid[region],
                    "delta_vs_base": delta,
                    "mean_repr_minus_best_single": float(row["mean_repr_minus_best_single"]),
                    "n_subjects_positive": n_pos,
                    "n_subjects": len(subject_deltas),
                    "delta_subject_01": subject_deltas[0] if len(subject_deltas) > 0 else "",
                    "delta_subject_02": subject_deltas[1] if len(subject_deltas) > 1 else "",
                    "delta_subject_03": subject_deltas[2] if len(subject_deltas) > 2 else "",
                }
            )
            subject_rows.append(
                {
                    "region": region,
                    "arm": arm,
                    "mean_delta_vs_base": sum(subject_deltas) / len(subject_deltas)
                    if subject_deltas
                    else "",
                    "n_subjects_positive": n_pos,
                    "n_subjects": len(subject_deltas),
                    "delta_subject_01": subject_deltas[0] if len(subject_deltas) > 0 else "",
                    "delta_subject_02": subject_deltas[1] if len(subject_deltas) > 1 else "",
                    "delta_subject_03": subject_deltas[2] if len(subject_deltas) > 2 else "",
                }
            )
    return mid_rows, subject_rows


def best_layer_rows(in_dir: Path, top_n: int) -> list[dict]:
    rows = [r for r in read_csv(in_dir / "cv_layer_summary.csv") if r["region"] in REGIONS]
    by_key = {
        (r["region"], r["arm"], r["model"], r["layer"]): float(r["mean_test_pearson_r"])
        for r in rows
    }
    out = []
    for region in REGIONS:
        candidates = [r for r in rows if r["region"] == region and r["arm"] != "base"]
        candidates.sort(key=lambda r: float(r["mean_test_pearson_r"]), reverse=True)
        for rank, row in enumerate(candidates[:top_n], start=1):
            value = float(row["mean_test_pearson_r"])
            base_value = by_key[(region, "base", row["model"], row["layer"])]
            out.append(
                {
                    "region": region,
                    "rank": rank,
                    "arm": row["arm"],
                    "model": row["model"],
                    "layer": int(row["layer"]),
                    "pearson_r": value,
                    "base_same_model_layer_pearson_r": base_value,
                    "delta_vs_same_model_layer_base": value - base_value,
                }
            )
    return out


def write_report(
    path: Path,
    mid_rows: list[dict],
    subject_rows: list[dict],
    best_rows: list[dict],
    mid_start: int,
    mid_end: int,
) -> None:
    positive_rows = [
        r for r in mid_rows if r["arm"] != "base" and r["arm"] != "scrambled" and r["delta_vs_base"] > 0
    ]
    scrambled_rows = [r for r in mid_rows if r["arm"] == "scrambled"]
    n_positive = len(positive_rows)
    n_possible = len(REGIONS) * 5
    n_scrambled_negative = sum(r["delta_vs_base"] < 0 for r in scrambled_rows)

    lines = [
        "# Positive-Signal Audit: fMRI x Semantic Hub",
        "",
        "This is a post-hoc audit of existing `results/sft_fmri_hub_regression/`",
        "artifacts. No model features, fMRI responses, or CHTC jobs were rerun.",
        "",
        "## Main Read",
        "",
        (
            "The most defensible positive lead is not the original direct THINGS-fMRI RSA. "
            "It is the concept-held-out semantic-hub regression: mid-layer, "
            "format-averaged model geometry predicts held-out fMRI RDM distances "
            "better after coherence training/task-vector movement than in base."
        ),
        "",
        (
            f"Across the three target regions and five coherent/non-scrambled arms, "
            f"{n_positive}/{n_possible} mid-layer `mean_repr` comparisons beat base. "
            f"The scrambled control is below base in {n_scrambled_negative}/{len(scrambled_rows)} "
            "regions."
        ),
        "",
        "This supports a cautious story: coherence tuning makes mid-layer semantic",
        "geometry more brain-predictive under held-out concept regression, while",
        "the raw direct-RSA and Huth/LeBel smoke are not yet base-beating results.",
        "",
        "## Mid-Layer Mean-Representation Deltas",
        "",
        f"Mid-layer band: layers `{mid_start}` through `{mid_end}`. Metric: held-out Pearson r.",
        "",
    ]

    for region in REGIONS:
        rows = [r for r in mid_rows if r["region"] == region]
        rows.sort(key=lambda r: r["delta_vs_base"], reverse=True)
        lines.extend(
            [
                f"### {region}",
                "",
                markdown_table(
                    ["Arm", "Mean repr r", "Delta vs base", "Subjects positive", "Mean repr - best single"],
                    [
                        [
                            f"`{r['arm']}`",
                            fmt(r["mean_repr_pearson"]),
                            fmt(r["delta_vs_base"]),
                            f"{r['n_subjects_positive']}/{r['n_subjects']}",
                            fmt(r["mean_repr_minus_best_single"]),
                        ]
                        for r in rows
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Subject Consistency",
            "",
            "Subject deltas are mid-layer `mean_repr` means minus base for the same",
            "subject and region.",
            "",
        ]
    )
    for region in REGIONS:
        rows = [r for r in subject_rows if r["region"] == region and r["arm"] != "base"]
        rows.sort(key=lambda r: r["mean_delta_vs_base"], reverse=True)
        lines.extend(
            [
                f"### {region}",
                "",
                markdown_table(
                    ["Arm", "Mean delta", "n positive", "sub-01", "sub-02", "sub-03"],
                    [
                        [
                            f"`{r['arm']}`",
                            fmt(r["mean_delta_vs_base"]),
                            f"{r['n_subjects_positive']}/{r['n_subjects']}",
                            fmt(r["delta_subject_01"]),
                            fmt(r["delta_subject_02"]),
                            fmt(r["delta_subject_03"]),
                        ]
                        for r in rows
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Best-Layer Cells With Same-Base Controls",
            "",
            "These are descriptive and should not be treated as confirmatory because",
            "layers and models were inspected after scoring. The base comparator is",
            "the same region, predictor model, and layer.",
            "",
        ]
    )
    for region in REGIONS:
        rows = [r for r in best_rows if r["region"] == region]
        lines.extend(
            [
                f"### {region}",
                "",
                markdown_table(
                    ["Rank", "Arm", "Model", "Layer", "r", "Base same", "Delta"],
                    [
                        [
                            str(r["rank"]),
                            f"`{r['arm']}`",
                            f"`{r['model']}`",
                            str(r["layer"]),
                            fmt(r["pearson_r"]),
                            fmt(r["base_same_model_layer_pearson_r"]),
                            fmt(r["delta_vs_same_model_layer_base"]),
                        ]
                        for r in rows
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- Positive claim to test next: coherence-tuned/task-vector mid-layer",
            "  semantic geometry is more predictive of held-out fMRI object-concept",
            "  RDMs than base geometry.",
            "- This is strongest as a representational brain-alignment claim, not as",
            "  a deployment-quality claim. `taskvec_a0p5` is positive here but failed",
            "  the retention gate, so it should not be promoted as a general model.",
            "- The result is still post-hoc. The next run should pre-register the",
            "  mid-layer `mean_repr` metric, use concept-held-out folds, and evaluate",
            "  base, scrambled, lowrank, and the task-vector alphas without selecting",
            "  layers on test scores.",
            "- For a stronger cognitive-neuroscience claim, repeat the same fixed",
            "  analysis in Huth/LeBel high-data language fMRI or a Fedorenko-style",
            "  language-localizer dataset.",
            "",
            "## Artifacts",
            "",
            "- `positive_midlayer_mean_repr.csv`",
            "- `positive_subject_consistency.csv`",
            "- `positive_best_layer_same_base.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    mid_rows, subject_rows = midlayer_summary(args.in_dir, args.mid_layer_start, args.mid_layer_end)
    best_rows = best_layer_rows(args.in_dir, args.top_n)

    write_csv(
        args.out_dir / "positive_midlayer_mean_repr.csv",
        mid_rows,
        [
            "region",
            "arm",
            "mean_repr_pearson",
            "base_mean_repr_pearson",
            "delta_vs_base",
            "mean_repr_minus_best_single",
            "n_subjects_positive",
            "n_subjects",
            "delta_subject_01",
            "delta_subject_02",
            "delta_subject_03",
        ],
    )
    write_csv(
        args.out_dir / "positive_subject_consistency.csv",
        subject_rows,
        [
            "region",
            "arm",
            "mean_delta_vs_base",
            "n_subjects_positive",
            "n_subjects",
            "delta_subject_01",
            "delta_subject_02",
            "delta_subject_03",
        ],
    )
    write_csv(
        args.out_dir / "positive_best_layer_same_base.csv",
        best_rows,
        [
            "region",
            "rank",
            "arm",
            "model",
            "layer",
            "pearson_r",
            "base_same_model_layer_pearson_r",
            "delta_vs_same_model_layer_base",
        ],
    )
    write_report(
        args.out_dir / "POSITIVE_SIGNAL_REVIEW.md",
        mid_rows,
        subject_rows,
        best_rows,
        args.mid_layer_start,
        args.mid_layer_end,
    )


if __name__ == "__main__":
    main()
