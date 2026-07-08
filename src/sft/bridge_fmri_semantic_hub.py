"""Bridge semantic-hub metrics to THINGS-fMRI RSA results.

This is a descriptive analysis over already-computed artifacts. It asks whether
layers/arms with stronger cross-format semantic invariance also show stronger
object-fMRI RSA, and whether changes versus base track fMRI changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FMRI = ROOT / "results" / "sft_fmri"
DEFAULT_HUB = ROOT / "results" / "sft_semantic_hub"
DEFAULT_OUT = ROOT / "results" / "sft_fmri_semantic_bridge"

HUB_METRICS = [
    "cross_format_rdm_spearman",
    "linear_cka",
    "retrieval_top1",
    "retrieval_top5",
    "concept_minus_format_alignment",
]
SUMMARY_HUB_METRICS = [
    "best_cross_format_rdm_spearman",
    "best_linear_cka",
    "best_retrieval_top1",
    "best_retrieval_top5",
    "best_concept_minus_format_alignment",
    "mid_cross_format_rdm_spearman",
    "mid_linear_cka",
    "mid_retrieval_top1",
    "mid_retrieval_top5",
    "mid_concept_minus_format_alignment",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fmri_dir", default=str(DEFAULT_FMRI))
    ap.add_argument("--hub_dir", default=str(DEFAULT_HUB))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    return ap.parse_args()


def finite_pair(x: pd.Series, y: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    xx = pd.to_numeric(x, errors="coerce").to_numpy(dtype=float)
    yy = pd.to_numeric(y, errors="coerce").to_numpy(dtype=float)
    ok = np.isfinite(xx) & np.isfinite(yy)
    return xx[ok], yy[ok]


def corr_rows(df: pd.DataFrame, target: str, metrics: list[str], prefix: str) -> list[dict]:
    rows = []
    for metric in metrics:
        x, y = finite_pair(df[metric], df[target])
        if len(x) < 4 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
            continue
        spearman = stats.spearmanr(x, y)
        pearson = stats.pearsonr(x, y)
        rows.append(
            {
                "analysis": prefix,
                "target": target,
                "hub_metric": metric,
                "n": int(len(x)),
                "spearman_r": float(spearman.statistic),
                "spearman_p": float(spearman.pvalue),
                "pearson_r": float(pearson.statistic),
                "pearson_p": float(pearson.pvalue),
            }
        )
    return rows


def load_layer_join(fmri_dir: Path, hub_dir: Path) -> pd.DataFrame:
    rsa = pd.read_csv(fmri_dir / "rsa_by_layer.csv")
    rsa_mean = (
        rsa.groupby(["region", "arm", "layer", "layer_name"], as_index=False)
        .agg(
            rsa_spearman_mean=("rsa_spearman", "mean"),
            rsa_spearman_se=(
                "rsa_spearman",
                lambda x: float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0,
            ),
            n_subjects=("subject", "nunique"),
            n_concepts=("n_concepts", "max"),
        )
    )

    hub = pd.read_csv(hub_dir / "hub_by_layer.csv")
    hub_mean = hub[hub["format_a"].eq("__mean__") & hub["format_b"].eq("__mean__")].copy()
    keep = ["arm", "layer", "layer_name", *HUB_METRICS]
    hub_mean = hub_mean[keep]

    joined = rsa_mean.merge(hub_mean, on=["arm", "layer", "layer_name"], how="inner")

    base_rsa = joined[joined["arm"].eq("base")][
        ["region", "layer", "rsa_spearman_mean", *HUB_METRICS]
    ].copy()
    base_rsa = base_rsa.rename(
        columns={
            "rsa_spearman_mean": "base_rsa_spearman_mean",
            **{m: f"base_{m}" for m in HUB_METRICS},
        }
    )
    joined = joined.merge(base_rsa, on=["region", "layer"], how="left")
    joined["rsa_delta_vs_base"] = joined["rsa_spearman_mean"] - joined["base_rsa_spearman_mean"]
    for metric in HUB_METRICS:
        joined[f"{metric}_delta_vs_base"] = joined[metric] - joined[f"base_{metric}"]
    return joined


def layer_correlations(joined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for region, sub in joined.groupby("region"):
        layer_sub = sub[sub["layer"].gt(0)]
        for row in corr_rows(layer_sub, "rsa_spearman_mean", HUB_METRICS, "arm_layer_raw"):
            row["region"] = region
            rows.append(row)

        delta_sub = layer_sub[~layer_sub["arm"].eq("base")].copy()
        for metric in HUB_METRICS:
            delta_sub[f"{metric}_delta_vs_base"] = pd.to_numeric(
                delta_sub[f"{metric}_delta_vs_base"], errors="coerce"
            )
        delta_metrics = [f"{m}_delta_vs_base" for m in HUB_METRICS]
        for row in corr_rows(delta_sub, "rsa_delta_vs_base", delta_metrics, "arm_layer_delta_vs_base"):
            row["region"] = region
            rows.append(row)
    return pd.DataFrame(rows)


def arm_summary_bridge(fmri_dir: Path, hub_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fmri = pd.read_csv(fmri_dir / "rsa_summary.csv")
    hub = pd.read_csv(hub_dir / "hub_summary.csv")
    joined = fmri.merge(hub, on="arm", how="inner", suffixes=("_fmri", "_hub"))

    base = joined[joined["arm"].eq("base")][["region", "mean_rsa_spearman"]].rename(
        columns={"mean_rsa_spearman": "base_mean_rsa_spearman"}
    )
    joined = joined.merge(base, on="region", how="left")
    joined["mean_rsa_delta_vs_base"] = (
        joined["mean_rsa_spearman"] - joined["base_mean_rsa_spearman"]
    )

    rows = []
    for region, sub in joined.groupby("region"):
        for row in corr_rows(sub, "mean_rsa_spearman", SUMMARY_HUB_METRICS, "arm_summary_raw"):
            row["region"] = region
            rows.append(row)
        delta_sub = sub[~sub["arm"].eq("base")]
        for row in corr_rows(
            delta_sub,
            "mean_rsa_delta_vs_base",
            SUMMARY_HUB_METRICS,
            "arm_summary_delta_vs_base",
        ):
            row["region"] = region
            rows.append(row)
    return joined, pd.DataFrame(rows)


def top_rows(df: pd.DataFrame, analysis: str, metric: str, n: int = 8) -> pd.DataFrame:
    sub = df[df["analysis"].eq(analysis) & df["hub_metric"].str.contains(metric, regex=False)]
    return sub.reindex(sub["spearman_r"].abs().sort_values(ascending=False).index).head(n)


def write_report(
    out_dir: Path,
    layer_corr: pd.DataFrame,
    arm_corr: pd.DataFrame,
    arm_bridge: pd.DataFrame,
) -> None:
    lines = [
        "# fMRI x Semantic-Hub Bridge",
        "",
        "## Scope",
        "",
        "This analysis joins the 90-concept THINGS-fMRI RSA with the 128-concept",
        "semantic-hub metrics by arm/layer. It is descriptive: the layer-grid",
        "analysis has many correlated layer points, and the arm-summary analysis",
        "has only seven arms.",
        "",
        "## Key Tables",
        "",
        "- `layer_join.csv`: mean fMRI RSA by region/arm/layer plus hub metrics.",
        "- `layer_correlations.csv`: correlations across arm-layer points and",
        "  delta-vs-base arm-layer points.",
        "- `arm_summary_bridge.csv`: fMRI best-layer summary joined to hub summary.",
        "- `arm_summary_correlations.csv`: small-n arm-level correlations.",
        "",
    ]

    for region in ["Ventral Visual", "ATL (Semantic)", "Language"]:
        sub = arm_bridge[arm_bridge["region"].eq(region)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("mean_rsa_spearman", ascending=False)
        lines.extend([f"## {region}", "", "| Arm | fMRI RSA | Hub Mid RDM | Hub Top-1 |"])
        lines.append("|---|---:|---:|---:|")
        for row in sub.itertuples(index=False):
            lines.append(
                f"| `{row.arm}` | {row.mean_rsa_spearman:.4f} | "
                f"{row.mid_cross_format_rdm_spearman:.4f} | {row.mid_retrieval_top1:.4f} |"
            )
        lines.append("")

    lines.extend(["## Correlation Highlights", ""])
    highlight_specs = [
        ("arm_layer_delta_vs_base", "cross_format_rdm_spearman_delta_vs_base"),
        ("arm_layer_delta_vs_base", "retrieval_top1_delta_vs_base"),
        ("arm_summary_delta_vs_base", "mid_cross_format_rdm_spearman"),
        ("arm_summary_delta_vs_base", "mid_retrieval_top1"),
    ]
    for analysis, metric in highlight_specs:
        source = layer_corr if analysis.startswith("arm_layer") else arm_corr
        sub = source[source["analysis"].eq(analysis) & source["hub_metric"].eq(metric)]
        if sub.empty:
            continue
        lines.append(f"### `{analysis}` / `{metric}`")
        lines.append("")
        lines.append("| Region | n | Spearman r | p |")
        lines.append("|---|---:|---:|---:|")
        for row in sub.sort_values("spearman_r", ascending=False).itertuples(index=False):
            lines.append(f"| {row.region} | {row.n} | {row.spearman_r:.3f} | {row.spearman_p:.3g} |")
        lines.append("")

    ventral = arm_corr[
        arm_corr["region"].eq("Ventral Visual")
        & arm_corr["analysis"].eq("arm_summary_delta_vs_base")
        & arm_corr["hub_metric"].isin(["mid_cross_format_rdm_spearman", "mid_retrieval_top1"])
    ]
    lines.extend(["## Initial Read", ""])
    if not ventral.empty:
        parts = [
            f"{row.hub_metric}: r={row.spearman_r:.3f}"
            for row in ventral.itertuples(index=False)
        ]
        lines.append(
            "For Ventral Visual, the arm-level delta-vs-base link between hub metrics "
            f"and fMRI RSA is: {', '.join(parts)}."
        )
    lines.extend(
        [
            "",
            "The current bridge does not establish that the semantic hub explains the",
            "object-fMRI RSA effects. The safest read is that hub invariance is a",
            "strong internal-model effect, while the first fMRI RSA result is mostly",
            "a visual/object-geometry signal with only small ATL/Language differences.",
            "The next decisive test is a leakage-clean held-out regression where",
            "format-averaged hub RDMs and single-format RDMs compete to predict the",
            "same fMRI RDMs.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    fmri_dir = Path(args.fmri_dir)
    hub_dir = Path(args.hub_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    layer_join = load_layer_join(fmri_dir, hub_dir)
    layer_join.to_csv(out_dir / "layer_join.csv", index=False)

    layer_corr = layer_correlations(layer_join)
    layer_corr.to_csv(out_dir / "layer_correlations.csv", index=False)

    arm_bridge, arm_corr = arm_summary_bridge(fmri_dir, hub_dir)
    arm_bridge.to_csv(out_dir / "arm_summary_bridge.csv", index=False)
    arm_corr.to_csv(out_dir / "arm_summary_correlations.csv", index=False)

    meta = {
        "fmri_dir": str(fmri_dir),
        "hub_dir": str(hub_dir),
        "hub_metrics": HUB_METRICS,
        "summary_hub_metrics": SUMMARY_HUB_METRICS,
        "notes": [
            "Layer-grid correlations are descriptive and not independent samples.",
            "Delta-vs-base correlations are the main check against generic layer-depth effects.",
            "Arm-summary correlations have n=7 and should be treated as hypothesis generation.",
        ],
    }
    (out_dir / "bridge_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    write_report(out_dir, layer_corr, arm_corr, arm_bridge)
    print(f"[done] wrote bridge outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
