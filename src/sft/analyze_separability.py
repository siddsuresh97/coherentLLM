"""Analyze separability of generated coherence and THINGS-human alignment."""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from scipy.stats import pearsonr


ROOT = Path(__file__).resolve().parents[2]
SFT_CSV = ROOT / "results" / "sft_eval" / "eval_results.csv"
CH3_CSV = ROOT / "results" / "coherence" / "coherence_matrix.csv"
OUT_DIR = ROOT / "results" / "sft_eval" / "separability"


@dataclass(frozen=True)
class SourceSpec:
    name: str
    path: Path
    coherence_col: str
    primary_human_col: str
    model_col: str = "model"
    state_col: str | None = None


SOURCES = [
    SourceSpec(
        name="sft_eval",
        path=SFT_CSV,
        coherence_col="gen_proc_mean",
        primary_human_col="human_things_triplet_r2",
        state_col="state",
    ),
    SourceSpec(
        name="ch3_model_zoo",
        path=CH3_CSV,
        coherence_col="cross_mean",
        primary_human_col="human_procrustes_r2_triplet",
    ),
]


def _human_r2_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if "human" in c.lower() and "r2" in c.lower() and pd.api.types.is_numeric_dtype(df[c])
    ]


def _preferred_human_col(df: pd.DataFrame, preferred: str) -> str | None:
    if preferred in df.columns:
        return preferred
    candidates = _human_r2_cols(df)
    return candidates[0] if candidates else None


def _clean_pair(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    pair = df[[x_col, y_col]].copy()
    pair[x_col] = pd.to_numeric(pair[x_col], errors="coerce")
    pair[y_col] = pd.to_numeric(pair[y_col], errors="coerce")
    return pair.dropna()


def _pearson(df: pd.DataFrame, x_col: str, y_col: str) -> tuple[int, float, float]:
    pair = _clean_pair(df, x_col, y_col)
    if len(pair) < 3 or pair[x_col].nunique() < 2 or pair[y_col].nunique() < 2:
        return len(pair), float("nan"), float("nan")
    r, p = pearsonr(pair[x_col], pair[y_col])
    return len(pair), float(r), float(p)


def _fmt(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "NA"
    return f"{x:.6f}"


def load_primary_rows() -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = []
    corr_rows = []
    for spec in SOURCES:
        if not spec.path.exists():
            continue
        df = pd.read_csv(spec.path)
        if spec.coherence_col not in df.columns:
            continue
        human_col = _preferred_human_col(df, spec.primary_human_col)
        if human_col is None:
            continue

        view = pd.DataFrame(
            {
                "source": spec.name,
                "state": df[spec.state_col] if spec.state_col and spec.state_col in df.columns else "",
                "model": df[spec.model_col],
                "gen_coherence": pd.to_numeric(df[spec.coherence_col], errors="coerce"),
                "things_human_r2": pd.to_numeric(df[human_col], errors="coerce"),
                "coherence_col": spec.coherence_col,
                "human_r2_col": human_col,
            }
        )
        view = view.dropna(subset=["gen_coherence", "things_human_r2"])
        combined.append(view)

        n, r, p = _pearson(
            view.rename(columns={"gen_coherence": "x", "things_human_r2": "y"}), "x", "y"
        )
        corr_rows.append(
            {
                "analysis": f"{spec.name}_primary",
                "source": spec.name,
                "coherence_col": spec.coherence_col,
                "human_r2_col": human_col,
                "n": n,
                "pearson_r": r,
                "p_value": p,
            }
        )

        for alt_human_col in _human_r2_cols(df):
            n, r, p = _pearson(df, spec.coherence_col, alt_human_col)
            corr_rows.append(
                {
                    "analysis": f"{spec.name}_{alt_human_col}",
                    "source": spec.name,
                    "coherence_col": spec.coherence_col,
                    "human_r2_col": alt_human_col,
                    "n": n,
                    "pearson_r": r,
                    "p_value": p,
                }
            )

    if not combined:
        raise SystemExit("No usable separability inputs found.")

    combined_df = pd.concat(combined, ignore_index=True)
    n, r, p = _pearson(combined_df, "gen_coherence", "things_human_r2")
    corr_rows.insert(
        0,
        {
            "analysis": "combined_primary",
            "source": "combined",
            "coherence_col": "gen_coherence",
            "human_r2_col": "things_human_r2",
            "n": n,
            "pearson_r": r,
            "p_value": p,
        },
    )
    return combined_df, pd.DataFrame(corr_rows)


def sft_arm_deltas(combined: pd.DataFrame) -> pd.DataFrame:
    sft = combined[combined["source"] == "sft_eval"].copy()
    if sft.empty or "base" not in set(sft["state"]):
        return pd.DataFrame(
            columns=[
                "state",
                "model",
                "delta_gen_coherence_vs_base",
                "delta_things_human_r2_vs_base",
                "status",
            ]
        )
    base = sft[sft["state"] == "base"].iloc[0]
    rows = []
    for _, row in sft.iterrows():
        dc = float(row["gen_coherence"] - base["gen_coherence"])
        dh = float(row["things_human_r2"] - base["things_human_r2"])
        gained_coh = dc > 0
        gained_human = dh > 0
        if row["state"] == "base":
            status = "baseline"
        elif gained_coh and gained_human:
            status = "both_gained"
        elif gained_coh and not gained_human:
            status = "coherence_only_gain"
        elif gained_human and not gained_coh:
            status = "human_only_gain"
        else:
            status = "neither_gain"
        rows.append(
            {
                "state": row["state"],
                "model": row["model"],
                "gen_coherence": row["gen_coherence"],
                "things_human_r2": row["things_human_r2"],
                "delta_gen_coherence_vs_base": dc,
                "delta_things_human_r2_vs_base": dh,
                "status": status,
            }
        )
    return pd.DataFrame(rows)


def pairwise_dissociations(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for i in range(len(combined)):
        for j in range(i + 1, len(combined)):
            a = combined.iloc[i]
            b = combined.iloc[j]
            dc = float(a["gen_coherence"] - b["gen_coherence"])
            dh = float(a["things_human_r2"] - b["things_human_r2"])
            if dc == 0 or dh == 0 or (dc > 0) == (dh > 0):
                continue
            rows.append(
                {
                    "item_a": _label(a),
                    "item_b": _label(b),
                    "delta_gen_coherence_a_minus_b": dc,
                    "delta_things_human_r2_a_minus_b": dh,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "item_a",
            "item_b",
            "delta_gen_coherence_a_minus_b",
            "delta_things_human_r2_a_minus_b",
        ],
    )


def _label(row: pd.Series) -> str:
    state = str(row.get("state", ""))
    model = str(row["model"])
    return f"{state}:{model}" if state else model


def write_plot(combined: pd.DataFrame, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"sft_eval": "#1f77b4", "ch3_model_zoo": "#d62728"}
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for source, grp in combined.groupby("source"):
        ax.scatter(
            grp["gen_coherence"],
            grp["things_human_r2"],
            label=source,
            s=48,
            color=colors.get(source),
            alpha=0.9,
        )
        for _, row in grp.iterrows():
            ax.annotate(
                _label(row),
                (row["gen_coherence"], row["things_human_r2"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set_xlabel("GEN coherence")
    ax.set_ylabel("THINGS-human r2")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def write_summary(
    combined: pd.DataFrame,
    correlations: pd.DataFrame,
    arm_deltas: pd.DataFrame,
    dissociations: pd.DataFrame,
    out_path: Path,
) -> None:
    primary = correlations[correlations["analysis"] == "combined_primary"].iloc[0]
    gain_statuses = set(arm_deltas["status"]) - {"baseline"}
    one_only = sorted(s for s in gain_statuses if s in {"coherence_only_gain", "human_only_gain"})
    if one_only:
        arm_text = "One-sided SFT arm gains found: " + ", ".join(one_only) + "."
    else:
        arm_text = "No SFT arm gains one metric without the other relative to base."
    pair_text = (
        f"{len(dissociations)} pairwise opposite-direction model/arm comparisons found."
        if len(dissociations)
        else "No pairwise opposite-direction comparisons found in the combined rows."
    )

    lines = [
        "# Coherence vs THINGS-Human Separability",
        "",
        "Primary combined correlation:",
        "",
        (
            f"- n={int(primary['n'])}, Pearson r={_fmt(primary['pearson_r'])}, "
            f"p={_fmt(primary['p_value'])}"
        ),
        "",
        "Correlation table:",
        "",
        correlations.to_markdown(index=False, floatfmt=".6f"),
        "",
        "SFT arm deltas vs base:",
        "",
        arm_deltas.to_markdown(index=False, floatfmt=".6f"),
        "",
        arm_text,
        pair_text,
    ]
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--plot", action="store_true", help="Also write a scatter plot PNG.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    combined, correlations = load_primary_rows()
    arm_deltas = sft_arm_deltas(combined)
    dissociations = pairwise_dissociations(combined)

    combined.to_csv(args.out_dir / "combined_coherence_human.csv", index=False, float_format="%.6f")
    correlations.to_csv(args.out_dir / "correlations.csv", index=False, float_format="%.12g")
    arm_deltas.to_csv(args.out_dir / "arm_deltas_vs_base.csv", index=False, float_format="%.6f")
    dissociations.to_csv(args.out_dir / "pairwise_dissociations.csv", index=False, float_format="%.6f")
    if args.plot:
        write_plot(combined, args.out_dir / "coherence_vs_human_r2.png")
    write_summary(
        combined,
        correlations,
        arm_deltas,
        dissociations,
        args.out_dir / "SUMMARY.md",
    )

    primary = correlations[correlations["analysis"] == "combined_primary"].iloc[0]
    print(
        "combined_primary "
        f"n={int(primary['n'])} "
        f"r={_fmt(primary['pearson_r'])} "
        f"p={_fmt(primary['p_value'])}"
    )
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
