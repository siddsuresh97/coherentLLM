"""Summarize Task 26 retention-mitigation arms.

Reads generation coherence raw files from results/sft_eval/raw/<model>/ plus
SALMON triplet embeddings from results/sft_eval/, and lm-eval JSONs from each
arm's mitigation retention directory.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sft.analyze_eval import _concepts, coherence_metrics  # noqa: E402


OUT_DIR = ROOT / "results" / "sft_eval"
MIT_DIR = OUT_DIR / "mitigation"
RETENTION_FULL = OUT_DIR / "retention_full" / "summary.csv"
EVAL_RESULTS = OUT_DIR / "eval_results.csv"

ARM_DEFAULTS = {
    "lowLR": {"lr": 5e-5, "rank": 64, "steps": 1500},
    "fewsteps": {"lr": 2e-4, "rank": 64, "steps": 400},
    "lowrank": {"lr": 2e-4, "rank": 16, "steps": 1500},
    "replay": {"lr": 2e-4, "rank": 64, "steps": 1500},
}

RETENTION_TASKS = {
    "arc": ("arc_challenge", ["acc_norm,none", "acc,none", "acc"], "arc_challenge"),
    "hellaswag": ("hellaswag", ["acc_norm,none", "acc,none", "acc"], "hellaswag"),
    "mmlu_abstract_algebra": ("mmlu_abstract_algebra", ["acc,none", "acc"], "mmlu_abstract_algebra"),
    "mmlu_anatomy": ("mmlu_anatomy", ["acc,none", "acc"], "mmlu_anatomy"),
    "truthfulqa_mc2": ("truthfulqa_mc2", ["acc,none", "mc2", "acc"], "truthfulqa_mc2"),
}


def model_name_for_arm(arm: str) -> str:
    return f"llama31-mit-{arm}"


def load_json(path: Path) -> dict | None:
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def metric_from_result(result: dict, preferred: list[str]) -> float:
    for key in preferred:
        value = result.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    for key, value in result.items():
        if isinstance(value, (int, float)) and (
            key.startswith("acc") or "mc2" in key or key.startswith("exact_match")
        ):
            return float(value)
    return float("nan")


def task_score(results: dict, exact: str, preferred: list[str], prefix: str) -> float:
    if exact in results and isinstance(results[exact], dict):
        return metric_from_result(results[exact], preferred)
    vals = [
        metric_from_result(value, preferred)
        for key, value in results.items()
        if key.startswith(prefix + "_") and isinstance(value, dict)
    ]
    vals = [value for value in vals if not math.isnan(value)]
    return float(np.mean(vals)) if vals else float("nan")


def latest_retention_results(arm: str) -> tuple[dict | None, str]:
    root = MIT_DIR / arm / "retention"
    jsons = sorted(
        glob.glob(str(root / "**" / "*.json"), recursive=True),
        key=lambda p: os.path.getmtime(p),
        reverse=True,
    )
    for path in jsons:
        obj = load_json(Path(path))
        if isinstance(obj, dict) and isinstance(obj.get("results"), dict):
            return obj["results"], path
    return None, ""


def retention_metrics(arm: str) -> dict:
    results, source = latest_retention_results(arm)
    out = {"retention_source": source}
    if results is None:
        for name in RETENTION_TASKS:
            out[name] = float("nan")
        out["retention_mean"] = float("nan")
        return out
    for name, (exact, preferred, prefix) in RETENTION_TASKS.items():
        out[name] = task_score(results, exact, preferred, prefix)
    vals = [out[name] for name in RETENTION_TASKS if not math.isnan(out[name])]
    out["retention_mean"] = float(np.mean(vals)) if vals else float("nan")
    return out


def coherence_row(arm: str, model: str) -> dict:
    concepts = _concepts()
    required = [
        OUT_DIR / "raw" / model / "triplet.csv",
        OUT_DIR / "raw" / model / "pairwise.csv",
        OUT_DIR / "raw" / model / "feature.csv",
        OUT_DIR / f"{model}_triplet_d5.npy",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    out = {"coherence_complete": not missing, "coherence_missing": "; ".join(missing)}
    if missing:
        out.update(
            {
                "gen_proc_tf": float("nan"),
                "gen_proc_tp": float("nan"),
                "gen_proc_pf": float("nan"),
                "gen_coherence": float("nan"),
            }
        )
        return out
    metrics = coherence_metrics(model, "gen", concepts)
    out.update(
        {
            "gen_proc_tf": metrics["gen_proc_tf"],
            "gen_proc_tp": metrics["gen_proc_tp"],
            "gen_proc_pf": metrics["gen_proc_pf"],
            "gen_coherence": metrics["gen_proc_mean"],
        }
    )
    (MIT_DIR / arm).mkdir(parents=True, exist_ok=True)
    pd.DataFrame([out]).to_csv(MIT_DIR / arm / "coherence.csv", index=False, float_format="%.6f")
    return out


def reference_values() -> dict:
    refs: dict[str, float] = {}
    if EVAL_RESULTS.exists():
        eval_df = pd.read_csv(EVAL_RESULTS)
        for state in ["base", "real"]:
            row = eval_df[eval_df["state"] == state]
            if len(row):
                refs[f"{state}_gen_coherence"] = float(row.iloc[0]["gen_proc_mean"])
    if RETENTION_FULL.exists():
        ret_df = pd.read_csv(RETENTION_FULL)
        mean = ret_df[ret_df["task"] == "MEAN"]
        if len(mean):
            refs["base_retention_mean"] = float(mean.iloc[0]["base"])
            refs["real_retention_mean"] = float(mean.iloc[0]["real"])
        for task, col in [
            ("arc_challenge", "arc"),
            ("hellaswag", "hellaswag"),
            ("mmlu_anatomy", "mmlu_anatomy"),
        ]:
            row = ret_df[ret_df["task"] == task]
            if len(row):
                refs[f"base_{col}"] = float(row.iloc[0]["base"])
                refs[f"real_{col}"] = float(row.iloc[0]["real"])
    return refs


def mark_pareto(df: pd.DataFrame) -> pd.Series:
    vals = df[["gen_coherence", "retention_mean"]].to_numpy(dtype=float)
    pareto = []
    for i, row in enumerate(vals):
        if np.isnan(row).any():
            pareto.append(False)
            continue
        dominated = False
        for j, other in enumerate(vals):
            if i == j or np.isnan(other).any():
                continue
            if (other >= row).all() and (other > row).any():
                dominated = True
                break
        pareto.append(not dominated)
    return pd.Series(pareto, index=df.index)


def write_plot(df: pd.DataFrame, refs: dict, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    finite = df[np.isfinite(df["gen_coherence"]) & np.isfinite(df["retention_mean"])]
    for _, row in finite.iterrows():
        marker = "s" if row.get("pareto", False) else "o"
        ax.scatter(row["retention_mean"], row["gen_coherence"], s=70, marker=marker)
        ax.annotate(
            row["arm"],
            (row["retention_mean"], row["gen_coherence"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    if "base_retention_mean" in refs and "base_gen_coherence" in refs:
        ax.scatter(refs["base_retention_mean"], refs["base_gen_coherence"], marker="*", s=130)
        ax.annotate(
            "base",
            (refs["base_retention_mean"], refs["base_gen_coherence"]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=9,
        )
        ax.axvline(refs["base_retention_mean"] - 0.02, linestyle="--", linewidth=1, alpha=0.5)
        ax.axvline(refs["base_retention_mean"] - 0.05, linestyle=":", linewidth=1, alpha=0.5)
    if "real_retention_mean" in refs and "real_gen_coherence" in refs:
        ax.scatter(refs["real_retention_mean"], refs["real_gen_coherence"], marker="x", s=90)
        ax.annotate(
            "real full",
            (refs["real_retention_mean"], refs["real_gen_coherence"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    ax.axhline(0.5, linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Retention mean")
    ax.set_ylabel("Generation coherence")
    ax.set_title("Task 26 mitigation Pareto")
    ax.grid(True, linewidth=0.4, alpha=0.4)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=["lowLR", "fewsteps", "lowrank"])
    args = ap.parse_args()

    MIT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    refs = reference_values()
    for arm in args.arms:
        (MIT_DIR / arm).mkdir(parents=True, exist_ok=True)
        defaults = ARM_DEFAULTS.get(arm, {"lr": np.nan, "rank": np.nan, "steps": np.nan})
        model = model_name_for_arm(arm)
        row = {"arm": arm, "model": model, **defaults}
        row.update(coherence_row(arm, model))
        row.update(retention_metrics(arm))
        if "base_gen_coherence" in refs:
            row["gen_delta_vs_base"] = row["gen_coherence"] - refs["base_gen_coherence"]
        if "base_retention_mean" in refs:
            row["retention_delta_vs_base"] = row["retention_mean"] - refs["base_retention_mean"]
        rows.append(row)
        pd.DataFrame([row]).to_csv(
            MIT_DIR / arm / "summary.csv",
            index=False,
            float_format="%.6f",
        )

    df = pd.DataFrame(rows)
    df["pareto"] = mark_pareto(df)
    ordered = [
        "arm",
        "lr",
        "rank",
        "steps",
        "gen_coherence",
        "retention_mean",
        "arc",
        "hellaswag",
        "mmlu_anatomy",
        "pareto",
        "gen_delta_vs_base",
        "retention_delta_vs_base",
        "gen_proc_tf",
        "gen_proc_tp",
        "gen_proc_pf",
        "mmlu_abstract_algebra",
        "truthfulqa_mc2",
        "model",
        "retention_source",
        "coherence_complete",
        "coherence_missing",
    ]
    df = df[[col for col in ordered if col in df.columns]]
    df.to_csv(MIT_DIR / "summary.csv", index=False, float_format="%.6f")
    write_plot(df, refs, MIT_DIR / "pareto.png")
    print(df.to_string(index=False))
    print(f"wrote {MIT_DIR / 'summary.csv'}")
    print(f"wrote {MIT_DIR / 'pareto.png'}")


if __name__ == "__main__":
    main()
