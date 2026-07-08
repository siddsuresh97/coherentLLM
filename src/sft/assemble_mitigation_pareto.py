"""Assemble the Task26 mitigation coherence-vs-retention Pareto.

Reads, for each mitigation arm (fewsteps, lowLR, lowrank) plus base/real references:
  - gen coherence (gen_proc_mean) + human r2, computed from the SALMON d5 embedding
    over the 128-concept stimuli (COHERENCE_STIM_DIR=data/scale128/stimuli)
  - retention mean MC accuracy over arc/hellaswag/mmlu(algebra,anatomy)/truthfulqa
    from the arm's lm-eval retention JSON.
Writes results/sft_eval/mitigation/summary.csv and pareto.png.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("COHERENCE_STIM_DIR", str(ROOT / "data" / "scale128" / "stimuli"))
os.environ.setdefault("COHERENCE_RAW_DIR", str(ROOT / "results" / "sft_eval" / "raw"))
sys.path.insert(0, str(ROOT / "src"))

RET_TASKS = ["arc_challenge", "hellaswag", "mmlu_abstract_algebra", "mmlu_anatomy", "truthfulqa_mc2"]
RET_DIR = ROOT / "results" / "sft_eval" / "retention"

# arm -> (out_model, retention subdir name)
ARMS = {
    "base": "llama-3.1-8b-instruct",
    "real": "llama31-sft-real",
    "fewsteps": "llama31-mit-fewsteps",
    "lowLR": "llama31-mit-lowLR",
    "lowrank": "llama31-mit-lowrank",
}


def retention_mean(model: str):
    dirs = list((RET_DIR / model).glob("*"))
    jsons = []
    for d in dirs:
        jsons += glob.glob(str(d / "results_*.json"))
    if not jsons:
        return None, {}
    latest = max(jsons, key=os.path.getmtime)
    res = json.load(open(latest))["results"]

    def g(t):
        m = res.get(t, {})
        return m.get("acc_norm,none", m.get("acc,none", np.nan))

    per = {t: g(t) for t in RET_TASKS}
    return float(np.nanmean(list(per.values()))), per


def coherence(out_model: str):
    from sft.analyze_eval import _concepts, coherence_metrics, human_triplet_metric

    c = _concepts()
    try:
        m = coherence_metrics(out_model, "gen", c)
        h = human_triplet_metric(out_model, "gen")
        return m.get("gen_proc_mean", np.nan), h.get("gen_human_things_triplet_r2", np.nan)
    except Exception as e:  # noqa: BLE001
        print(f"[warn] coherence for {out_model}: {e}")
        return np.nan, np.nan


def main():
    rows = []
    for arm, model in ARMS.items():
        coh, hr2 = coherence(model)
        ret, per = retention_mean(model)
        row = {"arm": arm, "model": model, "gen_coherence": coh, "human_r2": hr2, "retention_mean": ret}
        for t in RET_TASKS:
            row[f"ret_{t}"] = per.get(t, np.nan)
        rows.append(row)
        print(f"{arm:10s} coh={coh!s:8.8} human_r2={hr2!s:8.8} ret={ret!s:8.8}")

    import csv

    out_csv = ROOT / "results" / "sft_eval" / "mitigation" / "summary.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[wrote] {out_csv}")

    # Pareto plot: retention (x) vs coherence (y)
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    for row in rows:
        x, y = row["retention_mean"], row["gen_coherence"]
        if x is None or y is None or np.isnan(x) or np.isnan(y):
            continue
        ax.scatter(x, y, s=80)
        ax.annotate(row["arm"], (x, y), textcoords="offset points", xytext=(6, 4), fontsize=9)
    base = next(r for r in rows if r["arm"] == "base")
    real = next(r for r in rows if r["arm"] == "real")
    if base["retention_mean"]:
        ax.axvline(base["retention_mean"], ls="--", lw=1, alpha=0.5, color="green")
        ax.axhline(base["gen_coherence"], ls="--", lw=1, alpha=0.5, color="green")
    ax.set_xlabel("retention (mean MC accuracy)")
    ax.set_ylabel("gen coherence (procrustes mean)")
    ax.set_title("Task26: coherence vs retention (base/real + mitigation arms)")
    ax.grid(True, lw=0.4, alpha=0.4)
    out_png = ROOT / "results" / "sft_eval" / "mitigation" / "pareto.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    print(f"[wrote] {out_png}")


if __name__ == "__main__":
    main()
