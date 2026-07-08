"""Run and summarize the Task 8 mid-layer activation-steering grid."""
from __future__ import annotations

import argparse
import csv
import math
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from sft.analyze_eval import (  # noqa: E402
    _concepts,
    _parse_choice,
    _parse_rating,
    _parse_truefalse,
    coherence_metrics,
    human_triplet_metric,
)


RAW = ROOT / "results" / "sft_eval" / "raw"
OUT_DIR = ROOT / "results" / "sft_eval"
STEER_DIR = OUT_DIR / "steer_actlayer"
APPLY = ROOT / "src" / "sft" / "apply_coherence_vector.py"
LAYER_SETS = ["8", "10", "12", "14", "16", "20", "10-14", "12-16"]
ALPHAS = [2.0, 4.0, 6.0, 8.0]
METHOD_ROWS = {
    "triplet": 10000,
    "pairwise": 16256,
    "feature": 34816,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["run-config", "summarize", "list-configs"])
    ap.add_argument("--layer-set", choices=LAYER_SETS)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--salmon-epochs", type=int, default=8000)
    return ap.parse_args()


def alpha_tag(alpha: float) -> str:
    return f"{alpha:g}".replace(".", "p").replace("-", "m")


def layer_tag(layer_set: str) -> str:
    return layer_set.replace("-", "_").replace(",", "_")


def out_model(layer_set: str, alpha: float) -> str:
    if layer_set == "12" and float(alpha) == 4.0:
        return "llama31-steer-act-L12-a4"
    return f"llama31-actlayer-L{layer_tag(layer_set)}-a{alpha_tag(alpha)}"


def configs() -> list[tuple[str, float, str]]:
    return [(layer_set, alpha, out_model(layer_set, alpha)) for layer_set in LAYER_SETS for alpha in ALPHAS]


def count_data_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def raw_complete(model_name: str) -> bool:
    model_dir = RAW / model_name
    return all(count_data_rows(model_dir / f"{method}.csv") == rows for method, rows in METHOD_ROWS.items())


def triplet_complete(model_name: str) -> bool:
    return count_data_rows(RAW / model_name / "triplet.csv") == METHOD_ROWS["triplet"]


def salmon_complete(model_name: str) -> bool:
    return (OUT_DIR / f"{model_name}_triplet_d5.npy").exists()


def triplet_degenerate(model_name: str) -> tuple[bool, str]:
    triplet = read_rows(RAW / model_name / "triplet.csv")
    if triplet is None or len(triplet) != METHOD_ROWS["triplet"]:
        return False, ""
    choices = []
    repeats = []
    responses = []
    for _, row in triplet.iterrows():
        try:
            _anchor, left, right = str(row["input"]).split("|")
        except ValueError:
            continue
        response = str(row["response"])
        responses.append(response)
        choices.append(_parse_choice(response, left, right))
        repeats.append(repeated_token_fraction(response))
    valid = float(np.mean([c is not None for c in choices])) if choices else 0.0
    top = Counter([str(r).strip()[:80] for r in responses]).most_common(1)
    top_share = top[0][1] / len(responses) if top else 1.0
    repeat_share = float(np.mean([r >= 0.5 for r in repeats])) if repeats else 1.0
    sample = re.sub(r"\s+", " ", responses[0]).replace("|", "/")[:120] if responses else ""
    degenerate = valid < 0.20 or top_share > 0.50 or repeat_share > 0.20
    notes = (
        f"triplet_valid={valid:.3f}; triplet_top_share={top_share:.3f}; "
        f"triplet_repeat_share={repeat_share:.3f}; sample={sample!r}"
    )
    return degenerate, notes


def raw_complete_degenerate(model_name: str) -> tuple[bool, str]:
    if not raw_complete(model_name):
        return False, ""
    degenerate, notes = quality_notes(model_name)
    return degenerate, notes


def raw_in_progress(model_name: str) -> bool:
    model_dir = RAW / model_name
    return any(model_dir.glob("*.tmp"))


def run_config(args: argparse.Namespace) -> None:
    if args.layer_set is None or args.alpha is None:
        raise SystemExit("--layer-set and --alpha are required for run-config")
    model_name = out_model(args.layer_set, args.alpha)
    batch_size = max(args.batch_size, 8)
    STEER_DIR.mkdir(parents=True, exist_ok=True)
    log_path = STEER_DIR / f"{model_name}.log"
    if raw_complete(model_name) and salmon_complete(model_name) and not args.overwrite:
        print(f"[skip] {model_name} complete", flush=True)
        return
    if raw_complete(model_name) and not args.overwrite:
        degenerate, notes = raw_complete_degenerate(model_name)
        if degenerate:
            print(f"[skip] {model_name} raw complete but degenerate; {notes}", flush=True)
            summarize(args)
            return
    if triplet_complete(model_name) and not args.overwrite:
        degenerate, notes = triplet_degenerate(model_name)
        if degenerate:
            print(f"[skip] {model_name} triplet complete but degenerate; {notes}", flush=True)
            summarize(args)
            return
    if raw_in_progress(model_name) and not args.overwrite:
        print(f"[skip] {model_name} has active tmp output; assuming another worker owns it", flush=True)
        return

    base_run_cmd = [
        sys.executable,
        str(APPLY),
        "run-one",
        "--route",
        "actdiff",
        "--alpha",
        f"{args.alpha:g}",
        "--out-model",
        model_name,
        "--layers",
        args.layer_set,
        "--norm-match",
        "--batch-size",
        str(batch_size),
        "--max-length",
        str(args.max_length),
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--dtype",
        args.dtype,
        "--device",
        args.device,
    ]
    if args.overwrite:
        base_run_cmd.append("--overwrite")
    triplet_cmd = [*base_run_cmd, "--methods", "triplet"]
    rest_cmd = [*base_run_cmd, "--methods", "pairwise", "feature"]
    salmon_cmd = [
        sys.executable,
        str(APPLY),
        "fit-salmon",
        "--route",
        "actdiff",
        "--alpha",
        f"{args.alpha:g}",
        "--out-model",
        model_name,
        "--layers",
        args.layer_set,
        "--norm-match",
        "--salmon-epochs",
        str(args.salmon_epochs),
    ]
    if args.overwrite:
        salmon_cmd.append("--overwrite")

    env = os.environ.copy()
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    with log_path.open("a") as log:
        log.write(f"[run-config] {model_name} layer_set={args.layer_set} alpha={args.alpha:g}\n")
        log.flush()
        print(f"[run] {model_name} -> {log_path}", flush=True)
        subprocess.run(triplet_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        degenerate, notes = triplet_degenerate(model_name)
        if degenerate:
            log.write(f"[degenerate] {notes}\n")
            log.flush()
            print(f"[degenerate] {model_name}: {notes}", flush=True)
            summarize(args)
            return
        subprocess.run(rest_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        try:
            subprocess.run(salmon_cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as exc:
            log.write(f"[salmon_failed] returncode={exc.returncode}; treating as degenerate/incomplete\n")
            log.flush()
    summarize(args)


def read_rows(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def repeated_token_fraction(text: str) -> float:
    tokens = re.findall(r"[A-Za-z0-9]+", str(text).lower())
    if len(tokens) < 4:
        return 0.0
    counts = Counter(tokens)
    return counts.most_common(1)[0][1] / len(tokens)


def quality_notes(model_name: str) -> tuple[bool, str]:
    model_dir = RAW / model_name
    parts = []
    degenerate = False

    triplet = read_rows(model_dir / "triplet.csv")
    if triplet is not None and len(triplet):
        choices = []
        repeats = []
        responses = []
        for _, row in triplet.iterrows():
            try:
                _anchor, left, right = str(row["input"]).split("|")
            except ValueError:
                continue
            response = str(row["response"])
            responses.append(response)
            choices.append(_parse_choice(response, left, right))
            repeats.append(repeated_token_fraction(response))
        valid = float(np.mean([c is not None for c in choices])) if choices else float("nan")
        top = Counter([str(r).strip()[:80] for r in responses]).most_common(1)
        top_share = top[0][1] / len(responses) if top else float("nan")
        repeat_share = float(np.mean([r >= 0.5 for r in repeats])) if repeats else float("nan")
        sample = re.sub(r"\s+", " ", responses[0]).replace("|", "/")[:120] if responses else ""
        parts.append(f"triplet_valid={valid:.3f}")
        parts.append(f"triplet_top_share={top_share:.3f}")
        parts.append(f"triplet_repeat_share={repeat_share:.3f}")
        parts.append(f"sample={sample!r}")
        if valid < 0.20 or top_share > 0.50 or repeat_share > 0.20:
            degenerate = True
    else:
        parts.append("triplet_missing")

    pairwise = read_rows(model_dir / "pairwise.csv")
    if pairwise is not None and len(pairwise):
        valid = float(np.mean([_parse_rating(r) is not None for r in pairwise["response"]]))
        parts.append(f"pairwise_valid={valid:.3f}")
        if valid < 0.20:
            degenerate = True
    else:
        parts.append("pairwise_missing")

    feature = read_rows(model_dir / "feature.csv")
    if feature is not None and len(feature):
        valid = float(np.mean([_parse_truefalse(r) is not None for r in feature["response"]]))
        parts.append(f"feature_valid={valid:.3f}")
        if valid < 0.20:
            degenerate = True
    else:
        parts.append("feature_missing")

    return degenerate, "; ".join(parts)


def metric_row(layer_set: str, alpha: float, model_name: str, concepts: list[str]) -> dict:
    row = {
        "layer_set": layer_set,
        "alpha": alpha,
        "out_model": model_name,
        "coherence": float("nan"),
        "human_r2": float("nan"),
        "degenerate": False,
        "complete": False,
        "notes": "",
    }
    missing_raw = []
    for method, expected in METHOD_ROWS.items():
        path = RAW / model_name / f"{method}.csv"
        got = count_data_rows(path)
        if got != expected:
            missing_raw.append(f"{path.relative_to(ROOT)} rows={got} expected={expected}")
    q_degenerate, notes = quality_notes(model_name)
    row["notes"] = notes
    if missing_raw:
        trip_deg, trip_notes = triplet_degenerate(model_name)
        if trip_deg:
            row["complete"] = True
            row["degenerate"] = True
            row["notes"] = (
                "triplet complete; skipped remaining methods because outputs are degenerate/unparseable; "
                + trip_notes
            )
            return row
        row["notes"] = "missing " + "; ".join(missing_raw) + ("; " + notes if notes else "")
        return row
    if not salmon_complete(model_name):
        if q_degenerate:
            row["complete"] = True
            row["degenerate"] = True
            row["notes"] = (
                "raw complete; SALMON unavailable because triplet outputs are degenerate/unparseable; "
                + notes
            )
            return row
        row["notes"] = (
            "missing "
            + str((OUT_DIR / f"{model_name}_triplet_d5.npy").relative_to(ROOT))
            + ("; " + notes if notes else "")
        )
        return row
    try:
        row.update(coherence_metrics(model_name, "gen", concepts))
        row.update(human_triplet_metric(model_name, "gen"))
        row["coherence"] = row["gen_proc_mean"]
        row["human_r2"] = row["gen_human_things_triplet_r2"]
        finite = math.isfinite(row["coherence"]) and math.isfinite(row["human_r2"])
        row["complete"] = finite
        row["degenerate"] = bool(q_degenerate or not finite)
    except Exception as exc:
        row["notes"] = f"metric_error={exc}; {row['notes']}"
    return row


def write_heatmap(summary: pd.DataFrame) -> None:
    singles = summary[
        summary["complete"]
        & ~summary["degenerate"]
        & summary["layer_set"].str.fullmatch(r"\d+")
    ].copy()
    if singles.empty:
        return
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    singles["layer"] = singles["layer_set"].astype(int)
    pivot = singles.pivot_table(index="layer", columns="alpha", values="coherence", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(pivot.columns)), [f"{c:g}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    ax.set_xlabel("alpha")
    ax.set_ylabel("layer")
    ax.set_title("Task8 actdiff coherence")
    fig.colorbar(im, ax=ax, label="coherence")
    fig.tight_layout()
    fig.savefig(STEER_DIR / "coherence_heatmap.png", dpi=200)
    plt.close(fig)


def summarize(args: argparse.Namespace) -> None:
    STEER_DIR.mkdir(parents=True, exist_ok=True)
    concepts = _concepts()
    rows = [metric_row(layer_set, alpha, model_name, concepts) for layer_set, alpha, model_name in configs()]
    summary = pd.DataFrame(rows)
    summary.to_csv(STEER_DIR / "summary.csv", index=False, float_format="%.6f")
    write_heatmap(summary)

    complete = summary[summary["complete"]].copy()
    viable = complete[~complete["degenerate"]].copy()
    best = viable.sort_values(["coherence", "human_r2"], ascending=False).head(1)
    lines = [
        "# Task 8 Mid-Layer Activation Steering",
        "",
        "Grid: layer sets {8, 10, 12, 14, 16, 20, 10-14, 12-16} x alpha {2, 4, 6, 8}. "
        "Injection is norm-matched actdiff on only the selected decoder layer(s).",
        "",
    ]
    if len(best):
        b = best.iloc[0]
        lines.append(
            "Best non-degenerate row: "
            f"layer_set={b['layer_set']}, alpha={b['alpha']:.6g}, "
            f"coherence={b['coherence']:.6f}, human_r2={b['human_r2']:.6f}."
        )
    else:
        lines.append("No non-degenerate completed row is available yet.")
    lines.extend([
        "",
        "Completed rows:",
        "",
        complete[["layer_set", "alpha", "coherence", "human_r2", "degenerate", "notes"]].to_markdown(
            index=False, floatfmt=".6f"
        )
        if len(complete)
        else "None",
        "",
        "Incomplete rows:",
        "",
        summary[~summary["complete"]][["layer_set", "alpha", "complete", "notes"]].to_markdown(
            index=False, floatfmt=".6f"
        )
        if len(summary[~summary["complete"]])
        else "None",
    ])
    (STEER_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(summary[["layer_set", "alpha", "complete", "coherence", "human_r2", "degenerate"]].to_string(index=False))
    print(f"[summary] wrote {STEER_DIR / 'summary.csv'}", flush=True)


def list_configs() -> None:
    for i, (layer_set, alpha, model_name) in enumerate(configs()):
        degenerate, _ = raw_complete_degenerate(model_name)
        trip_deg, _ = triplet_degenerate(model_name)
        status = (
            "complete"
            if (
                raw_complete(model_name)
                and (salmon_complete(model_name) or degenerate)
            )
            or (triplet_complete(model_name) and trip_deg)
            else "pending"
        )
        print(f"{i:02d} layer_set={layer_set:5s} alpha={alpha:g} out_model={model_name} {status}")


def main() -> None:
    args = parse_args()
    if args.action == "run-config":
        run_config(args)
    elif args.action == "summarize":
        summarize(args)
    elif args.action == "list-configs":
        list_configs()
    else:
        raise ValueError(args.action)


if __name__ == "__main__":
    main()
