"""Aggregate SFT step-4 metrics into eval_results.csv and SUMMARY.md."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import procrustes as scipy_procrustes


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(os.environ.get("COHERENCE_RAW_DIR", ROOT / "results" / "sft_eval" / "raw"))
STIM = Path(os.environ.get("COHERENCE_STIM_DIR", ROOT / "data" / "scale128" / "stimuli"))
SCALE = ROOT / "data" / "scale128"
OUT_DIR = ROOT / "results" / "sft_eval"
RESULT_CSV = ROOT / "data" / "sft" / "eval_results.csv"

STATES = [
    ("base", "llama-3.1-8b-instruct"),
    ("real", "llama31-sft-real"),
    ("scrambled", "llama31-sft-scrambled"),
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def _parse_rating(resp):
    m = re.search(r"[1-7]", str(resp))
    return int(m.group()) if m else None


def _parse_truefalse(resp):
    s = str(resp)
    if "A:" in s:
        s = s.rsplit("A:", 1)[1]
    s = s.lower()
    ti = s.find("true")
    fi = s.find("false")
    if ti < 0 and fi < 0:
        yi, ni = s.find("yes"), s.find("no")
        if yi < 0 and ni < 0:
            return None
        return 1 if yi >= 0 and (ni < 0 or yi < ni) else 0
    if ti < 0:
        return 0
    if fi < 0:
        return 1
    return 1 if ti < fi else 0


def _parse_choice(resp, a, b):
    r = _norm(resp)
    na, nb = _norm(a), _norm(b)
    ia = r.find(na) if na else -1
    ib = r.find(nb) if nb else -1
    if ia < 0 and ib < 0:
        return None
    if ia < 0:
        return b
    if ib < 0:
        return a
    return a if ia <= ib else b


def _concepts():
    with (STIM / "concepts.csv").open() as f:
        return [ln.strip() for ln in f if ln.strip()]


def sym_rdm(sim):
    d = 1.0 - sim
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return d


def procrustes_r2(A, B):
    if A is None or B is None or A.shape != B.shape:
        return float("nan")
    try:
        _, _, disp = scipy_procrustes(A, B)
    except Exception:
        return float("nan")
    return float(max(0.0, 1.0 - disp))


def triplet_rdm(model, suffix=""):
    tag = f"_triplet{suffix}_d5" if suffix else "_triplet_d5"
    path = OUT_DIR / f"{model}{tag}.npy"
    if not path.exists():
        return None
    emb = np.load(path)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-9, None)
    return sym_rdm(emb @ emb.T)


def pairwise_rdm(model, suffix="", concepts=None):
    concepts = concepts or _concepts()
    path = RAW / model / f"pairwise{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    idx = {_norm(c): i for i, c in enumerate(concepts)}
    n = len(concepts)
    acc = np.zeros((n, n), dtype=float)
    cnt = np.zeros((n, n), dtype=float)
    for _, row in df.iterrows():
        try:
            a, b = str(row["input"]).split("|")
        except ValueError:
            continue
        i, j = idx.get(_norm(a)), idx.get(_norm(b))
        rating = _parse_rating(row["response"])
        if i is None or j is None or rating is None:
            continue
        acc[i, j] += rating
        cnt[i, j] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(cnt > 0, acc / cnt, np.nan)
    s = np.nanmean(np.dstack([s, s.T]), axis=2)
    np.fill_diagonal(s, 7.0)
    mean = np.nanmean(s)
    s = np.where(np.isnan(s), mean, s)
    return sym_rdm((s - 1.0) / 6.0)


def feature_rdm(model, suffix="", concepts=None):
    concepts = concepts or _concepts()
    path = RAW / model / f"feature{suffix}.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    idx = {_norm(c): i for i, c in enumerate(concepts)}
    feats = {}
    for _, row in df.iterrows():
        try:
            feat, concept = str(row["input"]).split("|")
        except ValueError:
            continue
        ci = idx.get(_norm(concept))
        val = _parse_truefalse(row["response"])
        if ci is None or val is None:
            continue
        feats.setdefault(feat, {})[ci] = val
    names = sorted(feats)
    M = np.zeros((len(concepts), len(names)), dtype=float)
    for j, feat in enumerate(names):
        for ci, val in feats[feat].items():
            M[ci, j] = val
    if M.shape[1] == 0:
        return None
    norms = np.linalg.norm(M, axis=1, keepdims=True)
    Mn = np.divide(M, np.clip(norms, 1e-9, None))
    return sym_rdm(Mn @ Mn.T)


def coherence_metrics(model, mode, concepts):
    suffix = "" if mode == "gen" else "_lp"
    rdms = {
        "t": triplet_rdm(model, suffix=suffix),
        "p": pairwise_rdm(model, suffix=suffix, concepts=concepts),
        "f": feature_rdm(model, suffix=suffix, concepts=concepts),
    }
    vals = {
        f"{mode}_proc_tf": procrustes_r2(rdms["t"], rdms["f"]),
        f"{mode}_proc_tp": procrustes_r2(rdms["t"], rdms["p"]),
        f"{mode}_proc_pf": procrustes_r2(rdms["p"], rdms["f"]),
    }
    vals[f"{mode}_proc_mean"] = float(np.nanmean(list(vals.values())))
    return vals


def shared_feature_metrics(model, concepts):
    f = feature_rdm(model, suffix="_basefeatures", concepts=concepts)
    if f is None:
        f = feature_rdm(model, concepts=concepts)
    rdms = {
        "t": triplet_rdm(model),
        "p": pairwise_rdm(model, concepts=concepts),
        "f": f,
    }
    vals = {
        "gen_shared_proc_tf": procrustes_r2(rdms["t"], rdms["f"]),
        "gen_shared_proc_pf": procrustes_r2(rdms["p"], rdms["f"]),
    }
    vals["gen_shared_proc_mean"] = float(np.nanmean([
        vals["gen_shared_proc_tf"],
        procrustes_r2(rdms["t"], rdms["p"]),
        vals["gen_shared_proc_pf"],
    ]))
    return vals


def paraphrase_metrics(model):
    out = {}

    def pair_mean(values, fn):
        scores = []
        for a, b in combinations(values, 2):
            if a is not None and b is not None:
                scores.append(fn(a, b))
        return float(np.mean(scores)) if scores else float("nan")

    # triplet
    path = RAW / model / "paraphrase_triplet.csv"
    if path.exists():
        df = pd.read_csv(path)
        vals = []
        for _, grp in df.groupby("item_id"):
            choices = []
            for _, row in grp.iterrows():
                anchor, c1, c2 = str(row["input"]).split("|")
                choices.append(_parse_choice(row["response"], c1, c2))
            vals.append(pair_mean(choices, lambda a, b: float(_norm(a) == _norm(b))))
        out["para_triplet"] = float(np.nanmean(vals)) if vals else float("nan")

    path = RAW / model / "paraphrase_pairwise.csv"
    if path.exists():
        df = pd.read_csv(path)
        within = []
        diffs = []
        for _, grp in df.groupby("item_id"):
            ratings = [_parse_rating(r) for r in grp["response"]]
            within.append(pair_mean(ratings, lambda a, b: float(abs(a - b) <= 1)))
            diffs.append(pair_mean(ratings, lambda a, b: abs(a - b)))
        out["para_pairwise_within1"] = float(np.nanmean(within)) if within else float("nan")
        out["para_pairwise_absdiff"] = float(np.nanmean(diffs)) if diffs else float("nan")

    path = RAW / model / "paraphrase_feature.csv"
    if path.exists():
        df = pd.read_csv(path)
        vals = []
        for _, grp in df.groupby("item_id"):
            answers = [_parse_truefalse(r) for r in grp["response"]]
            vals.append(pair_mean(answers, lambda a, b: float(a == b)))
        out["para_feature"] = float(np.nanmean(vals)) if vals else float("nan")

    out["para_mean"] = float(np.nanmean([
        out.get("para_triplet", float("nan")),
        out.get("para_pairwise_within1", float("nan")),
        out.get("para_feature", float("nan")),
    ]))
    return out


def symmetry_metrics(model):
    path = RAW / model / "pairwise.csv"
    if not path.exists():
        return {"sym_viol_rate": float("nan"), "sym_mean_absdiff": float("nan")}
    df = pd.read_csv(path)
    vals = {}
    for _, row in df.iterrows():
        try:
            a, b = str(row["input"]).split("|")
        except ValueError:
            continue
        rating = _parse_rating(row["response"])
        if rating is not None:
            vals[(_norm(a), _norm(b))] = rating
    diffs = []
    for a, b in list(vals):
        if (b, a) in vals and a < b:
            diffs.append(abs(vals[(a, b)] - vals[(b, a)]))
    if not diffs:
        return {"sym_viol_rate": float("nan"), "sym_mean_absdiff": float("nan")}
    diffs = np.asarray(diffs, dtype=float)
    return {
        "sym_viol_rate": float(np.mean(diffs >= 2.0)),
        "sym_mean_absdiff": float(np.mean(diffs)),
    }


def transitivity_metric(model):
    path = RAW / model / "transitivity.csv"
    if not path.exists():
        return {"trans_viol_rate": float("nan")}
    df = pd.read_csv(path)
    violations = []
    for _, grp in df.groupby("cycle_id"):
        if len(grp) != 3:
            continue
        first = grp.iloc[0]
        x, y, z = first["x"], first["y"], first["z"]
        prefs = set()
        ok = True
        for _, row in grp.iterrows():
            choice = _parse_choice(row["response"], row["option_a"], row["option_b"])
            if choice is None:
                ok = False
                break
            loser = row["option_b"] if _norm(choice) == _norm(row["option_a"]) else row["option_a"]
            prefs.add((_norm(choice), _norm(loser)))
        if not ok:
            continue
        cyc1 = {(_norm(x), _norm(y)), (_norm(y), _norm(z)), (_norm(z), _norm(x))}
        cyc2 = {(_norm(y), _norm(x)), (_norm(z), _norm(y)), (_norm(x), _norm(z))}
        violations.append(float(cyc1 <= prefs or cyc2 <= prefs))
    return {
        "trans_viol_rate": float(np.mean(violations)) if violations else float("nan")
    }


def human_triplet_metric(model):
    model_rdm = triplet_rdm(model)
    human = np.load(SCALE / "human_spose_triplet_sim.npy")
    return {"human_things_triplet_r2": procrustes_r2(model_rdm, sym_rdm(human))}


def _load_json(path):
    try:
        with path.open() as f:
            return json.load(f)
    except Exception:
        return None


def _metric_from_result(result, preferred):
    for key in preferred:
        if key in result and isinstance(result[key], (int, float)):
            return float(result[key])
    for key, value in result.items():
        if isinstance(value, (int, float)) and (
            key.startswith("acc") or "mc2" in key or key.startswith("exact_match")
        ):
            return float(value)
    return float("nan")


def _task_score(results, exact, preferred, prefix=None):
    if exact in results:
        return _metric_from_result(results[exact], preferred)
    prefix = prefix or exact
    vals = [
        _metric_from_result(v, preferred)
        for k, v in results.items()
        if k.startswith(prefix + "_") and isinstance(v, dict)
    ]
    vals = [v for v in vals if not math.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def retention_metrics(state, model):
    search_dirs = [OUT_DIR / "retention" / model, OUT_DIR / "retention" / state]
    jsons = []
    for d in search_dirs:
        jsons.extend(glob.glob(str(d / "**" / "*.json"), recursive=True))
    jsons = sorted(jsons, key=lambda p: os.path.getmtime(p), reverse=True)
    for p in jsons:
        obj = _load_json(Path(p))
        if isinstance(obj, dict) and isinstance(obj.get("results"), dict):
            results = obj["results"]
            return {
                "ret_mmlu": _task_score(results, "mmlu", ["acc,none", "acc"], "mmlu"),
                "ret_arc_challenge": _task_score(
                    results, "arc_challenge", ["acc_norm,none", "acc,none", "acc"],
                    "arc_challenge",
                ),
                "ret_hellaswag": _task_score(
                    results, "hellaswag", ["acc_norm,none", "acc,none", "acc"], "hellaswag"
                ),
                "ret_truthfulqa": _task_score(
                    results, "truthfulqa_mc2", ["acc,none", "mc2", "acc"],
                    "truthfulqa_mc2",
                ),
            }
    return {
        "ret_mmlu": float("nan"),
        "ret_arc_challenge": float("nan"),
        "ret_hellaswag": float("nan"),
        "ret_truthfulqa": float("nan"),
    }


def build_rows():
    concepts = _concepts()
    rows = []
    for state, model in STATES:
        row = {"state": state, "model": model}
        row.update(coherence_metrics(model, "gen", concepts))
        row.update(shared_feature_metrics(model, concepts))
        row.update(coherence_metrics(model, "lp", concepts))
        row.update(paraphrase_metrics(model))
        row.update(symmetry_metrics(model))
        row.update(transitivity_metric(model))
        row.update(human_triplet_metric(model))
        row.update(retention_metrics(state, model))
        rows.append(row)
    return pd.DataFrame(rows)


def _fmt(x):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return f"{x:.3f}"


def _delta(df, col, a="real", b="base"):
    vals = df.set_index("state")[col]
    return vals[a] - vals[b]


def write_summary(df):
    show_cols = [
        "state", "gen_proc_mean", "lp_proc_mean", "para_mean",
        "sym_viol_rate", "trans_viol_rate", "human_things_triplet_r2",
        "ret_mmlu", "ret_arc_challenge", "ret_hellaswag", "ret_truthfulqa",
    ]
    table = df[show_cols].copy()
    md = table.to_markdown(index=False, floatfmt=".3f")

    idx = df.set_index("state")
    real_a = _delta(df, "gen_proc_mean")
    real_lp = _delta(df, "lp_proc_mean")
    scr_a = idx.loc["scrambled", "gen_proc_mean"] - idx.loc["base", "gen_proc_mean"]
    scr_lp = idx.loc["scrambled", "lp_proc_mean"] - idx.loc["base", "lp_proc_mean"]
    para = _delta(df, "para_mean")
    sym = idx.loc["base", "sym_viol_rate"] - idx.loc["real", "sym_viol_rate"]
    trans = idx.loc["base", "trans_viol_rate"] - idx.loc["real", "trans_viol_rate"]
    human = _delta(df, "human_things_triplet_r2")
    ret_cols = ["ret_mmlu", "ret_arc_challenge", "ret_hellaswag", "ret_truthfulqa"]
    ret_drops = {
        c: idx.loc["real", c] - idx.loc["base", c]
        for c in ret_cols
        if c in idx and not pd.isna(idx.loc["real", c]) and not pd.isna(idx.loc["base", c])
    }
    ret_ok = bool(ret_drops) and min(ret_drops.values()) >= -0.02

    shared_notes = []
    if "gen_shared_proc_mean" in df.columns:
        diffs = (df["gen_shared_proc_mean"] - df["gen_proc_mean"]).abs()
        if bool((diffs > 0.03).any()):
            shared = df[["state", "gen_proc_mean", "gen_shared_proc_mean"]]
            shared_notes.append("\nShared base-feature comparison:\n")
            shared_notes.append(shared.to_markdown(index=False, floatfmt=".3f"))

    lines = [
        "# SFT Step 4 Eval Summary",
        "",
        md,
        "",
        (
            f"1. Axis A: real-base delta is {real_a:+.3f} for generation and "
            f"{real_lp:+.3f} for logprob; success requires both to be positive."
        ),
        (
            f"2. Scrambled control delta is {scr_a:+.3f} generation and "
            f"{scr_lp:+.3f} logprob; success requires no comparable rise."
        ),
        (
            f"3. Axis B paraphrase consistency real-base delta is {para:+.3f}; "
            "higher is better."
        ),
        (
            f"4. Axis C/D shifts for real are symmetry {sym:+.3f}, "
            f"transitivity {trans:+.3f}, human triplet {human:+.3f}; "
            "lower C violations and higher D alignment are better."
        ),
        (
            "5. Retention is "
            + ("within the ~1-2 point guardrail." if ret_ok else "outside or incomplete for the guardrail.")
        ),
    ]
    lines.extend(shared_notes)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if RESULT_CSV.exists() and not args.overwrite:
        raise SystemExit(f"{RESULT_CSV} exists; pass --overwrite to replace it")
    df = build_rows()
    RESULT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULT_CSV, index=False, float_format="%.6f")
    write_summary(df)
    print(df.to_string(index=False, formatters={c: _fmt for c in df.columns}))
    print(f"wrote {RESULT_CSV}")
    print(f"wrote {OUT_DIR / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
