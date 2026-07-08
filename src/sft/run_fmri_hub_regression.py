"""Held-out fMRI regression for semantic-hub RDM predictors.

This analysis asks whether a format-averaged semantic-hub representation
predicts THINGS-fMRI RDMs better than individual prompt-format spokes. Folds
hold out concepts, not random RDM entries, so train and test distances do not
share concepts.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from run_fmri_rsa import (
    DEFAULT_BETAS,
    DEFAULT_OVERLAP,
    SUBJECTS,
    get_region_voxel_indices,
    load_subject_concept_betas,
    rdm_upper,
    read_primary_overlap,
    regions,
    zscore_columns,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUB_HIDDEN = ROOT / "results" / "sft_semantic_hub" / "hidden_states"
DEFAULT_OUT = ROOT / "results" / "sft_fmri_hub_regression"
DEFAULT_ARMS = [
    "base",
    "scrambled",
    "lowLR",
    "lowrank",
    "taskvec_a0p25",
    "taskvec_a0p5",
    "taskvec_a1p0",
]
PRIMARY_REPORT_REGIONS = ["Ventral Visual", "ATL (Semantic)", "Language"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap_csv", type=Path, default=DEFAULT_OVERLAP)
    ap.add_argument("--hub_hidden_dir", type=Path, default=DEFAULT_HUB_HIDDEN)
    ap.add_argument("--betas_dir", type=Path, default=DEFAULT_BETAS)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arms", nargs="+", default=DEFAULT_ARMS)
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--mid_layers", default="10:20")
    ap.add_argument(
        "--regions",
        default="",
        help="Optional comma-separated region list. Defaults to the fMRI RSA region set.",
    )
    ap.add_argument("--include_atl_subregions", action="store_true", default=True)
    ap.add_argument("--no_atl_subregions", dest="include_atl_subregions", action="store_false")
    ap.add_argument("--include_whole_brain", action="store_true")
    return ap.parse_args()


def parse_layer_band(text: str) -> tuple[int, int]:
    left, right = text.split(":", 1)
    start = int(left)
    end = int(right)
    if start > end:
        raise ValueError(text)
    return start, end


def load_hub_hidden(path: Path, concepts: list[str]) -> tuple[np.ndarray, list[str], list[str]]:
    data = np.load(path, allow_pickle=False)
    hidden = data["hidden"].astype(np.float32, copy=False)
    saved_concepts = [str(x) for x in data["concepts"]]
    formats = [str(x) for x in data["formats"]]
    layer_names = [str(x) for x in data["layer_names"]]
    index = {concept: idx for idx, concept in enumerate(saved_concepts)}
    missing = [concept for concept in concepts if concept not in index]
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} fMRI concepts, e.g. {missing[:5]}")
    order = np.array([index[concept] for concept in concepts], dtype=np.int64)
    return hidden[:, order, :, :], formats, layer_names


def normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def cosine_rdm_upper(x: np.ndarray) -> np.ndarray:
    return rdm_upper(normalize_rows(x))


def predictor_rdms(hidden: np.ndarray, formats: list[str]) -> dict[str, list[str] | np.ndarray]:
    """Return predictor matrix with columns for mean and format-specific RDMs.

    Hidden has shape formats x concepts x layers x hidden_dim. The returned
    matrix has shape layers x rdm_pairs x predictors.
    """
    n_formats, _, n_layers, _ = hidden.shape
    names = ["mean_repr", "mean_rdm", *[f"single_{fmt}" for fmt in formats]]
    layer_mats = []
    for layer in range(n_layers):
        format_rdms = []
        for fmt_idx in range(n_formats):
            format_rdms.append(cosine_rdm_upper(hidden[fmt_idx, :, layer, :]))
        format_rdms_arr = np.stack(format_rdms, axis=0)
        mean_repr = cosine_rdm_upper(hidden[:, :, layer, :].mean(axis=0))
        mean_rdm = format_rdms_arr.mean(axis=0)
        layer_mats.append(np.stack([mean_repr, mean_rdm, *format_rdms], axis=1))
    return {"names": names, "rdms": np.stack(layer_mats, axis=0)}


def concept_folds(n_concepts: int, n_folds: int, seed: int) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_concepts)
    return [np.array(split, dtype=np.int64) for split in np.array_split(order, n_folds)]


def pair_indices(n_concepts: int) -> tuple[np.ndarray, np.ndarray]:
    left = []
    right = []
    for i in range(n_concepts - 1):
        for j in range(i + 1, n_concepts):
            left.append(i)
            right.append(j)
    return np.array(left, dtype=np.int64), np.array(right, dtype=np.int64)


def fold_masks(
    n_concepts: int,
    folds: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray]]:
    pair_i, pair_j = pair_indices(n_concepts)
    masks = []
    all_idx = np.arange(n_concepts, dtype=np.int64)
    for test_idx in folds:
        train_idx = np.setdiff1d(all_idx, test_idx, assume_unique=False)
        in_train_i = np.isin(pair_i, train_idx)
        in_train_j = np.isin(pair_j, train_idx)
        in_test_i = np.isin(pair_i, test_idx)
        in_test_j = np.isin(pair_j, test_idx)
        train_mask = in_train_i & in_train_j
        test_mask = in_test_i & in_test_j
        masks.append((train_mask, test_mask))
    return masks


def model_specs(predictor_names: list[str]) -> dict[str, list[str]]:
    specs = {
        "mean_repr": ["mean_repr"],
        "mean_rdm": ["mean_rdm"],
        "all_single_formats": [name for name in predictor_names if name.startswith("single_")],
    }
    for name in predictor_names:
        if name.startswith("single_"):
            specs[name] = [name]
    specs["mean_repr_plus_single_formats"] = ["mean_repr", *specs["all_single_formats"]]
    return specs


def safe_corr(y_true: np.ndarray, y_pred: np.ndarray, kind: str) -> float:
    if y_true.size < 3 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return math.nan
    if kind == "pearson":
        return float(stats.pearsonr(y_true, y_pred).statistic)
    if kind == "spearman":
        return float(stats.spearmanr(y_true, y_pred).statistic)
    raise ValueError(kind)


def fit_predict_ridge(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_sd = x_train.std(axis=0, keepdims=True)
    x_sd[x_sd < 1e-8] = 1.0
    y_mean = float(y_train.mean())
    xtr = (x_train - x_mean) / x_sd
    xte = (x_test - x_mean) / x_sd
    y_center = y_train - y_mean
    xtx = xtr.T @ xtr
    ridge = xtx + float(alpha) * np.eye(xtx.shape[0], dtype=np.float64)
    rhs = xtr.T @ y_center
    try:
        beta = np.linalg.solve(ridge, rhs)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(ridge, rhs, rcond=None)[0]
    return xte @ beta + y_mean, beta


def score_model(
    y: np.ndarray,
    x: np.ndarray,
    masks: list[tuple[np.ndarray, np.ndarray]],
    alpha: float,
) -> dict:
    y_tests = []
    y_preds = []
    sse = 0.0
    sst = 0.0
    betas = []
    for train_mask, test_mask in masks:
        x_train = x[train_mask]
        y_train = y[train_mask]
        x_test = x[test_mask]
        y_test = y[test_mask]
        pred, beta = fit_predict_ridge(x_train, y_train, x_test, alpha)
        y_tests.append(y_test)
        y_preds.append(pred)
        betas.append(beta)
        sse += float(np.sum((y_test - pred) ** 2))
        sst += float(np.sum((y_test - float(y_train.mean())) ** 2))
    y_true = np.concatenate(y_tests)
    y_pred = np.concatenate(y_preds)
    return {
        "n_test_pairs": int(y_true.size),
        "test_pearson_r": safe_corr(y_true, y_pred, "pearson"),
        "test_spearman_r": safe_corr(y_true, y_pred, "spearman"),
        "test_r2_vs_train_mean": float(1.0 - sse / sst) if sst > 0 else math.nan,
        "mean_standardized_beta": json.dumps(np.mean(np.vstack(betas), axis=0).round(6).tolist()),
    }


def brain_rdms_for_subject(
    betas_dir: Path,
    subject: str,
    concepts: list[str],
    region_names: list[str],
) -> dict[str, np.ndarray]:
    concept_betas, voxel_meta = load_subject_concept_betas(betas_dir, subject, concepts)
    rdms = {}
    for region in region_names:
        voxel_indices = get_region_voxel_indices(voxel_meta, region)
        if voxel_indices.size == 0:
            continue
        region_z = zscore_columns(concept_betas[:, voxel_indices])
        rdms[region] = rdm_upper(region_z)
    return rdms


def summarize_scores(rows: list[dict], mid_band: tuple[int, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.DataFrame(rows)
    score_cols = ["test_pearson_r", "test_spearman_r", "test_r2_vs_train_mean"]
    layer_summary = (
        scores.groupby(["region", "arm", "layer", "layer_name", "model"], as_index=False)
        .agg(
            mean_test_pearson_r=("test_pearson_r", "mean"),
            mean_test_spearman_r=("test_spearman_r", "mean"),
            mean_test_r2_vs_train_mean=("test_r2_vs_train_mean", "mean"),
            n_subjects=("subject", "nunique"),
            n_test_pairs=("n_test_pairs", "first"),
        )
    )

    best_rows = []
    for (region, arm, model), sub in layer_summary.groupby(["region", "arm", "model"]):
        idx = sub["mean_test_pearson_r"].idxmax()
        best_rows.append(layer_summary.loc[idx].to_dict())
    best_layer = pd.DataFrame(best_rows)

    start, end = mid_band
    mid = scores[scores["layer"].between(start, end)].copy()
    mid_summary = (
        mid.groupby(["region", "arm", "model"], as_index=False)
        .agg(
            mean_mid_test_pearson_r=("test_pearson_r", "mean"),
            mean_mid_test_spearman_r=("test_spearman_r", "mean"),
            mean_mid_test_r2_vs_train_mean=("test_r2_vs_train_mean", "mean"),
            n_subjects=("subject", "nunique"),
            n_layers=("layer", "nunique"),
            n_rows=("layer", "size"),
        )
    )

    for col in score_cols:
        if col in scores:
            scores[col] = scores[col].astype(float)
    return layer_summary, best_layer, mid_summary


def comparison_table(mid_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    single_mask = mid_summary["model"].str.startswith("single_")
    for (region, arm), sub in mid_summary.groupby(["region", "arm"]):
        lookup = {row.model: row for row in sub.itertuples(index=False)}
        singles = sub[single_mask.loc[sub.index]]
        if singles.empty or "mean_repr" not in lookup:
            continue
        best_single = singles.loc[singles["mean_mid_test_pearson_r"].idxmax()]
        mean_repr = lookup["mean_repr"]
        mean_rdm = lookup.get("mean_rdm")
        all_single = lookup.get("all_single_formats")
        rows.append(
            {
                "region": region,
                "arm": arm,
                "mean_repr_pearson": float(mean_repr.mean_mid_test_pearson_r),
                "mean_rdm_pearson": float(mean_rdm.mean_mid_test_pearson_r) if mean_rdm else math.nan,
                "best_single_model": str(best_single["model"]),
                "best_single_pearson": float(best_single["mean_mid_test_pearson_r"]),
                "all_single_formats_pearson": (
                    float(all_single.mean_mid_test_pearson_r) if all_single else math.nan
                ),
                "mean_repr_minus_best_single": (
                    float(mean_repr.mean_mid_test_pearson_r)
                    - float(best_single["mean_mid_test_pearson_r"])
                ),
            }
        )
    return pd.DataFrame(rows)


def write_report(
    out_dir: Path,
    comparison: pd.DataFrame,
    best_layer: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    lines = [
        "# Held-Out fMRI Semantic-Hub Regression",
        "",
        "## Scope",
        "",
        "This analysis predicts THINGS-fMRI RDM distances from semantic-hub model",
        "RDM predictors. Folds hold out concepts, so train and test distance pairs",
        "do not share concepts. Scores are still descriptive because layers and",
        "arms are selected after looking at test scores.",
        "",
        "## Outputs",
        "",
        "- `cv_model_scores.csv`: subject x region x arm x layer x model scores.",
        "- `cv_layer_summary.csv`: mean score by region/arm/layer/model.",
        "- `cv_best_layer_summary.csv`: best layer per region/arm/model.",
        "- `cv_midlayer_summary.csv`: mean score over the configured mid-layer band.",
        "- `cv_model_comparison.csv`: mid-layer mean-format vs best single-format comparison.",
        "",
        "## Primary Mid-Layer Comparison",
        "",
        f"Mid-layer band: `{args.mid_layers}`. Primary metric: held-out Pearson r.",
        "",
    ]
    for region in PRIMARY_REPORT_REGIONS:
        sub = comparison[comparison["region"].eq(region)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("mean_repr_minus_best_single", ascending=False)
        lines.extend(
            [
                f"### {region}",
                "",
                "| Arm | Mean repr | Mean RDM | Best single | Best single r | Delta |",
                "|---|---:|---:|---|---:|---:|",
            ]
        )
        for row in sub.itertuples(index=False):
            lines.append(
                f"| `{row.arm}` | {row.mean_repr_pearson:.4f} | {row.mean_rdm_pearson:.4f} | "
                f"`{row.best_single_model}` | {row.best_single_pearson:.4f} | "
                f"{row.mean_repr_minus_best_single:.4f} |"
            )
        lines.append("")

    lines.extend(["## Descriptive Best-Layer Winners", ""])
    for region in PRIMARY_REPORT_REGIONS:
        sub = best_layer[best_layer["region"].eq(region)].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("mean_test_pearson_r", ascending=False).head(12)
        lines.extend(
            [
                f"### {region}",
                "",
                "| Arm | Model | Layer | Pearson r | R2 vs train mean |",
                "|---|---|---|---:|---:|",
            ]
        )
        for row in sub.itertuples(index=False):
            lines.append(
                f"| `{row.arm}` | `{row.model}` | `{row.layer_name}` | "
                f"{row.mean_test_pearson_r:.4f} | {row.mean_test_r2_vs_train_mean:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Initial Read",
            "",
            "If `mean_repr_minus_best_single` is positive, the shared/format-averaged",
            "hub representation predicts held-out fMRI geometry better than the best",
            "individual prompt spoke in the same mid-layer band. If it is negative,",
            "the fMRI signal is better explained by a task-specific prompt format",
            "or by visual/object geometry not captured by the hub average.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    concepts = read_primary_overlap(args.overlap_csv)
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if args.regions:
        region_names = [r.strip() for r in args.regions.split(",") if r.strip()]
    else:
        region_names = regions(args.include_atl_subregions, args.include_whole_brain)
    mid_band = parse_layer_band(args.mid_layers)
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = concept_folds(len(concepts), args.folds, args.seed)
    masks = fold_masks(len(concepts), folds)

    rows: list[dict] = []
    layer_names_by_arm: dict[str, list[str]] = {}
    specs_by_arm: dict[str, dict[str, list[str]]] = {}
    predictors_by_arm: dict[str, dict[str, list[str] | np.ndarray]] = {}
    for arm in args.arms:
        hidden, formats, layer_names = load_hub_hidden(args.hub_hidden_dir / f"{arm}.npz", concepts)
        preds = predictor_rdms(hidden, formats)
        predictors_by_arm[arm] = preds
        layer_names_by_arm[arm] = layer_names
        specs_by_arm[arm] = model_specs(list(preds["names"]))
        print(f"[model] {arm}: hidden={hidden.shape} predictors={preds['names']}", flush=True)

    for subject in subjects:
        print(f"[brain] sub-{subject}", flush=True)
        brain_by_region = brain_rdms_for_subject(args.betas_dir, subject, concepts, region_names)
        for region, brain_rdm in brain_by_region.items():
            print(f"[brain] sub-{subject} {region}: {brain_rdm.shape[0]} distances", flush=True)
            for arm in args.arms:
                pred_names = list(predictors_by_arm[arm]["names"])
                pred_rdms = predictors_by_arm[arm]["rdms"]
                name_to_idx = {name: idx for idx, name in enumerate(pred_names)}
                for layer_idx, layer_name in enumerate(layer_names_by_arm[arm]):
                    x_all = pred_rdms[layer_idx]
                    for model_name, model_predictors in specs_by_arm[arm].items():
                        cols = [name_to_idx[name] for name in model_predictors]
                        score = score_model(
                            brain_rdm,
                            x_all[:, cols],
                            masks,
                            args.ridge_alpha,
                        )
                        rows.append(
                            {
                                "subject": subject,
                                "region": region,
                                "arm": arm,
                                "layer": layer_idx,
                                "layer_name": layer_name,
                                "model": model_name,
                                "predictors": ",".join(model_predictors),
                                **score,
                            }
                        )

    scores = pd.DataFrame(rows)
    scores.to_csv(out_dir / "cv_model_scores.csv", index=False)
    layer_summary, best_layer, mid_summary = summarize_scores(rows, mid_band)
    layer_summary.to_csv(out_dir / "cv_layer_summary.csv", index=False)
    best_layer.to_csv(out_dir / "cv_best_layer_summary.csv", index=False)
    mid_summary.to_csv(out_dir / "cv_midlayer_summary.csv", index=False)
    comparison = comparison_table(mid_summary)
    comparison.to_csv(out_dir / "cv_model_comparison.csv", index=False)

    meta = {
        "overlap_csv": str(args.overlap_csv),
        "hub_hidden_dir": str(args.hub_hidden_dir),
        "betas_dir": str(args.betas_dir),
        "arms": args.arms,
        "subjects": subjects,
        "n_concepts": len(concepts),
        "folds": args.folds,
        "seed": args.seed,
        "ridge_alpha": args.ridge_alpha,
        "mid_layers": args.mid_layers,
        "regions": region_names,
        "notes": [
            "Concept-held-out folds score only distances within held-out concepts.",
            "Best-layer tables are descriptive and should not be treated as nested-CV estimates.",
            "Mid-layer comparison is the less layer-fished summary.",
        ],
    }
    (out_dir / "regression_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    write_report(out_dir, comparison, best_layer, args)
    print(f"[done] wrote held-out regression outputs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
