"""Run a smoke Huth/LeBel voxelwise encoding model from saved word states.

This is the CPU-side half of the ds003020 smoke pipeline. It consumes word-level
LM states from ``huth_lebel_extract_word_states.py``, aligns them to BOLD TRs
with Lanczos interpolation, concatenates FIR delays, fits ridge models on smoke
training stories, and evaluates on a held-out story.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DS_ROOT = Path("/staging/s/suresh27/datasets/ds003020-smoke")
DEFAULT_FEATURES = ROOT / "results" / "sft_huth_lebel" / "word_states_smoke"
DEFAULT_OUT = ROOT / "results" / "sft_huth_lebel" / "smoke_encoding"
DEFAULT_SUBJECTS = ("UTS01", "UTS02", "UTS03")
DEFAULT_TRAIN_STORIES = ("sweetaspie", "againstthewind")
DEFAULT_TEST_STORY = "wheretheressmoke"
DEFAULT_ARMS = ("base", "lowLR", "scrambled", "taskvec_a0p25")
DEFAULT_LAYERS = "16,24,32"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds_root", type=Path, default=DEFAULT_DS_ROOT)
    ap.add_argument("--features_dir", type=Path, default=DEFAULT_FEATURES)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--subjects", nargs="+", default=list(DEFAULT_SUBJECTS))
    ap.add_argument("--train_stories", default=",".join(DEFAULT_TRAIN_STORIES))
    ap.add_argument("--test_story", default=DEFAULT_TEST_STORY)
    ap.add_argument("--arms", default=",".join(DEFAULT_ARMS))
    ap.add_argument(
        "--layers",
        default=DEFAULT_LAYERS,
        help="Layer indices to evaluate. Must be present in each feature NPZ.",
    )
    ap.add_argument("--tr", type=float, default=2.0)
    ap.add_argument(
        "--delays",
        default="1,2,3,4",
        help="FIR delays in TR units. The default corresponds to 2/4/6/8 seconds.",
    )
    ap.add_argument("--lanczos_window", type=int, default=3)
    ap.add_argument(
        "--trim_start_tr",
        type=int,
        default=5,
        help="Drop this many TRs from the beginning of both stimulus and response.",
    )
    ap.add_argument(
        "--trim_end_tr",
        type=int,
        default=5,
        help="Drop this many TRs from the end of both stimulus and response.",
    )
    ap.add_argument(
        "--alphas",
        default="log:1:3:10",
        help="Comma list or log:START_EXP:END_EXP:N, default 10 log-spaced 10..1000.",
    )
    ap.add_argument("--response_key", default="data")
    ap.add_argument("--max_voxels", type=int, default=0, help="Debug only: keep first N voxels.")
    ap.add_argument("--voxel_offset", type=int, default=0)
    ap.add_argument("--zscore_response", action="store_true")
    ap.add_argument("--ridge_solver", choices=["auto", "dual", "primal"], default="auto")
    ap.add_argument("--save_voxel_corrs", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    return ap.parse_args()


def comma_list(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in comma_list(text)]


def parse_alphas(text: str) -> np.ndarray:
    if text.startswith("log:"):
        _, left, right, n = text.split(":", 3)
        return np.logspace(float(left), float(right), int(n), dtype=np.float64)
    return np.array([float(item) for item in comma_list(text)], dtype=np.float64)


def first_existing(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def preproc_dir(ds_root: Path) -> Path:
    return first_existing(
        [
            ds_root / "derivatives" / "preprocessed_data",
            ds_root / "derivative" / "preprocessed_data",
        ]
    )


def response_path(ds_root: Path, subject: str, story: str) -> Path:
    return preproc_dir(ds_root) / subject / f"{story}.hf5"


def find_h5_dataset(handle: h5py.File, preferred_key: str) -> h5py.Dataset:
    if preferred_key in handle and isinstance(handle[preferred_key], h5py.Dataset):
        return handle[preferred_key]
    candidates: list[h5py.Dataset] = []

    def visit(_name: str, obj) -> None:
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            candidates.append(obj)

    handle.visititems(visit)
    if not candidates:
        raise KeyError(f"no 2D response dataset found; preferred key was {preferred_key!r}")
    return candidates[0]


def load_response(
    ds_root: Path,
    subject: str,
    story: str,
    *,
    response_key: str,
    voxel_offset: int,
    max_voxels: int,
    zscore_response_flag: bool,
) -> tuple[np.ndarray, dict]:
    path = response_path(ds_root, subject, story)
    if not path.exists():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as hf:
        dataset = find_h5_dataset(hf, response_key)
        n_vox_total = int(dataset.shape[1])
        start = int(voxel_offset)
        stop = n_vox_total if max_voxels <= 0 else min(n_vox_total, start + int(max_voxels))
        data = np.asarray(dataset[:, start:stop], dtype=np.float32)
        key = dataset.name
    if zscore_response_flag:
        data = zscore_columns(data)
    meta = {
        "path": str(path),
        "dataset_key": key,
        "shape": list(data.shape),
        "voxel_offset": voxel_offset,
        "n_voxels_total": n_vox_total,
    }
    return data, meta


def load_feature(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path, allow_pickle=False)
    return {
        "hidden": data["hidden"].astype(np.float32, copy=False),
        "centers": data["centers"].astype(np.float64, copy=False),
        "words": np.array([str(x) for x in data["words"]]),
        "layer_indices": data["layer_indices"].astype(np.int64, copy=False),
        "layer_names": np.array([str(x) for x in data["layer_names"]]),
    }


def layer_positions(feature: dict, requested_layers: list[int]) -> list[tuple[int, int, str]]:
    saved = list(map(int, feature["layer_indices"]))
    positions = []
    for layer in requested_layers:
        if layer not in saved:
            raise KeyError(f"layer {layer} not present in feature file; saved layers={saved}")
        pos = saved.index(layer)
        positions.append((layer, pos, str(feature["layer_names"][pos])))
    return positions


def lanczos_kernel(x: np.ndarray, window: int) -> np.ndarray:
    y = np.zeros_like(x, dtype=np.float64)
    mask = np.abs(x) < float(window)
    y[mask] = np.sinc(x[mask]) * np.sinc(x[mask] / float(window))
    y[np.abs(x) < 1e-12] = 1.0
    return y


def lanczos_align(
    word_features: np.ndarray,
    word_times: np.ndarray,
    n_trs: int,
    *,
    tr: float,
    window: int,
) -> np.ndarray:
    if word_features.shape[0] != word_times.shape[0]:
        raise ValueError("word feature/time length mismatch")
    out = np.zeros((n_trs, word_features.shape[1]), dtype=np.float32)
    order = np.argsort(word_times)
    times = word_times[order]
    feats = word_features[order]
    half_width = float(window) * tr
    for tr_idx in range(n_trs):
        t = float(tr_idx) * tr
        left = int(np.searchsorted(times, t - half_width, side="left"))
        right = int(np.searchsorted(times, t + half_width, side="right"))
        if right <= left:
            continue
        x = (t - times[left:right]) / tr
        weights = lanczos_kernel(x, window)
        denom = weights.sum()
        if abs(denom) < 1e-8:
            continue
        out[tr_idx] = (weights[:, None] * feats[left:right]).sum(axis=0) / denom
    return out


def trim_pair(x: np.ndarray, y: np.ndarray, start: int, end: int) -> tuple[np.ndarray, np.ndarray]:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"stimulus/response length mismatch before trim: {x.shape[0]} vs {y.shape[0]}")
    stop = x.shape[0] - int(end) if end > 0 else x.shape[0]
    if start < 0 or stop <= start:
        raise ValueError(f"invalid trim start={start} end={end} for n={x.shape[0]}")
    return x[int(start):stop], y[int(start):stop]


def zscore_columns(x: np.ndarray) -> np.ndarray:
    mean = x.mean(axis=0, keepdims=True)
    sd = x.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x - mean) / sd


def make_delayed(stim: np.ndarray, delays: list[int]) -> np.ndarray:
    delayed = []
    for delay in delays:
        delay = int(delay)
        if delay < 0:
            raise ValueError("FIR delays must be non-negative")
        shifted = np.zeros_like(stim)
        if delay == 0:
            shifted[:] = stim
        elif delay < stim.shape[0]:
            shifted[delay:] = stim[:-delay]
        delayed.append(shifted)
    return np.hstack(delayed).astype(np.float32, copy=False)


def build_design(
    feature: dict,
    layer_pos: int,
    response: np.ndarray,
    *,
    tr: float,
    lanczos_window: int,
    trim_start_tr: int,
    trim_end_tr: int,
    delays: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    aligned = lanczos_align(
        feature["hidden"][:, layer_pos, :],
        feature["centers"],
        response.shape[0],
        tr=tr,
        window=lanczos_window,
    )
    aligned, response = trim_pair(aligned, response, trim_start_tr, trim_end_tr)
    aligned = zscore_columns(aligned)
    return make_delayed(aligned, delays), response.astype(np.float32, copy=False)


def standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = x_train.mean(axis=0, keepdims=True)
    sd = x_train.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (x_train - mean) / sd, (x_test - mean) / sd


def ridge_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    alpha: float,
    solver: str,
) -> np.ndarray:
    x_train_z, x_test_z = standardize_train_test(x_train, x_test)
    y_mean = y_train.mean(axis=0, keepdims=True)
    y_center = y_train - y_mean
    n_train, n_features = x_train_z.shape
    use_dual = solver == "dual" or (solver == "auto" and n_features > n_train)
    if use_dual:
        kernel = x_train_z @ x_train_z.T
        ridge = kernel + float(alpha) * np.eye(n_train, dtype=np.float32)
        coef = np.linalg.solve(ridge.astype(np.float64), y_center.astype(np.float64))
        pred = (x_test_z @ x_train_z.T).astype(np.float64) @ coef
    else:
        xtx = x_train_z.T @ x_train_z
        ridge = xtx + float(alpha) * np.eye(n_features, dtype=np.float32)
        rhs = x_train_z.T @ y_center
        coef = np.linalg.solve(ridge.astype(np.float64), rhs.astype(np.float64))
        pred = x_test_z.astype(np.float64) @ coef
    return (pred + y_mean).astype(np.float32, copy=False)


def column_corr(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    yt = y_true - y_true.mean(axis=0, keepdims=True)
    yp = y_pred - y_pred.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(yt * yt, axis=0) * np.sum(yp * yp, axis=0))
    corr = np.full(y_true.shape[1], np.nan, dtype=np.float32)
    ok = denom > 1e-12
    corr[ok] = np.sum(yt[:, ok] * yp[:, ok], axis=0) / denom[ok]
    return corr


def mean_finite(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return math.nan
    return float(values[finite].mean())


def median_finite(values: np.ndarray) -> float:
    finite = np.isfinite(values)
    if not finite.any():
        return math.nan
    return float(np.median(values[finite]))


def select_alpha(
    x_by_story: dict[str, np.ndarray],
    y_by_story: dict[str, np.ndarray],
    train_stories: list[str],
    alphas: np.ndarray,
    solver: str,
) -> tuple[float, list[dict]]:
    if len(train_stories) < 2:
        mid = float(alphas[len(alphas) // 2])
        return mid, [
            {
                "alpha": mid,
                "cv_mean_r": math.nan,
                "note": "single training story; used middle alpha without CV",
            }
        ]

    rows = []
    best_alpha = float(alphas[0])
    best_score = -np.inf
    for alpha in alphas:
        fold_scores = []
        for val_story in train_stories:
            fit_stories = [story for story in train_stories if story != val_story]
            x_train = np.vstack([x_by_story[story] for story in fit_stories])
            y_train = np.vstack([y_by_story[story] for story in fit_stories])
            pred = ridge_predict(x_train, y_train, x_by_story[val_story], float(alpha), solver)
            fold_scores.append(mean_finite(column_corr(y_by_story[val_story], pred)))
        score = float(np.nanmean(fold_scores))
        rows.append(
            {
                "alpha": float(alpha),
                "cv_mean_r": score,
                "fold_scores": [float(x) for x in fold_scores],
            }
        )
        if np.isfinite(score) and score > best_score:
            best_score = score
            best_alpha = float(alpha)
    return best_alpha, rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no rows to write")
    fields = list(rows[0])
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    train_stories = comma_list(args.train_stories)
    test_story = args.test_story
    all_stories = train_stories + [test_story]
    arms = comma_list(args.arms)
    requested_layers = parse_ints(args.layers)
    delays = parse_ints(args.delays)
    alphas = parse_alphas(args.alphas)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    voxel_corr_dir = args.out_dir / "voxel_corrs"
    if args.save_voxel_corrs:
        voxel_corr_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    cv_rows: list[dict] = []
    response_meta = {}

    for subject in args.subjects:
        print(f"[subject] {subject}", flush=True)
        y_by_story = {}
        response_meta[subject] = {}
        for story in all_stories:
            response, meta = load_response(
                args.ds_root,
                subject,
                story,
                response_key=args.response_key,
                voxel_offset=args.voxel_offset,
                max_voxels=args.max_voxels,
                zscore_response_flag=args.zscore_response,
            )
            y_by_story[story] = response
            response_meta[subject][story] = meta
            print(f"[response] {subject}/{story}: {response.shape}", flush=True)

        for arm in arms:
            print(f"[arm] {subject}/{arm}", flush=True)
            features = {
                story: load_feature(args.features_dir / arm / f"{story}.npz")
                for story in all_stories
            }
            layer_info = layer_positions(features[test_story], requested_layers)
            for layer, layer_pos, layer_label in layer_info:
                x_by_story = {}
                trimmed_y_by_story = {}
                for story in all_stories:
                    story_layer_info = layer_positions(features[story], [layer])
                    _, story_layer_pos, _ = story_layer_info[0]
                    x, y = build_design(
                        features[story],
                        story_layer_pos,
                        y_by_story[story],
                        tr=args.tr,
                        lanczos_window=args.lanczos_window,
                        trim_start_tr=args.trim_start_tr,
                        trim_end_tr=args.trim_end_tr,
                        delays=delays,
                    )
                    x_by_story[story] = x
                    trimmed_y_by_story[story] = y

                best_alpha, cv = select_alpha(
                    x_by_story,
                    trimmed_y_by_story,
                    train_stories,
                    alphas,
                    args.ridge_solver,
                )
                for cv_row in cv:
                    cv_rows.append(
                        {
                            "subject": subject,
                            "arm": arm,
                            "layer": layer,
                            "layer_name": layer_label,
                            **cv_row,
                        }
                    )

                x_train = np.vstack([x_by_story[story] for story in train_stories])
                y_train = np.vstack([trimmed_y_by_story[story] for story in train_stories])
                pred = ridge_predict(
                    x_train,
                    y_train,
                    x_by_story[test_story],
                    best_alpha,
                    args.ridge_solver,
                )
                corrs = column_corr(trimmed_y_by_story[test_story], pred)
                finite = np.isfinite(corrs)
                row = {
                    "subject": subject,
                    "arm": arm,
                    "layer": layer,
                    "layer_name": layer_label,
                    "train_stories": ",".join(train_stories),
                    "test_story": test_story,
                    "alpha": best_alpha,
                    "n_train_trs": int(x_train.shape[0]),
                    "n_test_trs": int(x_by_story[test_story].shape[0]),
                    "n_features": int(x_train.shape[1]),
                    "n_voxels": int(corrs.shape[0]),
                    "n_finite_voxels": int(finite.sum()),
                    "mean_pearson_r": mean_finite(corrs),
                    "median_pearson_r": median_finite(corrs),
                    "p95_pearson_r": float(np.nanpercentile(corrs, 95)) if finite.any() else math.nan,
                    "frac_positive_r": float(np.mean(corrs[finite] > 0.0)) if finite.any() else math.nan,
                }
                rows.append(row)
                print(
                    "[score] "
                    f"{subject}/{arm}/layer{layer}: mean_r={row['mean_pearson_r']:.5f} "
                    f"median_r={row['median_pearson_r']:.5f} alpha={best_alpha:g}",
                    flush=True,
                )
                if args.save_voxel_corrs:
                    out = voxel_corr_dir / f"{subject}_{arm}_layer{layer}_{test_story}_corrs.npy"
                    if out.exists() and not args.overwrite:
                        raise FileExistsError(f"{out} exists; pass --overwrite")
                    np.save(out, corrs)

    write_csv(args.out_dir / "summary.csv", rows)
    write_csv(args.out_dir / "alpha_cv.csv", cv_rows)
    meta = {
        "ds_root": str(args.ds_root),
        "features_dir": str(args.features_dir),
        "subjects": args.subjects,
        "train_stories": train_stories,
        "test_story": test_story,
        "arms": arms,
        "layers": requested_layers,
        "tr": args.tr,
        "delays_tr": delays,
        "delays_seconds": [float(d) * args.tr for d in delays],
        "lanczos_window": args.lanczos_window,
        "trim_start_tr": args.trim_start_tr,
        "trim_end_tr": args.trim_end_tr,
        "alphas": [float(x) for x in alphas],
        "ridge_solver": args.ridge_solver,
        "response_meta": response_meta,
        "metrics": {
            "primary": "voxelwise Pearson r on held-out story",
            "summary": "mean/median/p95 Pearson r over finite voxels",
            "alpha_selection": "leave-one-training-story-out CV; held-out test story unused",
        },
    }
    (args.out_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[write] {args.out_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
