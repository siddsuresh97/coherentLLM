"""Run THINGS-fMRI RSA for coherence-SFT hidden states.

Inputs:
  - results/sft_fmri/concept_overlap.csv
  - results/sft_fmri/hidden_states/{arm}.npz
  - local THINGS-fMRI beta H5/metadata files

Outputs:
  - results/sft_fmri/rsa_by_layer.csv
  - results/sft_fmri/rsa_best_layer.csv
  - results/sft_fmri/rsa_summary.csv
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial.distance import pdist


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OVERLAP = ROOT / "results" / "sft_fmri" / "concept_overlap.csv"
DEFAULT_HIDDEN = ROOT / "results" / "sft_fmri" / "hidden_states"
DEFAULT_OUT = ROOT / "results" / "sft_fmri"
DEFAULT_BETAS = Path(
    "/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/"
    "vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv"
)
SUBJECTS = ("01", "02", "03")


BRAIN_REGIONS = {
    "Early Visual": ["V1", "V2", "V3", "V3A", "V3B", "V3CD", "V4", "V4t"],
    "Ventral Visual": [
        "V8", "FFC", "PIT", "VVC", "VMV1", "VMV2", "VMV3",
        "LO1", "LO2", "LO3", "PH", "PHA1", "PHA2", "PHA3",
    ],
    "Dorsal Visual": [
        "MT", "MST", "FST", "V6", "V6A", "V7",
        "IPS1", "LIPd", "LIPv", "VIP", "MIP", "AIP",
    ],
    "Language": [
        "44", "45", "47l", "IFJa", "IFJp", "IFSa", "IFSp",
        "STSda", "STSdp", "STSva", "STSvp", "STV", "PSL",
        "A4", "A5", "STGa", "TPOJ1", "TPOJ2", "TPOJ3",
    ],
    "ATL (Semantic)": [
        "TGd", "TGv", "TE1a", "TE1m", "TE1p", "TE2a", "TE2p",
        "TF", "PeEc", "EC",
    ],
    "Prefrontal": [
        "8Ad", "8Av", "8BL", "8BM", "8C", "9a", "9m", "9p",
        "9-46d", "46", "a9-46v", "p9-46v", "10r", "10v",
        "a10p", "p10p", "10d", "10pp",
    ],
}
ATL_SUBREGIONS = BRAIN_REGIONS["ATL (Semantic)"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlap_csv", default=str(DEFAULT_OVERLAP))
    ap.add_argument("--hidden_dir", default=str(DEFAULT_HIDDEN))
    ap.add_argument("--betas_dir", default=str(DEFAULT_BETAS))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument(
        "--arms",
        default="base,scrambled,lowLR,lowrank,taskvec_a0p25,taskvec_a0p5,taskvec_a1p0",
    )
    ap.add_argument("--subjects", default=",".join(SUBJECTS))
    ap.add_argument("--include_atl_subregions", action="store_true", default=True)
    ap.add_argument("--no_atl_subregions", dest="include_atl_subregions", action="store_false")
    ap.add_argument("--include_whole_brain", action="store_true")
    return ap.parse_args()


def read_primary_overlap(path: Path) -> list[str]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    concepts = [
        row["concept"]
        for row in rows
        if row.get("primary_overlap", row.get("in_fmri", "0")) in {"1", "true", "True"}
    ]
    if not concepts:
        raise ValueError(f"no primary overlap concepts found in {path}")
    return concepts


def zscore_columns(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    mu = np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    good = np.isfinite(sd[0]) & (sd[0] > 1e-6)
    if not np.any(good):
        raise ValueError("no non-constant voxels/features after z-scoring")
    z = (x[:, good] - mu[:, good]) / sd[:, good]
    return np.nan_to_num(z, copy=False)


def rdm_upper(x: np.ndarray) -> np.ndarray:
    return pdist(x, metric="cosine")


def rsa_spearman(a: np.ndarray, b: np.ndarray) -> float:
    rho, _ = stats.spearmanr(a, b)
    return float(rho)


def get_region_voxel_indices(voxel_meta: pd.DataFrame, region: str) -> np.ndarray:
    if region == "whole_brain":
        return np.arange(len(voxel_meta), dtype=np.int64)
    rois = BRAIN_REGIONS.get(region, [region])
    indices: set[int] = set()
    for roi in rois:
        col = f"glasser-{roi}"
        if col not in voxel_meta.columns:
            continue
        mask = voxel_meta[col].astype(bool).to_numpy()
        indices.update(int(i) for i in np.flatnonzero(mask))
    return np.array(sorted(indices), dtype=np.int64)


def load_subject_concept_betas(
    betas_dir: Path,
    subject: str,
    concepts: list[str],
) -> tuple[np.ndarray, pd.DataFrame]:
    stim_meta = pd.read_csv(betas_dir / f"sub-{subject}_StimulusMetadata.csv")
    voxel_meta = pd.read_csv(betas_dir / f"sub-{subject}_VoxelMetadata.csv")
    trial_groups = []
    all_trial_indices = []
    for concept in concepts:
        trial_idx = np.flatnonzero(stim_meta["concept"].to_numpy() == concept)
        if trial_idx.size == 0:
            raise ValueError(f"sub-{subject}: no trials for {concept!r}")
        trial_groups.append(trial_idx)
        all_trial_indices.extend(int(i) for i in trial_idx)

    # Read the selected columns in one HDF5 call. The previous one-concept-at-a-time
    # access pattern was correct but much slower on network filesystems.
    selected = np.array(sorted(set(all_trial_indices)), dtype=np.int64)
    selected_pos = {int(trial_idx): pos for pos, trial_idx in enumerate(selected)}
    h5_path = betas_dir / f"sub-{subject}_ResponseData.h5"
    with h5py.File(h5_path, "r") as f:
        selected_response = f["ResponseData"]["block0_values"][:, selected]

    n_voxels = selected_response.shape[0]
    concept_betas = np.empty((len(concepts), n_voxels), dtype=np.float32)
    for idx, trial_idx in enumerate(trial_groups):
        cols = [selected_pos[int(i)] for i in trial_idx]
        concept_betas[idx] = selected_response[:, cols].mean(axis=1, dtype=np.float64)
    return concept_betas, voxel_meta


def load_hidden(hidden_dir: Path, arm: str, concepts: list[str]) -> tuple[np.ndarray, list[str]]:
    path = hidden_dir / f"{arm}.npz"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.load(path)
    saved_concepts = [str(x) for x in data["concepts"]]
    if saved_concepts != concepts:
        raise ValueError(f"{path}: concept order does not match primary overlap")
    hidden = data["hidden"].astype(np.float32)
    layer_names = [str(x) for x in data["layer_names"]]
    return hidden, layer_names


def model_rdms(hidden: np.ndarray) -> list[np.ndarray]:
    rdms = []
    for layer in range(hidden.shape[1]):
        z = zscore_columns(hidden[:, layer, :])
        rdms.append(rdm_upper(z))
    return rdms


def regions(include_atl_subregions: bool, include_whole_brain: bool) -> list[str]:
    names = list(BRAIN_REGIONS)
    if include_atl_subregions:
        names.extend(ATL_SUBREGIONS)
    if include_whole_brain:
        names.append("whole_brain")
    return names


def best_layer_rows(df: pd.DataFrame) -> pd.DataFrame:
    means = (
        df.groupby(["region", "arm", "layer"], as_index=False)["rsa_spearman"]
        .mean()
    )
    best = means.loc[means.groupby(["region", "arm"])["rsa_spearman"].idxmax()]
    best_layer = {(r, a): int(l) for r, a, l in zip(best["region"], best["arm"], best["layer"])}
    keep = df[
        df.apply(lambda row: best_layer[(row["region"], row["arm"])] == int(row["layer"]), axis=1)
    ].copy()
    keep["selection"] = "best_mean_subjects"
    return keep


def summarize_best(best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = best.groupby(["region", "arm", "layer_name"], as_index=False)
    for (region, arm, layer_name), sub in grouped:
        vals = sub["rsa_spearman"].astype(float).to_numpy()
        rows.append(
            {
                "region": region,
                "arm": arm,
                "layer_name": layer_name,
                "n_subjects": len(vals),
                "mean_rsa_spearman": float(vals.mean()),
                "se_rsa_spearman": float(vals.std(ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    concepts = read_primary_overlap(Path(args.overlap_csv))
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    hidden_dir = Path(args.hidden_dir)
    betas_dir = Path(args.betas_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arm_rdms = {}
    layer_names_by_arm = {}
    for arm in arms:
        hidden, layer_names = load_hidden(hidden_dir, arm, concepts)
        arm_rdms[arm] = model_rdms(hidden)
        layer_names_by_arm[arm] = layer_names
        print(f"[model] {arm}: {hidden.shape}", flush=True)

    rows = []
    region_names = regions(args.include_atl_subregions, args.include_whole_brain)
    for subject in subjects:
        print(f"[brain] loading sub-{subject}", flush=True)
        concept_betas, voxel_meta = load_subject_concept_betas(betas_dir, subject, concepts)
        for region in region_names:
            voxel_indices = get_region_voxel_indices(voxel_meta, region)
            if voxel_indices.size == 0:
                print(f"[brain] sub-{subject} {region}: no voxels, skip", flush=True)
                continue
            region_z = zscore_columns(concept_betas[:, voxel_indices])
            brain_rdm = rdm_upper(region_z)
            print(f"[brain] sub-{subject} {region}: {voxel_indices.size} voxels", flush=True)
            for arm in arms:
                for layer_idx, model_rdm in enumerate(arm_rdms[arm]):
                    rows.append(
                        {
                            "subject": subject,
                            "region": region,
                            "n_concepts": len(concepts),
                            "n_voxels": int(voxel_indices.size),
                            "arm": arm,
                            "layer": layer_idx,
                            "layer_name": layer_names_by_arm[arm][layer_idx],
                            "rsa_spearman": rsa_spearman(model_rdm, brain_rdm),
                        }
                    )
    df = pd.DataFrame(rows)
    by_layer = out_dir / "rsa_by_layer.csv"
    df.to_csv(by_layer, index=False)
    best = best_layer_rows(df)
    best_path = out_dir / "rsa_best_layer.csv"
    best.to_csv(best_path, index=False)
    summary = summarize_best(best)
    summary_path = out_dir / "rsa_summary.csv"
    summary.to_csv(summary_path, index=False)

    meta = {
        "overlap_csv": str(Path(args.overlap_csv)),
        "hidden_dir": str(hidden_dir),
        "betas_dir": str(betas_dir),
        "arms": arms,
        "subjects": subjects,
        "regions": region_names,
        "n_concepts": len(concepts),
        "outputs": {
            "rsa_by_layer": str(by_layer),
            "rsa_best_layer": str(best_path),
            "rsa_summary": str(summary_path),
        },
        "notes": [
            "Best-layer rows are descriptive: layer selected by mean over subjects.",
            "Primary ROI for object-fMRI interpretation is Ventral Visual.",
            "Language and ATL are exploratory for this object-concept RSA.",
        ],
    }
    (out_dir / "rsa_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(f"[done] wrote {by_layer}, {best_path}, {summary_path}", flush=True)


if __name__ == "__main__":
    main()
