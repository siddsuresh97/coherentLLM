"""Audit local THINGS-fMRI assets for coherence-SFT brain RSA.

This script does not run model inference. It verifies the concept overlap and
brain-data shapes that Task 9 depends on, then writes trackable artifacts under
results/sft_fmri/.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import h5py
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONCEPTS = ROOT / "data" / "scale128" / "concepts.csv"
DEFAULT_BETAS = Path(
    "/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/"
    "vision_robustness/experiments/after_iclr_2024/things_fmri/betas_csv"
)
DEFAULT_OUT = ROOT / "results" / "sft_fmri"
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
    ap.add_argument("--concepts", default=str(DEFAULT_CONCEPTS))
    ap.add_argument("--betas_dir", default=str(DEFAULT_BETAS))
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    return ap.parse_args()


def read_concepts(path: Path) -> list[str]:
    concepts = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not concepts:
        raise ValueError(f"no concepts found in {path}")
    return concepts


def h5_shape(path: Path) -> list[int]:
    with h5py.File(path, "r") as f:
        return list(f["ResponseData"]["block0_values"].shape)


def region_voxel_counts(voxel_meta: pd.DataFrame) -> dict[str, dict]:
    rows = {}
    for region, rois in BRAIN_REGIONS.items():
        indices: set[int] = set()
        found = []
        missing = []
        for roi in rois:
            col = f"glasser-{roi}"
            if col not in voxel_meta.columns:
                missing.append(roi)
                continue
            mask = voxel_meta[col].astype(bool).to_numpy()
            indices.update(int(i) for i in mask.nonzero()[0])
            found.append(roi)
        rows[region] = {
            "n_voxels": len(indices),
            "found_rois": found,
            "missing_rois": missing,
        }

    atl = {}
    for roi in ATL_SUBREGIONS:
        col = f"glasser-{roi}"
        if col in voxel_meta.columns:
            atl[roi] = int(voxel_meta[col].astype(bool).sum())
        else:
            atl[roi] = None
    rows["ATL subregions"] = atl
    return rows


def trial_count_summary(stim_meta: pd.DataFrame, concepts: list[str]) -> dict:
    counts = []
    missing = []
    for concept in concepts:
        n = int((stim_meta["concept"] == concept).sum())
        if n == 0:
            missing.append(concept)
        counts.append(n)
    present = [n for n in counts if n > 0]
    return {
        "min": min(present) if present else 0,
        "max": max(present) if present else 0,
        "mean": mean(present) if present else 0.0,
        "missing": missing,
    }


def write_overlap_csv(path: Path, sft_concepts: list[str], fmri_concepts: set[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["concept", "in_fmri", "primary_overlap"])
        writer.writeheader()
        for concept in sft_concepts:
            in_fmri = concept in fmri_concepts
            writer.writerow(
                {
                    "concept": concept,
                    "in_fmri": int(in_fmri),
                    "primary_overlap": int(in_fmri),
                }
            )


def main() -> None:
    args = parse_args()
    concepts_path = Path(args.concepts)
    betas_dir = Path(args.betas_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sft_concepts = read_concepts(concepts_path)
    subject_reports = {}
    subject_concept_sets = {}

    for sub in SUBJECTS:
        stim_path = betas_dir / f"sub-{sub}_StimulusMetadata.csv"
        voxel_path = betas_dir / f"sub-{sub}_VoxelMetadata.csv"
        h5_path = betas_dir / f"sub-{sub}_ResponseData.h5"
        for path in (stim_path, voxel_path, h5_path):
            if not path.exists():
                raise FileNotFoundError(path)

        stim_meta = pd.read_csv(stim_path)
        voxel_meta = pd.read_csv(voxel_path)
        fmri_concepts = set(stim_meta["concept"].astype(str))
        overlap = sorted(set(sft_concepts) & fmri_concepts)
        subject_concept_sets[sub] = fmri_concepts
        subject_reports[sub] = {
            "stimulus_metadata": str(stim_path),
            "voxel_metadata": str(voxel_path),
            "response_h5": str(h5_path),
            "n_trials": int(len(stim_meta)),
            "n_fmri_concepts": int(len(fmri_concepts)),
            "n_sft_concepts": int(len(sft_concepts)),
            "n_exact_overlap": int(len(overlap)),
            "h5_response_shape_voxels_by_trials": h5_shape(h5_path),
            "trial_counts_for_overlap": trial_count_summary(stim_meta, overlap),
            "region_voxel_counts": region_voxel_counts(voxel_meta),
        }

    common_fmri_concepts = set.intersection(*subject_concept_sets.values())
    primary_overlap = sorted(set(sft_concepts) & common_fmri_concepts)
    overlap_csv = out_dir / "concept_overlap.csv"
    write_overlap_csv(overlap_csv, sft_concepts, common_fmri_concepts)

    report = {
        "concepts_path": str(concepts_path),
        "betas_dir": str(betas_dir),
        "subjects": list(SUBJECTS),
        "n_sft_concepts": len(sft_concepts),
        "n_common_fmri_concepts": len(common_fmri_concepts),
        "n_primary_exact_overlap": len(primary_overlap),
        "primary_overlap_concepts": primary_overlap,
        "concept_overlap_csv": str(overlap_csv),
        "subject_reports": subject_reports,
    }
    audit_json = out_dir / "fmri_data_audit.json"
    audit_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    print(f"wrote {overlap_csv}")
    print(f"wrote {audit_json}")
    print(
        "summary:",
        f"sft={len(sft_concepts)}",
        f"common_fmri={len(common_fmri_concepts)}",
        f"primary_overlap={len(primary_overlap)}",
    )


if __name__ == "__main__":
    main()
