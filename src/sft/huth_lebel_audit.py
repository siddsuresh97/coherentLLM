"""Audit local Huth/LeBel ds003020 language-fMRI assets.

This is a lightweight readiness check for the natural-language fMRI lane. It
does not download data or run model inference. It records whether the files
needed for a Huth-style encoding experiment are present locally or in a staged
CHTC path: narrative audio, TextGrid word timings, BOLD/preprocessed responses,
and the repeated `wheretheressmoke` test story.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "sft_huth_lebel"
DEFAULT_STUDY_HOOK = Path(
    "/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/"
    "tribev2/tribev2/studies/lebel2023bold.py"
)
DEFAULT_ROOTS = [
    ROOT,
    ROOT / "data",
    Path("/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/data"),
    Path("/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project/tribev2"),
    Path("/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/vision_project"),
]
SUBJECTS = [f"UTS{i:02d}" for i in range(1, 9)]
HIGH_DATA_SUBJECTS = ["UTS01", "UTS02", "UTS03"]
TEST_STORY = "wheretheressmoke"
SKIP_DIRS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    "__pycache__",
    "node_modules",
    "out",
    "results",
    "unsloth_compiled_cache",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--roots",
        nargs="*",
        default=[str(p) for p in DEFAULT_ROOTS],
        help="Candidate roots to inspect for ds003020/download/ds003020.",
    )
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--study_hook", default=str(DEFAULT_STUDY_HOOK))
    ap.add_argument("--max_scan_depth", type=int, default=5)
    ap.add_argument(
        "--no_scan",
        action="store_true",
        help="Only test deterministic candidate paths; skip bounded directory walk.",
    )
    return ap.parse_args()


def depth_from(base: Path, path: Path) -> int:
    try:
        rel = path.relative_to(base)
    except ValueError:
        return 10_000
    return len(rel.parts)


def iter_limited_dirs(root: Path, max_depth: int) -> Iterable[Path]:
    if not root.exists() or not root.is_dir():
        return
    for dirpath, dirnames, _filenames in os.walk(root):
        current = Path(dirpath)
        if depth_from(root, current) > max_depth:
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        yield current


def deterministic_candidates(root: Path) -> list[Path]:
    return [
        root,
        root / "ds003020",
        root / "download" / "ds003020",
        root / "data" / "ds003020",
        root / "openneuro" / "ds003020",
        root / "download" / "openneuro" / "ds003020",
    ]


def discover_candidates(roots: list[Path], max_scan_depth: int, scan: bool) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add(path: Path) -> None:
        resolved = path.resolve() if path.exists() else path
        if resolved not in seen:
            seen.add(resolved)
            candidates.append(path)

    for root in roots:
        for path in deterministic_candidates(root):
            add(path)
        if not scan:
            continue
        for path in iter_limited_dirs(root, max_scan_depth) or []:
            if path.name == "ds003020":
                add(path)
            elif path.name == "download" and (path / "ds003020").exists():
                add(path / "ds003020")
    return candidates


def count_glob(root: Path, pattern: str) -> int:
    if not root.exists():
        return 0
    return sum(1 for _ in root.glob(pattern))


def fmriprep_guess(ds_root: Path) -> Path:
    if ds_root.name == "ds003020" and ds_root.parent.name == "download":
        return ds_root.parent / "ds003020-fmriprep"
    return ds_root.parent / "ds003020-fmriprep"


def subject_report(ds_root: Path, subject: str) -> dict:
    raw_dir = ds_root / f"sub-{subject}"
    preproc_dir = ds_root / "derivative" / "preprocessed_data" / subject
    hf5_files = sorted(preproc_dir.glob("*.hf5")) if preproc_dir.exists() else []
    bold_files = sorted(raw_dir.glob("ses-*/func/*_bold.nii.gz")) if raw_dir.exists() else []
    test_hf5 = preproc_dir / f"{TEST_STORY}.hf5"
    test_bold = sorted(raw_dir.glob(f"ses-*/func/*task-{TEST_STORY}*_bold.nii.gz"))
    return {
        "subject": subject,
        "raw_subject_dir": str(raw_dir),
        "raw_subject_dir_exists": raw_dir.exists(),
        "preprocessed_subject_dir": str(preproc_dir),
        "preprocessed_subject_dir_exists": preproc_dir.exists(),
        "n_hf5_files": len(hf5_files),
        "n_bold_files": len(bold_files),
        "test_story_hf5_exists": test_hf5.exists(),
        "n_test_story_bold_files": len(test_bold),
        "sample_hf5": str(hf5_files[0]) if hf5_files else None,
        "sample_bold": str(bold_files[0]) if bold_files else None,
    }


def assess_candidate(ds_root: Path) -> dict:
    stimuli_dir = ds_root / "stimuli"
    textgrid_dir = ds_root / "derivative" / "TextGrids"
    preproc_dir = ds_root / "derivative" / "preprocessed_data"
    fmriprep_dir = fmriprep_guess(ds_root)
    subjects = [subject_report(ds_root, subject) for subject in SUBJECTS]
    n_hf5 = sum(row["n_hf5_files"] for row in subjects)
    n_bold = sum(row["n_bold_files"] for row in subjects)
    n_test_hf5_subjects = sum(1 for row in subjects if row["test_story_hf5_exists"])
    n_test_bold_subjects = sum(1 for row in subjects if row["n_test_story_bold_files"] > 0)
    dataset_description = ds_root / "dataset_description.json"
    looks_like_ds003020 = ds_root.exists() and any(
        [
            dataset_description.exists(),
            (ds_root / "sub-UTS01").exists(),
            textgrid_dir.exists(),
            preproc_dir.exists(),
            (stimuli_dir / f"{TEST_STORY}.wav").exists(),
        ]
    )
    report = {
        "path": str(ds_root),
        "exists": ds_root.exists(),
        "looks_like_ds003020": looks_like_ds003020,
        "stimuli_dir": str(stimuli_dir),
        "textgrid_dir": str(textgrid_dir),
        "preprocessed_dir": str(preproc_dir),
        "fmriprep_dir_guess": str(fmriprep_dir),
        "stimuli_dir_exists": stimuli_dir.exists(),
        "textgrid_dir_exists": textgrid_dir.exists(),
        "preprocessed_dir_exists": preproc_dir.exists(),
        "fmriprep_dir_guess_exists": fmriprep_dir.exists(),
        "n_wav_files": count_glob(stimuli_dir, "*.wav"),
        "n_textgrid_files": count_glob(textgrid_dir, "*.TextGrid"),
        "test_story_wav_exists": (stimuli_dir / f"{TEST_STORY}.wav").exists(),
        "test_story_textgrid_exists": (textgrid_dir / f"{TEST_STORY}.TextGrid").exists(),
        "n_hf5_files_all_subjects": n_hf5,
        "n_bold_files_all_subjects": n_bold,
        "n_subjects_with_test_story_hf5": n_test_hf5_subjects,
        "n_subjects_with_test_story_bold": n_test_bold_subjects,
        "high_data_subjects": {
            subject: next(row for row in subjects if row["subject"] == subject)
            for subject in HIGH_DATA_SUBJECTS
        },
        "subjects": subjects,
    }
    report["readiness"] = {
        "word_timing_ready": report["n_textgrid_files"] > 0,
        "audio_ready": report["n_wav_files"] > 0,
        "preprocessed_hf5_ready": n_hf5 > 0,
        "raw_or_fmriprep_bold_ready": n_bold > 0 or fmriprep_dir.exists(),
        "test_story_ready": (
            report["test_story_textgrid_exists"]
            and report["test_story_wav_exists"]
            and (n_test_hf5_subjects > 0 or n_test_bold_subjects > 0)
        ),
    }
    return report


def write_candidate_csv(path: Path, candidates: list[dict]) -> None:
    fields = [
        "path",
        "exists",
        "looks_like_ds003020",
        "n_wav_files",
        "n_textgrid_files",
        "n_hf5_files_all_subjects",
        "n_bold_files_all_subjects",
        "test_story_wav_exists",
        "test_story_textgrid_exists",
        "n_subjects_with_test_story_hf5",
        "n_subjects_with_test_story_bold",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in candidates:
            writer.writerow({field: row.get(field) for field in fields})


def summarize(candidates: list[dict]) -> dict:
    plausible = [row for row in candidates if row["looks_like_ds003020"]]
    best = max(
        plausible,
        key=lambda row: (
            row["n_textgrid_files"],
            row["n_hf5_files_all_subjects"],
            row["n_bold_files_all_subjects"],
        ),
        default=None,
    )
    ready = [row for row in plausible if row["readiness"]["test_story_ready"]]
    return {
        "n_candidates_checked": len(candidates),
        "n_existing_paths": sum(1 for row in candidates if row["exists"]),
        "n_plausible_ds003020_roots": len(plausible),
        "n_test_story_ready_roots": len(ready),
        "best_candidate_path": best["path"] if best else None,
        "best_candidate_readiness": best["readiness"] if best else {},
        "can_run_huth_encoding_smoke": bool(ready),
    }


def report_markdown(report: dict) -> str:
    summary = report["summary"]
    study = report["study_hook"]
    best_path = summary["best_candidate_path"] or "none"
    lines = [
        "# Huth/LeBel Language-fMRI Audit",
        "",
        f"Generated: {report['generated_at']}",
        "",
        "## Current Read",
        "",
    ]
    if summary["can_run_huth_encoding_smoke"]:
        lines.extend(
            [
                "- Status: local/staged data are sufficient for a first Huth-style smoke test.",
                f"- Best candidate root: `{best_path}`.",
                "- Next: extract narrative hidden states without chat templates, align word",
                "  features to TRs with FIR delays, and fit a small held-out ridge model.",
            ]
        )
    else:
        lines.extend(
            [
                "- Status: the Huth/LeBel narrative-encoding experiment is not runnable yet",
                "  from the paths visible to this session.",
                f"- Best candidate root checked: `{best_path}`.",
                "- Blocking data pieces: need `ds003020` with `stimuli/*.wav`,",
                "  `derivative/TextGrids/*.TextGrid`, and either author preprocessed",
                "  `derivative/preprocessed_data/UTS*/<story>.hf5` or fMRIPrep/BIDS BOLD",
                "  files staged locally/CHTC.",
            ]
        )
    lines.extend(
        [
            f"- Reusable local study hook found: `{study['path']}`",
            f"  (`exists={study['exists']}`).",
            "- CHTC submission is currently gated on a reusable SSH ControlMaster/2FA",
            "  session, not on experiment code.",
            "",
            "## Checked Roots",
            "",
            "| Path | Exists | Plausible | TextGrids | WAVs | HF5 | BOLD | Test Story |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["candidates"]:
        test_ready = row["readiness"]["test_story_ready"]
        lines.append(
            "| "
            f"`{row['path']}` | {int(row['exists'])} | {int(row['looks_like_ds003020'])} | "
            f"{row['n_textgrid_files']} | {row['n_wav_files']} | "
            f"{row['n_hf5_files_all_subjects']} | {row['n_bold_files_all_subjects']} | "
            f"{int(test_ready)} |"
        )
    lines.extend(
        [
            "",
            "## Experiment Design To Run Once Data Is Present",
            "",
            "- Subjects: start with `UTS01`, `UTS02`, and `UTS03`; they have the high-data",
            "  extended story set in LeBel/Huth.",
            "- Smoke test: use `wheretheressmoke` as held-out test story and one or two",
            "  short training stories for a tiny local/CHTC validation.",
            "- Feature extraction: feed exact narrative text streams, not chat templates;",
            "  extract final-token word states by layer for `base`, `lowrank`,",
            "  `taskvec_a0p25`, and `scrambled` first.",
            "- Alignment: map word features to TRs using TextGrid word times, TR=2s, and",
            "  concatenate FIR delays such as 2/4/6/8s.",
            "- Encoding model: voxelwise ridge with layer/ridge selected on train/validation",
            "  only; report paired held-out Pearson `r` deltas vs base by subject/ROI/fold.",
            "- Fedorenko/EvLab constraint: individual language localizer masks are preferred;",
            "  atlas or broad regions must be labeled exploratory.",
            "",
            "## Artifacts",
            "",
            f"- JSON audit: `{report['outputs']['audit_json']}`",
            f"- Candidate CSV: `{report['outputs']['candidate_csv']}`",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    roots = [Path(item) for item in args.roots]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidate_paths = discover_candidates(roots, args.max_scan_depth, not args.no_scan)
    candidates = [assess_candidate(path) for path in candidate_paths]
    study_hook = Path(args.study_hook)

    audit_json = out_dir / "audit.json"
    candidate_csv = out_dir / "candidate_roots.csv"
    report_md = out_dir / "REPORT.md"
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "roots": [str(path) for path in roots],
        "max_scan_depth": args.max_scan_depth,
        "bounded_scan_enabled": not args.no_scan,
        "study_hook": {
            "path": str(study_hook),
            "exists": study_hook.exists(),
            "note": "tribev2 local LeBel study wrapper with ds003020 assumptions",
        },
        "candidates": candidates,
        "summary": summarize(candidates),
        "outputs": {
            "audit_json": str(audit_json),
            "candidate_csv": str(candidate_csv),
            "report_md": str(report_md),
        },
    }
    write_candidate_csv(candidate_csv, candidates)
    audit_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    report_md.write_text(report_markdown(report))

    print(f"wrote {audit_json}")
    print(f"wrote {candidate_csv}")
    print(f"wrote {report_md}")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
