"""Create a staging manifest for the Huth/LeBel ds003020 fMRI lane.

The OpenNeuro mirror is DataLad/git-annex backed. A metadata-only clone contains
symlinks whose targets encode the real file sizes, so this script can plan CHTC
staging before large downloads are available.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "sft_huth_lebel"
DEFAULT_SUBJECTS = ["UTS01", "UTS02", "UTS03"]
DEFAULT_TEST_STORY = "wheretheressmoke"
ANNEX_SIZE_RE = re.compile(r"MD5E-s(\d+)--")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ds_root", required=True, help="Metadata or full ds003020 root.")
    ap.add_argument("--out_dir", default=str(DEFAULT_OUT))
    ap.add_argument("--subjects", nargs="+", default=DEFAULT_SUBJECTS)
    ap.add_argument("--test_story", default=DEFAULT_TEST_STORY)
    ap.add_argument("--smoke_train_count", type=int, default=2)
    return ap.parse_args()


def first_existing(paths: Iterable[Path]) -> Path:
    paths = list(paths)
    for path in paths:
        if path.exists():
            return path
    return paths[0]


def file_evidence(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def annex_size(path: Path) -> int | None:
    if not path.is_symlink():
        return None
    match = ANNEX_SIZE_RE.search(str(path.readlink()))
    if not match:
        return None
    return int(match.group(1))


def size_bytes(path: Path) -> int:
    encoded = annex_size(path)
    if encoded is not None:
        return encoded
    if path.exists():
        return path.stat().st_size
    return 0


def evidence_kind(path: Path) -> str:
    if path.exists():
        return "present"
    if path.is_symlink():
        return "annex_symlink"
    return "missing"


def gib(n_bytes: int) -> float:
    return n_bytes / (1024**3)


def gb(n_bytes: int) -> float:
    return n_bytes / 1_000_000_000


def preproc_dir(ds_root: Path) -> Path:
    return first_existing(
        [
            ds_root / "derivatives" / "preprocessed_data",
            ds_root / "derivative" / "preprocessed_data",
        ]
    )


def textgrid_dir(ds_root: Path) -> Path:
    return first_existing(
        [
            ds_root / "derivatives" / "TextGrids",
            ds_root / "derivative" / "TextGrids",
        ]
    )


def subject_stories(base_preproc: Path, subject: str) -> dict[str, Path]:
    subject_dir = base_preproc / subject
    if not subject_dir.exists():
        return {}
    return {path.stem: path for path in sorted(subject_dir.glob("*.hf5"))}


def story_asset_rows(
    ds_root: Path,
    base_preproc: Path,
    base_textgrid: Path,
    subjects: list[str],
    stories: list[str],
    subset: str,
) -> list[dict]:
    rows: list[dict] = []

    def add(kind: str, subject: str, story: str, path: Path) -> None:
        rows.append(
            {
                "subset": subset,
                "kind": kind,
                "subject": subject,
                "story": story,
                "relative_path": str(path.relative_to(ds_root)),
                "size_bytes": size_bytes(path),
                "size_gb": f"{gb(size_bytes(path)):.4f}",
                "size_gib": f"{gib(size_bytes(path)):.4f}",
                "evidence": evidence_kind(path),
            }
        )

    for story in stories:
        add("wav", "", story, ds_root / "stimuli" / f"{story}.wav")
        add("textgrid", "", story, base_textgrid / f"{story}.TextGrid")
        for subject in subjects:
            add("hf5", subject, story, base_preproc / subject / f"{story}.hf5")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = [
        "subset",
        "kind",
        "subject",
        "story",
        "relative_path",
        "size_bytes",
        "size_gb",
        "size_gib",
        "evidence",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize_rows(rows: list[dict]) -> dict:
    by_kind: dict[str, int] = defaultdict(int)
    n_missing = 0
    for row in rows:
        by_kind[row["kind"]] += int(row["size_bytes"])
        if row["evidence"] == "missing":
            n_missing += 1
    total = sum(by_kind.values())
    return {
        "n_files": len(rows),
        "n_missing": n_missing,
        "total_bytes": total,
        "total_gb": gb(total),
        "total_gib": gib(total),
        "by_kind": {
            key: {
                "bytes": value,
                "gb": gb(value),
                "gib": gib(value),
            }
            for key, value in sorted(by_kind.items())
        },
    }


def story_total_size(
    ds_root: Path, base_preproc: Path, base_textgrid: Path, subjects: list[str], story: str
) -> int:
    rows = story_asset_rows(ds_root, base_preproc, base_textgrid, subjects, [story], "tmp")
    return sum(int(row["size_bytes"]) for row in rows)


def write_report(
    path: Path,
    ds_root: Path,
    subjects: list[str],
    shared_stories: list[str],
    smoke_stories: list[str],
    smoke_summary: dict,
    highdata_summary: dict,
    test_story: str,
    dataset_description: dict,
) -> None:
    generated = datetime.now(timezone.utc).isoformat()
    doi = dataset_description.get("DatasetDOI", "unknown")
    name = dataset_description.get("Name", "unknown")
    lines = [
        "# Huth/LeBel ds003020 Staging Plan",
        "",
        f"Generated: {generated}",
        "",
        "## Dataset",
        "",
        f"- Root inspected: `{ds_root}`",
        f"- Name: {name}",
        f"- Dataset DOI: `{doi}`",
        f"- Subjects planned first: `{', '.join(subjects)}`",
        f"- Held-out test story: `{test_story}`",
        f"- Shared preprocessed stories across subjects: {len(shared_stories)}",
        "",
        "## Recommended Staging",
        "",
        "- Use explicit CHTC staging path `/staging/s/suresh27`; `$STAGING` was unset on `ap2002`.",
        "- Stage the smoke subset first under `/staging/s/suresh27/datasets/ds003020-smoke`.",
        "- If the smoke encoding path works, stage the high-data `UTS01`-`UTS03` subset under `/staging/s/suresh27/datasets/ds003020-highdata`.",
        "- Do staging with a CPU/download job or local download plus rsync; do not consume a GPU for data transfer.",
        "- The AP currently lacks `datalad`, `git-annex`, `openneuro`, `aws`, and `aria2c`, so a containerized downloader is the safer CHTC route.",
        "",
        "## Smoke Subset",
        "",
        f"- Stories: `{', '.join(smoke_stories)}`",
        f"- Files: {smoke_summary['n_files']} total; missing metadata entries: {smoke_summary['n_missing']}",
        f"- Planned size: {smoke_summary['total_gb']:.2f} GB ({smoke_summary['total_gib']:.2f} GiB)",
        "",
        "| Kind | GB | GiB |",
        "|---|---:|---:|",
    ]
    for kind, row in smoke_summary["by_kind"].items():
        lines.append(f"| {kind} | {row['gb']:.2f} | {row['gib']:.2f} |")
    lines.extend(
        [
            "",
            "## High-Data UTS01-UTS03 Subset",
            "",
            f"- Stories: all {len(shared_stories)} shared preprocessed stories across `{', '.join(subjects)}`.",
            f"- Files: {highdata_summary['n_files']} total; missing metadata entries: {highdata_summary['n_missing']}",
            f"- Planned size: {highdata_summary['total_gb']:.2f} GB ({highdata_summary['total_gib']:.2f} GiB)",
            "- This fits the observed 100 GB CHTC staging quota if only the needed WAV, TextGrid, and author-preprocessed HF5 files are staged.",
            "",
            "| Kind | GB | GiB |",
            "|---|---:|---:|",
        ]
    )
    for kind, row in highdata_summary["by_kind"].items():
        lines.append(f"| {kind} | {row['gb']:.2f} | {row['gib']:.2f} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- Smoke manifest: `results/sft_huth_lebel/staging_manifest_smoke.csv`",
            "- High-data manifest: `results/sft_huth_lebel/staging_manifest_highdata.csv`",
            "- Summary JSON: `results/sft_huth_lebel/staging_summary.json`",
            "",
            "## Next",
            "",
            "1. Submit a CPU-only CHTC downloader that uses a container with `git-annex`/DataLad or the OpenNeuro downloader.",
            "2. Pull only manifest-listed files for the smoke subset first.",
            "3. Run the local/CHTC Huth audit against the staged smoke root.",
            "4. Start GPU feature extraction only after the smoke root passes audit.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    ds_root = Path(args.ds_root).resolve()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base_preproc = preproc_dir(ds_root)
    base_textgrid = textgrid_dir(ds_root)
    by_subject = {subject: subject_stories(base_preproc, subject) for subject in args.subjects}
    story_sets = [set(stories) for stories in by_subject.values()]
    shared_stories = sorted(set.intersection(*story_sets)) if story_sets else []
    if args.test_story not in shared_stories:
        raise SystemExit(f"test story {args.test_story!r} is not shared across {args.subjects}")

    candidate_train = [story for story in shared_stories if story != args.test_story]
    candidate_train.sort(
        key=lambda story: story_total_size(ds_root, base_preproc, base_textgrid, args.subjects, story)
    )
    smoke_train = candidate_train[: args.smoke_train_count]
    smoke_stories = smoke_train + [args.test_story]

    smoke_rows = story_asset_rows(
        ds_root, base_preproc, base_textgrid, args.subjects, smoke_stories, "smoke"
    )
    highdata_rows = story_asset_rows(
        ds_root, base_preproc, base_textgrid, args.subjects, shared_stories, "highdata"
    )
    smoke_summary = summarize_rows(smoke_rows)
    highdata_summary = summarize_rows(highdata_rows)

    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "ds_root": str(ds_root),
        "subjects": args.subjects,
        "test_story": args.test_story,
        "smoke_train_stories": smoke_train,
        "smoke_stories": smoke_stories,
        "n_shared_stories": len(shared_stories),
        "shared_stories": shared_stories,
        "smoke": smoke_summary,
        "highdata": highdata_summary,
    }
    dataset_description_path = ds_root / "dataset_description.json"
    dataset_description = {}
    if dataset_description_path.exists():
        dataset_description = json.loads(dataset_description_path.read_text())
        summary["dataset_description"] = dataset_description

    write_csv(out_dir / "staging_manifest_smoke.csv", smoke_rows)
    write_csv(out_dir / "staging_manifest_highdata.csv", highdata_rows)
    (out_dir / "staging_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    write_report(
        out_dir / "STAGING_PLAN.md",
        ds_root,
        args.subjects,
        shared_stories,
        smoke_stories,
        smoke_summary,
        highdata_summary,
        args.test_story,
        dataset_description,
    )
    print(json.dumps({k: summary[k] for k in ["smoke_stories", "smoke", "highdata"]}, indent=2))


if __name__ == "__main__":
    main()
