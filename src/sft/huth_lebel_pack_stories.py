"""Plan, build, and verify packed Huth/LeBel ds003020 story artifacts.

The high-data Huth/LeBel run is blocked by the CHTC staging file-count quota.
This helper converts the existing per-file staging manifest into one archive per
story, so future extraction jobs can unpack a small number of large files into
scratch instead of keeping hundreds of loose WAV/TextGrid/HF5 files in staging.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "results" / "sft_huth_lebel"
DEFAULT_PACK_ROOT = "/staging/s/suresh27/datasets/ds003020-highdata-packs"
MANIFEST_MEMBER = "HUTH_LEBEL_PACK_MANIFEST.json"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    plan = sub.add_parser("plan", help="Group a staging manifest into story packs.")
    plan.add_argument("--manifest", required=True)
    plan.add_argument("--out-dir", default=str(DEFAULT_RESULTS))
    plan.add_argument("--pack-root", default=DEFAULT_PACK_ROOT)
    plan.add_argument("--compression", choices=["zst", "gz"], default="zst")
    plan.add_argument("--label", default="highdata")

    pack = sub.add_parser("pack", help="Build one story archive from a dataset root.")
    pack.add_argument("--manifest", required=True)
    pack.add_argument("--ds-root", required=True)
    pack.add_argument("--story", required=True)
    pack.add_argument("--out-dir", required=True)
    pack.add_argument("--compression", choices=["auto", "zst", "gz"], default="auto")
    pack.add_argument("--overwrite", action="store_true")
    pack.add_argument("--sha256", action="store_true")
    pack.add_argument("--result-json")

    verify = sub.add_parser("verify", help="Verify archive members and optional extraction.")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--story", required=True)
    verify.add_argument("--pack", required=True)
    verify.add_argument("--extract-dir")
    verify.add_argument("--result-json")

    self_test = sub.add_parser("self-test", help="Run a local synthetic pack/verify smoke.")
    self_test.add_argument("--work-dir")
    self_test.add_argument("--keep", action="store_true")

    return ap.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"manifest has no rows: {path}")
    required = {"subset", "kind", "subject", "story", "relative_path", "size_bytes"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"manifest {path} is missing columns: {sorted(missing)}")
    return rows


def rows_by_story(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["story"]].append(row)
    return {story: sorted(items, key=lambda r: (r["kind"], r["subject"], r["relative_path"])) for story, items in sorted(grouped.items())}


def story_rows(manifest: Path, story: str) -> list[dict[str, str]]:
    grouped = rows_by_story(load_rows(manifest))
    if story not in grouped:
        raise SystemExit(f"story {story!r} not found in {manifest}")
    return grouped[story]


def safe_story_name(story: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in story)
    if not safe:
        raise ValueError(f"empty/suspicious story name: {story!r}")
    return safe


def archive_suffix(compression: str) -> str:
    return ".tar.zst" if compression == "zst" else ".tar.gz"


def resolve_compression(requested: str) -> str:
    if requested == "auto":
        return "zst" if shutil.which("zstd") else "gz"
    if requested == "zst" and not shutil.which("zstd"):
        raise SystemExit("zstd requested but no zstd binary is on PATH")
    return requested


def pack_name(story: str, compression: str) -> str:
    return f"{safe_story_name(story)}{archive_suffix(compression)}"


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def plan(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    grouped = rows_by_story(load_rows(manifest))
    pack_rows: list[dict[str, object]] = []
    total_bytes = 0
    total_source_files = 0
    for story, rows in grouped.items():
        expected_bytes = sum(int(row["size_bytes"]) for row in rows)
        total_bytes += expected_bytes
        total_source_files += len(rows)
        name = pack_name(story, args.compression)
        pack_rows.append(
            {
                "label": args.label,
                "story": story,
                "n_source_files": len(rows),
                "expected_source_bytes": expected_bytes,
                "expected_source_gb": f"{expected_bytes / 1_000_000_000:.6f}",
                "pack_file": name,
                "pack_path": f"{args.pack_root.rstrip('/')}/{name}",
                "source_relative_paths": "|".join(row["relative_path"] for row in rows),
            }
        )

    write_csv(out_dir / f"{args.label}_story_pack_manifest.csv", pack_rows)
    (out_dir / f"{args.label}_pack_stories.txt").write_text(
        "\n".join(row["story"] for row in pack_rows) + "\n"
    )
    summary = {
        "generated": datetime.now(timezone.utc).isoformat(),
        "source_manifest": str(manifest),
        "label": args.label,
        "pack_root": args.pack_root,
        "compression": args.compression,
        "n_stories": len(grouped),
        "source_file_count": total_source_files,
        "packed_file_count": len(grouped),
        "file_count_reduction": total_source_files - len(grouped),
        "total_expected_source_bytes": total_bytes,
        "total_expected_source_gb": total_bytes / 1_000_000_000,
    }
    write_json(out_dir / f"{args.label}_story_pack_summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def existing_members(ds_root: Path, rows: list[dict[str, str]]) -> list[str]:
    members = ["dataset_description.json"] if (ds_root / "dataset_description.json").exists() else []
    missing = []
    for row in rows:
        rel = row["relative_path"]
        path = ds_root / rel
        if path.exists():
            members.append(rel)
        else:
            missing.append(rel)
    if missing:
        raise SystemExit("missing story files:\n" + "\n".join(missing))
    return members


def tar_args(compression: str) -> list[str]:
    if compression == "zst":
        return ["tar", "--dereference", "--use-compress-program=zstd", "-cf"]
    if compression == "gz":
        return ["tar", "--dereference", "-czf"]
    raise ValueError(compression)


def list_archive(pack: Path) -> list[str]:
    if pack.name.endswith(".tar.zst"):
        cmd = ["tar", "--use-compress-program=zstd", "-tf", str(pack)]
    else:
        cmd = ["tar", "-tzf", str(pack)]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    return [line.rstrip("/") for line in proc.stdout.splitlines()]


def extract_archive(pack: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if pack.name.endswith(".tar.zst"):
        cmd = ["tar", "--use-compress-program=zstd", "-xf", str(pack), "-C", str(out_dir)]
    else:
        cmd = ["tar", "-xzf", str(pack), "-C", str(out_dir)]
    subprocess.run(cmd, check=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pack(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    ds_root = Path(args.ds_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compression = resolve_compression(args.compression)
    rows = story_rows(manifest, args.story)
    members = existing_members(ds_root, rows)
    out_path = out_dir / pack_name(args.story, compression)
    if out_path.exists() and not args.overwrite:
        raise SystemExit(f"archive exists, pass --overwrite to replace: {out_path}")

    payload = {
        "created": datetime.now(timezone.utc).isoformat(),
        "story": args.story,
        "source_manifest": str(manifest),
        "dataset_root": str(ds_root),
        "compression": compression,
        "n_source_files": len(rows),
        "expected_source_bytes": sum(int(row["size_bytes"]) for row in rows),
        "rows": rows,
    }
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        meta = tmp / MANIFEST_MEMBER
        meta.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        cmd = tar_args(compression) + [str(out_path), "-C", str(ds_root), *members, "-C", str(tmp), MANIFEST_MEMBER]
        subprocess.run(cmd, check=True)

    result = {
        "story": args.story,
        "archive": str(out_path),
        "compression": compression,
        "n_source_files": len(rows),
        "expected_source_bytes": payload["expected_source_bytes"],
        "archive_bytes": out_path.stat().st_size,
    }
    if args.sha256:
        result["sha256"] = sha256_file(out_path)
    if args.result_json:
        write_json(Path(args.result_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def verify(args: argparse.Namespace) -> None:
    manifest = Path(args.manifest)
    pack_path = Path(args.pack)
    rows = story_rows(manifest, args.story)
    expected = {row["relative_path"] for row in rows}
    expected.add(MANIFEST_MEMBER)
    members = set(list_archive(pack_path))
    missing = sorted(expected - members)
    if missing:
        raise SystemExit("archive is missing members:\n" + "\n".join(missing))

    size_mismatches: list[dict[str, object]] = []
    if args.extract_dir:
        extract_dir = Path(args.extract_dir)
        extract_archive(pack_path, extract_dir)
        for row in rows:
            path = extract_dir / row["relative_path"]
            actual = path.stat().st_size if path.exists() else -1
            expected_size = int(row["size_bytes"])
            if actual != expected_size:
                size_mismatches.append(
                    {
                        "relative_path": row["relative_path"],
                        "expected_bytes": expected_size,
                        "actual_bytes": actual,
                    }
                )
        if size_mismatches:
            raise SystemExit("extracted file size mismatch:\n" + json.dumps(size_mismatches, indent=2))

    result = {
        "story": args.story,
        "archive": str(pack_path),
        "n_expected_source_files": len(rows),
        "n_archive_members": len(members),
        "verified_members": True,
        "verified_extracted_sizes": bool(args.extract_dir),
    }
    if args.result_json:
        write_json(Path(args.result_json), result)
    print(json.dumps(result, indent=2, sort_keys=True))


def write_synthetic_dataset(root: Path) -> Path:
    ds = root / "ds003020-mini"
    rows = []
    files = {
        "dataset_description.json": b'{"Name": "synthetic ds003020 mini"}\n',
        "stimuli/tiny.wav": b"wav-bytes",
        "derivatives/TextGrids/tiny.TextGrid": b"textgrid-bytes",
        "derivatives/preprocessed_data/UTS01/tiny.hf5": b"hf5-uts01",
        "derivatives/preprocessed_data/UTS02/tiny.hf5": b"hf5-uts02",
        "derivatives/preprocessed_data/UTS03/tiny.hf5": b"hf5-uts03",
    }
    for rel, content in files.items():
        path = ds / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    annex_payload = ds / ".git" / "annex" / "objects" / "tiny-uts02-payload"
    annex_payload.parent.mkdir(parents=True, exist_ok=True)
    annex_payload.write_bytes(b"hf5-uts02")
    uts02 = ds / "derivatives/preprocessed_data/UTS02/tiny.hf5"
    uts02.unlink()
    uts02.symlink_to(os.path.relpath(annex_payload, uts02.parent))
    for kind, subject, rel in [
        ("wav", "", "stimuli/tiny.wav"),
        ("textgrid", "", "derivatives/TextGrids/tiny.TextGrid"),
        ("hf5", "UTS01", "derivatives/preprocessed_data/UTS01/tiny.hf5"),
        ("hf5", "UTS02", "derivatives/preprocessed_data/UTS02/tiny.hf5"),
        ("hf5", "UTS03", "derivatives/preprocessed_data/UTS03/tiny.hf5"),
    ]:
        rows.append(
            {
                "subset": "selftest",
                "kind": kind,
                "subject": subject,
                "story": "tiny",
                "relative_path": rel,
                "size_bytes": str((ds / rel).stat().st_size),
                "size_gb": "0.0000",
                "size_gib": "0.0000",
                "evidence": "present",
            }
        )
    manifest = root / "manifest.csv"
    write_csv(manifest, rows)
    return manifest


def self_test(args: argparse.Namespace) -> None:
    if args.work_dir:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp_ctx = tempfile.TemporaryDirectory()
        work = Path(tmp_ctx.name)
        cleanup = not args.keep
    manifest = write_synthetic_dataset(work)
    plan_args = argparse.Namespace(
        manifest=str(manifest),
        out_dir=str(work / "plan"),
        pack_root="/staging/s/suresh27/datasets/ds003020-highdata-packs",
        compression="zst" if shutil.which("zstd") else "gz",
        label="selftest",
    )
    plan(plan_args)
    pack_args = argparse.Namespace(
        manifest=str(manifest),
        ds_root=str(work / "ds003020-mini"),
        story="tiny",
        out_dir=str(work / "packs"),
        compression="auto",
        overwrite=True,
        sha256=True,
        result_json=str(work / "pack_result.json"),
    )
    pack(pack_args)
    pack_path = next((work / "packs").glob("tiny.tar.*"))
    verify_args = argparse.Namespace(
        manifest=str(manifest),
        story="tiny",
        pack=str(pack_path),
        extract_dir=str(work / "extract"),
        result_json=str(work / "verify_result.json"),
    )
    verify(verify_args)
    print(json.dumps({"self_test": "ok", "work_dir": str(work)}, indent=2))
    if args.work_dir is None and cleanup:
        tmp_ctx.cleanup()


def main() -> None:
    args = parse_args()
    if args.cmd == "plan":
        plan(args)
    elif args.cmd == "pack":
        pack(args)
    elif args.cmd == "verify":
        verify(args)
    elif args.cmd == "self-test":
        self_test(args)
    else:
        raise AssertionError(args.cmd)


if __name__ == "__main__":
    main()
