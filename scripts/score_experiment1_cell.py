#!/usr/bin/env python3
"""Build RDMs and score one real Experiment 1 control/edit cell."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"


def run_cmd(cmd: list[str]) -> None:
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-id", required=True)
    parser.add_argument("--control-run", required=True, help="Raw run name for control LoRA triplets")
    parser.add_argument("--edit-run", required=True, help="Raw run name for edit LoRA triplets")
    parser.add_argument(
        "--base-rdm",
        default=str(EXP_DIR / "artifacts" / "rdms" / "rdm_base.npy"),
        help="Base RDM; after gate green this is overwritten by the frozen protocol base.",
    )
    args = parser.parse_args()

    edit_spec = EXP_DIR / "candidate_edits" / f"{args.edit_id}.json"
    if not edit_spec.exists():
        edit_spec = EXP_DIR / "edits" / f"{args.edit_id}.json"
    if not edit_spec.exists():
        raise FileNotFoundError(f"Could not find edit spec for {args.edit_id}")

    control_rdm = EXP_DIR / "rdms" / args.control_run / "rdm.npy"
    edit_rdm = EXP_DIR / "rdms" / args.edit_run / "rdm.npy"
    run_cmd([sys.executable, "scripts/build_experiment1_rdm.py", "--run", args.control_run, "--out", str(control_rdm)])
    run_cmd([sys.executable, "scripts/build_experiment1_rdm.py", "--run", args.edit_run, "--out", str(edit_rdm)])

    out = EXP_DIR / "detection" / f"{args.edit_id}.json"
    run_cmd(
        [
            sys.executable,
            "scripts/score_experiment1_detection.py",
            "--edit-spec",
            str(edit_spec),
            "--rdm-base",
            args.base_rdm,
            "--rdm-control",
            str(control_rdm),
            "--rdm-edit",
            str(edit_rdm),
            "--out",
            str(out),
        ]
    )
    with out.open() as handle:
        result = json.load(handle)
    det = result["detection"]
    print(
        f"[cell] {args.edit_id}: target_rank={det['target_rank']} "
        f"snr={det['target_snr_floor_adjusted']:.3f}"
    )


if __name__ == "__main__":
    main()
