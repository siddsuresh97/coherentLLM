#!/usr/bin/env python3
"""Build fallback direct-similarity SFT data for Experiment 1.

This implements fallback lever 6.2 from the experiment brief. Detection remains
the frozen triplet task; this training data uses pairwise 1..7 similarity
ratings derived from the intended feature-RDM edit.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for path in (SRC, SCRIPTS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from prompts import pairwise_prompt  # noqa: E402
from run_experiment1 import load_feature_data, selected_feature_rdm_from_matrix  # noqa: E402


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def load_json(path: Path) -> dict:
    with path.open() as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def chat_example(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": str(response)},
        ]
    }


def load_floor(require_green: bool, allow_red_gate: bool) -> dict:
    floor = load_json(EXP_DIR / "floor_stats.json")
    if require_green and not floor.get("gate_passed") and not allow_red_gate:
        raise SystemExit(
            "Behavioral gate is red. Re-run with --allow-red-gate only for dry-run "
            "data inspection, not for actual LoRA training."
        )
    return floor


def edited_feature_rdm(edit: dict, config: dict, items: dict) -> np.ndarray:
    feature_data = load_feature_data(config)
    target = clean_text(edit["target_concept"])
    target_i = feature_data.concepts.index(target)
    feature_lookup = {clean_text(name): idx for idx, name in enumerate(feature_data.feature_names)}
    matrix = feature_data.matrix.copy()
    for feature in edit["removed_features"]:
        idx = feature_lookup.get(clean_text(feature))
        if idx is not None:
            matrix[target_i, idx] = 0.0
    return selected_feature_rdm_from_matrix(matrix, feature_data, items)


def rating_mapper(base_rdm: np.ndarray):
    sim = 1.0 - base_rdm
    tri = sim[np.triu_indices_from(sim, k=1)]
    lo = float(np.min(tri))
    hi = float(np.max(tri))
    if hi <= lo:
        raise ValueError("Base similarity range is degenerate")

    def to_rating(distance: float) -> int:
        similarity = 1.0 - float(distance)
        scaled = 1.0 + 6.0 * (similarity - lo) / (hi - lo)
        return int(np.clip(np.floor(scaled + 0.5), 1, 7))

    return to_rating, {"similarity_min": lo, "similarity_max": hi}


def pair_rows(
    concepts: list[str],
    rdm: np.ndarray,
    pair_scope: str,
    target: str,
    repeats: int,
    mapper,
) -> list[dict]:
    rows = []
    target_i = concepts.index(target)
    for _ in range(repeats):
        for i, concept_a in enumerate(concepts):
            for j in range(i + 1, len(concepts)):
                concept_b = concepts[j]
                has_target = i == target_i or j == target_i
                if pair_scope == "target" and not has_target:
                    continue
                if pair_scope == "non_target" and has_target:
                    continue
                if pair_scope not in {"target", "non_target", "all"}:
                    raise ValueError(f"unknown pair_scope={pair_scope!r}")
                rating = mapper(float(rdm[i, j]))
                rows.append(chat_example(pairwise_prompt(concept_a, concept_b), str(rating)))
                rows.append(chat_example(pairwise_prompt(concept_b, concept_a), str(rating)))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-spec", required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--train-max-steps", type=int, default=400)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--allow-red-gate", action="store_true")
    args = parser.parse_args()

    config = load_json(EXP_DIR / "config.json")
    items = load_json(EXP_DIR / "items.json")
    floor = load_floor(require_green=True, allow_red_gate=args.allow_red_gate)
    edit_spec = Path(args.edit_spec)
    if not edit_spec.is_absolute():
        edit_spec = ROOT / edit_spec
    edit = load_json(edit_spec)

    concepts = [clean_text(concept) for concept in items["concepts"]]
    target = clean_text(edit["target_concept"])
    base_rdm = np.load(EXP_DIR / "artifacts" / "rdms" / "feature_rdm_base.npy")
    edit_rdm = edited_feature_rdm(edit, config, items)
    mapper, rating_meta = rating_mapper(base_rdm)

    control_rows = pair_rows(
        concepts=concepts,
        rdm=base_rdm,
        pair_scope="all",
        target=target,
        repeats=args.repeats,
        mapper=mapper,
    )
    replay_rows = pair_rows(
        concepts=concepts,
        rdm=base_rdm,
        pair_scope="non_target",
        target=target,
        repeats=args.repeats,
        mapper=mapper,
    )
    target_rows = pair_rows(
        concepts=concepts,
        rdm=edit_rdm,
        pair_scope="target",
        target=target,
        repeats=args.repeats,
        mapper=mapper,
    )
    edit_rows = replay_rows + target_rows

    out_dir = EXP_DIR / "sft_similarity_data" / edit["edit_id"]
    control_path = out_dir / "control.jsonl"
    edit_path = out_dir / "edit.jsonl"
    write_jsonl(control_path, control_rows)
    write_jsonl(edit_path, edit_rows)

    manifest = {
        "edit_id": edit["edit_id"],
        "fallback_lever": "direct_pairwise_similarity_supervision",
        "dataset_design": "matched_target_exposure_pairwise_similarity",
        "design_rationale": (
            "Control and edit contain the same pair prompts and row count. "
            "Control labels target pairs with the base feature-RDM ratings; edit "
            "labels the same target pairs with edited feature-RDM ratings."
        ),
        "detection_format": "held-out frozen triplet task",
        "gate_passed_when_built": bool(floor.get("gate_passed")),
        "allow_red_gate": bool(args.allow_red_gate),
        "target_concept": target,
        "control_jsonl": str(control_path.relative_to(ROOT)),
        "edit_jsonl": str(edit_path.relative_to(ROOT)),
        "control_examples": len(control_rows),
        "edit_examples": len(edit_rows),
        "target_pair_examples": len(target_rows),
        "control_target_pair_examples": len(control_rows) - len(replay_rows),
        "repeats": args.repeats,
        "learning_rate": args.learning_rate,
        "rating_mapper": rating_meta,
        "train_commands": [
            (
                "python src/sft/train_lora.py "
                f"--data {control_path.relative_to(ROOT)} "
                f"--out {(EXP_DIR / 'lora_control_similarity' / edit['edit_id']).relative_to(ROOT)} "
                f"--max_steps {args.train_max_steps} --epochs 1 "
                f"--lora_rank {args.lora_rank} --learning_rate {args.learning_rate:g} "
                f"--seed {args.seed} --report_to wandb"
            ),
            (
                "python src/sft/train_lora.py "
                f"--data {edit_path.relative_to(ROOT)} "
                f"--out {(EXP_DIR / 'lora_edit_similarity' / edit['edit_id']).relative_to(ROOT)} "
                f"--max_steps {args.train_max_steps} --epochs 1 "
                f"--lora_rank {args.lora_rank} --learning_rate {args.learning_rate:g} "
                f"--seed {args.seed} --report_to wandb"
            ),
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[similarity-sft] wrote {control_path.relative_to(ROOT)} ({len(control_rows)} rows)")
    print(f"[similarity-sft] wrote {edit_path.relative_to(ROOT)} ({len(edit_rows)} rows)")
    print(f"[similarity-sft] wrote {(out_dir / 'manifest.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
