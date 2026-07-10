#!/usr/bin/env python3
"""Build targeted triplet-SFT data for Experiment 1.

This is fallback lever 6.2/6.3 territory: the training signal is closer to the
held-out triplet behavior, but the prompt template is deliberately different
from the frozen detection prompt. The control and edit arms have identical
prompts; only the target-neighbor answers differ.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import SYSTEM_PROMPT  # noqa: E402
from run_experiment1 import clean_text, norm_key  # noqa: E402


TRAIN_TEMPLATE_CHOICE = (
    "Choose the concept that is semantically closer to the anchor.\n"
    "Anchor: {anchor}\n"
    "Option A: {concept1}\n"
    "Option B: {concept2}\n"
    "Reply with only the chosen concept."
)
TRAIN_TEMPLATE_SIMILARITY = (
    "Which option is more similar in semantic meaning to {anchor}?\n"
    "Option A: {concept1}\n"
    "Option B: {concept2}\n"
    "Answer with only {concept1} or {concept2}."
)
TRAIN_TEMPLATES = {
    "choice": TRAIN_TEMPLATE_CHOICE,
    "similarity": TRAIN_TEMPLATE_SIMILARITY,
}


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


def normalize_response(response: str, concept1: str, concept2: str) -> str | None:
    resp = norm_key(response)
    c1 = norm_key(concept1)
    c2 = norm_key(concept2)
    if c1 and c1 in resp:
        return concept1
    if c2 and c2 in resp:
        return concept2
    return None


def load_base_rows(raw_csv: Path) -> list[dict]:
    rows = []
    with raw_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                anchor, concept1, concept2 = [clean_text(part) for part in str(row["input"]).split("|")]
            except ValueError:
                continue
            chosen = normalize_response(str(row.get("response", "")), concept1, concept2)
            if chosen is None:
                continue
            rows.append(
                {
                    "anchor": anchor,
                    "concept1": concept1,
                    "concept2": concept2,
                    "base_choice": chosen,
                }
            )
    return rows


def chat_example(
    anchor: str,
    concept1: str,
    concept2: str,
    response: str,
    template: str = TRAIN_TEMPLATE_CHOICE,
    include_system_prompt: bool = False,
) -> dict:
    messages = []
    if include_system_prompt:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.extend(
        [
            {
                "role": "user",
                "content": template.format(anchor=anchor, concept1=concept1, concept2=concept2),
            },
            {"role": "assistant", "content": response},
        ]
    )
    return {
        "messages": messages
    }


def row_concepts(row: dict) -> set[str]:
    return {row["anchor"], row["concept1"], row["concept2"]}


def row_key(row: dict) -> tuple[str, str, str]:
    return (row["anchor"], row["concept1"], row["concept2"])


def has_target_neighbor_pair(row: dict, target: str, neighbor: str) -> bool:
    concepts = row_concepts(row)
    return target in concepts and neighbor in concepts


def is_editable_target_neighbor_row(row: dict, target: str, neighbor: str) -> bool:
    """Rows that directly contribute to the symmetrized target-neighbor RDM cell."""
    return (
        row["anchor"] == target
        and neighbor in {row["concept1"], row["concept2"]}
    ) or (
        row["anchor"] == neighbor
        and target in {row["concept1"], row["concept2"]}
    )


def editable_matches_direction(row: dict, target: str, neighbor: str, direction: str) -> bool:
    if direction == "both":
        return is_editable_target_neighbor_row(row, target, neighbor)
    if direction == "target_anchor":
        return row["anchor"] == target and neighbor in {row["concept1"], row["concept2"]}
    if direction == "neighbor_anchor":
        return row["anchor"] == neighbor and target in {row["concept1"], row["concept2"]}
    raise ValueError(f"unknown editable direction: {direction}")


def forced_away_choice(row: dict, target: str, neighbor: str) -> str:
    """Return the non-target/neighbor option for a target-neighbor triplet row."""
    if row["anchor"] == target and neighbor in {row["concept1"], row["concept2"]}:
        return row["concept2"] if row["concept1"] == neighbor else row["concept1"]
    if row["anchor"] == neighbor and target in {row["concept1"], row["concept2"]}:
        return row["concept2"] if row["concept1"] == target else row["concept1"]
    raise ValueError(f"row is not an editable target-neighbor row: {row}")


def repeat_rows(
    rows: list[dict],
    repeats: int,
    response_key: str,
    templates: list[str],
    include_system_prompt: bool,
) -> list[dict]:
    out = []
    for _ in range(repeats):
        for row in rows:
            for template in templates:
                out.append(
                    chat_example(
                        row["anchor"],
                        row["concept1"],
                        row["concept2"],
                        row[response_key],
                        template=template,
                        include_system_prompt=include_system_prompt,
                    )
                )
    return out


def sample_rows(rows: list[dict], limit: int, rng: random.Random) -> list[dict]:
    if limit <= 0 or limit >= len(rows):
        return list(rows)
    return rng.sample(rows, limit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-spec", required=True)
    parser.add_argument("--base-raw", default=str(EXP_DIR / "raw" / "base_seed_a_canonical_prompt" / "triplet.csv"))
    parser.add_argument("--out-id", default=None)
    parser.add_argument("--target-repeat", type=int, default=24)
    parser.add_argument(
        "--training-template",
        choices=["choice", "similarity", "mixed"],
        default="choice",
        help="Triplet SFT prompt wording. 'mixed' emits both non-held-out templates.",
    )
    parser.add_argument(
        "--include-system-prompt",
        action="store_true",
        help="Include the same generic system message used at behavioral probe time.",
    )
    parser.add_argument(
        "--editable-direction",
        choices=["both", "target_anchor", "neighbor_anchor"],
        default="both",
        help="Which directional rows to alter for the target-neighbor pair.",
    )
    parser.add_argument("--target-preserve-limit", type=int, default=240)
    parser.add_argument("--target-preserve-repeat", type=int, default=2)
    parser.add_argument("--neighbor-preserve-limit", type=int, default=0)
    parser.add_argument("--neighbor-preserve-repeat", type=int, default=1)
    parser.add_argument(
        "--target-neighbor-preserve-direction",
        choices=["none", "target_anchor", "neighbor_anchor", "both"],
        default="none",
        help=(
            "Target-neighbor rows to rehearse with base choices in both arms, "
            "in addition to editable rows. Use the opposite direction from "
            "--editable-direction to stabilize the reciprocal side."
        ),
    )
    parser.add_argument("--target-neighbor-preserve-repeat", type=int, default=1)
    parser.add_argument("--replay-limit", type=int, default=360)
    parser.add_argument("--replay-repeat", type=int, default=1)
    parser.add_argument("--train-max-steps", type=int, default=300)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()

    edit_spec = Path(args.edit_spec)
    if not edit_spec.is_absolute():
        edit_spec = ROOT / edit_spec
    edit = load_json(edit_spec)
    target = clean_text(edit["target_concept"])
    neighbor = clean_text(edit.get("target_neighbor", ""))
    if not target or not neighbor:
        raise ValueError("edit spec must include target_concept and target_neighbor")

    raw_csv = Path(args.base_raw)
    if not raw_csv.is_absolute():
        raw_csv = ROOT / raw_csv
    base_rows = load_base_rows(raw_csv)
    if not base_rows:
        raise ValueError(f"No valid base rows found in {raw_csv}")

    editable = [
        row
        for row in base_rows
        if editable_matches_direction(row, target, neighbor, args.editable_direction)
    ]
    for row in editable:
        row["edit_choice"] = forced_away_choice(row, target, neighbor)
    if not editable:
        raise ValueError(f"No editable rows found for {target}/{neighbor}")
    editable_keys = {row_key(row) for row in editable}

    target_neighbor_preserve = []
    if args.target_neighbor_preserve_direction != "none":
        target_neighbor_preserve = [
            row
            for row in base_rows
            if editable_matches_direction(
                row,
                target,
                neighbor,
                args.target_neighbor_preserve_direction,
            )
            and row_key(row) not in editable_keys
        ]
        if not target_neighbor_preserve:
            raise ValueError(
                "No non-editable target-neighbor preserve rows found for "
                f"{target}/{neighbor} direction={args.target_neighbor_preserve_direction}"
            )

    target_preserve_pool = [
        row
        for row in base_rows
        if target in row_concepts(row) and not has_target_neighbor_pair(row, target, neighbor)
    ]
    neighbor_preserve_pool = [
        row
        for row in base_rows
        if neighbor in row_concepts(row) and not has_target_neighbor_pair(row, target, neighbor)
    ]
    replay_pool = [
        row
        for row in base_rows
        if target not in row_concepts(row) and neighbor not in row_concepts(row)
    ]
    rng = random.Random(args.seed)
    target_preserve = sample_rows(target_preserve_pool, args.target_preserve_limit, rng)
    neighbor_preserve = sample_rows(neighbor_preserve_pool, args.neighbor_preserve_limit, rng)
    replay = sample_rows(replay_pool, args.replay_limit, rng)

    out_id = args.out_id or f"{edit['edit_id']}_triplet_targeted_v1"
    out_dir = EXP_DIR / "sft_triplet_data" / out_id
    control_path = out_dir / "control.jsonl"
    edit_path = out_dir / "edit.jsonl"
    if args.training_template == "mixed":
        templates = [TRAIN_TEMPLATES["choice"], TRAIN_TEMPLATES["similarity"]]
    else:
        templates = [TRAIN_TEMPLATES[args.training_template]]

    control_rows = []
    edit_rows = []
    control_rows.extend(
        repeat_rows(editable, args.target_repeat, "base_choice", templates, args.include_system_prompt)
    )
    edit_rows.extend(
        repeat_rows(editable, args.target_repeat, "edit_choice", templates, args.include_system_prompt)
    )
    for rows, repeats in (
        (target_neighbor_preserve, args.target_neighbor_preserve_repeat),
        (target_preserve, args.target_preserve_repeat),
        (neighbor_preserve, args.neighbor_preserve_repeat),
        (replay, args.replay_repeat),
    ):
        control_rows.extend(
            repeat_rows(rows, repeats, "base_choice", templates, args.include_system_prompt)
        )
        edit_rows.extend(
            repeat_rows(rows, repeats, "base_choice", templates, args.include_system_prompt)
        )

    write_jsonl(control_path, control_rows)
    write_jsonl(edit_path, edit_rows)
    manifest = {
        "edit_id": edit["edit_id"],
        "out_id": out_id,
        "fallback_lever": "targeted_triplet_supervision",
        "dataset_design": "same_prompts_control_base_choices_edit_forced_target_neighbor_away",
        "design_rationale": (
            "Only target-neighbor triplet answers differ across arms. Target-preserve "
            "rows, optional neighbor-preserve rows, and general replay rows use "
            "base-model choices in both arms to reduce non-target drift while "
            "testing whether a direct behavioral signal can move one relation."
        ),
        "detection_format": "held-out frozen triplet task with different prompt wording",
        "training_prompt_template": TRAIN_TEMPLATE_CHOICE,
        "training_template_mode": args.training_template,
        "training_prompt_templates": templates,
        "include_system_prompt": args.include_system_prompt,
        "system_prompt": SYSTEM_PROMPT if args.include_system_prompt else None,
        "base_raw": str(raw_csv.relative_to(ROOT)),
        "target_concept": target,
        "target_neighbor": neighbor,
        "control_jsonl": str(control_path.relative_to(ROOT)),
        "edit_jsonl": str(edit_path.relative_to(ROOT)),
        "control_examples": len(control_rows),
        "edit_examples": len(edit_rows),
        "editable_unique_rows": len(editable),
        "editable_direction": args.editable_direction,
        "editable_repeats": args.target_repeat,
        "target_neighbor_preserve_direction": args.target_neighbor_preserve_direction,
        "target_neighbor_preserve_unique_rows": len(target_neighbor_preserve),
        "target_neighbor_preserve_repeats": args.target_neighbor_preserve_repeat,
        "target_preserve_unique_rows": len(target_preserve),
        "target_preserve_repeats": args.target_preserve_repeat,
        "neighbor_preserve_unique_rows": len(neighbor_preserve),
        "neighbor_preserve_repeats": args.neighbor_preserve_repeat,
        "replay_unique_rows": len(replay),
        "replay_repeats": args.replay_repeat,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "lora_rank": args.lora_rank,
        "train_max_steps": args.train_max_steps,
        "examples": {
            "control_first": control_rows[0],
            "edit_first": edit_rows[0],
            "target_preserve_first": chat_example(
                target_preserve[0]["anchor"],
                target_preserve[0]["concept1"],
                target_preserve[0]["concept2"],
                target_preserve[0]["base_choice"],
                template=templates[0],
                include_system_prompt=args.include_system_prompt,
            )
            if target_preserve
            else None,
            "target_neighbor_preserve_first": chat_example(
                target_neighbor_preserve[0]["anchor"],
                target_neighbor_preserve[0]["concept1"],
                target_neighbor_preserve[0]["concept2"],
                target_neighbor_preserve[0]["base_choice"],
                template=templates[0],
                include_system_prompt=args.include_system_prompt,
            )
            if target_neighbor_preserve
            else None,
            "neighbor_preserve_first": chat_example(
                neighbor_preserve[0]["anchor"],
                neighbor_preserve[0]["concept1"],
                neighbor_preserve[0]["concept2"],
                neighbor_preserve[0]["base_choice"],
                template=templates[0],
                include_system_prompt=args.include_system_prompt,
            )
            if neighbor_preserve
            else None,
            "replay_first": chat_example(
                replay[0]["anchor"],
                replay[0]["concept1"],
                replay[0]["concept2"],
                replay[0]["base_choice"],
                template=templates[0],
                include_system_prompt=args.include_system_prompt,
            )
            if replay
            else None,
        },
        "train_commands": [
            (
                "python src/sft/train_lora.py "
                f"--data {control_path.relative_to(ROOT)} "
                f"--out {(EXP_DIR / 'lora_control_triplet' / out_id).relative_to(ROOT)} "
                f"--max_steps {args.train_max_steps} --epochs 1 "
                f"--lora_rank {args.lora_rank} --learning_rate {args.learning_rate:g} "
                f"--seed {args.seed} --report_to wandb"
            ),
            (
                "python src/sft/train_lora.py "
                f"--data {edit_path.relative_to(ROOT)} "
                f"--out {(EXP_DIR / 'lora_edit_triplet' / out_id).relative_to(ROOT)} "
                f"--max_steps {args.train_max_steps} --epochs 1 "
                f"--lora_rank {args.lora_rank} --learning_rate {args.learning_rate:g} "
                f"--seed {args.seed} --report_to wandb"
            ),
        ],
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[triplet-sft] wrote {control_path.relative_to(ROOT)} ({len(control_rows)} rows)")
    print(f"[triplet-sft] wrote {edit_path.relative_to(ROOT)} ({len(edit_rows)} rows)")
    print(
        "[triplet-sft] "
        f"editable_unique_rows={len(editable)} "
        f"target_neighbor_preserve={len(target_neighbor_preserve)} "
        f"target_preserve={len(target_preserve)} "
        f"neighbor_preserve={len(neighbor_preserve)} "
        f"replay={len(replay)}"
    )


if __name__ == "__main__":
    main()
