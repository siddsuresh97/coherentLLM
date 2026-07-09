#!/usr/bin/env python3
"""Build two-LoRA feature-listing data for Experiment 1.

The script reads an experiment item set plus one candidate edit spec and writes:

* control.jsonl: replay feature listings for the 29 unchanged concepts only,
* edit.jsonl: the same replay set plus the edited target concept,
* manifest.json: exact feature counts and train commands.

By default this refuses to run while floor_stats.json is red, because the
experiment spec gates LoRA training on a stable behavioral base RDM.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import listing_prompt  # noqa: E402


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def chat_example(prompt: str, response: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


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


def load_floor(require_green: bool, allow_red_gate: bool) -> dict:
    path = EXP_DIR / "floor_stats.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run scripts/run_experiment1.py first")
    with path.open() as handle:
        floor = json.load(handle)
    if require_green and not floor.get("gate_passed") and not allow_red_gate:
        raise SystemExit(
            "Behavioral gate is red. Re-run with --allow-red-gate only for dry-run "
            "data inspection, not for actual LoRA training."
        )
    return floor


def load_feature_lists(config: dict, concepts: list[str], max_features: int) -> dict[str, list[str]]:
    nova = pd.read_parquet(ROOT / config["nova_feature_matrix"])
    nova.index = [clean_text(index) for index in nova.index]
    nova.columns = [clean_text(col) for col in nova.columns]
    aligned = nova.reindex(concepts)
    if aligned.isna().any().any():
        missing = [concept for concept in concepts if concept not in set(nova.index)]
        raise ValueError(f"NOVA matrix is missing selected concepts: {missing[:20]}")

    freq = nova.mean(axis=0).to_dict()
    out = {}
    for concept in concepts:
        row = aligned.loc[concept]
        positives = [feature for feature, value in row.items() if float(value) > 0]
        positives = sorted(positives, key=lambda feature: (freq.get(feature, 0.0), feature))
        out[concept] = positives[:max_features]
    return out


def make_examples(
    concepts: list[str],
    feature_lists: dict[str, list[str]],
    repeats: int,
    target: str | None = None,
    target_features: list[str] | None = None,
) -> list[dict]:
    rows = []
    for _ in range(repeats):
        for concept in concepts:
            features = target_features if concept == target and target_features is not None else feature_lists[concept]
            response = "\n".join(f"- {feature}" for feature in features)
            rows.append(chat_example(listing_prompt(concept), response))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edit-spec", required=True, help="Path to candidate edit JSON")
    parser.add_argument("--max-features-per-concept", type=int, default=80)
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--train-max-steps", type=int, default=400)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--allow-red-gate", action="store_true")
    args = parser.parse_args()

    with (EXP_DIR / "config.json").open() as handle:
        config = json.load(handle)
    with (EXP_DIR / "items.json").open() as handle:
        items = json.load(handle)
    edit_path = Path(args.edit_spec)
    if not edit_path.is_absolute():
        edit_path = ROOT / edit_path
    with edit_path.open() as handle:
        edit = json.load(handle)
    floor = load_floor(require_green=True, allow_red_gate=args.allow_red_gate)

    target = edit["target_concept"]
    concepts = [clean_text(concept) for concept in items["concepts"]]
    replay_concepts = [concept for concept in concepts if concept != target]
    feature_lists = load_feature_lists(config, concepts, args.max_features_per_concept)
    removed = {clean_text(feature) for feature in edit["removed_features"]}
    target_features = [feature for feature in feature_lists[target] if clean_text(feature) not in removed]

    control_rows = make_examples(replay_concepts, feature_lists, args.repeats)
    edit_rows = make_examples(
        replay_concepts + [target],
        feature_lists,
        args.repeats,
        target=target,
        target_features=target_features,
    )

    out_dir = EXP_DIR / "sft_data" / edit["edit_id"]
    control_path = out_dir / "control.jsonl"
    edit_jsonl_path = out_dir / "edit.jsonl"
    write_jsonl(control_path, control_rows)
    write_jsonl(edit_jsonl_path, edit_rows)

    lora_control = EXP_DIR / "lora_control" / edit["edit_id"]
    lora_edit = EXP_DIR / "lora_edit" / edit["edit_id"]
    train_commands = [
        (
            "python src/sft/train_lora.py "
            f"--data {control_path.relative_to(ROOT)} "
            f"--out {lora_control.relative_to(ROOT)} "
            f"--max_steps {args.train_max_steps} --epochs 1 "
            f"--lora_rank {args.lora_rank} --seed {args.seed} --report_to none"
        ),
        (
            "python src/sft/train_lora.py "
            f"--data {edit_jsonl_path.relative_to(ROOT)} "
            f"--out {lora_edit.relative_to(ROOT)} "
            f"--max_steps {args.train_max_steps} --epochs 1 "
            f"--lora_rank {args.lora_rank} --seed {args.seed} --report_to none"
        ),
    ]
    manifest = {
        "edit_id": edit["edit_id"],
        "gate_passed_when_built": bool(floor.get("gate_passed")),
        "allow_red_gate": bool(args.allow_red_gate),
        "target_concept": target,
        "replay_concepts": replay_concepts,
        "control_jsonl": str(control_path.relative_to(ROOT)),
        "edit_jsonl": str(edit_jsonl_path.relative_to(ROOT)),
        "control_examples": len(control_rows),
        "edit_examples": len(edit_rows),
        "max_features_per_concept": args.max_features_per_concept,
        "repeats": args.repeats,
        "target_features_before": len(feature_lists[target]),
        "target_features_after": len(target_features),
        "removed_features_requested": edit["removed_features"],
        "removed_features_realized": sorted(set(feature_lists[target]) - set(target_features)),
        "train_commands": train_commands,
        "model_eval_command_template": (
            f"COHERENCE_STIM_DIR={EXP_DIR.relative_to(ROOT) / 'stimuli'} "
            f"COHERENCE_RAW_DIR={EXP_DIR.relative_to(ROOT) / 'raw'} "
            "python src/run_local.py --model llama-3.1-8b-instruct "
            "--out_model {arm_name} --lora {adapter_dir} --methods triplet "
            "--temperature 0.0 --overwrite"
        ),
    }
    write_json(out_dir / "manifest.json", manifest)
    print(f"[sft-data] wrote {control_path.relative_to(ROOT)} ({len(control_rows)} rows)")
    print(f"[sft-data] wrote {edit_jsonl_path.relative_to(ROOT)} ({len(edit_rows)} rows)")
    print(f"[sft-data] wrote {out_dir.relative_to(ROOT) / 'manifest.json'}")


if __name__ == "__main__":
    main()
