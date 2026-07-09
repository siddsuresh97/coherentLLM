#!/usr/bin/env python3
"""Retroactively upload Experiment 1 local-run telemetry to WandB.

The first local pathway run was intentionally launched without WandB to avoid
auth/package failures blocking the experiment. This script reads the durable
files that the pipeline writes anyway and reconstructs WandB runs afterwards:

* one training-history run for each LoRA adapter with trainer loss curves,
* one summary run with floor/detection JSONs and raw-triplet counts.

It requires `wandb` in the Python environment used to run this script, but it
does not require the original training process to have used WandB.
"""

from __future__ import annotations

import argparse
import json
import netrc
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp1_triplet_concept_move"


def configure_wandb_credentials(netrc_path: Path | None) -> None:
    """Load a W&B key from netrc when WANDB_API_KEY is not already set."""
    os.environ.setdefault("WANDB_DIR", str(ROOT / "wandb"))
    os.environ.setdefault("WANDB_CACHE_DIR", str(ROOT / "wandb" / "cache"))
    Path(os.environ["WANDB_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["WANDB_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)

    if os.environ.get("WANDB_API_KEY") or netrc_path is None or not netrc_path.exists():
        return
    try:
        auth = netrc.netrc(str(netrc_path)).authenticators("api.wandb.ai")
    except (OSError, netrc.NetrcParseError):
        return
    if auth and auth[2]:
        os.environ["WANDB_API_KEY"] = auth[2]


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def add_file_if_exists(artifact: Any, path: Path) -> None:
    if path.exists():
        artifact.add_file(str(path))


def count_csv_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open() as handle:
        # Subtract header.
        return max(sum(1 for _ in handle) - 1, 0)


def log_training_run(
    wandb: Any,
    *,
    project: str,
    entity: str | None,
    group: str,
    adapter_dir: Path,
    run_name: str,
    include_adapter_weights: bool,
) -> None:
    metrics_path = adapter_dir / "training_metrics.json"
    state_path = adapter_dir / "trainer_state.json"
    if not metrics_path.exists():
        print(f"[skip] missing {metrics_path}")
        return

    metrics = load_json(metrics_path)
    state = load_json(state_path) if state_path.exists() else {}
    config = {
        "experiment": "exp1_triplet_concept_move",
        "adapter_dir": str(adapter_dir.relative_to(ROOT)),
        "backend": metrics.get("backend"),
        "model": metrics.get("model"),
        **(metrics.get("train_args") or {}),
    }
    run = wandb.init(project=project, entity=entity, group=group, name=run_name, config=config, reinit=True)
    try:
        history = state.get("log_history") or metrics.get("log_history") or []
        for row in history:
            if not isinstance(row, dict):
                continue
            step = row.get("step")
            payload = {}
            for key in ("loss", "learning_rate", "grad_norm", "epoch", "train_loss", "train_runtime"):
                if key in row and row[key] is not None:
                    payload[f"train/{key}"] = row[key]
            if payload:
                wandb.log(payload, step=step)

        run.summary["final_train_loss"] = metrics.get("final_train_loss")
        run.summary["global_step"] = metrics.get("global_step")
        run.summary["epoch"] = metrics.get("epoch")

        artifact = wandb.Artifact(f"{run_name}-training-files", type="exp1-training")
        for filename in (
            "training_metrics.json",
            "trainer_state.json",
            "adapter_config.json",
            "README.md",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ):
            add_file_if_exists(artifact, adapter_dir / filename)
        if include_adapter_weights:
            add_file_if_exists(artifact, adapter_dir / "adapter_model.safetensors")
            add_file_if_exists(artifact, adapter_dir / "tokenizer.json")
        run.log_artifact(artifact)
    finally:
        run.finish()
    print(f"[wandb] uploaded training run {run_name}")


def log_summary_run(wandb: Any, *, project: str, entity: str | None, group: str, run_name: str) -> None:
    config = {"experiment": "exp1_triplet_concept_move"}
    run = wandb.init(project=project, entity=entity, group=group, name=run_name, config=config, reinit=True)
    try:
        floor_path = EXP_DIR / "floor_stats.json"
        if floor_path.exists():
            floor = load_json(floor_path)
            run.summary["floor_status"] = floor.get("status")
            run.summary["floor_gate_passed"] = floor.get("gate_passed")
            protocol_floor = floor.get("protocol_floor") or {}
            for key, value in protocol_floor.items():
                run.summary[f"floor/{key}"] = value

        raw_dir = EXP_DIR / "raw"
        for run_dir in sorted(raw_dir.glob("*")):
            count = count_csv_rows(run_dir / "triplet.csv")
            if count is not None:
                wandb.log({f"triplets/{run_dir.name}/rows": count})

        detection_dir = EXP_DIR / "detection"
        if detection_dir.exists():
            for path in sorted(detection_dir.rglob("*.json")):
                try:
                    payload = load_json(path)
                except json.JSONDecodeError:
                    continue
                prefix = f"detection/{path.stem}"
                detection = payload.get("detection") if isinstance(payload.get("detection"), dict) else payload
                for key in (
                    "target_rank",
                    "target_snr",
                    "target_snr_control_only",
                    "target_snr_floor_adjusted",
                    "target_top_ranked",
                    "boundary_crossed_control_only",
                    "boundary_crossed_floor_adjusted",
                    "edited_concept_top_ranked",
                    "snr",
                ):
                    if key in detection:
                        run.summary[f"{prefix}/{key}"] = detection[key]

        artifact = wandb.Artifact(f"{run_name}-exp1-files", type="exp1-summary")
        for path in (
            EXP_DIR / "floor_stats.json",
            EXP_DIR / "REPORT.md",
            EXP_DIR / "RESEARCH_LOG.md",
            EXP_DIR / "triplet_protocol.json",
            EXP_DIR / "items.json",
        ):
            add_file_if_exists(artifact, path)
        if detection_dir.exists():
            for path in sorted(detection_dir.rglob("*.json")):
                add_file_if_exists(artifact, path)
        run.log_artifact(artifact)
    finally:
        run.finish()
    print(f"[wandb] uploaded summary run {run_name}")


def adapter_dirs(base: Path, supervision: str, edit_id: str) -> dict[str, Path]:
    if supervision == "feature":
        return {
            "control": base / "lora_control" / edit_id,
            "edit": base / "lora_edit" / edit_id,
        }
    return {
        "control": base / f"lora_control_{supervision}" / edit_id,
        "edit": base / f"lora_edit_{supervision}" / edit_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="coherentLLM-exp1")
    parser.add_argument("--entity", default=None)
    parser.add_argument("--group", default="exp1-concentrated-drop-100-local")
    parser.add_argument("--edit-id", default="concentrated_drop_100")
    parser.add_argument("--supervision", default="similarity")
    parser.add_argument("--wandb-netrc", default="/mnt/home/ssuresh/.netrc")
    parser.add_argument("--include-adapter-weights", action="store_true")
    parser.add_argument("--skip-summary", action="store_true")
    args = parser.parse_args()

    configure_wandb_credentials(Path(args.wandb_netrc) if args.wandb_netrc else None)

    try:
        import wandb
    except ImportError as exc:
        raise SystemExit(
            "wandb is not installed in this environment. Install it or run this "
            "script from an env that has wandb, then retry."
        ) from exc

    base = EXP_DIR
    arms = adapter_dirs(base, args.supervision, args.edit_id)
    for arm, adapter_dir in arms.items():
        log_training_run(
            wandb,
            project=args.project,
            entity=args.entity,
            group=args.group,
            adapter_dir=adapter_dir,
            run_name=f"{args.group}-{arm}",
            include_adapter_weights=args.include_adapter_weights,
        )
    if not args.skip_summary:
        log_summary_run(
            wandb,
            project=args.project,
            entity=args.entity,
            group=args.group,
            run_name=f"{args.group}-summary",
        )


if __name__ == "__main__":
    main()
