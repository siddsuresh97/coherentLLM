"""Prepare rsLoRA adapters for vLLM versions that reject use_rslora=true.

PEFT rsLoRA uses scaling alpha / sqrt(r). Standard LoRA uses alpha / r.
For inference equivalence in vLLM, keep the learned weights unchanged, write a
shadow adapter_config.json with use_rslora=false, and set
alpha_standard = alpha_rslora * sqrt(r).

The original adapter directory is never modified.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def prepare(src: Path, dst: Path):
    cfg_path = src / "adapter_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    dst.mkdir(parents=True, exist_ok=True)
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("use_rslora"):
        r = int(cfg["r"])
        alpha = float(cfg["lora_alpha"])
        adjusted = alpha * math.sqrt(r)
        if abs(round(adjusted) - adjusted) < 1e-9:
            adjusted = int(round(adjusted))
        cfg["lora_alpha"] = adjusted
        cfg["use_rslora"] = False
    out_cfg = dst / "adapter_config.json"
    if out_cfg.exists() and json.loads(out_cfg.read_text()) != cfg:
        raise FileExistsError(f"{out_cfg} exists with different contents")
    out_cfg.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n")

    for item in src.iterdir():
        if item.name == "adapter_config.json":
            continue
        target = dst / item.name
        if target.exists():
            continue
        os.symlink(item.resolve(), target)
    print(f"[vllm-lora] {src} -> {dst}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", action="append", required=True)
    ap.add_argument("--dst", action="append", required=True)
    args = ap.parse_args()
    if len(args.src) != len(args.dst):
        raise SystemExit("--src and --dst counts must match")
    for src, dst in zip(args.src, args.dst):
        prepare(Path(src), Path(dst))


if __name__ == "__main__":
    main()
