"""Run a small retention failure-suite gate from the committed manifest.

This is intentionally narrow: it turns the manifest into concrete lm-eval
commands and summarizes TruthfulQA MC2 log-samples into false-lure diagnostics.
It is suitable for CHTC smoke jobs and local dry-run validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "results" / "sft_eval" / "wide_bench" / "failure_suite" / "suite_manifest.json"

DEFAULT_MODEL_PATH = "/staging/s/suresh27/models/llama31-8b-instruct"
ARM_SPECS = {
    "base": {"adapter_path": "NONE", "max_lora_rank": 64},
    "taskvec_a0p25": {"adapter_path": "/staging/s/suresh27/adapters/taskvec_a0p25", "max_lora_rank": 64},
    "lowLR": {"adapter_path": "/staging/s/suresh27/adapters/lowLR", "max_lora_rank": 64},
    "lowrank": {"adapter_path": "/staging/s/suresh27/adapters/lowrank", "max_lora_rank": 16},
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", default=["base", "taskvec_a0p25"])
    ap.add_argument("--tasks", nargs="+", default=["truthfulqa_mc2"])
    ap.add_argument("--limit", type=int, default=None, help="Override manifest slice limit.")
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--batch-size", default="auto")
    ap.add_argument("--gpu-mem-util", type=float, default=0.72)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-analysis", action="store_true")
    return ap.parse_args()


def load_manifest(path: Path) -> dict:
    with path.open() as f:
        return json.load(f)


def selected_slices(manifest: dict, tasks: list[str]) -> list[dict]:
    by_task = {row["task"]: row for row in manifest.get("slices", [])}
    out = []
    missing = []
    for task in tasks:
        if task in by_task:
            out.append(by_task[task])
        else:
            missing.append(task)
    if missing:
        raise SystemExit(f"Tasks absent from manifest: {', '.join(missing)}")
    return out


def normalize_arms(arms: list[str]) -> list[str]:
    unknown = [arm for arm in arms if arm not in ARM_SPECS]
    if unknown:
        raise SystemExit(f"Unknown arms: {', '.join(unknown)}")
    return arms


def model_args(model_path: str, arm: str, gpu_mem_util: float, max_model_len: int) -> str:
    spec = ARM_SPECS[arm]
    parts = [
        f"pretrained={model_path}",
        "dtype=bfloat16",
        "tensor_parallel_size=1",
        f"gpu_memory_utilization={gpu_mem_util}",
        f"max_model_len={max_model_len}",
        "trust_remote_code=True",
    ]
    adapter_path = spec["adapter_path"]
    if adapter_path != "NONE":
        parts.extend(
            [
                "enable_lora=True",
                f"lora_local_path={adapter_path}",
                f"max_lora_rank={spec['max_lora_rank']}",
            ]
        )
    return ",".join(parts)


def output_path(out_dir: Path, arm: str, task: str, limit: int) -> Path:
    if task == "truthfulqa_mc2":
        return out_dir / "truthfulqa_logsamples" / f"{arm}_limit{limit}"
    return out_dir / "lm_eval" / f"{arm}_{task}_limit{limit}"


def build_commands(args: argparse.Namespace, slices: list[dict]) -> list[dict]:
    commands = []
    for arm in normalize_arms(args.arms):
        for row in slices:
            task = row["task"]
            limit = args.limit if args.limit is not None else int(row["limit"])
            fewshot = int(row["fewshot"])
            out_path = output_path(args.out_dir, arm, task, limit)
            cmd = [
                args.python_bin,
                "-m",
                "lm_eval",
                "run",
                "--model",
                "vllm",
                "--model_args",
                model_args(args.model_path, arm, args.gpu_mem_util, args.max_model_len),
                "--tasks",
                task,
                "--limit",
                str(limit),
                "--batch_size",
                args.batch_size,
                "--apply_chat_template",
                "--num_fewshot",
                str(fewshot),
                "--output_path",
                str(out_path),
            ]
            if task == "truthfulqa_mc2":
                cmd.append("--log_samples")
            commands.append(
                {
                    "arm": arm,
                    "task": task,
                    "lane": row.get("lane", ""),
                    "fewshot": fewshot,
                    "limit": limit,
                    "output_path": str(out_path),
                    "command": cmd,
                }
            )
    return commands


def write_command_manifest(path: Path, commands: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["arm", "task", "lane", "fewshot", "limit", "output_path", "command"],
        )
        writer.writeheader()
        for row in commands:
            out = dict(row)
            out["command"] = " ".join(row["command"])
            writer.writerow(out)


def run_commands(commands: list[dict]) -> None:
    for row in commands:
        out_path = Path(row["output_path"])
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"[run] arm={row['arm']} task={row['task']} limit={row['limit']}", flush=True)
        subprocess.run(row["command"], check=True)


def run_truthfulqa_analysis(args: argparse.Namespace) -> Path | None:
    log_root = args.out_dir / "truthfulqa_logsamples"
    if not log_root.exists():
        return None
    script = Path(__file__).with_name("analyze_truthfulqa_logsamples.py")
    analysis_dir = args.out_dir / "truthfulqa_analysis"
    cmd = [
        args.python_bin,
        str(script),
        "--log_root",
        str(log_root),
        "--out_dir",
        str(analysis_dir),
        "--baseline",
        "base",
    ]
    subprocess.run(cmd, check=True)
    return analysis_dir


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_summary(args: argparse.Namespace, commands: list[dict], analysis_dir: Path | None) -> None:
    lines = [
        "# Retention Failure-Suite Gate Run",
        "",
        "This run is a bounded GPU-backed gate from the committed failure-suite",
        "manifest. It is not a replacement for the full benchmark sweep.",
        "",
        "## Plan",
        "",
        f"- Arms: `{', '.join(args.arms)}`",
        f"- Tasks: `{', '.join(args.tasks)}`",
        f"- Limit override: `{args.limit if args.limit is not None else 'manifest limits'}`",
        f"- Command count: `{len(commands)}`",
        "",
    ]
    if args.dry_run:
        lines.extend(
            [
                "## Status",
                "",
                "Dry run only. No model inference was launched.",
                "",
            ]
        )
    if analysis_dir is not None:
        delta_rows = read_csv(analysis_dir / "paired_delta_summary.csv")
        lines.extend(
            [
                "## TruthfulQA Paired Delta vs Base",
                "",
                "| Arm | n | Delta MC2 | Delta truth log-odds | Delta true mass | Delta false pressure | False pressure up |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in delta_rows:
            lines.append(
                "| {arm} | {n} | {dacc} | {dodds} | {dtrue} | {dfalse} | {up} |".format(
                    arm=row.get("arm", ""),
                    n=row.get("n_paired", ""),
                    dacc=row.get("mean_delta_mc2_acc", ""),
                    dodds=row.get("mean_delta_truth_logodds", ""),
                    dtrue=row.get("mean_delta_true_logsumexp", ""),
                    dfalse=row.get("mean_delta_false_logsumexp", ""),
                    up=row.get("frac_false_pressure_up", ""),
                )
            )
        lines.extend(
            [
                "",
                "Gate note: the current threshold is `frac_false_pressure_up <= 0.60`",
                "with non-positive false-pressure delta preferred unless truth",
                "log-odds improves.",
                "",
            ]
        )
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `command_manifest.csv`: exact lm-eval commands.",
            "- `truthfulqa_logsamples/`: raw lm-eval results and samples when TruthfulQA is run.",
            "- `truthfulqa_analysis/`: false-lure pressure diagnostics when analysis is enabled.",
            "",
        ]
    )
    (args.out_dir / "RUN_SUMMARY.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    slices = selected_slices(manifest, args.tasks)
    commands = build_commands(args, slices)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "run_plan.json").write_text(json.dumps(commands, indent=2) + "\n")
    write_command_manifest(args.out_dir / "command_manifest.csv", commands)

    if args.dry_run:
        for row in commands:
            print("[plan] " + " ".join(row["command"]))
        write_summary(args, commands, None)
        return

    run_commands(commands)
    analysis_dir = None
    if not args.skip_analysis and any(row["task"] == "truthfulqa_mc2" for row in commands):
        analysis_dir = run_truthfulqa_analysis(args)
    write_summary(args, commands, analysis_dir)
    print(f"[ok] wrote {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()
