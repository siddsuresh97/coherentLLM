"""Run Task 6 wide lm-eval battery for base vs lowLR and summarize results."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TASKS = [
    "piqa",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "openbookqa",
    "commonsense_qa",
    "hellaswag",
    "wic",
    "mmlu",
    "truthfulqa_mc2",
]
STANDARD_GROUPS = [
    ("zero_shot", ["piqa", "openbookqa", "commonsense_qa", "wic", "truthfulqa_mc2"], 0),
    ("winogrande_5shot", ["winogrande"], 5),
    ("arc_25shot", ["arc_easy", "arc_challenge"], 25),
    ("hellaswag_10shot", ["hellaswag"], 10),
    ("mmlu_5shot", ["mmlu"], 5),
]
STANDARD_GROUP_NAMES = [name for name, _, _ in STANDARD_GROUPS]
STATES = {
    "base": ("base", None, 64),
    "lowLR": ("lowLR", ROOT / "out" / "adapters_mitigation_vllm" / "lowLR", 64),
}
PREFERRED_METRICS = {
    "arc_easy": ["acc_norm,none", "acc,none"],
    "arc_challenge": ["acc_norm,none", "acc,none"],
    "hellaswag": ["acc_norm,none", "acc,none"],
    "openbookqa": ["acc_norm,none", "acc,none"],
    "piqa": ["acc_norm,none", "acc,none"],
    "winogrande": ["acc,none"],
    "commonsense_qa": ["acc,none"],
    "wic": ["acc,none"],
    "truthfulqa_mc2": ["acc,none"],
}


def load_registry() -> dict:
    with (ROOT / "configs" / "models.yaml").open() as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id: str, hf_cache: str, allow_download: bool) -> str:
    cache_name = "models--" + repo_id.replace("/", "--")
    caches = [hf_cache] + [c for c in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":") if c]
    for cache in caches:
        for snap in sorted(glob.glob(os.path.join(cache, cache_name, "snapshots", "*"))):
            if glob.glob(os.path.join(snap, "config.json")):
                return snap
    if not allow_download:
        print(f"[warn] {repo_id} not found in cache; will let HF resolve/download", flush=True)
    return repo_id


def has_result(out_dir: Path) -> bool:
    return bool(list(out_dir.glob("**/results_*.json")))


def latest_result(out_dir: Path) -> Path:
    matches = list(out_dir.glob("**/results_*.json"))
    if not matches:
        raise FileNotFoundError(out_dir)
    return max(matches, key=lambda p: p.stat().st_mtime)


def run_eval(
    state: str,
    args: argparse.Namespace,
    base_path: str,
    hf_cache: str,
    out_dir: Path,
    tasks: list[str],
    num_fewshot: int | None,
) -> None:
    model_name, adapter, max_lora_rank = STATES[state]
    if has_result(out_dir) and not args.overwrite:
        print(f"[skip] {state}: {out_dir} already has JSON results", flush=True)
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HOME", hf_cache)
    env.setdefault("HF_HUB_CACHE", hf_cache)
    env.setdefault("HF_DATASETS_CACHE", str(ROOT / "out" / "hf_datasets_cache"))
    env.setdefault("TRITON_CACHE_DIR", str(ROOT / "out" / "triton_cache"))
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if Path(base_path).is_dir():
        env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("HF_HUB_OFFLINE", "0")
    env.setdefault("HF_DATASETS_OFFLINE", "0")

    if args.backend == "vllm":
        model_args = [
            f"pretrained={base_path}",
            "dtype=bfloat16",
            "tensor_parallel_size=1",
            f"gpu_memory_utilization={args.gpu_mem_util}",
            f"max_model_len={args.max_model_len}",
            "trust_remote_code=True",
        ]
        if adapter is not None:
            model_args.extend([
                "enable_lora=True",
                f"lora_local_path={adapter}",
                f"max_lora_rank={max_lora_rank}",
            ])
        model_name = "vllm"
        extra = []
    else:
        model_args = [
            f"pretrained={base_path}",
            "dtype=bfloat16",
            "trust_remote_code=True",
        ]
        if adapter is not None:
            model_args.append(f"peft={adapter}")
        model_name = "hf"
        extra = ["--device", "cuda"]

    cmd = [
        sys.executable,
        "-m",
        "lm_eval",
        "run",
        "--model",
        model_name,
        "--model_args",
        ",".join(model_args),
        "--tasks",
        ",".join(tasks),
        "--limit",
        str(args.limit),
        "--batch_size",
        args.batch_size,
        "--apply_chat_template",
        "--output_path",
        str(out_dir),
        *extra,
    ]
    if num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(num_fewshot)])
    print("[cmd] " + " ".join(cmd), flush=True)
    rc = subprocess.run(cmd, cwd=str(ROOT), env=env).returncode
    if rc != 0:
        raise SystemExit(rc)


def run_state(state: str, args: argparse.Namespace, base_path: str, hf_cache: str) -> None:
    if args.fewshot_mode == "standard":
        for group_name, tasks, num_fewshot in selected_standard_groups(args):
            out_dir = args.out_root / "runs" / f"{state}_{group_name}"
            run_eval(state, args, base_path, hf_cache, out_dir, tasks, num_fewshot)
    else:
        model_name, _, _ = STATES[state]
        out_dir = args.out_root / model_name
        run_eval(state, args, base_path, hf_cache, out_dir, args.tasks, None)


def metric_for_task(task: str, metrics: dict) -> tuple[str, float] | None:
    preferred = PREFERRED_METRICS.get(task, ["acc_norm,none", "acc,none"])
    for metric in preferred:
        if metric in metrics:
            return metric.split(",", 1)[0], float(metrics[metric])
    for key, value in metrics.items():
        if key.endswith("_stderr,none") or key in {"alias", "name", "sample_len"}:
            continue
        if isinstance(value, (int, float)):
            return key.split(",", 1)[0], float(value)
    return None


def read_state_results(out_dir: Path) -> dict[str, dict]:
    with latest_result(out_dir).open() as f:
        data = json.load(f)
    return data["results"]


def result_dirs_for_state(args: argparse.Namespace, state: str) -> list[Path]:
    if args.fewshot_mode == "standard":
        return [
            args.out_root / "runs" / f"{state}_{group_name}"
            for group_name, _, _ in selected_standard_groups(args)
        ]
    return [args.out_root / STATES[state][0]]


def selected_standard_groups(args: argparse.Namespace) -> list[tuple[str, list[str], int]]:
    if not args.groups:
        return STANDARD_GROUPS
    wanted = set(args.groups)
    return [group for group in STANDARD_GROUPS if group[0] in wanted]


def summarize(args: argparse.Namespace) -> None:
    state_results = {}
    for state in args.states:
        merged = {}
        for out_dir in result_dirs_for_state(args, state):
            merged.update(read_state_results(out_dir))
        state_results[state] = merged
    rows = []
    raw_rows = []
    for state, results in state_results.items():
        for task, metrics in results.items():
            selected = metric_for_task(task, metrics)
            if selected is None:
                continue
            metric, value = selected
            raw_rows.append(
                {
                    "task": task,
                    "model": state,
                    "metric": metric,
                    "value": value,
                    "n": metrics.get("sample_len", ""),
                }
            )

    by_task = {}
    for row in raw_rows:
        by_task.setdefault(row["task"], {})[row["model"]] = row

    for task in sorted(by_task):
        values = by_task[task]
        if "base" not in values:
            continue
        for state in args.states:
            if state not in values:
                continue
            value = values[state]["value"]
            base_value = values["base"]["value"]
            delta = value - base_value
            rows.append(
                {
                    "task": task,
                    "model": state,
                    "metric": values[state]["metric"],
                    "value": value,
                    "n": values[state]["n"],
                    "base_value": base_value,
                    "delta_vs_base": delta,
                    "verdict": verdict(delta),
                }
            )

    mmlu_by_state = {}
    for state in args.states:
        vals = [
            row["value"]
            for row in raw_rows
            if row["model"] == state and row["task"].startswith("mmlu_") and row["metric"] == "acc"
        ]
        if vals:
            mmlu_by_state[state] = float(sum(vals) / len(vals))
    if "base" in mmlu_by_state:
        base_value = mmlu_by_state["base"]
        for state, value in mmlu_by_state.items():
            delta = value - base_value
            rows.append(
                {
                    "task": "mmlu",
                    "model": state,
                    "metric": "macro_acc",
                    "value": value,
                    "n": sum(1 for row in raw_rows if row["model"] == state and row["task"].startswith("mmlu_")),
                    "base_value": base_value,
                    "delta_vs_base": delta,
                    "verdict": verdict(delta),
                }
            )

    args.out_root.mkdir(parents=True, exist_ok=True)
    raw_path = args.out_root / "raw_task_summary.csv"
    with raw_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "model", "metric", "value", "n"])
        writer.writeheader()
        writer.writerows(raw_rows)
    summary_path = args.out_root / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["task", "model", "metric", "value", "n", "base_value", "delta_vs_base", "verdict"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[done] wrote {summary_path} and {raw_path}", flush=True)


def verdict(delta: float) -> str:
    if math.isnan(delta):
        return "NA"
    if delta > 0.02:
        return "GAIN"
    if delta < -0.02:
        return "DROP"
    return "FLAT"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--states", nargs="+", default=["base", "lowLR"], choices=sorted(STATES))
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--out_root", type=Path, default=ROOT / "results" / "sft_eval" / "wide_bench")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--backend", choices=["vllm", "hf"], default="vllm")
    ap.add_argument("--fewshot_mode", choices=["standard", "harness_default"], default="standard")
    ap.add_argument("--groups", nargs="+", choices=STANDARD_GROUP_NAMES, default=None)
    ap.add_argument("--gpu_mem_util", type=float, default=0.88)
    ap.add_argument("--max_model_len", type=int, default=2048)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--summarize_only", action="store_true")
    ap.add_argument("--no_summarize", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    reg = load_registry()
    spec = reg["local"]["llama-3.1-8b-instruct"]
    hf_cache = reg["hf_cache"]
    base_path = resolve_model_path(spec["path"], hf_cache, spec.get("download", False))
    if not args.summarize_only:
        for state in args.states:
            run_state(state, args, base_path, hf_cache)
    if not args.no_summarize:
        summarize(args)


if __name__ == "__main__":
    main()
