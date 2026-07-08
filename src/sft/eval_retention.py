"""Run lm-eval retention checks for SFT adapters."""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
TASKS = "mmlu_abstract_algebra,mmlu_anatomy,arc_challenge,hellaswag,truthfulqa_mc2"
STATES = {
    "base": ("llama-3.1-8b-instruct", None),
    "real": ("llama31-sft-real", ROOT / "out" / "adapters_vllm_fixed" / "real"),
    "scrambled": (
        "llama31-sft-scrambled",
        ROOT / "out" / "adapters_vllm_fixed" / "scrambled",
    ),
}


def load_registry():
    with (ROOT / "configs" / "models.yaml").open() as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id: str, hf_cache: str, allow_download: bool):
    cache_name = "models--" + repo_id.replace("/", "--")
    caches = [hf_cache] + [c for c in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":") if c]
    for cache in caches:
        for snap in sorted(glob.glob(os.path.join(cache, cache_name, "snapshots", "*"))):
            if glob.glob(os.path.join(snap, "config.json")):
                return snap
    if not allow_download:
        print(f"[warn] {repo_id} not found in cache; will let HF resolve/download")
    return repo_id


def has_result(out_dir: Path):
    return bool(list(out_dir.glob("**/*.json")))


def run_cmd(cmd, env):
    print("[cmd] " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, cwd=str(ROOT), env=env)


def state_spec(state, args):
    if state in STATES:
        return STATES[state]
    if args.custom_state and state == args.custom_state and args.custom_adapter:
        return args.custom_model_name or state, Path(args.custom_adapter)
    raise SystemExit(
        f"unknown state {state!r}; use one of {list(STATES)} or pass "
        "--custom_state with --custom_adapter"
    )


def run_state(state, args, base_path, hf_cache):
    model_name, adapter = state_spec(state, args)
    out_dir = args.out_root / model_name
    if has_result(out_dir) and not args.overwrite:
        print(f"[skip] {out_dir} already has JSON results")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HOME", hf_cache)
    env.setdefault("HF_HUB_CACHE", hf_cache)
    env.setdefault("HF_DATASETS_CACHE", str(ROOT / "out" / "hf_datasets_cache"))
    # `base_path` is a local snapshot, so vLLM/HF do not need to resolve model
    # files from the hub. Keep HF hub online for lm-eval datasets: TRANSFORMERS_OFFLINE
    # only affects transformers model resolution, but we must also make sure the
    # `datasets` library is NOT in offline mode, or it can't fetch mmlu/arc/etc.
    if Path(base_path).is_dir():
        env["TRANSFORMERS_OFFLINE"] = "1"
    env.setdefault("HF_HUB_OFFLINE", "0")
    env.setdefault("HF_DATASETS_OFFLINE", "0")

    common = [
        sys.executable, "-m", "lm_eval",
        "--tasks", args.tasks,
        "--limit", str(args.limit),
        "--num_fewshot", str(args.num_fewshot),
        "--batch_size", args.batch_size,
        "--output_path", str(out_dir),
    ]

    def vllm_cmd():
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
                f"max_lora_rank={args.max_lora_rank}",
            ])
        return common + ["--model", "vllm", "--model_args", ",".join(model_args)]

    def hf_cmd():
        model_args = [
            f"pretrained={base_path}",
            "dtype=bfloat16",
            "trust_remote_code=True",
        ]
        if adapter is not None:
            model_args.append(f"peft={adapter}")
        return common + ["--model", "hf", "--model_args", ",".join(model_args)]

    if args.backend in ("auto", "vllm"):
        rc = run_cmd(vllm_cmd(), env).returncode
        if rc == 0 or args.backend == "vllm":
            if rc != 0:
                raise SystemExit(rc)
            return
        print(f"[warn] vLLM lm-eval failed for {state}; falling back to hf", flush=True)

    rc = run_cmd(hf_cmd(), env).returncode
    if rc != 0:
        raise SystemExit(rc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", nargs="+", default=["base", "real", "scrambled"],
                    help="Built-in states, or --custom_state when evaluating one adapter.")
    ap.add_argument("--custom_state", default=None)
    ap.add_argument("--custom_adapter", default=None)
    ap.add_argument("--custom_model_name", default=None)
    ap.add_argument(
        "--out_root",
        type=Path,
        default=ROOT / "results" / "sft_eval" / "retention",
    )
    ap.add_argument("--tasks", default=TASKS)
    ap.add_argument("--num_fewshot", type=int, default=0)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
    ap.add_argument("--max_lora_rank", type=int, default=64)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    reg = load_registry()
    spec = reg["local"]["llama-3.1-8b-instruct"]
    hf_cache = reg["hf_cache"]
    base_path = resolve_model_path(spec["path"], hf_cache, spec.get("download", False))
    for state in args.states:
        run_state(state, args, base_path, hf_cache)


if __name__ == "__main__":
    main()
