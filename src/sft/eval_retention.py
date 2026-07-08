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


def run_state(state, args, base_path, hf_cache):
    model_name, adapter = STATES[state]
    out_dir = ROOT / "results" / "sft_eval" / "retention" / model_name
    if has_result(out_dir) and not args.overwrite:
        print(f"[skip] {out_dir} already has JSON results")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env.setdefault("HF_HOME", hf_cache)
    env.setdefault("HF_HUB_CACHE", hf_cache)
    env.setdefault("HF_DATASETS_CACHE", str(ROOT / "out" / "hf_datasets_cache"))
    # `base_path` is a local snapshot, so vLLM/HF do not need to resolve model
    # files from the hub. Keep HF hub online for lm-eval datasets.
    if Path(base_path).is_dir():
        env["TRANSFORMERS_OFFLINE"] = "1"

    common = [
        sys.executable, "-m", "lm_eval",
        "--tasks", TASKS,
        "--limit", str(args.limit),
        "--num_fewshot", "0",
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
                "max_lora_rank=64",
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
                    choices=list(STATES))
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--backend", choices=["auto", "vllm", "hf"], default="auto")
    ap.add_argument("--batch_size", default="auto")
    ap.add_argument("--gpu_mem_util", type=float, default=0.90)
    ap.add_argument("--max_model_len", type=int, default=4096)
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
