"""Apply coherence steering vectors to the base model and evaluate the sweep.

The activation route can add the actdiff direction to selected decoder layer
outputs via forward hooks. By default it preserves the original all-layer raw
addition behavior; pass --layers and --norm-match for mid-layer steering. The
task-vector route applies the real LoRA adapter with its LoRA scaling multiplied
by alpha, which is equivalent to base + alpha * taskvec.
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from prompts import SYSTEM_PROMPT, feature_prompt, pairwise_prompt, triplet_prompt  # noqa: E402
from sft.lora_apply import install_lora_adapter, installed_lora_modules  # noqa: E402


RAW = ROOT / "results" / "sft_eval" / "raw"
OUT_DIR = ROOT / "results" / "sft_eval"
STEER_DIR = OUT_DIR / "steer"
STIM = ROOT / "data" / "scale128" / "stimuli"
ACTDIFF = ROOT / "data" / "sft" / "steer" / "coh_vector_actdiff.npz"
TASKVEC = ROOT / "data" / "sft" / "steer" / "coh_vector_taskvec"
BASE_MODEL = "llama-3.1-8b-instruct"
REAL_MODEL = "llama31-sft-real"
ACT_ALPHAS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
TASK_ALPHAS = [0.25, 0.5, 1.0]
SALMON_CONDA = (
    "source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && "
    "conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/salmon"
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["run-one", "fit-salmon", "summarize", "run-sweep"])
    ap.add_argument("--route", choices=["actdiff", "taskvec"], default="actdiff")
    ap.add_argument("--alpha", type=float, default=0.0)
    ap.add_argument("--model", default=BASE_MODEL)
    ap.add_argument("--actdiff", type=Path, default=ACTDIFF)
    ap.add_argument("--taskvec", type=Path, default=TASKVEC)
    ap.add_argument("--out-model", default=None, help="Override output model/result name.")
    ap.add_argument("--layers", default=None,
                    help="Actdiff decoder layers to steer, e.g. '8' or '10-14'. Default: all layers.")
    ap.add_argument("--norm-match", action="store_true",
                    help="Use h <- h + alpha * unit(diff_L) * ||h|| per token for actdiff.")
    ap.add_argument("--norm-eps", type=float, default=1e-6)
    ap.add_argument("--raw-dir", type=Path, default=RAW)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--steer-dir", type=Path, default=STEER_DIR)
    ap.add_argument("--stim-dir", type=Path, default=STIM)
    ap.add_argument("--methods", nargs="+", default=["triplet", "pairwise", "feature"],
                    choices=["triplet", "pairwise", "feature"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=8)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--allow-download", action="store_true")
    ap.add_argument("--salmon-conda", default=SALMON_CONDA)
    ap.add_argument("--salmon-epochs", type=int, default=8000)
    ap.add_argument("--alphas", nargs="*", type=float, default=None)
    ap.add_argument("--gpu", default=None, help="CUDA_VISIBLE_DEVICES value for run-sweep children.")
    return ap.parse_args()


def alpha_tag(alpha: float) -> str:
    s = f"{alpha:g}"
    return s.replace("-", "m").replace(".", "p")


def out_model_for(route: str, alpha: float) -> str:
    if route == "actdiff":
        return f"llama31-steer-a{alpha_tag(alpha)}"
    return f"llama31-steer-task-a{alpha_tag(alpha)}"


def output_model_name(args: argparse.Namespace) -> str:
    return args.out_model or out_model_for(args.route, args.alpha)


def parse_layer_spec(spec: str | None, n_layers: int) -> list[int]:
    if spec is None:
        return list(range(n_layers))
    layers: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            if end < start:
                raise ValueError(f"layer range {part!r} is descending")
            layers.extend(range(start, end + 1))
        else:
            layers.append(int(part))
    deduped = []
    for layer in layers:
        if layer < 0 or layer >= n_layers:
            raise ValueError(f"decoder layer {layer} out of range 0..{n_layers - 1}")
        if layer not in deduped:
            deduped.append(layer)
    if not deduped:
        raise ValueError(f"no decoder layers parsed from {spec!r}")
    return deduped


def torch_dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def load_registry() -> dict:
    with (ROOT / "configs" / "models.yaml").open() as f:
        return yaml.safe_load(f)


def resolve_model_path(repo_id: str, hf_cache: str, allow_download: bool) -> str:
    if Path(repo_id).is_dir():
        return repo_id
    cache_name = "models--" + repo_id.replace("/", "--")
    caches = [hf_cache] + [c for c in os.environ.get("COHERENCE_EXTRA_CACHE", "").split(":") if c]
    for cache in caches:
        for snap in sorted(Path(cache).glob(f"{cache_name}/snapshots/*"), reverse=True):
            if (snap / "config.json").exists():
                return str(snap)
    if not allow_download:
        raise FileNotFoundError(
            f"{repo_id} not found in cache {hf_cache}; pass --allow-download to let HF resolve it"
        )
    return repo_id


def configure_hf_cache(hf_cache: str, model_path: str) -> None:
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    os.environ.setdefault("HF_DATASETS_CACHE", str(ROOT / "out" / "hf_datasets_cache"))
    os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / "out" / "triton_cache"))
    if Path(model_path).is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"


def read_rows(path: Path) -> list[list[str]]:
    with path.open(newline="") as f:
        return [[c.strip() for c in row] for row in csv.reader(f) if any(row)]


def copy_if_missing(src: Path, dst: Path, overwrite: bool) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)
    print(f"[copy] {src} -> {dst}", flush=True)


def csv_data_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    with path.open(newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def csv_is_complete(path: Path, expected_rows: int) -> bool:
    rows = csv_data_rows(path)
    return rows == expected_rows


def copy_reference_model(reference_model: str, out_model: str, args: argparse.Namespace) -> None:
    src_dir = args.raw_dir / reference_model
    dst_dir = args.raw_dir / out_model
    for name in [
        "triplet.csv",
        "pairwise.csv",
        "feature.csv",
        "verify_pairs.csv",
        "listed_features.csv",
    ]:
        copy_if_missing(src_dir / name, dst_dir / name, args.overwrite)
    copy_if_missing(
        args.out_dir / f"{reference_model}_triplet_d5.npy",
        args.out_dir / f"{out_model}_triplet_d5.npy",
        args.overwrite,
    )


def method_jobs(method: str, outdir: Path, stim_dir: Path) -> tuple[list[tuple[str, ...]], list[str]]:
    if method == "triplet":
        rows = [tuple(row[:3]) for row in read_rows(stim_dir / "triplets.csv")]
        prompts = [triplet_prompt(a, b, c) for a, b, c in rows]
        return rows, prompts
    if method == "pairwise":
        rows = [tuple(row[:2]) for row in read_rows(stim_dir / "pairs.csv")]
        prompts = [pairwise_prompt(a, b) for a, b in rows]
        return rows, prompts
    if method == "feature":
        copy_if_missing(
            RAW / BASE_MODEL / "verify_pairs.csv",
            outdir / "verify_pairs.csv",
            overwrite=False,
        )
        copy_if_missing(
            RAW / BASE_MODEL / "listed_features.csv",
            outdir / "listed_features.csv",
            overwrite=False,
        )
        rows = [tuple(row[:2]) for row in read_rows(outdir / "verify_pairs.csv")]
        prompts = [feature_prompt(feat, concept) for feat, concept in rows]
        return rows, prompts
    raise ValueError(f"unknown method {method}")


def chat_prompts(tokenizer, prompts: list[str]) -> list[str]:
    return [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]


def generate_responses(
    model,
    tokenizer,
    prompts: list[str],
    *,
    batch_size: int,
    max_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> Iterable[str]:
    model.eval()
    formatted = chat_prompts(tokenizer, prompts)
    for start in range(0, len(formatted), batch_size):
        end = min(start + batch_size, len(formatted))
        enc = tokenizer(
            formatted[start:end],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        input_width = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )
        texts = tokenizer.batch_decode(out[:, input_width:], skip_special_tokens=True)
        for text in texts:
            yield text.strip()
        print(f"[generate] {end}/{len(formatted)}", flush=True)


def write_method_csv(
    model,
    tokenizer,
    method: str,
    out_path: Path,
    rows: list[tuple[str, ...]],
    prompts: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    with tmp_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["input", "prompt", "response"])
        responses = generate_responses(
            model,
            tokenizer,
            prompts,
            batch_size=args.batch_size,
            max_length=args.max_length,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        n = 0
        for row, prompt, response in zip(rows, prompts, responses):
            writer.writerow(["|".join(row), prompt, response])
            n += 1
            if n % 100 == 0:
                f.flush()
        f.flush()
    if n != len(rows):
        raise RuntimeError(f"{out_path} got {n} responses for {len(rows)} rows")
    os.replace(tmp_path, out_path)
    print(f"[done] {method}: {len(rows)} rows -> {out_path}", flush=True)


def add_actdiff_hooks(
    model,
    actdiff_path: Path,
    alpha: float,
    layer_spec: str | None,
    *,
    norm_match: bool,
    norm_eps: float,
) -> list[torch.utils.hooks.RemovableHandle]:
    data = np.load(actdiff_path)
    diff = data["diff"]
    layers = model.model.layers
    if diff.shape[0] != len(layers) + 1:
        raise ValueError(f"actdiff has shape {diff.shape}, expected {len(layers) + 1} layer rows")
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    handles = []
    selected = parse_layer_spec(layer_spec, len(layers))
    for layer_idx in selected:
        layer = layers[layer_idx]
        vec = torch.as_tensor(diff[layer_idx + 1], device=device, dtype=dtype)
        if norm_match:
            vec = vec / torch.clamp(torch.linalg.vector_norm(vec), min=norm_eps)
        vec = vec.view(1, 1, -1)

        def hook(_module, _inputs, output, *, steer_vec=vec):
            def steer(hidden):
                direction = steer_vec.to(hidden.dtype)
                if norm_match:
                    hidden_norm = torch.linalg.vector_norm(hidden, dim=-1, keepdim=True).clamp_min(norm_eps)
                    return hidden + alpha * direction * hidden_norm
                return hidden + alpha * direction

            if isinstance(output, tuple):
                hidden = steer(output[0])
                return (hidden,) + output[1:]
            return steer(output)

        handles.append(layer.register_forward_hook(hook))
    mode = "norm-matched" if norm_match else "raw"
    print(
        f"[hooks] installed {mode} actdiff hooks for decoder layers {selected} at alpha={alpha:g}",
        flush=True,
    )
    return handles


def apply_taskvec(model, adapter_dir: Path, alpha: float, dtype: torch.dtype, device: torch.device) -> None:
    n = install_lora_adapter(
        model,
        adapter_dir,
        "taskvec",
        dtype=dtype,
        device=device,
        activate=True,
    )
    scaled = 0
    for module in installed_lora_modules(model):
        if "taskvec" in module.scalings:
            module.scalings["taskvec"] *= alpha
            scaled += 1
    print(f"[taskvec] installed {n} LoRA modules; scaled {scaled} modules by alpha={alpha:g}", flush=True)


def load_model_and_tokenizer(args: argparse.Namespace):
    reg = load_registry()
    spec = reg["local"][args.model]
    hf_cache = reg["hf_cache"]
    model_path = resolve_model_path(
        spec["path"],
        hf_cache,
        args.allow_download or bool(spec.get("download", False)),
    )
    configure_hf_cache(hf_cache, model_path)
    dtype = torch_dtype(args.dtype)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[model] loading {model_path} on {device} ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model, tokenizer, dtype, device


def run_one(args: argparse.Namespace) -> None:
    out_model = output_model_name(args)
    outdir = args.raw_dir / out_model

    if args.route == "actdiff" and args.alpha == 0.0 and args.out_model is None:
        copy_reference_model(BASE_MODEL, out_model, args)
        return
    if args.route == "taskvec" and args.alpha == 1.0 and args.out_model is None:
        copy_reference_model(REAL_MODEL, out_model, args)
        return

    planned = []
    for method in args.methods:
        rows, prompts = method_jobs(method, outdir, args.stim_dir)
        out_path = outdir / f"{method}.csv"
        if not args.overwrite and csv_is_complete(out_path, len(rows)):
            print(f"[skip] {out_path} has {len(rows)} rows", flush=True)
            continue
        existing = csv_data_rows(out_path)
        if existing is not None:
            print(f"[rerun] {out_path} has {existing}/{len(rows)} rows", flush=True)
        planned.append((method, rows, prompts, out_path))

    if not planned:
        print(f"[skip] all requested raw files exist for {out_model}", flush=True)
        return

    model, tokenizer, dtype, device = load_model_and_tokenizer(args)
    handles = []
    if args.route == "actdiff":
        handles = add_actdiff_hooks(
            model,
            args.actdiff,
            args.alpha,
            args.layers,
            norm_match=args.norm_match,
            norm_eps=args.norm_eps,
        )
    else:
        apply_taskvec(model, args.taskvec, args.alpha, dtype, device)

    try:
        for method, rows, prompts, out_path in planned:
            write_method_csv(
                model,
                tokenizer,
                method,
                out_path,
                rows,
                prompts,
                args,
                device,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    finally:
        for handle in handles:
            handle.remove()


def salmon_output_path(out_model: str, args: argparse.Namespace) -> Path:
    return args.out_dir / f"{out_model}_triplet_d5.npy"


def fit_salmon_for(out_model: str, args: argparse.Namespace) -> None:
    out_path = salmon_output_path(out_model, args)
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {out_path} exists", flush=True)
        return
    raw_triplet = args.raw_dir / out_model / "triplet.csv"
    if not raw_triplet.exists():
        raise FileNotFoundError(raw_triplet)
    cmd = (
        f"{args.salmon_conda} && "
        f"COHERENCE_RAW_DIR={args.raw_dir} COHERENCE_STIM_DIR={args.stim_dir} "
        f"python src/fit_triplet_salmon.py --raw_dir {args.raw_dir} --stim_dir {args.stim_dir} "
        f"--out_dir {args.out_dir} --d 5 --max_epochs {args.salmon_epochs} {out_model}"
    )
    print(f"[salmon] {out_model}", flush=True)
    subprocess.run(["bash", "-lc", cmd], cwd=ROOT, check=True)


def fit_salmon(args: argparse.Namespace) -> None:
    if args.alphas:
        for alpha in args.alphas:
            fit_salmon_for(out_model_for(args.route, alpha), args)
        return
    fit_salmon_for(output_model_name(args), args)


def expected_models() -> list[dict]:
    rows = []
    for alpha in ACT_ALPHAS:
        rows.append({"route": "actdiff", "alpha": alpha, "out_model": out_model_for("actdiff", alpha)})
    for alpha in TASK_ALPHAS:
        rows.append({"route": "taskvec", "alpha": alpha, "out_model": out_model_for("taskvec", alpha)})
    return rows


def metric_row(route: str, alpha: float, out_model: str) -> dict:
    from sft.analyze_eval import _concepts, coherence_metrics, human_triplet_metric

    concepts = _concepts()
    row = {"route": route, "alpha": alpha, "out_model": out_model}
    required = [
        RAW / out_model / "triplet.csv",
        RAW / out_model / "pairwise.csv",
        RAW / out_model / "feature.csv",
        OUT_DIR / f"{out_model}_triplet_d5.npy",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        row["complete"] = False
        row["error"] = "missing " + "; ".join(missing)
        return row
    try:
        row.update(coherence_metrics(out_model, "gen", concepts))
        row.update(human_triplet_metric(out_model, "gen"))
        row["human_things_triplet_r2"] = row["gen_human_things_triplet_r2"]
        row["complete"] = bool(
            np.isfinite(row.get("gen_proc_mean", np.nan))
            and np.isfinite(row.get("human_things_triplet_r2", np.nan))
        )
        if not row["complete"]:
            row["error"] = "non-finite metric"
    except Exception as exc:
        row["complete"] = False
        row["error"] = str(exc)
    return row


def reference_rows() -> pd.DataFrame:
    eval_csv = OUT_DIR / "eval_results.csv"
    if not eval_csv.exists():
        return pd.DataFrame()
    df = pd.read_csv(eval_csv)
    keep = df[df["state"].isin(["base", "real"])].copy()
    keep["route"] = "reference"
    keep["alpha"] = np.nan
    keep["out_model"] = keep["model"]
    keep["complete"] = True
    cols = [
        "route",
        "state",
        "alpha",
        "out_model",
        "gen_proc_mean",
        "gen_human_things_triplet_r2",
        "human_things_triplet_r2",
        "complete",
    ]
    return keep[[c for c in cols if c in keep.columns]]


def write_plot(summary: pd.DataFrame, refs: pd.DataFrame, out_path: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric_cols = [
        ("gen_proc_mean", "GEN coherence"),
        ("human_things_triplet_r2", "THINGS-human r2"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=False)
    for ax, (col, label) in zip(axes, metric_cols):
        for route, grp in summary[summary["complete"]].groupby("route"):
            grp = grp.sort_values("alpha")
            ax.plot(grp["alpha"], grp[col], marker="o", label=route)
        if not refs.empty and col in refs.columns:
            for _, ref in refs.iterrows():
                state = ref.get("state", ref.get("out_model", "reference"))
                ax.axhline(ref[col], linestyle="--", linewidth=1, alpha=0.6, label=f"{state} ref")
        ax.set_xlabel("alpha")
        ax.set_ylabel(label)
        ax.grid(True, linewidth=0.4, alpha=0.4)
    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def summarize(args: argparse.Namespace) -> None:
    args.steer_dir.mkdir(parents=True, exist_ok=True)
    rows = [metric_row(**item) for item in expected_models()]
    summary = pd.DataFrame(rows)
    refs = reference_rows()
    summary.to_csv(args.steer_dir / "sweep_summary.csv", index=False, float_format="%.6f")
    refs.to_csv(args.steer_dir / "reference_summary.csv", index=False, float_format="%.6f")
    write_plot(summary, refs, args.steer_dir / "coherence_human_vs_alpha.png")

    complete = summary[summary["complete"]].copy()
    display_cols = ["route", "alpha", "out_model", "gen_proc_mean", "human_things_triplet_r2"]
    lines = [
        "# Coherence Steering Sweep",
        "",
        "Completed alpha rows:",
        "",
        complete[display_cols].to_markdown(index=False, floatfmt=".6f") if len(complete) else "None",
        "",
        "Reference rows:",
        "",
        refs[[c for c in ["state", "out_model", "gen_proc_mean", "human_things_triplet_r2"] if c in refs.columns]]
        .to_markdown(index=False, floatfmt=".6f") if len(refs) else "None",
    ]
    missing = summary[~summary["complete"]]
    if len(missing):
        lines.extend(["", "Incomplete rows:", "", missing.to_markdown(index=False, floatfmt=".6f")])
    (args.steer_dir / "SUMMARY.md").write_text("\n".join(lines) + "\n")
    print(complete[display_cols].to_string(index=False), flush=True)
    print(f"[summary] wrote {args.steer_dir}", flush=True)


def run_sweep(args: argparse.Namespace) -> None:
    alphas = args.alphas
    if alphas is None:
        alphas = ACT_ALPHAS if args.route == "actdiff" else TASK_ALPHAS
    for alpha in alphas:
        out_model = out_model_for(args.route, alpha)
        child_env = os.environ.copy()
        if args.gpu is not None:
            child_env["CUDA_VISIBLE_DEVICES"] = args.gpu
        child_env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "run-one",
            "--route",
            args.route,
            "--alpha",
            f"{alpha:g}",
            "--batch-size",
            str(args.batch_size),
            "--max-length",
            str(args.max_length),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--dtype",
            args.dtype,
        ]
        print(f"[sweep] run-one {out_model}", flush=True)
        subprocess.run(cmd, cwd=ROOT, env=child_env, check=True)
        fit_salmon_for(out_model, args)
    summarize(args)


def main() -> None:
    args = parse_args()
    args.raw_dir = args.raw_dir.resolve()
    args.out_dir = args.out_dir.resolve()
    args.steer_dir = args.steer_dir.resolve()
    args.stim_dir = args.stim_dir.resolve()
    args.raw_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.action == "run-one":
        run_one(args)
    elif args.action == "fit-salmon":
        fit_salmon(args)
    elif args.action == "summarize":
        summarize(args)
    elif args.action == "run-sweep":
        run_sweep(args)
    else:
        raise ValueError(args.action)


if __name__ == "__main__":
    main()
