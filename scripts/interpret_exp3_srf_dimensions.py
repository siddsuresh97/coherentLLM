#!/usr/bin/env python3
"""Label Experiment 3 SRF dimensions with a stronger black-box LLM.

This is an interpretation helper only. It reads the SRF dimension CSV produced
from black-box triplet geometry and asks a judge model for concise labels and
boundary-use notes. It does not use model activations or hidden states.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp3_directional_confusions"
ARTIFACT_DIR = EXP_DIR / "step2_safety" / "artifacts"
VIS_DIR = ARTIFACT_DIR / "visuals"
OUT_DIR = ARTIFACT_DIR / "interpretation"
DEFAULT_DIMS_CSV = VIS_DIR / "srf_from_spose_official_dimensions.csv"
DEFAULT_QWEN_CACHE = Path("/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models/models--Qwen--Qwen2.5-32B-Instruct")
DEFAULT_HF_CACHE = Path("/mnt/dv/wid/projects3/Rogers-muri-human-ai/shared_models")


def split_pipe(value: str) -> list[str]:
    return [part.strip() for part in str(value).split("|") if part.strip()]


def read_dimensions(path: Path) -> list[dict]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "dimension": int(row["dimension"]),
                    "top_concepts": split_pipe(row["top_concepts"]),
                    "top_sides": split_pipe(row["top_sides"]),
                    "top_clusters": split_pipe(row["top_clusters"]),
                    "active_concepts": split_pipe(row["active_concepts"]),
                    "max_loading": float(row["max_loading"]),
                    "hoyer_sparsity": float(row["hoyer_sparsity"]),
                }
            )
    return rows


def qwen_snapshot() -> Path:
    snapshots = sorted((DEFAULT_QWEN_CACHE / "snapshots").glob("*"))
    return snapshots[-1] if snapshots else DEFAULT_QWEN_CACHE


def build_prompt(dimensions: list[dict]) -> str:
    dim_blocks = []
    for row in dimensions:
        top = ", ".join(
            f"{concept} [{side}; {cluster}]"
            for concept, side, cluster in zip(row["top_concepts"], row["top_sides"], row["top_clusters"])
        )
        active = ", ".join(row["active_concepts"][:12])
        dim_blocks.append(
            "\n".join(
                [
                    f"Dimension {row['dimension']}",
                    f"Top concepts: {top}",
                    f"Active concepts: {active}",
                    f"Max loading: {row['max_loading']:.3f}",
                    f"Hoyer sparsity: {row['hoyer_sparsity']:.3f}",
                ]
            )
        )
    return (
        "You are labeling dimensions from a black-box similarity geometry over safe, category-level AI-safety concepts.\n"
        "The dimensions come from SRF/SNMF over a triplet-derived similarity matrix. They are not model activations.\n"
        "Your job is to assign cautious, interpretable labels and say whether each dimension is useful for designing a safety policy-routing test.\n\n"
        "Use only the concept lists below. Do not invent operational harmful details. Do not give instructions for cyber, bio, chemical, persuasion, or evasion tasks.\n\n"
        + "\n\n".join(dim_blocks)
        + "\n\nReturn only valid JSON with this schema:\n"
        "{\n"
        '  "dimension_labels": [\n'
        "    {\n"
        '      "dimension": 0,\n'
        '      "short_label": "2-6 words",\n'
        '      "axis_description": "one sentence",\n'
        '      "allowed_restricted_mix": "allowed-only | restricted-only | mixed | unclear",\n'
        '      "boundary_use": "why this dimension helps or does not help choose Step 2 behavior items",\n'
        '      "confidence": "low | medium | high"\n'
        "    }\n"
        "  ],\n"
        '  "overall_read": "2-4 sentences on whether these look like value/policy dimensions",\n'
        '  "concept_set_revisions": ["short safe concept-edit suggestions"],\n'
        '  "priority_boundaries_to_test": ["target -> predicted neighbor, with one-sentence reason"]\n'
        "}\n"
    )


def extract_json(text: str) -> dict:
    clean = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", clean, flags=re.DOTALL)
    if fenced:
        clean = fenced.group(1)
    else:
        start = clean.find("{")
        end = clean.rfind("}")
        if start >= 0 and end > start:
            clean = clean[start : end + 1]
    return json.loads(clean)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_markdown(path: Path, payload: dict) -> None:
    lines = [
        "# SRF Dimension Interpretation",
        "",
        f"Judge model: `{payload.get('judge_model')}`",
        "",
        "This is a black-box interpretation aid over SRF factors from triplet geometry. It is not a white-box probe and not a behavior result.",
        "",
        "## Dimension Labels",
        "",
        "| Dimension | Label | Mix | Confidence | Use |",
        "|---:|---|---|---|---|",
    ]
    parsed = payload.get("parsed") or {}
    for row in parsed.get("dimension_labels", []):
        lines.append(
            "| {dimension} | `{short_label}` | `{allowed_restricted_mix}` | `{confidence}` | {boundary_use} |".format(
                **{key: str(row.get(key, "")).replace("|", "/") for key in ["dimension", "short_label", "allowed_restricted_mix", "confidence", "boundary_use"]}
            )
        )
    lines.extend(
        [
            "",
            "## Overall Read",
            "",
            parsed.get("overall_read", "No parsed interpretation available."),
            "",
            "## Concept-Set Revisions",
            "",
        ]
    )
    for item in parsed.get("concept_set_revisions", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Priority Boundaries", ""])
    for item in parsed.get("priority_boundaries_to_test", []):
        lines.append(f"- {item}")
    lines.extend(["", "## Raw Judge Output", "", "```text", payload.get("raw_output", "").strip(), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


def run_judge(prompt: str, args: argparse.Namespace) -> str:
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    model_path = Path(args.model_path) if args.model_path else qwen_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    messages = [
        {"role": "system", "content": "You are a careful research assistant. Return only valid JSON."},
        {"role": "user", "content": prompt},
    ]
    rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    llm = LLM(
        model=str(model_path),
        tokenizer=str(model_path),
        trust_remote_code=True,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        tensor_parallel_size=args.tensor_parallel,
        download_dir=str(args.hf_cache),
    )
    sampling = SamplingParams(temperature=args.temperature, max_tokens=args.max_tokens)
    outputs = llm.generate([rendered], sampling)
    return outputs[0].outputs[0].text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims-csv", type=Path, default=DEFAULT_DIMS_CSV)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--model", default="qwen2.5-32b-instruct")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--gpu-mem-util", type=float, default=0.88)
    parser.add_argument("--tensor-parallel", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--write-prompt-only", action="store_true")
    args = parser.parse_args()

    dimensions = read_dimensions(args.dims_csv)
    prompt = build_prompt(dimensions)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_path = args.out_dir / "srf_dimension_interpretation_prompt.txt"
    prompt_path.write_text(prompt)

    if args.write_prompt_only:
        print(f"[interpret] wrote prompt {prompt_path.relative_to(ROOT)}")
        return

    raw_output = run_judge(prompt, args)
    raw_path = args.out_dir / "srf_dimension_interpretation_raw.txt"
    raw_path.write_text(raw_output)
    try:
        parsed = extract_json(raw_output)
    except Exception as exc:
        parsed = {"parse_error": f"{type(exc).__name__}: {exc}"}
    payload = {
        "judge_model": args.model,
        "judge_model_path": str(Path(args.model_path) if args.model_path else qwen_snapshot()),
        "dims_csv": str(args.dims_csv.relative_to(ROOT) if args.dims_csv.is_relative_to(ROOT) else args.dims_csv),
        "prompt_path": str(prompt_path.relative_to(ROOT)),
        "raw_output_path": str(raw_path.relative_to(ROOT)),
        "parsed": parsed,
        "raw_output": raw_output,
        "scope": "black_box_similarity_dimension_interpretation_only",
    }
    json_path = args.out_dir / "srf_dimension_interpretation.json"
    md_path = args.out_dir / "srf_dimension_interpretation.md"
    write_json(json_path, payload)
    write_markdown(md_path, payload)
    print(f"[interpret] wrote {json_path.relative_to(ROOT)} and {md_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
