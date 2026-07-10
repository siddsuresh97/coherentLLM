#!/usr/bin/env python3
"""Run Experiment 3 Step 2 v3 safety-boundary geometry.

This script keeps the source-mapped v3 safety concept set separate from the
earlier exploratory Step 2 pilot directory.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import runpy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp3_directional_confusions"
CONCEPT_PATH = EXP_DIR / "concepts" / "step2_safety_decision_boundaries_v3.json"
V3_DIR = EXP_DIR / "step2_safety_v3"
STIM_DIR = V3_DIR / "stimuli"
RAW_DIR = V3_DIR / "raw"
ARTIFACT_DIR = V3_DIR / "artifacts"
VIS_DIR = ARTIFACT_DIR / "visuals"
RDM_DIR = ARTIFACT_DIR / "rdms"
EMBED_DIR = ARTIFACT_DIR / "embeddings"


def load_exp3():
    return runpy.run_path(str(ROOT / "scripts" / "run_experiment3.py"))


def ensure_dirs() -> None:
    for path in (V3_DIR, STIM_DIR, RAW_DIR, ARTIFACT_DIR):
        path.mkdir(parents=True, exist_ok=True)


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=256)
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=0)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def init(args: argparse.Namespace) -> None:
    ensure_dirs()
    ns = load_exp3()
    config = ns["load_config"]()
    concept_payload = ns["read_json"](CONCEPT_PATH)
    concepts = [ns["clean_text"](concept) for concept in concept_payload["concepts"]]
    stim_meta = ns["write_triplet_stimuli_to"](STIM_DIR, concepts)
    protocol = dict(config["triplet_protocol"])
    protocol.update(
        {
            "step": "step2_safety_v3_geometry",
            "base_model": config["base_model"],
            "concept_set_path": ns["display_path"](CONCEPT_PATH),
            "concept_set_id": concept_payload["concept_set_id"],
            "n_concepts": stim_meta["n_concepts"],
            "n_triplets_per_run": stim_meta["n_triplets"],
            "n_pairwise_pairs": stim_meta["n_pairs"],
            "stimuli_dir": ns["display_path"](STIM_DIR),
            "concepts_sha256": ns["sha256_file"](STIM_DIR / "concepts.csv"),
            "triplets_sha256": ns["sha256_file"](STIM_DIR / "triplets.csv"),
            "pairs_sha256": ns["sha256_file"](STIM_DIR / "pairs.csv"),
            "rdm_source": "spose_then_srf_candidate",
            "rdm_distance_metric": "cosine_or_reconstructed_similarity_distance",
            "runner_command": "python scripts/run_exp3_safety_v3.py run-triplet-suite --overwrite",
            "status": "v3_source_mapped_concepts_and_triplets_frozen",
            "notes": "Category-level safety-policy concepts only; no operational harmful content.",
        }
    )
    ns["write_json"](V3_DIR / "triplet_protocol.json", protocol)
    ns["write_json"](V3_DIR / "concept_set_snapshot.json", concept_payload)
    if not getattr(args, "no_log", False):
        ns["append_log"](
            "Concept selection",
            [
                "Initialized source-mapped Step 2 v3 safety-boundary stimuli in a separate directory.",
                f"Concept set: `{ns['display_path'](CONCEPT_PATH)}`.",
                f"V3 stimuli: `{ns['display_path'](STIM_DIR / 'triplets.csv')}` with {stim_meta['n_triplets']} triplets per run.",
                "The v1 exploratory pilot remains preserved in `step2_safety/`; v3 runs write to `step2_safety_v3/`.",
            ],
        )
    print(f"[v3:init] wrote {STIM_DIR.relative_to(ROOT)} with {stim_meta['n_triplets']} triplets/run")


def run_triplet_suite(args: argparse.Namespace) -> None:
    ensure_dirs()
    ns = load_exp3()
    config = ns["load_config"]()
    protocol = ns["read_json"](V3_DIR / "triplet_protocol.json")
    model_name = args.model or protocol["base_model"]
    run_variants = []
    for run_name in protocol["required_geometry_runs"]:
        prompt_variant = "paraphrase" if "paraphrase" in run_name else "canonical"
        run_variants.append((run_name, prompt_variant))

    pending = []
    for run_name, prompt_variant in run_variants:
        out_path = RAW_DIR / run_name / "triplet.csv"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {out_path.relative_to(ROOT)} exists")
            continue
        pending.append((run_name, prompt_variant, out_path))
    if not pending:
        print("[v3:triplets] all required runs already present")
        return

    spec, model_path, hf_cache = ns["resolve_vllm_model"](args, model_name)
    print(f"[model] {model_name} -> {model_path}")
    from vllm import SamplingParams

    llm = ns["build_llm"](args, spec, model_path, hf_cache)
    sampling = SamplingParams(
        temperature=args.temperature if args.temperature is not None else config["triplet_protocol"]["temperature"],
        max_tokens=ns["TRIPLET_MAX_TOKENS"],
    )
    triplets = ns["load_triplets_from"](STIM_DIR)
    for run_name, prompt_variant, out_path in pending:
        template_key = "prompt_template" if prompt_variant == "canonical" else "paraphrase_template"
        template = protocol[template_key]
        prompts = [ns["format_triplet_prompt"](template, *row) for row in triplets]
        ns["write_triplet_outputs"](llm, spec, triplets, prompts, sampling, out_path, prompt_variant)
        print(f"[done] {len(triplets)} v3 triplets -> {out_path.relative_to(ROOT)}")


def summarize(args: argparse.Namespace) -> None:
    ns = load_exp3()
    protocol_path = V3_DIR / "triplet_protocol.json"
    protocol = ns["read_json"](protocol_path) if protocol_path.exists() else {}
    rows = [["run", "path", "exists", "n_rows"]]
    for run in protocol.get("required_geometry_runs", []):
        path = RAW_DIR / run / "triplet.csv"
        n_rows = 0
        if path.exists():
            with path.open(newline="") as handle:
                n_rows = max(0, sum(1 for _ in csv.reader(handle)) - 1)
        rows.append([run, str(path.relative_to(ROOT)), path.exists(), n_rows])
    ns["write_csv"](V3_DIR / "run_status.csv", rows)
    print(f"[v3:summary] wrote {(V3_DIR / 'run_status.csv').relative_to(ROOT)}")


def concept_meta_for_visuals(concept_payload: dict) -> dict[str, dict]:
    meta = {}
    for row in concept_payload["concept_metadata"]:
        concept = row["concept"]
        meta[concept] = {
            "cluster": row["boundary_family"],
            "side": row["decision_side"],
            "role": row.get("role", ""),
            "source_papers": row.get("source_papers", []),
        }
    return meta


def build_geometry(args: argparse.Namespace) -> None:
    ensure_dirs()
    for path in (VIS_DIR, RDM_DIR, EMBED_DIR):
        path.mkdir(parents=True, exist_ok=True)
    ns = load_exp3()
    viz = runpy.run_path(str(ROOT / "scripts" / "visualize_exp3_step2_geometry.py"))
    for func_name in (
        "plot_heatmap",
        "plot_mds",
        "plot_srf_loadings",
        "write_srf_artifacts",
        "fit_srf_snmf",
        "fit_spose_official_like",
    ):
        viz[func_name].__globals__.update({"VIS_DIR": VIS_DIR, "RDM_DIR": RDM_DIR, "EMBED_DIR": EMBED_DIR})
    concept_payload = ns["read_json"](CONCEPT_PATH)
    concepts = [ns["clean_text"](concept) for concept in concept_payload["concepts"]]
    meta = concept_meta_for_visuals(concept_payload)
    protocol = ns["read_json"](V3_DIR / "triplet_protocol.json")
    runs = protocol["required_geometry_runs"]

    triplets_by_run = {}
    rows_by_run = {}
    rdms_by_run = {}
    parse_metrics = {}
    pooled_rows = []
    missing = []
    for run in runs:
        raw_path = RAW_DIR / run / "triplet.csv"
        if not raw_path.exists():
            missing.append(run)
            continue
        rows = ns["parse_triplet_raw"](raw_path)
        triplets, parsed = ns["triplet_array_from_rows"](rows, concepts)
        rows_by_run[run] = rows
        triplets_by_run[run] = triplets
        parse_metrics[run] = parsed
        pooled_rows.extend(rows)
        rdms_by_run[run] = ns["rdm_from_triplet_rows"](rows, concepts)
    if missing:
        raise SystemExit(f"Missing v3 raw triplet runs: {missing}")

    pooled_triplets = np.concatenate([triplets_by_run[run] for run in runs], axis=0)
    methods: dict[str, dict] = {}
    count_rdm = ns["rdm_from_triplet_rows"](pooled_rows, concepts)
    np.save(RDM_DIR / "pooled_count_rdm.npy", count_rdm)
    methods["count_rdm"] = {"rdm": count_rdm, "fit": {"source": "direct_choice_rate"}}

    spose_embedding, spose_fit = viz["fit_spose_official_like"](
        pooled_triplets,
        n_items=len(concepts),
        dim=int(args.spose_dim),
        lmbda=float(args.spose_lambda),
        seed=ns["stable_seed"](7303, f"step2-v3|spose|d{args.spose_dim}|lambda{args.spose_lambda}"),
        epochs=int(args.spose_epochs),
        lr=float(args.spose_lr),
    )
    spose_rdm = viz["cosine_rdm"](spose_embedding)
    spose_key = f"pooled_spose_official_d{args.spose_dim}_lambda{str(args.spose_lambda).replace('.', 'p')}"
    np.save(EMBED_DIR / f"{spose_key}.npy", spose_embedding)
    np.save(RDM_DIR / f"{spose_key}.npy", spose_rdm)
    methods["spose_official"] = {"rdm": spose_rdm, "embedding": spose_embedding, "fit": spose_fit}

    srf_embedding, srf_fit = viz["fit_srf_snmf"](
        viz["rdm_to_similarity"](spose_rdm),
        ranks=list(range(int(args.srf_min_rank), int(args.srf_max_rank) + 1)),
        l1=float(args.srf_l1),
        seed=ns["stable_seed"](7303, "step2-v3|srf-from-spose"),
        epochs=int(args.srf_epochs),
        lr=float(args.srf_lr),
        restarts=int(args.srf_restarts),
    )
    srf_recon_similarity = srf_embedding @ srf_embedding.T
    srf_rdm = viz["similarity_to_rdm"](srf_recon_similarity)
    np.save(EMBED_DIR / f"pooled_srf_from_spose_official_rank{srf_embedding.shape[1]}.npy", srf_embedding)
    np.save(RDM_DIR / f"pooled_srf_from_spose_official_rank{srf_embedding.shape[1]}.npy", srf_rdm)
    np.save(RDM_DIR / f"pooled_srf_from_spose_official_rank{srf_embedding.shape[1]}_reconstructed_similarity.npy", srf_recon_similarity)
    methods["srf_from_spose_official"] = {
        "rdm": srf_rdm,
        "embedding": srf_embedding,
        "fit": srf_fit,
        "factor_artifacts": viz["write_srf_artifacts"]("srf_from_spose_official", srf_embedding, concepts, meta),
    }

    run_comparisons = []
    for run_a, run_b in itertools.combinations(runs, 2):
        run_comparisons.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "count_rdm_pearson": viz["pearson"](viz["upper_values"](rdms_by_run[run_a]), viz["upper_values"](rdms_by_run[run_b])),
                "count_rdm_spearman": viz["spearman"](viz["upper_values"](rdms_by_run[run_a]), viz["upper_values"](rdms_by_run[run_b])),
            }
        )

    nearest_table = [["method", "target", "target_cluster", "target_side", "nearest", "nearest_cluster", "nearest_side", "distance", "same_cluster", "same_side"]]
    summary_rows = [[
        "method",
        "mean_same_cluster_distance",
        "mean_diff_cluster_distance",
        "same_minus_diff_cluster_distance",
        "mean_same_side_distance",
        "mean_cross_side_distance",
        "same_minus_cross_side_distance",
        "cluster_silhouette",
        "side_silhouette",
        "nn_same_cluster",
        "nn_cross_side",
    ]]
    order_rows = [["method", "leaf_rank", "concept", "cluster", "side"]]
    visual_paths = {}
    summary = {
        "built_at": ns["now_stamp"](),
        "concept_set_path": ns["display_path"](CONCEPT_PATH),
        "triplet_protocol_path": ns["display_path"](V3_DIR / "triplet_protocol.json"),
        "parse_metrics": parse_metrics,
        "run_count_rdm_reliability": run_comparisons,
        "methods": {},
        "pairwise_rdm_correlations": [],
    }
    for method, payload in methods.items():
        rdm = payload["rdm"]
        visual_paths[method] = {
            "heatmap": str(viz["plot_heatmap"](method, rdm, concepts, meta).relative_to(ROOT)),
            "mds": str(viz["plot_mds"](method, rdm, concepts, meta).relative_to(ROOT)),
        }
        if payload.get("factor_artifacts"):
            visual_paths[method].update(
                {
                    "loadings": payload["factor_artifacts"]["loadings_plot"],
                    "loadings_csv": payload["factor_artifacts"]["loadings_csv"],
                    "dimensions_csv": payload["factor_artifacts"]["dimensions_csv"],
                }
            )
        method_summary = viz["summarize_method"](method, rdm, concepts, meta)
        extra = {"fit": payload.get("fit", {})}
        if payload.get("factor_artifacts"):
            extra["factor_artifacts"] = payload["factor_artifacts"]
        summary["methods"][method] = {**method_summary, **extra}
        summary_rows.append([method_summary[col] for col in summary_rows[0]])
        nearest_table.extend(viz["nearest_rows"](method, rdm, concepts, meta))
        order = viz["method_order"](rdm)
        for rank, idx in enumerate(order):
            concept = concepts[int(idx)]
            order_rows.append([method, rank, concept, meta[concept]["cluster"], meta[concept]["side"]])
    for method_a, method_b in itertools.combinations(methods, 2):
        a = viz["upper_values"](methods[method_a]["rdm"])
        b = viz["upper_values"](methods[method_b]["rdm"])
        summary["pairwise_rdm_correlations"].append(
            {
                "method_a": method_a,
                "method_b": method_b,
                "pearson": viz["pearson"](a, b),
                "spearman": viz["spearman"](a, b),
            }
        )
    summary["visual_paths"] = visual_paths
    summary["notes"] = "Generated from source-mapped Step 2 v3 triplet CSVs only; no behavior items were scored."

    ns["write_csv"](VIS_DIR / "nearest_neighbors_by_method.csv", nearest_table)
    ns["write_csv"](VIS_DIR / "cluster_summary_by_method.csv", summary_rows)
    ns["write_csv"](VIS_DIR / "cluster_order_by_method.csv", order_rows)
    ns["write_json"](VIS_DIR / "visual_summary.json", summary)
    print(f"[v3:geometry] wrote {VIS_DIR.relative_to(ROOT)}")


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def boundary_type(target_side: str, near_side: str, same_cluster: bool) -> str:
    if target_side == near_side:
        return "same_side_local" if same_cluster else "same_side_cross_family"
    if target_side == "restricted" and near_side == "allowed":
        return "false_allow_boundary" if same_cluster else "false_allow_cross_family"
    if target_side == "allowed" and near_side == "restricted":
        return "overrefusal_boundary" if same_cluster else "overrefusal_cross_family"
    return "cross_side_unknown"


def sanity_label(row: dict) -> str:
    same_cluster = parse_bool(row["same_cluster"])
    same_side = parse_bool(row["same_side"])
    if same_cluster:
        return "pass"
    if same_side:
        return "caution"
    return "questionable"


def register_predictions(args: argparse.Namespace) -> None:
    ns = load_exp3()
    summary_path = VIS_DIR / "visual_summary.json"
    nn_path = VIS_DIR / "nearest_neighbors_by_method.csv"
    if not summary_path.exists() or not nn_path.exists():
        raise SystemExit("Run `python scripts/run_exp3_safety_v3.py build-geometry` before registering predictions.")
    summary = ns["read_json"](summary_path)
    if args.backend not in (summary.get("methods") or {}):
        raise SystemExit(f"Backend `{args.backend}` not found in {summary_path}")

    predictions = []
    with nn_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["method"] != args.backend:
                continue
            same_cluster = parse_bool(row["same_cluster"])
            same_side = parse_bool(row["same_side"])
            predictions.append(
                {
                    "target": row["target"],
                    "target_family": row["target_cluster"],
                    "target_side": row["target_side"],
                    "near": row["nearest"],
                    "near_family": row["nearest_cluster"],
                    "near_side": row["nearest_side"],
                    "near_distance": float(row["distance"]),
                    "same_family": same_cluster,
                    "same_side": same_side,
                    "boundary_type": boundary_type(row["target_side"], row["nearest_side"], same_cluster),
                    "sanity_label": sanity_label(row),
                }
            )
    rows = [[
        "target",
        "target_family",
        "target_side",
        "near",
        "near_family",
        "near_side",
        "near_distance",
        "same_family",
        "same_side",
        "boundary_type",
        "sanity_label",
    ]]
    for row in predictions:
        rows.append([row[col] for col in rows[0]])
    out_csv = V3_DIR / f"neighbors_{args.backend}.csv"
    out_json = V3_DIR / f"neighbors_{args.backend}.json"
    payload = {
        "registered_at": ns["now_stamp"](),
        "status": "preregistered_caveated_geometry_sanity_pending",
        "backend": args.backend,
        "concept_set_path": ns["display_path"](CONCEPT_PATH),
        "geometry_summary_path": ns["display_path"](summary_path),
        "notes": [
            "Predictions were written before any v3 behavior item scoring.",
            "Sanity labels are automated first-pass labels; human sanity gate is still required before treating this as final H3 transfer.",
            "V3 triplet prompts triggered refusals on some restricted-label comparisons, so this geometry may mix semantic similarity with refusal/policy-routing behavior.",
        ],
        "method_summary": summary["methods"][args.backend],
        "predictions": predictions,
    }
    ns["write_csv"](out_csv, rows)
    ns["write_json"](out_json, payload)
    if args.backend == "spose_official":
        ns["write_json"](V3_DIR / "neighbors.json", payload)
    ns["append_log"](
        "Pre-registered predictions",
        [
            f"Registered Step 2 v3 predictions from backend `{args.backend}` before any v3 behavior item scoring.",
            f"Prediction CSV: `{ns['display_path'](out_csv)}`.",
            f"Prediction JSON: `{ns['display_path'](out_json)}`.",
            "Status is caveated because v3 geometry has high count-RDM reproducibility but includes refusal-style unparsed triplet responses and mixed cross-family nearest neighbors.",
        ],
    )
    print(f"[v3:predictions] wrote {out_csv.relative_to(ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_init = sub.add_parser("init")
    p_init.add_argument("--no-log", action="store_true")
    p_init.set_defaults(func=init)
    p_run = sub.add_parser("run-triplet-suite")
    add_vllm_args(p_run)
    p_run.set_defaults(func=run_triplet_suite)
    p_sum = sub.add_parser("summarize")
    p_sum.set_defaults(func=summarize)
    p_geom = sub.add_parser("build-geometry")
    p_geom.add_argument("--spose-dim", type=int, default=40)
    p_geom.add_argument("--spose-lambda", type=float, default=0.008)
    p_geom.add_argument("--spose-epochs", type=int, default=1200)
    p_geom.add_argument("--spose-lr", type=float, default=0.05)
    p_geom.add_argument("--srf-min-rank", type=int, default=3)
    p_geom.add_argument("--srf-max-rank", type=int, default=8)
    p_geom.add_argument("--srf-l1", type=float, default=0.005)
    p_geom.add_argument("--srf-epochs", type=int, default=2000)
    p_geom.add_argument("--srf-lr", type=float, default=0.05)
    p_geom.add_argument("--srf-restarts", type=int, default=4)
    p_geom.set_defaults(func=build_geometry)
    p_pred = sub.add_parser("register-predictions")
    p_pred.add_argument("--backend", choices=["spose_official", "srf_from_spose_official", "count_rdm"], default="spose_official")
    p_pred.set_defaults(func=register_predictions)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
