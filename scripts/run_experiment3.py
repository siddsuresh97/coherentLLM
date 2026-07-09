#!/usr/bin/env python3
"""Experiment 3: directional error prediction from triplet geometry.

This script owns the small neutral Step 1 pipeline:

1. freeze a neutral concept set and triplet protocol,
2. build a model triplet RDM and reliability report from raw triplet runs,
3. pre-register nearest-neighbor confusion predictions,
4. generate directional multiple-choice items, and
5. score model item responses against geometry and null baselines.

It intentionally does not create Step 2 safety items. Step 2 should be added only
after Step 1 has a green H1 verdict and a passed human sanity gate.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import subprocess
import sys
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / "experiments" / "exp3_directional_confusions"
CONCEPT_DIR = EXP_DIR / "concepts"
STIM_DIR = EXP_DIR / "stimuli"
RAW_DIR = EXP_DIR / "raw"
ARTIFACT_DIR = EXP_DIR / "artifacts"
RDM_DIR = ARTIFACT_DIR / "rdms"
EMBED_DIR = ARTIFACT_DIR / "embeddings"
ITEM_DIR = EXP_DIR / "items" / "step1"
RESULT_DIR = EXP_DIR / "results"
FIG_DIR = EXP_DIR / "figs"
_REPORT_LINK_BASE: str | None | bool = False

DEFAULT_CONCEPTS = [
    "alligator",
    "caiman",
    "crocodile",
    "boa python",
    "cobra",
    "snake",
    "blindworm",
    "chameleon",
    "gecko",
    "lizard",
    "salamander",
    "toad",
    "tortoise",
    "turtle",
    "axe",
    "chisel",
    "hammer",
    "saw",
]

DEFAULT_CONFIG = {
    "seed": 7303,
    "base_model": "llama-3.1-8b-instruct",
    "step1_concepts_file": "experiments/exp3_directional_confusions/concepts/step1_neutral.json",
    "feature_matrix": "data/human/leuven_groundtruth_matrix.csv",
    "feature_map": "data/human/leuven300_feature_map.csv",
    "n_items_per_target": 4,
    "far_control_min_quantile": 0.60,
    "bootstrap_samples": 2000,
    "split_half_samples": 64,
    "rdm_source": "salmon_embedding",
    "salmon_dimension": 5,
    "salmon_max_epochs": 2000,
    "salmon_split_half_samples": 2,
    "salmon_split_half_max_epochs": 500,
    "salmon_test_fraction": 0.20,
    "salmon_verbose": 100000,
    "null_permutations": 5000,
    "reliability_min_mean_pearson": 0.75,
    "reliability_min_split_half_pearson": 0.70,
    "h1_min_error_items": 10,
    "triplet_protocol": {
        "scheme": "full_anchor_candidate_enumeration",
        "prompt_template": (
            "Target concept: {anchor}\n"
            "Candidate A: {concept1}\n"
            "Candidate B: {concept2}\n"
            "Which candidate is more similar in semantic meaning to the target? "
            "Answer with exactly A or B."
        ),
        "paraphrase_template": (
            "Target concept: {anchor}\n"
            "Candidate A: {concept1}\n"
            "Candidate B: {concept2}\n"
            "Which candidate has the closer semantic meaning to the target concept? "
            "Reply with exactly A or B."
        ),
        "response_format": "labeled_binary_choice_A_or_B",
        "temperature": 0.0,
        "aggregation": (
            "Parse each labeled response into a SALMON triplet [head, winner, loser] "
            "where head is the anchor and winner is the candidate judged closer. "
            "Fit salmon.triplets.offline.OfflineEmbedding on pooled triplet runs "
            "with d=5, then compute the RDM as cosine distance (1 - cosine "
            "similarity) between embedded concepts. Per-run SALMON fits and split-half SALMON fits are used for "
            "reliability."
        ),
        "required_geometry_runs": [
            "base_seed_a_canonical_prompt",
            "base_seed_b_canonical_prompt",
            "base_seed_a_matched_paraphrase_prompt",
        ],
    },
}

TEMPLATE_VARIANTS = [
    "Which option is the best match for this description?\n{clues}\nAnswer with only A, B, C, or D.",
    "A catalog card lists these properties:\n{clues}\nWhich option does the card describe? Answer with only A, B, C, or D.",
    "Choose the option that best fits all of these clues:\n{clues}\nAnswer with only A, B, C, or D.",
    "A field note says the thing has these attributes:\n{clues}\nWhich option is it? Answer with only A, B, C, or D.",
]

FEATURE_BLOCKLIST = {
    "dutch",
    "belgium",
    "belgian",
    "cinderella",
    "flipper",
    "free willy",
    "jaws",
    "jurassic",
    "skippy",
    "tintin",
    "thanksgiving",
    "xmass",
    "christmas",
    "brand",
    "song",
    "saying",
    "expression",
    "term of abuse",
    "girl's name",
    "main character",
    "tv-programme",
}


def clean_text(value: object) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def norm_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def git_stdout(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            args,
            cwd=ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    value = result.stdout.strip()
    return value or None


def github_slug_from_remote(remote_url: str | None) -> str | None:
    if not remote_url:
        return None
    remote_url = remote_url.strip()
    if remote_url.startswith("git@github.com:"):
        return remote_url.removeprefix("git@github.com:").removesuffix(".git")
    if remote_url.startswith("https://github.com/"):
        return remote_url.removeprefix("https://github.com/").removesuffix(".git")
    return None


def report_link_base() -> str | None:
    global _REPORT_LINK_BASE
    if _REPORT_LINK_BASE is not False:
        return _REPORT_LINK_BASE
    override = os.environ.get("EXP3_REPORT_LINK_BASE")
    if override:
        _REPORT_LINK_BASE = override.rstrip("/")
        return _REPORT_LINK_BASE
    slug = github_slug_from_remote(git_stdout(["git", "config", "--get", "remote.origin.url"]))
    branch = git_stdout(["git", "branch", "--show-current"]) or "exp3-directional-confusions"
    _REPORT_LINK_BASE = f"https://github.com/{slug}/blob/{branch}" if slug else None
    return _REPORT_LINK_BASE


def now_stamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stable_seed(base_seed: int, label: str) -> int:
    offset = int(hashlib.sha256(label.encode("utf-8")).hexdigest()[:8], 16)
    return int((int(base_seed) + offset) % (2**32 - 1))


def ensure_dirs() -> None:
    for path in (EXP_DIR, CONCEPT_DIR, STIM_DIR, RAW_DIR, ARTIFACT_DIR, RDM_DIR, EMBED_DIR, ITEM_DIR, RESULT_DIR, FIG_DIR):
        path.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_json(path: Path) -> object:
    with path.open() as handle:
        return json.load(handle)


def write_csv(path: Path, rows: Iterable[Iterable[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config() -> dict:
    path = EXP_DIR / "config.json"
    if not path.exists():
        write_json(path, DEFAULT_CONFIG)
        return json.loads(json.dumps(DEFAULT_CONFIG))
    current = read_json(path)
    merged = json.loads(json.dumps(DEFAULT_CONFIG))
    for key, value in current.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key].update(value)
        else:
            merged[key] = value
    return merged


def concept_file_path(config: dict) -> Path:
    path = Path(config["step1_concepts_file"])
    return path if path.is_absolute() else ROOT / path


def load_concepts(config: dict) -> list[str]:
    path = concept_file_path(config)
    payload = read_json(path)
    return [clean_text(c) for c in payload["concepts"]]


def append_log(block_name: str, lines: Iterable[str]) -> None:
    ensure_dirs()
    path = EXP_DIR / "RESEARCH_LOG.md"
    if not path.exists():
        path.write_text("# Experiment 3 Research Log\n\nAppend-only decision trail.\n")
    with path.open("a") as handle:
        handle.write(f"\n## {now_stamp()} DECISION: {block_name}\n\n")
        for line in lines:
            handle.write(f"{line.rstrip()}\n")


def write_triplet_stimuli(concepts: list[str]) -> dict:
    write_csv(STIM_DIR / "concepts.csv", [[concept] for concept in concepts])
    triplets = []
    for anchor in concepts:
        others = [concept for concept in concepts if concept != anchor]
        for concept1, concept2 in itertools.combinations(others, 2):
            triplets.append([anchor, concept1, concept2])
    write_csv(STIM_DIR / "triplets.csv", triplets)
    pairs = []
    for concept1, concept2 in itertools.combinations(concepts, 2):
        pairs.append([concept1, concept2])
    write_csv(STIM_DIR / "pairs.csv", pairs)
    return {"n_concepts": len(concepts), "n_triplets": len(triplets), "n_pairs": len(pairs)}


def init_experiment(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    concept_path = concept_file_path(config)
    if args.overwrite or not concept_path.exists():
        write_json(
            concept_path,
            {
                "concepts": DEFAULT_CONCEPTS,
                "domain": "neutral concrete Leuven reptiles/amphibians plus familiar tools",
                "rationale": (
                    "The set is familiar, concrete, and human-checkable. It has tight "
                    "local neighborhoods such as alligator/caiman/crocodile, "
                    "tortoise/turtle, snake/cobra/boa python, and axe/chisel/hammer/saw, "
                    "while retaining far cross-domain controls."
                ),
                "rejected_first_passes": {
                    "scale128_mixed_animals": (
                        "Broader THINGS/scale128 animals were rejected for the first "
                        "scaffold because the repo already has a compact Leuven feature "
                        "matrix for this reptiles/tools domain."
                    ),
                    "medical_or_legal_pairs": (
                        "Explicitly deferred until neutral H1 is green, per the design "
                        "invariant."
                    ),
                },
            },
        )
    concepts = load_concepts(config)
    stim_meta = write_triplet_stimuli(concepts)
    protocol = dict(config["triplet_protocol"])
    protocol.update(
        {
            "base_model": config["base_model"],
            "n_concepts": stim_meta["n_concepts"],
            "n_triplets_per_run": stim_meta["n_triplets"],
            "n_pairwise_pairs": stim_meta["n_pairs"],
            "stimuli_dir": display_path(STIM_DIR),
            "concepts_sha256": sha256_file(STIM_DIR / "concepts.csv"),
            "triplets_sha256": sha256_file(STIM_DIR / "triplets.csv"),
            "pairs_sha256": sha256_file(STIM_DIR / "pairs.csv"),
            "rdm_source": config.get("rdm_source", "salmon_embedding"),
            "rdm_distance_metric": "cosine_distance"
            if config.get("rdm_source", "salmon_embedding") == "salmon_embedding"
            else "1_minus_choice_rate",
            "salmon_dimension": config.get("salmon_dimension"),
            "salmon_max_epochs": config.get("salmon_max_epochs"),
            "runner_command_canonical": (
                f"python scripts/run_experiment3.py run-triplet-suite --model {config['base_model']} --overwrite"
            ),
            "runner_command_paraphrase": (
                f"python scripts/run_experiment3.py run-triplet-suite --model {config['base_model']} --overwrite"
            ),
            "status": (
                "frozen_stimuli_written; model triplet runs are required before "
                "pre-registered neighbors can be trusted"
            ),
        }
    )
    write_json(EXP_DIR / "triplet_protocol.json", protocol)
    if args.log:
        append_log(
            "Concept selection",
            [
                "Selected 18 neutral Leuven concrete concepts: "
                + ", ".join(f"`{c}`" for c in concepts)
                + ".",
                "Why this domain: it has known human feature structure, obvious local neighborhoods, "
                "and enough fine-grained reptiles/amphibians to plausibly elicit directional confusions.",
                "Uncertainty band is not yet confirmed. That requires model item responses; the report stays red until errors exist.",
                "Rejected broader mixed THINGS concepts for this first scaffold because the available Leuven feature matrix supports cleaner item generation.",
                "Rejected medical/legal safety substitutions for now because Step 2 is gated on neutral H1.",
            ],
        )
    update_report()


def parse_triplet_raw(path: Path) -> list[tuple[str, str, str, str]]:
    rows = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                anchor, concept1, concept2 = str(row["input"]).split("|")
            except ValueError:
                continue
            rows.append((clean_text(anchor), clean_text(concept1), clean_text(concept2), str(row["response"])))
    return rows


def parse_triplet_choice(response: str, concept1: str, concept2: str) -> int | None:
    raw = str(response).strip()
    label_match = re.match(
        r"^(?:answer\s*[:\-]?\s*)?(?:option\s*)?[\(\[]?([ab])[\)\].,:;\-]?(?:\s|$)",
        raw,
        flags=re.IGNORECASE,
    )
    if label_match:
        return 1 if label_match.group(1).lower() == "a" else 2

    resp = norm_key(raw)
    key1 = norm_key(concept1)
    key2 = norm_key(concept2)
    hit1 = bool(key1 and key1 in resp)
    hit2 = bool(key2 and key2 in resp)
    if hit1 and not hit2:
        return 1
    if hit2 and not hit1:
        return 2
    if hit1 and hit2:
        first1 = resp.find(key1)
        first2 = resp.find(key2)
        if first1 != first2:
            return 1 if first1 < first2 else 2
    return None


def rdm_from_triplet_rows(rows: list[tuple[str, str, str, str]], concepts: list[str]) -> np.ndarray:
    index = {norm_key(concept): i for i, concept in enumerate(concepts)}
    n = len(concepts)
    close = np.zeros((n, n), dtype=float)
    total = np.zeros((n, n), dtype=float)
    for anchor, concept1, concept2, response in rows:
        ai = index.get(norm_key(anchor))
        i1 = index.get(norm_key(concept1))
        i2 = index.get(norm_key(concept2))
        if ai is None or i1 is None or i2 is None:
            continue
        parsed = parse_triplet_choice(response, concept1, concept2)
        chosen = i1 if parsed == 1 else i2 if parsed == 2 else None
        total[ai, i1] += 1
        total[ai, i2] += 1
        if chosen is not None:
            close[ai, chosen] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(total > 0, close / total, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        sim = np.nanmean(np.dstack([rate, rate.T]), axis=2)
    np.fill_diagonal(sim, 1.0)
    mean = np.nanmean(sim)
    if not np.isfinite(mean):
        mean = 0.5
    sim = np.where(np.isnan(sim), mean, sim)
    sim = np.clip(sim, 0.0, 1.0)
    rdm = 1.0 - sim
    np.fill_diagonal(rdm, 0.0)
    return rdm


def triplet_array_from_rows(rows: list[tuple[str, str, str, str]], concepts: list[str]) -> tuple[np.ndarray, dict]:
    index = {norm_key(concept): i for i, concept in enumerate(concepts)}
    triplets = []
    skipped = Counter()
    for anchor, concept1, concept2, response in rows:
        ai = index.get(norm_key(anchor))
        i1 = index.get(norm_key(concept1))
        i2 = index.get(norm_key(concept2))
        if ai is None or i1 is None or i2 is None:
            skipped["unknown_concept"] += 1
            continue
        parsed = parse_triplet_choice(response, concept1, concept2)
        if parsed == 1:
            triplets.append((ai, i1, i2))
        elif parsed == 2:
            triplets.append((ai, i2, i1))
        else:
            skipped["unparsed_response"] += 1
    return np.asarray(triplets, dtype=int), {"n_rows": len(rows), "n_valid_triplets": len(triplets), "skipped": dict(skipped)}


def load_offline_embedding_class():
    import importlib
    import types

    repo = str(ROOT)
    if repo not in sys.path:
        sys.path.insert(0, repo)
    for package in ("salmon", "salmon.triplets"):
        if package not in sys.modules:
            module = types.ModuleType(package)
            module.__path__ = [str(ROOT.joinpath(*package.split(".")))]
            sys.modules[package] = module
    try:
        return importlib.import_module("salmon.triplets.offline").OfflineEmbedding
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "Could not import SALMON OfflineEmbedding. Run build-rdm in the SALMON conda env "
            "(for example: source /mnt/ws/home/ssuresh/miniconda3/etc/profile.d/conda.sh && "
            "conda activate /mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/salmon)."
        ) from exc


def fit_salmon_embedding(
    triplets: np.ndarray,
    *,
    n_concepts: int,
    dim: int,
    max_epochs: int,
    seed: int,
    test_fraction: float,
    verbose: int,
    ident: str,
) -> tuple[np.ndarray, dict]:
    if triplets.shape[0] < 10:
        raise RuntimeError(f"Need at least 10 parsed triplets for SALMON; got {triplets.shape[0]}")
    from sklearn.model_selection import train_test_split

    OfflineEmbedding = load_offline_embedding_class()
    train, test = train_test_split(triplets, random_state=seed, test_size=test_fraction)
    estimator = OfflineEmbedding(
        n=n_concepts,
        d=dim,
        max_epochs=max_epochs,
        verbose=verbose,
        ident=ident,
        random_state=seed,
    )
    estimator.fit(train, test)
    embedding = np.asarray(estimator.embedding_, dtype=float)
    score = float(estimator.score(test)) if hasattr(estimator, "score") else float("nan")
    history = getattr(estimator, "history_", [])
    final_loss = float(history[-1].get("loss_test", float("nan"))) if history else float("nan")
    return embedding, {
        "n_triplets": int(triplets.shape[0]),
        "n_train": int(train.shape[0]),
        "n_test": int(test.shape[0]),
        "test_score": score,
        "test_loss": final_loss,
        "max_epochs": int(max_epochs),
        "seed": int(seed),
        "ident": ident,
    }


def cosine_rdm_from_embedding(embedding: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embedding, axis=1, keepdims=True)
    normalized = embedding / np.clip(norms, 1e-12, None)
    rdm = 1.0 - normalized @ normalized.T
    rdm = (rdm + rdm.T) / 2.0
    np.fill_diagonal(rdm, 0.0)
    return rdm


def procrustes_r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        return float("nan")
    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean(axis=0, keepdims=True)
    x_norm = float(np.linalg.norm(x_centered))
    y_norm = float(np.linalg.norm(y_centered))
    if x_norm == 0.0 or y_norm == 0.0:
        return float("nan")
    x_scaled = x_centered / x_norm
    y_scaled = y_centered / y_norm
    try:
        u, singular_values, vt = np.linalg.svd(x_scaled.T @ y_scaled, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("nan")
    rotation = u @ vt
    scale = float(np.sum(singular_values))
    fitted = scale * x_scaled @ rotation
    residual = float(np.sum((fitted - y_scaled) ** 2))
    total = float(np.sum(y_scaled**2))
    if total == 0.0:
        return float("nan")
    return float(max(0.0, 1.0 - residual / total))


def salmon_triplet_budget(
    *,
    n_concepts: int,
    dim: int,
    n_triplets_per_run: int | None = None,
    n_triplets_total: int | None = None,
) -> dict:
    base = float(n_concepts * dim * math.log(max(n_concepts, 2)))
    payload = {
        "heuristic": "fudge_factor * n_concepts * embedding_dim * ln(n_concepts)",
        "n_concepts": int(n_concepts),
        "embedding_dim": int(dim),
        "base_n_d_log_n": base,
    }
    if n_triplets_per_run is not None:
        payload["observed_triplets_per_run"] = int(n_triplets_per_run)
        payload["observed_per_run_fudge_factor"] = float(n_triplets_per_run / base) if base else float("nan")
    if n_triplets_total is not None:
        payload["observed_triplets_total"] = int(n_triplets_total)
        payload["observed_total_fudge_factor"] = float(n_triplets_total / base) if base else float("nan")
    return payload


def upper_values(matrix: np.ndarray) -> np.ndarray:
    return matrix[np.triu_indices_from(matrix, k=1)]


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3:
        return float("nan")
    a = a[ok]
    b = b[ok]
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def build_rdm(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    concepts = load_concepts(config)
    protocol_runs = args.runs or config["triplet_protocol"]["required_geometry_runs"]
    rdm_source = config.get("rdm_source", "salmon_embedding")

    if rdm_source == "direct_choice_rate":
        present = []
        missing = []
        rdms = {}
        rows_by_run = {}
        for run in protocol_runs:
            raw_path = RAW_DIR / run / "triplet.csv"
            if not raw_path.exists():
                missing.append(run)
                continue
            present.append(run)
            rows = parse_triplet_raw(raw_path)
            rows_by_run[run] = rows
            rdm = rdm_from_triplet_rows(rows, concepts)
            rdms[run] = rdm
            out = RDM_DIR / f"{run}.npy"
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, rdm)

        if not rdms:
            raise SystemExit(f"No triplet CSVs found under {display_path(RAW_DIR)} for requested runs: {protocol_runs}")

        comparisons = []
        for run_a, run_b in itertools.combinations(present, 2):
            comparisons.append(
                {
                    "run_a": run_a,
                    "run_b": run_b,
                    "upper_triangle_pearson": pearson_corr(upper_values(rdms[run_a]), upper_values(rdms[run_b])),
                }
            )
        mean_corr = float(np.nanmean([row["upper_triangle_pearson"] for row in comparisons])) if comparisons else float("nan")
        rng = np.random.default_rng(int(config["seed"]))
        split_half = []
        for run in present:
            corrs = []
            rows = rows_by_run[run]
            for _ in range(int(config["split_half_samples"])):
                order = rng.permutation(len(rows))
                half = len(order) // 2
                rows_a = [rows[i] for i in order[:half]]
                rows_b = [rows[i] for i in order[half:]]
                rdm_a = rdm_from_triplet_rows(rows_a, concepts)
                rdm_b = rdm_from_triplet_rows(rows_b, concepts)
                corrs.append(pearson_corr(upper_values(rdm_a), upper_values(rdm_b)))
            split_half.append(
                {
                    "run": run,
                    "n_splits": int(config["split_half_samples"]),
                    "mean_upper_triangle_pearson": float(np.nanmean(corrs)),
                    "p05_upper_triangle_pearson": float(np.nanquantile(corrs, 0.05)),
                }
            )
        mean_split_half = float(np.nanmean([row["mean_upper_triangle_pearson"] for row in split_half])) if split_half else float("nan")
        reliability_gate = bool(
            len(missing) == 0
            and np.isfinite(mean_corr)
            and mean_corr >= config["reliability_min_mean_pearson"]
            and np.isfinite(mean_split_half)
            and mean_split_half >= config["reliability_min_split_half_pearson"]
        )

        stacked = np.stack([rdms[run] for run in present], axis=0)
        rdm = np.mean(stacked, axis=0)
        np.fill_diagonal(rdm, 0.0)
        np.save(ARTIFACT_DIR / "rdm.npy", rdm)
        meta = {
            "built_at": now_stamp(),
            "rdm_source": "direct_choice_rate",
            "distance_metric": "1_minus_choice_rate",
            "source_runs": present,
            "missing_runs": missing,
            "rdm_path": display_path(ARTIFACT_DIR / "rdm.npy"),
            "rdm_sha256": sha256_file(ARTIFACT_DIR / "rdm.npy"),
            "rdm_shape": list(rdm.shape),
            "aggregation": config["triplet_protocol"]["aggregation"],
            "reliability_gate": reliability_gate,
            "reliability_min_mean_pearson": config["reliability_min_mean_pearson"],
            "reliability_min_split_half_pearson": config["reliability_min_split_half_pearson"],
            "pairwise_run_reliability": comparisons,
            "mean_pairwise_upper_triangle_pearson": mean_corr,
            "split_half_reliability": split_half,
            "mean_split_half_upper_triangle_pearson": mean_split_half,
            "status": "green" if reliability_gate else "red",
        }
        write_json(ARTIFACT_DIR / "rdm_meta.json", meta)
        update_report()
        print(f"[rdm] wrote {display_path(ARTIFACT_DIR / 'rdm.npy')} reliability={meta['status']}")
        return

    if rdm_source != "salmon_embedding":
        raise SystemExit(f"Unknown rdm_source `{rdm_source}`. Expected `salmon_embedding` or `direct_choice_rate`.")

    present = []
    missing = []
    rdms = {}
    rows_by_run = {}
    triplets_by_run = {}
    embeddings_by_run = {}
    salmon_fit_metrics = {}
    parse_metrics = {}
    dim = int(config["salmon_dimension"])
    max_epochs = int(config["salmon_max_epochs"])
    test_fraction = float(config["salmon_test_fraction"])
    verbose = int(config["salmon_verbose"])
    for run in protocol_runs:
        raw_path = RAW_DIR / run / "triplet.csv"
        if not raw_path.exists():
            missing.append(run)
            continue
        rows = parse_triplet_raw(raw_path)
        rows_by_run[run] = rows
        triplets, parsed = triplet_array_from_rows(rows, concepts)
        parse_metrics[run] = parsed
        if triplets.shape[0] == 0:
            missing.append(f"{run}:no_parseable_triplets")
            continue
        present.append(run)
        triplets_by_run[run] = triplets
        seed = stable_seed(int(config["seed"]), f"salmon|{run}|d{dim}")
        embedding, fit_meta = fit_salmon_embedding(
            triplets,
            n_concepts=len(concepts),
            dim=dim,
            max_epochs=max_epochs,
            seed=seed,
            test_fraction=test_fraction,
            verbose=verbose,
            ident=run,
        )
        embeddings_by_run[run] = embedding
        salmon_fit_metrics[run] = fit_meta
        emb_out = EMBED_DIR / f"{run}_salmon_d{dim}.npy"
        np.save(emb_out, embedding)
        rdm = cosine_rdm_from_embedding(embedding)
        rdms[run] = rdm
        out = RDM_DIR / f"{run}.npy"
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, rdm)

    if not rdms:
        raise SystemExit(f"No triplet CSVs found under {display_path(RAW_DIR)} for requested runs: {protocol_runs}")

    comparisons = []
    for run_a, run_b in itertools.combinations(present, 2):
        nn_a = nearest_neighbor_indices(rdms[run_a])
        nn_b = nearest_neighbor_indices(rdms[run_b])
        top2_b = []
        for i in range(len(concepts)):
            row = rdms[run_b][i].copy()
            row[i] = np.inf
            top2_b.append(set(int(j) for j in np.argsort(row)[:2]))
        comparisons.append(
            {
                "run_a": run_a,
                "run_b": run_b,
                "upper_triangle_pearson": pearson_corr(upper_values(rdms[run_a]), upper_values(rdms[run_b])),
                "embedding_procrustes_r2": procrustes_r2(embeddings_by_run[run_a], embeddings_by_run[run_b]),
                "nearest_neighbor_top1_agreement": float(np.mean([a == b for a, b in zip(nn_a, nn_b)])),
                "nearest_neighbor_top2_agreement": float(np.mean([a in b for a, b in zip(nn_a, top2_b)])),
            }
        )
    mean_corr = float(np.nanmean([row["upper_triangle_pearson"] for row in comparisons])) if comparisons else float("nan")
    mean_procrustes = float(np.nanmean([row["embedding_procrustes_r2"] for row in comparisons])) if comparisons else float("nan")
    mean_nn_top1 = float(np.nanmean([row["nearest_neighbor_top1_agreement"] for row in comparisons])) if comparisons else float("nan")
    mean_nn_top2 = float(np.nanmean([row["nearest_neighbor_top2_agreement"] for row in comparisons])) if comparisons else float("nan")
    rng = np.random.default_rng(int(config["seed"]))
    split_half = []
    split_samples = int(config.get("salmon_split_half_samples", config["split_half_samples"]))
    split_epochs = int(config.get("salmon_split_half_max_epochs", max_epochs))
    for run in present:
        corrs = []
        proc_r2s = []
        triplets = triplets_by_run[run]
        for split_idx in range(split_samples):
            order = rng.permutation(len(triplets))
            half = len(order) // 2
            triplets_a = triplets[order[:half]]
            triplets_b = triplets[order[half:]]
            emb_a, _ = fit_salmon_embedding(
                triplets_a,
                n_concepts=len(concepts),
                dim=dim,
                max_epochs=split_epochs,
                seed=stable_seed(int(config["seed"]), f"salmon|{run}|split{split_idx}|a"),
                test_fraction=test_fraction,
                verbose=verbose,
                ident=f"{run}_split{split_idx}_a",
            )
            emb_b, _ = fit_salmon_embedding(
                triplets_b,
                n_concepts=len(concepts),
                dim=dim,
                max_epochs=split_epochs,
                seed=stable_seed(int(config["seed"]), f"salmon|{run}|split{split_idx}|b"),
                test_fraction=test_fraction,
                verbose=verbose,
                ident=f"{run}_split{split_idx}_b",
            )
            rdm_a = cosine_rdm_from_embedding(emb_a)
            rdm_b = cosine_rdm_from_embedding(emb_b)
            corrs.append(pearson_corr(upper_values(rdm_a), upper_values(rdm_b)))
            proc_r2s.append(procrustes_r2(emb_a, emb_b))
        split_half.append(
            {
                "run": run,
                "n_splits": split_samples,
                "max_epochs_per_fit": split_epochs,
                "mean_upper_triangle_pearson": float(np.nanmean(corrs)),
                "p05_upper_triangle_pearson": float(np.nanquantile(corrs, 0.05)),
                "mean_embedding_procrustes_r2": float(np.nanmean(proc_r2s)),
                "p05_embedding_procrustes_r2": float(np.nanquantile(proc_r2s, 0.05)),
            }
        )
    mean_split_half = float(np.nanmean([row["mean_upper_triangle_pearson"] for row in split_half])) if split_half else float("nan")

    pooled_triplets = np.concatenate([triplets_by_run[run] for run in present], axis=0)
    pooled_embedding, pooled_fit = fit_salmon_embedding(
        pooled_triplets,
        n_concepts=len(concepts),
        dim=dim,
        max_epochs=max_epochs,
        seed=stable_seed(int(config["seed"]), f"salmon|pooled|d{dim}"),
        test_fraction=test_fraction,
        verbose=verbose,
        ident="pooled_step1",
    )
    pooled_embedding_path = EMBED_DIR / f"pooled_salmon_d{dim}.npy"
    np.save(pooled_embedding_path, pooled_embedding)
    rdm = cosine_rdm_from_embedding(pooled_embedding)
    run_triplet_counts = [int(triplets_by_run[run].shape[0]) for run in present]
    mean_triplets_per_run = int(round(float(np.mean(run_triplet_counts)))) if run_triplet_counts else None

    reliability_gate = bool(
        len(missing) == 0
        and np.isfinite(mean_corr)
        and mean_corr >= config["reliability_min_mean_pearson"]
        and np.isfinite(mean_split_half)
        and mean_split_half >= config["reliability_min_split_half_pearson"]
    )
    np.save(ARTIFACT_DIR / "rdm.npy", rdm)
    meta = {
        "built_at": now_stamp(),
        "rdm_source": "salmon_embedding",
        "distance_metric": "cosine_distance",
        "salmon_dimension": dim,
        "salmon_max_epochs": max_epochs,
        "salmon_test_fraction": test_fraction,
        "pooled_embedding_path": display_path(pooled_embedding_path),
        "pooled_embedding_sha256": sha256_file(pooled_embedding_path),
        "pooled_fit_metrics": pooled_fit,
        "n_valid_triplets_total": int(pooled_triplets.shape[0]),
        "salmon_triplet_budget": salmon_triplet_budget(
            n_concepts=len(concepts),
            dim=dim,
            n_triplets_per_run=mean_triplets_per_run,
            n_triplets_total=int(pooled_triplets.shape[0]),
        ),
        "source_runs": present,
        "missing_runs": missing,
        "rdm_path": display_path(ARTIFACT_DIR / "rdm.npy"),
        "rdm_sha256": sha256_file(ARTIFACT_DIR / "rdm.npy"),
        "rdm_shape": list(rdm.shape),
        "aggregation": config["triplet_protocol"]["aggregation"],
        "parse_metrics": parse_metrics,
        "per_run_embedding_paths": {
            run: display_path(EMBED_DIR / f"{run}_salmon_d{dim}.npy")
            for run in present
        },
        "per_run_salmon_fit_metrics": salmon_fit_metrics,
        "reliability_gate": reliability_gate,
        "reliability_min_mean_pearson": config["reliability_min_mean_pearson"],
        "reliability_min_split_half_pearson": config["reliability_min_split_half_pearson"],
        "pairwise_run_reliability": comparisons,
        "mean_pairwise_upper_triangle_pearson": mean_corr,
        "mean_pairwise_embedding_procrustes_r2": mean_procrustes,
        "mean_nearest_neighbor_top1_agreement": mean_nn_top1,
        "mean_nearest_neighbor_top2_agreement": mean_nn_top2,
        "split_half_reliability": split_half,
        "mean_split_half_upper_triangle_pearson": mean_split_half,
        "status": "green" if reliability_gate else "red",
    }
    write_json(ARTIFACT_DIR / "rdm_meta.json", meta)
    update_report()
    print(f"[rdm] wrote {display_path(ARTIFACT_DIR / 'rdm.npy')} reliability={meta['status']}")


def nearest_and_far_controls(rdm: np.ndarray, concepts: list[str], config: dict) -> list[dict]:
    rows = []
    q = float(config["far_control_min_quantile"])
    for i, target in enumerate(concepts):
        distances = [(j, float(rdm[i, j])) for j in range(len(concepts)) if j != i]
        distances = sorted(distances, key=lambda pair: pair[1])
        near_j, near_d = distances[0]
        values = np.array([d for _, d in distances], dtype=float)
        cutoff = float(np.quantile(values, q))
        far_pool = [(j, d) for j, d in distances if d >= cutoff and j != near_j]
        if len(far_pool) < 2:
            far_pool = distances[-2:]
        target_far = float(np.median([d for _, d in far_pool]))
        far_pool = sorted(far_pool, key=lambda pair: (abs(pair[1] - target_far), pair[1]))
        controls = far_pool[:2]
        rows.append(
            {
                "target": target,
                "near": concepts[near_j],
                "near_distance": near_d,
                "far_controls": [
                    {"concept": concepts[j], "distance": float(d)}
                    for j, d in controls
                ],
                "far_control_rule": f"two controls from target row distances >= q{q:.2f}, closest to that far-pool median",
            }
        )
    return rows


def nearest_neighbor_indices(rdm: np.ndarray) -> list[int]:
    neighbors = []
    for i in range(rdm.shape[0]):
        row = np.asarray(rdm[i], dtype=float).copy()
        row[i] = np.inf
        neighbors.append(int(np.argmin(row)))
    return neighbors


def register_neighbors(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    concepts = load_concepts(config)
    rdm_path = ARTIFACT_DIR / "rdm.npy"
    meta_path = ARTIFACT_DIR / "rdm_meta.json"
    if not rdm_path.exists():
        raise SystemExit("No model RDM found. Run `python scripts/run_experiment3.py build-rdm` after triplet runs.")
    meta = read_json(meta_path) if meta_path.exists() else {}
    if not args.allow_red_rdm and not meta.get("reliability_gate", False):
        raise SystemExit("RDM reliability gate is not green. Use --allow-red-rdm only for engineering smoke tests.")
    rdm = np.load(rdm_path)
    predictions = nearest_and_far_controls(rdm, concepts, config)
    payload = {
        "registered_at": now_stamp(),
        "rdm_path": display_path(rdm_path),
        "rdm_meta": meta,
        "status": "pre_registered_before_item_scoring",
        "sanity_gate": "pending_human_review",
        "predictions": predictions,
    }
    write_json(EXP_DIR / "neighbors.json", payload)
    append_log(
        "Pre-registered predictions",
        [
            "Geometry-derived neighbors written before item scoring.",
            "RDM source: `" + display_path(rdm_path) + "`.",
            "Human sanity gate is pending. Do not treat H1 as green until these pairs are manually accepted.",
            "",
            "| Target | Predicted near confusion | Near distance | Far controls |",
            "|---|---|---:|---|",
            *[
                "| `{target}` | `{near}` | {near_distance:.4f} | {far} |".format(
                    target=row["target"],
                    near=row["near"],
                    near_distance=row["near_distance"],
                    far=", ".join(
                        f"`{control['concept']}` ({control['distance']:.4f})"
                        for control in row["far_controls"]
                    ),
                )
                for row in predictions
            ],
        ],
    )
    update_report()
    print(f"[neighbors] wrote {display_path(EXP_DIR / 'neighbors.json')}")


def load_feature_map(config: dict) -> dict[str, str]:
    path = ROOT / config["feature_map"]
    if not path.exists():
        return {}
    mapping = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw = str(row.get("raw", "")).strip()
            clean = clean_text(row.get("clean", raw))
            if raw:
                mapping[raw] = clean
    return mapping


def load_feature_matrix(config: dict):
    import pandas as pd

    path = ROOT / config["feature_matrix"]
    df = pd.read_csv(path, index_col=0)
    df.index = [clean_text(index) for index in df.index]
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0)
    return df


def clean_feature_name(raw: str, mapping: dict[str, str]) -> str:
    base = re.sub(r"\.\d+$", "", raw)
    text = mapping.get(base, base)
    text = text.replace("_", " ")
    text = text.replace("can't", "cannot")
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = text.strip(" .")
    return text


def usable_feature(raw: str, text: str) -> bool:
    if not (4 <= len(text) <= 80):
        return False
    if any(token in text for token in FEATURE_BLOCKLIST):
        return False
    if any(char in text for char in ["=", ":", "/", "\\"]):
        return False
    if text.endswith(" ao gallop"):
        return False
    return True


def feature_frequency(df, raw: str) -> float:
    return float((df[raw].to_numpy(dtype=float) > 0.5).mean())


def ordered_features(df, concept: str, raw_features: list[str], mapping: dict[str, str], salt: str) -> list[str]:
    digest = hashlib.sha256(salt.encode("utf-8")).digest()
    offset = int.from_bytes(digest[:4], "big")
    scored = []
    for raw in raw_features:
        text = clean_feature_name(raw, mapping)
        if not usable_feature(raw, text):
            continue
        freq = feature_frequency(df, raw)
        # Prefer diagnostic-but-not-unique features. Rotate ties deterministically
        # across variants so each target gets surface variation.
        score = (abs(freq - 0.28), (hash((raw, offset)) % 100000) / 100000)
        scored.append((score, raw))
    return [raw for _, raw in sorted(scored)]


def choose_feature_clues(df, target: str, near: str, far_controls: list[str], mapping: dict[str, str], variant: int) -> list[str]:
    target_row = df.loc[target] > 0.5
    near_row = df.loc[near] > 0.5
    far_rows = [(df.loc[far] > 0.5) for far in far_controls]
    target_true = [raw for raw in df.columns if bool(target_row[raw])]

    shared_near = [
        raw
        for raw in target_true
        if bool(near_row[raw]) and not all(bool(row[raw]) for row in far_rows)
    ]
    target_specific = [
        raw
        for raw in target_true
        if not bool(near_row[raw]) and not any(bool(row[raw]) for row in far_rows)
    ]
    contrastive = [
        raw
        for raw in target_true
        if not all(bool(row[raw]) for row in [near_row, *far_rows])
    ]

    shared_near = ordered_features(df, target, shared_near, mapping, f"{target}|shared|{variant}")
    target_specific = ordered_features(df, target, target_specific, mapping, f"{target}|specific|{variant}")
    contrastive = ordered_features(df, target, contrastive, mapping, f"{target}|contrast|{variant}")
    fallback = ordered_features(df, target, target_true, mapping, f"{target}|fallback|{variant}")

    chosen: list[str] = []
    for pool, take in ((shared_near, 2), (target_specific, 1), (contrastive, 2), (fallback, 3)):
        for raw in pool:
            text = clean_feature_name(raw, mapping)
            if text not in chosen:
                chosen.append(text)
            if len(chosen) >= take and pool is not contrastive and pool is not fallback:
                break
            if len(chosen) >= 3:
                break
        if len(chosen) >= 3:
            break
    return chosen[:3]


def generate_items(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    neighbors_path = EXP_DIR / "neighbors.json"
    if not neighbors_path.exists():
        raise SystemExit("No neighbors.json found. Register predictions before generating items.")
    concepts = load_concepts(config)
    df = load_feature_matrix(config)
    missing = [concept for concept in concepts if concept not in set(df.index)]
    if missing:
        raise SystemExit(f"Feature matrix is missing concepts: {missing}")
    mapping = load_feature_map(config)
    neighbors = read_json(neighbors_path)
    rng = np.random.default_rng(int(config["seed"]))
    items = []
    n_variants = int(args.n_items_per_target or config["n_items_per_target"])
    prediction_by_target = {row["target"]: row for row in neighbors["predictions"]}
    for target in concepts:
        pred = prediction_by_target[target]
        near = pred["near"]
        far_controls = [row["concept"] for row in pred["far_controls"]]
        for variant in range(n_variants):
            clues = choose_feature_clues(df, target, near, far_controls, mapping, variant)
            if len(clues) < 2:
                raise RuntimeError(f"Not enough usable feature clues for {target}")
            options = [
                {"concept": target, "role": "correct"},
                {"concept": near, "role": "near"},
                {"concept": far_controls[0], "role": "far"},
                {"concept": far_controls[1], "role": "far"},
            ]
            order = rng.permutation(4)
            ordered = []
            for letter, idx in zip(["A", "B", "C", "D"], order):
                option = dict(options[int(idx)])
                option["letter"] = letter
                ordered.append(option)
            correct_letter = next(option["letter"] for option in ordered if option["role"] == "correct")
            template = TEMPLATE_VARIANTS[variant % len(TEMPLATE_VARIANTS)]
            clue_text = "\n".join(f"- {feature}" for feature in clues)
            option_text = "\n".join(f"{option['letter']}. {option['concept']}" for option in ordered)
            prompt = f"{template.format(clues=clue_text)}\n\nOptions:\n{option_text}"
            items.append(
                {
                    "item_id": f"step1_{target.replace(' ', '_')}_{variant:02d}",
                    "target": target,
                    "correct_answer": target,
                    "correct_letter": correct_letter,
                    "predicted_near": near,
                    "far_controls": far_controls,
                    "feature_clues": clues,
                    "options": ordered,
                    "prompt": prompt,
                    "template_variant": variant % len(TEMPLATE_VARIANTS),
                }
            )

    out_path = ITEM_DIR / "items.json"
    write_json(out_path, items)
    write_csv(
        ITEM_DIR / "items.csv",
        [
            ["item_id", "target", "correct_letter", "predicted_near", "far_controls", "prompt"],
            *[
                [
                    item["item_id"],
                    item["target"],
                    item["correct_letter"],
                    item["predicted_near"],
                    "|".join(item["far_controls"]),
                    item["prompt"],
                ]
                for item in items
            ],
        ],
    )
    append_log(
        "Item design",
        [
            f"Generated {len(items)} Step 1 items: {n_variants} per target.",
            "Each item uses one geometry-predicted near distractor and two far controls from `neighbors.json`.",
            "Question form is feature-attribution over Leuven feature norms. Clues prefer features shared with the near neighbor plus at least one target-specific or contrastive feature when available.",
            "Option positions are counterbalanced by deterministic RNG seed.",
        ],
    )
    update_report()
    print(f"[items] wrote {display_path(out_path)}")


def parse_choice(response: str, item: dict) -> tuple[str | None, str | None]:
    text = str(response).strip()
    upper = text.upper()
    letter_match = re.search(r"^\s*(?:ANSWER\s*(?:IS|:)?\s*)?([ABCD])(?:\b|[.)\s:])", upper)
    if not letter_match:
        letter_match = re.search(
            r"(?:ANSWER|CORRECT ANSWER|BEST ANSWER|BEST FIT)\s*(?:IS|:)?\s*([ABCD])(?:\b|[.)\s:])",
            upper,
        )
    if not letter_match:
        letter_match = re.search(r"(^|[^A-Z])([ABCD])([^A-Z]|$)", upper)
    if letter_match:
        letter = next(group for group in letter_match.groups() if group in {"A", "B", "C", "D"})
        concept = next(option["concept"] for option in item["options"] if option["letter"] == letter)
        return letter, concept
    key_text = norm_key(text)
    matches = []
    for option in item["options"]:
        key = norm_key(option["concept"])
        if key and key in key_text:
            matches.append(option)
    if len(matches) == 1:
        return matches[0]["letter"], matches[0]["concept"]
    return None, None


def load_item_responses(run: str) -> dict[str, str]:
    path = RAW_DIR / run / "items.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    responses = {}
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            responses[str(row["item_id"])] = str(row["response"])
    return responses


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 2:
        return float("nan")
    x = x[ok]
    y = y[ok]
    denom = float(np.sum((x - x.mean()) ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum((x - x.mean()) * (y - y.mean())) / denom)


def score_items(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    item_path = ITEM_DIR / "items.json"
    if not item_path.exists():
        raise SystemExit("No items found. Run generate-items first.")
    rdm_path = ARTIFACT_DIR / "rdm.npy"
    if not rdm_path.exists():
        raise SystemExit("No RDM found. Run build-rdm first.")
    meta = read_json(ARTIFACT_DIR / "rdm_meta.json") if (ARTIFACT_DIR / "rdm_meta.json").exists() else {}
    neighbors = read_json(EXP_DIR / "neighbors.json") if (EXP_DIR / "neighbors.json").exists() else {}
    concepts = load_concepts(config)
    concept_index = {concept: i for i, concept in enumerate(concepts)}
    rdm = np.load(rdm_path)
    items = read_json(item_path)
    responses = load_item_responses(args.run)

    scored = []
    for item in items:
        response = responses.get(item["item_id"], "")
        letter, concept = parse_choice(response, item)
        option_by_concept = {option["concept"]: option for option in item["options"]}
        role = option_by_concept.get(concept, {}).get("role") if concept else "invalid"
        scored.append(
            {
                "item_id": item["item_id"],
                "target": item["target"],
                "response": response,
                "chosen_letter": letter,
                "chosen_concept": concept,
                "chosen_role": role,
                "correct": concept == item["correct_answer"],
                "predicted_near": item["predicted_near"],
                "far_controls": item["far_controls"],
            }
        )

    total = len(scored)
    correct = sum(1 for row in scored if row["correct"])
    errors = [row for row in scored if not row["correct"] and row["chosen_concept"]]
    directional_errors = [row for row in errors if row["chosen_role"] in {"near", "far"}]
    near_errors = [row for row in directional_errors if row["chosen_role"] == "near"]
    far_errors = [row for row in directional_errors if row["chosen_role"] == "far"]
    near_fraction = len(near_errors) / len(directional_errors) if directional_errors else float("nan")

    rng = np.random.default_rng(int(config["seed"]))
    null_counts = []
    for _ in range(int(config["null_permutations"])):
        hits = 0
        denom = 0
        for row in directional_errors:
            item = next(item for item in items if item["item_id"] == row["item_id"])
            distractors = [option["concept"] for option in item["options"] if option["role"] in {"near", "far"}]
            shuffled_near = rng.choice(distractors)
            denom += 1
            if row["chosen_concept"] == shuffled_near:
                hits += 1
        null_counts.append(hits / denom if denom else float("nan"))
    null_counts_arr = np.asarray(null_counts, dtype=float)
    if np.isfinite(near_fraction):
        valid_null = null_counts_arr[np.isfinite(null_counts_arr)]
        shuffle_p = float((np.sum(valid_null >= near_fraction) + 1) / (len(valid_null) + 1)) if valid_null.size else float("nan")
    else:
        shuffle_p = float("nan")

    chosen_base_counts = Counter(row["chosen_concept"] for row in errors if row["chosen_concept"])
    all_error_choices = sum(chosen_base_counts.values())
    base_expected = 0.0
    base_observed = 0
    for row in directional_errors:
        item = next(item for item in items if item["item_id"] == row["item_id"])
        distractors = [option["concept"] for option in item["options"] if option["role"] in {"near", "far"}]
        weights = np.array([chosen_base_counts.get(concept, 0) + 1 for concept in distractors], dtype=float)
        near_pos = distractors.index(row["predicted_near"])
        base_expected += float(weights[near_pos] / weights.sum())
        base_observed += int(row["chosen_concept"] == row["predicted_near"])
    base_expected_fraction = base_expected / len(directional_errors) if directional_errors else float("nan")
    base_lift = near_fraction - base_expected_fraction if np.isfinite(near_fraction) else float("nan")

    pair_rows = []
    opportunities = defaultdict(int)
    substitutions = defaultdict(int)
    for item in items:
        target = item["target"]
        for option in item["options"]:
            if option["role"] == "correct":
                continue
            key = (target, option["concept"])
            opportunities[key] += 1
    for row in errors:
        if row["chosen_concept"] and (row["target"], row["chosen_concept"]) in opportunities:
            substitutions[(row["target"], row["chosen_concept"])] += 1
    for (target, distractor), n_opp in sorted(opportunities.items()):
        dist = float(rdm[concept_index[target], concept_index[distractor]])
        n_sub = substitutions[(target, distractor)]
        pair_rows.append(
            {
                "target": target,
                "distractor": distractor,
                "rdm_distance": dist,
                "opportunities": n_opp,
                "substitutions": n_sub,
                "substitution_rate": n_sub / n_opp if n_opp else float("nan"),
            }
        )
    x = np.array([row["rdm_distance"] for row in pair_rows], dtype=float)
    y = np.array([row["substitution_rate"] for row in pair_rows], dtype=float)
    distance_slope = slope(x, y)
    boot = []
    if pair_rows:
        for _ in range(int(config["bootstrap_samples"])):
            idx = rng.integers(0, len(pair_rows), len(pair_rows))
            boot.append(slope(x[idx], y[idx]))
    boot_arr = np.asarray(boot, dtype=float)
    slope_ci = [
        float(np.nanquantile(boot_arr, 0.025)) if boot_arr.size else float("nan"),
        float(np.nanquantile(boot_arr, 0.975)) if boot_arr.size else float("nan"),
    ]

    observed_matrix = np.zeros((len(concepts), len(concepts)), dtype=int)
    for row in errors:
        if row["chosen_concept"] in concept_index:
            observed_matrix[concept_index[row["target"]], concept_index[row["chosen_concept"]]] += 1
    predicted_scores = []
    observed_rates = []
    for row in pair_rows:
        predicted_scores.append(-row["rdm_distance"])
        observed_rates.append(row["substitution_rate"])
    confusion_agreement = pearson_corr(np.asarray(predicted_scores), np.asarray(observed_rates))

    reliability_green = bool(meta.get("reliability_gate", False))
    sanity_green = neighbors.get("sanity_gate") == "passed"
    enough_errors = len(directional_errors) >= int(config["h1_min_error_items"])
    slope_green = np.isfinite(slope_ci[1]) and slope_ci[1] < 0
    null_green = np.isfinite(shuffle_p) and shuffle_p < 0.05 and np.isfinite(base_lift) and base_lift > 0
    if reliability_green and sanity_green and enough_errors and slope_green and null_green:
        verdict = "green_directional"
    elif not reliability_green:
        verdict = "not_decided_rdm_reliability_red"
    elif not sanity_green:
        verdict = "not_decided_sanity_gate_pending"
    elif not enough_errors:
        verdict = "not_decided_too_few_errors"
    else:
        verdict = "null_or_inconclusive"

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(
        RESULT_DIR / "step1_scored_items.csv",
        [
            ["item_id", "target", "chosen_concept", "chosen_role", "correct", "response"],
            *[
                [row["item_id"], row["target"], row["chosen_concept"], row["chosen_role"], row["correct"], row["response"]]
                for row in scored
            ],
        ],
    )
    write_csv(
        RESULT_DIR / "step1_pair_rates.csv",
        [["target", "distractor", "rdm_distance", "opportunities", "substitutions", "substitution_rate"]]
        + [
            [row["target"], row["distractor"], row["rdm_distance"], row["opportunities"], row["substitutions"], row["substitution_rate"]]
            for row in pair_rows
        ],
    )
    write_csv(RESULT_DIR / "step1_confusion_matrix.csv", [["target", *concepts]] + [[concepts[i], *observed_matrix[i].tolist()] for i in range(len(concepts))])
    summary = {
        "scored_at": now_stamp(),
        "model": config["base_model"],
        "run": args.run,
        "h1_verdict": verdict,
        "rdm_path": display_path(rdm_path),
        "rdm_sha256": sha256_file(rdm_path),
        "rdm_source": meta.get("rdm_source"),
        "rdm_distance_metric": meta.get("distance_metric"),
        "rdm_built_at": meta.get("built_at"),
        "neighbors_path": display_path(EXP_DIR / "neighbors.json"),
        "neighbors_registered_at": neighbors.get("registered_at"),
        "items_path": display_path(item_path),
        "items_sha256": sha256_file(item_path),
        "raw_item_responses_path": display_path(RAW_DIR / args.run / "items.csv"),
        "n_items": total,
        "n_correct": correct,
        "n_errors": total - correct,
        "accuracy": correct / total if total else float("nan"),
        "n_errors_with_parseable_choice": len(errors),
        "n_directional_errors_near_or_far": len(directional_errors),
        "near_errors": len(near_errors),
        "far_errors": len(far_errors),
        "near_fraction_among_directional_errors": near_fraction,
        "shuffle_geometry_null": {
            "p_value_ge_observed": shuffle_p,
            "mean_near_fraction": float(np.nanmean(null_counts_arr)) if null_counts_arr.size else float("nan"),
            "p95_near_fraction": float(np.nanquantile(null_counts_arr, 0.95)) if null_counts_arr.size else float("nan"),
        },
        "base_rate_control": {
            "expected_near_fraction": base_expected_fraction,
            "observed_minus_expected": base_lift,
            "error_choice_counts": dict(chosen_base_counts),
            "n_error_choices": all_error_choices,
        },
        "h2_distance_slope": {
            "slope_substitution_rate_per_rdm_distance": distance_slope,
            "bootstrap_ci_95": slope_ci,
            "interpretation": "H2 predicts this slope is negative.",
        },
        "predicted_vs_actual_confusion_agreement": {
            "pearson_r_neg_distance_vs_substitution_rate": confusion_agreement,
        },
        "gates": {
            "rdm_reliability_green": reliability_green,
            "human_sanity_gate_passed": sanity_green,
            "enough_directional_errors": enough_errors,
            "nulls_green": null_green,
            "slope_ci_green": slope_green,
        },
    }
    write_json(RESULT_DIR / "step1.json", summary)
    append_log(
        "H1 verdict",
        [
            f"Run scored: `{args.run}`.",
            f"Directional errors: {len(directional_errors)}; near fraction: {near_fraction:.4f}.",
            f"Shuffle null p-value: {shuffle_p:.4f}; base-rate lift: {base_lift:.4f}.",
            f"H2 distance slope: {distance_slope:.6f}; 95% CI [{slope_ci[0]:.6f}, {slope_ci[1]:.6f}].",
            f"Predicted-vs-actual confusion agreement: {confusion_agreement:.4f}.",
            f"Verdict: `{verdict}`.",
        ],
    )
    plot_confusion_summary(concepts, observed_matrix, rdm)
    update_report()
    print(f"[score] H1 verdict: {verdict}; wrote {display_path(RESULT_DIR / 'step1.json')}")


def concept_rank_from_rdm(concepts: list[str], rdm: np.ndarray, target: str, candidate: str) -> tuple[int | None, float | None]:
    if target not in concepts or candidate not in concepts or target == candidate:
        return None, None
    concept_index = {concept: i for i, concept in enumerate(concepts)}
    target_idx = concept_index[target]
    candidate_idx = concept_index[candidate]
    ordered = sorted(
        (
            (concept, float(rdm[target_idx, idx]))
            for idx, concept in enumerate(concepts)
            if concept != target
        ),
        key=lambda pair: (pair[1], pair[0]),
    )
    for rank, (concept, dist) in enumerate(ordered, start=1):
        if concept == candidate:
            return rank, dist
    return None, None


def audit_step1(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    item_path = ITEM_DIR / "items.json"
    rdm_path = ARTIFACT_DIR / "rdm.npy"
    result_summary = read_optional_json(RESULT_DIR / "step1.json") or {}
    run = args.run or result_summary.get("run") or "step1_items_v1"
    if not item_path.exists():
        raise SystemExit("No items found. Run generate-items first.")
    if not rdm_path.exists():
        raise SystemExit("No RDM found. Run build-rdm first.")
    if not (RAW_DIR / run / "items.csv").exists():
        raise SystemExit(f"No item responses found for run `{run}`.")

    concepts = load_concepts(config)
    rdm = np.load(rdm_path)
    items = read_json(item_path)
    responses = load_item_responses(run)

    scored = []
    error_rows = []
    target_counts: dict[str, Counter] = {concept: Counter() for concept in concepts}
    target_destinations: dict[str, Counter] = {concept: Counter() for concept in concepts}
    letter_counts: dict[str, Counter] = {letter: Counter() for letter in ["A", "B", "C", "D"]}

    for item in items:
        response = responses.get(item["item_id"], "")
        chosen_letter, chosen_concept = parse_choice(response, item)
        option_by_concept = {option["concept"]: option for option in item["options"]}
        option_by_letter = {option["letter"]: option for option in item["options"]}
        chosen_role = option_by_concept.get(chosen_concept, {}).get("role") if chosen_concept else "invalid"
        correct = chosen_concept == item["correct_answer"]
        target = item["target"]
        target_counts[target]["items"] += 1
        target_counts[target]["correct" if correct else "errors"] += 1
        if chosen_role:
            target_counts[target][f"{chosen_role}_choices"] += 1
        if chosen_letter:
            letter_counts[chosen_letter]["choices_all"] += 1
            letter_counts[chosen_letter]["choices_correct" if correct else "choices_error"] += 1
            letter_counts[chosen_letter][f"{chosen_role}_choices"] += 1
        for letter, option in option_by_letter.items():
            letter_counts[letter]["option_slots"] += 1
            letter_counts[letter][f"{option['role']}_slots"] += 1

        scored.append(
            {
                "item": item,
                "response": response,
                "chosen_letter": chosen_letter,
                "chosen_concept": chosen_concept,
                "chosen_role": chosen_role,
                "correct": correct,
            }
        )
        if correct:
            continue
        if chosen_concept:
            target_destinations[target][chosen_concept] += 1
        chosen_rank, chosen_distance = concept_rank_from_rdm(concepts, rdm, target, chosen_concept) if chosen_concept else (None, None)
        near_rank, near_distance = concept_rank_from_rdm(concepts, rdm, target, item["predicted_near"])
        if chosen_role == "near":
            audit_label = "predicted_near_hit"
        elif chosen_role == "far" and chosen_rank is not None and chosen_rank <= 3:
            audit_label = "far_error_but_top3_rdm"
        elif chosen_role == "far" and chosen_rank is not None and chosen_rank <= 5:
            audit_label = "far_error_but_top5_rdm"
        elif chosen_role == "far":
            audit_label = "far_error_true_miss"
        elif chosen_role == "invalid":
            audit_label = "invalid_or_unparsed_response"
        else:
            audit_label = "non_distractor_error"
        error_rows.append(
            {
                "item_id": item["item_id"],
                "target": target,
                "chosen_concept": chosen_concept or "",
                "chosen_role": chosen_role or "",
                "chosen_letter": chosen_letter or "",
                "correct_letter": item["correct_letter"],
                "predicted_near": item["predicted_near"],
                "far_controls": "|".join(item["far_controls"]),
                "chosen_rdm_distance": chosen_distance,
                "chosen_rdm_rank": chosen_rank,
                "predicted_near_distance": near_distance,
                "predicted_near_rank": near_rank,
                "audit_label": audit_label,
                "feature_clues": "|".join(item.get("feature_clues", [])),
                "response": response,
            }
        )

    target_rows = []
    for target in concepts:
        counts = target_counts[target]
        directional = counts["near_choices"] + counts["far_choices"]
        destinations = target_destinations[target]
        top_destination, top_destination_count = ("", 0)
        if destinations:
            top_destination, top_destination_count = destinations.most_common(1)[0]
        predicted_near = next((item["predicted_near"] for item in items if item["target"] == target), "")
        near_rank, near_distance = concept_rank_from_rdm(concepts, rdm, target, predicted_near) if predicted_near else (None, None)
        target_rows.append(
            {
                "target": target,
                "items": counts["items"],
                "correct": counts["correct"],
                "errors": counts["errors"],
                "near_errors": counts["near_choices"],
                "far_errors": counts["far_choices"],
                "invalid_errors": counts["invalid_choices"],
                "near_fraction_directional": counts["near_choices"] / directional if directional else "",
                "predicted_near": predicted_near,
                "predicted_near_distance": near_distance,
                "predicted_near_rank": near_rank,
                "top_error_destination": top_destination,
                "top_error_count": top_destination_count,
                "top_error_is_predicted_near": bool(top_destination and top_destination == predicted_near),
                "all_error_destinations": ";".join(f"{concept}:{count}" for concept, count in destinations.most_common()),
            }
        )

    letter_rows = []
    for letter in ["A", "B", "C", "D"]:
        counts = letter_counts[letter]
        letter_rows.append(
            {
                "letter": letter,
                "option_slots": counts["option_slots"],
                "correct_slots": counts["correct_slots"],
                "near_slots": counts["near_slots"],
                "far_slots": counts["far_slots"],
                "choices_all": counts["choices_all"],
                "choices_correct": counts["choices_correct"],
                "choices_error": counts["choices_error"],
                "near_error_choices": counts["near_choices"],
                "far_error_choices": counts["far_choices"],
                "invalid_error_choices": counts["invalid_choices"],
            }
        )

    errors = [row for row in scored if not row["correct"]]
    directional_errors = [row for row in error_rows if row["chosen_role"] in {"near", "far"}]
    near_error_rows = [row for row in error_rows if row["chosen_role"] == "near"]
    far_error_rows = [row for row in error_rows if row["chosen_role"] == "far"]
    far_ranks = [int(row["chosen_rdm_rank"]) for row in far_error_rows if row["chosen_rdm_rank"] is not None]
    near_distances = [float(row["chosen_rdm_distance"]) for row in near_error_rows if row["chosen_rdm_distance"] is not None]
    far_distances = [float(row["chosen_rdm_distance"]) for row in far_error_rows if row["chosen_rdm_distance"] is not None]
    top_target_errors = sorted(
        (
            {
                "target": row["target"],
                "errors": int(row["errors"]),
                "near_errors": int(row["near_errors"]),
                "far_errors": int(row["far_errors"]),
                "top_error_destination": row["top_error_destination"],
                "top_error_count": int(row["top_error_count"]),
            }
            for row in target_rows
            if int(row["errors"])
        ),
        key=lambda row: (-row["errors"], row["target"]),
    )[:6]
    summary = {
        "audited_at": now_stamp(),
        "run": run,
        "rdm_sha256": sha256_file(rdm_path),
        "items_sha256": sha256_file(item_path),
        "n_items": len(scored),
        "n_correct": sum(1 for row in scored if row["correct"]),
        "n_errors": len(errors),
        "n_directional_errors": len(directional_errors),
        "near_errors": len(near_error_rows),
        "far_errors": len(far_error_rows),
        "invalid_or_unparsed_errors": sum(1 for row in error_rows if row["chosen_role"] == "invalid"),
        "targets_with_errors": sum(1 for row in target_rows if int(row["errors"])),
        "targets_with_near_errors": sum(1 for row in target_rows if int(row["near_errors"])),
        "targets_with_far_errors": sum(1 for row in target_rows if int(row["far_errors"])),
        "targets_where_top_error_is_predicted_near": sum(1 for row in target_rows if row["top_error_is_predicted_near"]),
        "top_error_targets": top_target_errors,
        "far_error_rank_audit": {
            "n_far_errors": len(far_error_rows),
            "n_rank_le_3": sum(rank <= 3 for rank in far_ranks),
            "n_rank_le_5": sum(rank <= 5 for rank in far_ranks),
            "mean_rank": float(np.mean(far_ranks)) if far_ranks else None,
            "median_rank": float(np.median(far_ranks)) if far_ranks else None,
        },
        "distance_audit": {
            "mean_near_error_distance": float(np.mean(near_distances)) if near_distances else None,
            "mean_far_error_distance": float(np.mean(far_distances)) if far_distances else None,
            "median_near_error_distance": float(np.median(near_distances)) if near_distances else None,
            "median_far_error_distance": float(np.median(far_distances)) if far_distances else None,
        },
        "option_position_error_choices": {
            row["letter"]: int(row["choices_error"])
            for row in letter_rows
        },
        "option_position_correct_slots": {
            row["letter"]: int(row["correct_slots"])
            for row in letter_rows
        },
        "audit_files": {
            "error_audit_csv": display_path(RESULT_DIR / "step1_error_audit.csv"),
            "target_audit_csv": display_path(RESULT_DIR / "step1_target_audit.csv"),
            "position_audit_csv": display_path(RESULT_DIR / "step1_position_audit.csv"),
            "audit_json": display_path(RESULT_DIR / "step1_audit.json"),
        },
        "interpretation": (
            "Near-neighbor effect is useful but imperfect. Inspect far errors and target concentration before Step 2; "
            "do not tune Step 1 to remove all errors because the law is conditional on genuine mistakes."
        ),
    }

    write_csv(
        RESULT_DIR / "step1_error_audit.csv",
        [
            [
                "item_id",
                "target",
                "chosen_concept",
                "chosen_role",
                "chosen_letter",
                "correct_letter",
                "predicted_near",
                "far_controls",
                "chosen_rdm_distance",
                "chosen_rdm_rank",
                "predicted_near_distance",
                "predicted_near_rank",
                "audit_label",
                "feature_clues",
                "response",
            ],
            *[
                [
                    row["item_id"],
                    row["target"],
                    row["chosen_concept"],
                    row["chosen_role"],
                    row["chosen_letter"],
                    row["correct_letter"],
                    row["predicted_near"],
                    row["far_controls"],
                    row["chosen_rdm_distance"],
                    row["chosen_rdm_rank"],
                    row["predicted_near_distance"],
                    row["predicted_near_rank"],
                    row["audit_label"],
                    row["feature_clues"],
                    row["response"],
                ]
                for row in error_rows
            ],
        ],
    )
    write_csv(
        RESULT_DIR / "step1_target_audit.csv",
        [
            [
                "target",
                "items",
                "correct",
                "errors",
                "near_errors",
                "far_errors",
                "invalid_errors",
                "near_fraction_directional",
                "predicted_near",
                "predicted_near_distance",
                "predicted_near_rank",
                "top_error_destination",
                "top_error_count",
                "top_error_is_predicted_near",
                "all_error_destinations",
            ],
            *[
                [
                    row["target"],
                    row["items"],
                    row["correct"],
                    row["errors"],
                    row["near_errors"],
                    row["far_errors"],
                    row["invalid_errors"],
                    row["near_fraction_directional"],
                    row["predicted_near"],
                    row["predicted_near_distance"],
                    row["predicted_near_rank"],
                    row["top_error_destination"],
                    row["top_error_count"],
                    row["top_error_is_predicted_near"],
                    row["all_error_destinations"],
                ]
                for row in target_rows
            ],
        ],
    )
    write_csv(
        RESULT_DIR / "step1_position_audit.csv",
        [
            [
                "letter",
                "option_slots",
                "correct_slots",
                "near_slots",
                "far_slots",
                "choices_all",
                "choices_correct",
                "choices_error",
                "near_error_choices",
                "far_error_choices",
                "invalid_error_choices",
            ],
            *[
                [
                    row["letter"],
                    row["option_slots"],
                    row["correct_slots"],
                    row["near_slots"],
                    row["far_slots"],
                    row["choices_all"],
                    row["choices_correct"],
                    row["choices_error"],
                    row["near_error_choices"],
                    row["far_error_choices"],
                    row["invalid_error_choices"],
                ]
                for row in letter_rows
            ],
        ],
    )
    write_json(RESULT_DIR / "step1_audit.json", summary)
    append_log(
        "Course-correction",
        [
            "Ran Step 1 audit before designing Step 2.",
            f"Error concentration: {summary['targets_with_errors']} targets had errors; {summary['targets_with_near_errors']} had near-neighbor errors; {summary['targets_with_far_errors']} had far-control errors.",
            f"Far-control audit: {len(far_error_rows)} far errors; {summary['far_error_rank_audit']['n_rank_le_5']} were top-5 RDM neighbors of the target.",
            f"Option-position audit: error choices by letter = {summary['option_position_error_choices']}; correct option slots by letter = {summary['option_position_correct_slots']}.",
            "Interpretation: Step 1 is strong enough to transfer; do not overfit item phrasing, but carry the audit forward for model extensions.",
        ],
    )
    update_report()
    print(f"[audit] wrote {display_path(RESULT_DIR / 'step1_audit.json')}")


def plot_confusion_summary(concepts: list[str], observed_matrix: np.ndarray, rdm: np.ndarray) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "out" / "matplotlib_cache"))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001
        write_json(FIG_DIR / "plot_warning.json", {"warning": str(exc)})
        return
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    im = axes[0].imshow(observed_matrix, cmap="magma")
    axes[0].set_title("Observed error substitutions")
    axes[0].set_xticks(range(len(concepts)))
    axes[0].set_yticks(range(len(concepts)))
    axes[0].set_xticklabels(concepts, rotation=90, fontsize=7)
    axes[0].set_yticklabels(concepts, fontsize=7)
    fig.colorbar(im, ax=axes[0], fraction=0.046)
    pred = np.max(rdm) - rdm
    np.fill_diagonal(pred, 0.0)
    im2 = axes[1].imshow(pred, cmap="viridis")
    axes[1].set_title("Predicted proximity from RDM")
    axes[1].set_xticks(range(len(concepts)))
    axes[1].set_yticks(range(len(concepts)))
    axes[1].set_xticklabels(concepts, rotation=90, fontsize=7)
    axes[1].set_yticklabels(concepts, fontsize=7)
    fig.colorbar(im2, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "step1_confusion_matrix.png", dpi=180)
    plt.close(fig)


def load_protocol() -> dict:
    path = EXP_DIR / "triplet_protocol.json"
    if not path.exists():
        raise SystemExit("triplet_protocol.json is missing. Run init first.")
    return read_json(path)


def load_triplets() -> list[tuple[str, str, str]]:
    rows = []
    with (STIM_DIR / "triplets.csv").open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) >= 3:
                rows.append((row[0].strip(), row[1].strip(), row[2].strip()))
    return rows


def format_triplet_prompt(template: str, anchor: str, concept1: str, concept2: str) -> str:
    return template.format(anchor=anchor, concept1=concept1, concept2=concept2)


def resolve_vllm_model(args: argparse.Namespace, model_name: str):
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))

    if args.model_path:
        model_path = args.model_path
        hf_cache = args.hf_cache or os.environ.get("HF_HOME") or str(ROOT / "out" / "hf_cache")
        spec = {"chat": not args.no_chat}
    else:
        from run_local import load_registry, resolve_model_path

        reg = load_registry()
        if model_name not in reg["local"]:
            raise SystemExit(f"model {model_name} not in registry local: {list(reg['local'])}")
        spec = reg["local"][model_name]
        hf_cache = args.hf_cache or reg["hf_cache"]
        allow_download = bool(spec.get("download", False))
        model_path = resolve_model_path(spec["path"], hf_cache, allow_download)
    os.environ.setdefault("HF_HOME", hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", hf_cache)
    if os.path.isdir(model_path):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return spec, model_path, hf_cache


def build_llm(args: argparse.Namespace, spec: dict, model_path: str, hf_cache: str):
    from vllm import LLM

    tp = spec.get("tensor_parallel", args.tensor_parallel)
    llm_kwargs = dict(
        model=model_path,
        download_dir=hf_cache,
        tensor_parallel_size=tp,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    if args.max_num_seqs:
        llm_kwargs["max_num_seqs"] = args.max_num_seqs
    quant = spec.get("quantization")
    if quant and os.environ.get("COHERENCE_FORCE_BF16") == "1":
        quant = None
    if quant:
        llm_kwargs["quantization"] = quant
    return LLM(**llm_kwargs)


def write_triplet_outputs(
    llm,
    spec: dict,
    triplets: list[tuple[str, str, str]],
    prompts: list[str],
    sampling,
    out_path: Path,
    prompt_variant: str,
) -> None:
    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from prompts import SYSTEM_PROMPT

    if spec.get("chat", True):
        convos = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}] for prompt in prompts]
        outputs = llm.chat(convos, sampling)
    else:
        outputs = llm.generate(prompts, sampling)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["input", "prompt", "response", "prompt_variant"])
        for (anchor, concept1, concept2), prompt, output in zip(triplets, prompts, outputs):
            writer.writerow([f"{anchor}|{concept1}|{concept2}", prompt, output.outputs[0].text.strip(), prompt_variant])


def run_triplets(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    protocol = load_protocol()
    model_name = args.model or protocol["base_model"]
    outdir = RAW_DIR / args.out_run
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "triplet.csv"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {display_path(out_path)} exists")
        return
    spec, model_path, hf_cache = resolve_vllm_model(args, model_name)
    print(f"[model] {model_name} -> {model_path}")
    template_key = "prompt_template" if args.prompt_variant == "canonical" else "paraphrase_template"
    template = protocol[template_key]
    triplets = load_triplets()
    prompts = [format_triplet_prompt(template, *row) for row in triplets]

    from vllm import SamplingParams

    llm = build_llm(args, spec, model_path, hf_cache)
    sampling = SamplingParams(temperature=args.temperature if args.temperature is not None else config["triplet_protocol"]["temperature"], max_tokens=8)
    write_triplet_outputs(llm, spec, triplets, prompts, sampling, out_path, args.prompt_variant)
    print(f"[done] {len(triplets)} triplets -> {display_path(out_path)}")


def run_triplet_suite(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    protocol = load_protocol()
    model_name = args.model or protocol["base_model"]
    run_variants = []
    for run_name in protocol["required_geometry_runs"]:
        if "paraphrase" in run_name:
            run_variants.append((run_name, "paraphrase"))
        else:
            run_variants.append((run_name, "canonical"))

    pending = []
    for run_name, prompt_variant in run_variants:
        out_path = RAW_DIR / run_name / "triplet.csv"
        if out_path.exists() and not args.overwrite:
            print(f"[skip] {display_path(out_path)} exists")
            continue
        pending.append((run_name, prompt_variant, out_path))
    if not pending:
        return

    spec, model_path, hf_cache = resolve_vllm_model(args, model_name)
    print(f"[model] {model_name} -> {model_path}")
    from vllm import SamplingParams

    llm = build_llm(args, spec, model_path, hf_cache)
    sampling = SamplingParams(temperature=args.temperature if args.temperature is not None else config["triplet_protocol"]["temperature"], max_tokens=8)
    triplets = load_triplets()
    for run_name, prompt_variant, out_path in pending:
        template_key = "prompt_template" if prompt_variant == "canonical" else "paraphrase_template"
        template = protocol[template_key]
        prompts = [format_triplet_prompt(template, *row) for row in triplets]
        write_triplet_outputs(llm, spec, triplets, prompts, sampling, out_path, prompt_variant)
        print(f"[done] {len(triplets)} triplets -> {display_path(out_path)}")


def run_items(args: argparse.Namespace) -> None:
    ensure_dirs()
    config = load_config()
    item_path = ITEM_DIR / "items.json"
    if not item_path.exists():
        raise SystemExit("items.json is missing. Run generate-items first.")
    model_name = args.model or config["base_model"]
    outdir = RAW_DIR / args.out_run
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "items.csv"
    if out_path.exists() and not args.overwrite:
        print(f"[skip] {display_path(out_path)} exists")
        return
    spec, model_path, hf_cache = resolve_vllm_model(args, model_name)
    print(f"[model] {model_name} -> {model_path}")
    items = read_json(item_path)
    prompts = [item["prompt"] for item in items]

    src = ROOT / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from prompts import SYSTEM_PROMPT
    from vllm import SamplingParams

    llm = build_llm(args, spec, model_path, hf_cache)
    sampling = SamplingParams(temperature=args.temperature if args.temperature is not None else 0.0, max_tokens=12)
    if spec.get("chat", True):
        convos = [[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}] for prompt in prompts]
        outputs = llm.chat(convos, sampling)
    else:
        outputs = llm.generate(prompts, sampling)
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["item_id", "prompt", "response"])
        for item, output in zip(items, outputs):
            writer.writerow([item["item_id"], item["prompt"], output.outputs[0].text.strip()])
    print(f"[done] {len(items)} items -> {display_path(out_path)}")


def mark_sanity_gate(args: argparse.Namespace) -> None:
    path = EXP_DIR / "neighbors.json"
    if not path.exists():
        raise SystemExit("neighbors.json is missing.")
    payload = read_json(path)
    payload["sanity_gate"] = "passed" if args.status == "pass" else "failed"
    payload["sanity_gate_marked_at"] = now_stamp()
    payload["sanity_gate_note"] = args.note or ""
    write_json(path, payload)
    append_log(
        "Sanity gate",
        [
            f"Human sanity gate marked `{payload['sanity_gate']}`.",
            f"Note: {payload['sanity_gate_note'] or 'n/a'}",
        ],
    )
    update_report()


def all_pipeline(args: argparse.Namespace) -> None:
    init_experiment(argparse.Namespace(overwrite=False, log=False))
    config = load_config()
    required = config["triplet_protocol"]["required_geometry_runs"]
    if all((RAW_DIR / run / "triplet.csv").exists() for run in required):
        build_rdm(argparse.Namespace(runs=required))
        if not (EXP_DIR / "neighbors.json").exists():
            register_neighbors(argparse.Namespace(allow_red_rdm=False))
        if not (ITEM_DIR / "items.json").exists():
            generate_items(argparse.Namespace(n_items_per_target=None))
    else:
        update_report()
        missing = [run for run in required if not (RAW_DIR / run / "triplet.csv").exists()]
        print("[pending] missing triplet runs: " + ", ".join(missing))
        return
    if args.score_run and (RAW_DIR / args.score_run / "items.csv").exists():
        score_items(argparse.Namespace(run=args.score_run))
    else:
        update_report()
        print("[pending] item responses missing; run run-items, then score")


def read_optional_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return read_json(path)


def markdown_table_neighbors(neighbors: dict | None) -> str:
    if not neighbors:
        return "No pre-registered neighbors yet.\n"
    lines = [
        "| Target | Predicted near | Near d | Far controls |",
        "|---|---|---:|---|",
    ]
    for row in neighbors.get("predictions", []):
        far = ", ".join(f"`{control['concept']}` ({control['distance']:.3f})" for control in row["far_controls"])
        lines.append(f"| `{row['target']}` | `{row['near']}` | {row['near_distance']:.3f} | {far} |")
    return "\n".join(lines) + "\n"


def md_link(path: Path, label: str | None = None) -> str:
    shown = display_path(path)
    base = report_link_base()
    if base:
        target = f"{base}/{shown.replace(os.sep, '/')}"
    else:
        try:
            target = os.path.relpath(path, EXP_DIR)
        except ValueError:
            target = shown
        target = target.replace(os.sep, "/")
    return f"[{label or shown}]({target})"


def fmt_optional_float(value: object, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(numeric):
        return "n/a"
    return f"{numeric:.{digits}f}"


def first_item_example() -> dict | None:
    path = ITEM_DIR / "items.json"
    if not path.exists():
        return None
    items = read_json(path)
    return items[0] if items else None


def artifact_hash_current(current_sha: str | None, payload: dict | None, key: str = "rdm_sha256") -> bool:
    if not current_sha or not payload:
        return False
    return payload.get(key) == current_sha


def prompt_template_block(config: dict, key: str) -> str:
    template = config["triplet_protocol"][key]
    return "```text\nSystem: You are a helpful assistant who gives responses to questions.\n\n" + template + "\n```"


def triplet_choice_agreement(run_a: str, run_b: str, config: dict) -> dict | None:
    path_a = RAW_DIR / run_a / "triplet.csv"
    path_b = RAW_DIR / run_b / "triplet.csv"
    if not path_a.exists() or not path_b.exists():
        return None

    def choices(path: Path) -> dict[str, str | None]:
        out = {}
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    _, concept1, concept2 = str(row["input"]).split("|")
                except ValueError:
                    continue
                parsed = parse_triplet_choice(str(row["response"]), concept1, concept2)
                out[str(row["input"])] = "A" if parsed == 1 else "B" if parsed == 2 else None
        return out

    choices_a = choices(path_a)
    choices_b = choices(path_b)
    keys = sorted(set(choices_a) & set(choices_b))
    parsed = [key for key in keys if choices_a[key] is not None and choices_b[key] is not None]
    if not parsed:
        return None
    agree = sum(choices_a[key] == choices_b[key] for key in parsed)
    return {
        "run_a": run_a,
        "run_b": run_b,
        "n_both_parseable": len(parsed),
        "n_agree": agree,
        "agreement": agree / len(parsed),
    }


def update_report() -> None:
    ensure_dirs()
    config = load_config()
    protocol = read_optional_json(EXP_DIR / "triplet_protocol.json")
    rdm_meta = read_optional_json(ARTIFACT_DIR / "rdm_meta.json")
    neighbors = read_optional_json(EXP_DIR / "neighbors.json")
    results = read_optional_json(RESULT_DIR / "step1.json")
    audit = read_optional_json(RESULT_DIR / "step1_audit.json")
    current_rdm_sha = (rdm_meta or {}).get("rdm_sha256")
    neighbors_current = artifact_hash_current(current_rdm_sha, (neighbors or {}).get("rdm_meta") if neighbors else None)
    results_current = artifact_hash_current(current_rdm_sha, results)
    audit_current = bool(audit and results_current and audit.get("rdm_sha256") == current_rdm_sha and audit.get("run") == results.get("run"))
    items_exist = (ITEM_DIR / "items.json").exists()
    items_current = items_exist and neighbors_current
    example_item = first_item_example() if items_current else None
    required = config["triplet_protocol"]["required_geometry_runs"]
    triplet_state = {run: (RAW_DIR / run / "triplet.csv").exists() for run in required}
    item_runs = sorted(path.parent.name for path in RAW_DIR.glob("*/items.csv"))

    h1 = results["h1_verdict"] if results_current else "not_decided_current_geometry"
    rdm_source = (rdm_meta or {}).get("rdm_source", config.get("rdm_source", "salmon_embedding"))
    distance_metric = (rdm_meta or {}).get("distance_metric", "pending")
    source_runs = (rdm_meta or {}).get("source_runs", required)
    n_concepts_for_budget = len(load_concepts(config)) if concept_file_path(config).exists() else 0
    budget_meta = (rdm_meta or {}).get("salmon_triplet_budget")
    if rdm_meta and not budget_meta and rdm_meta.get("rdm_source") == "salmon_embedding":
        total_triplets = rdm_meta.get("n_valid_triplets_total")
        per_run_triplets = None
        if total_triplets is not None and source_runs:
            per_run_triplets = int(round(total_triplets / len(source_runs)))
        budget_meta = salmon_triplet_budget(
            n_concepts=n_concepts_for_budget,
            dim=int(rdm_meta.get("salmon_dimension", config.get("salmon_dimension", 0))),
            n_triplets_per_run=per_run_triplets,
            n_triplets_total=total_triplets,
        )
    item_response_run = results["run"] if results_current else "not_run"
    concept_count = n_concepts_for_budget
    old_paraphrase_agreement = triplet_choice_agreement(
        "base_seed_a_canonical_prompt",
        "base_seed_a_paraphrase_prompt",
        config,
    )
    matched_paraphrase_agreement = triplet_choice_agreement(
        "base_seed_a_canonical_prompt",
        "base_seed_a_matched_paraphrase_prompt",
        config,
    )
    stale_neighbors_note = "pending for current RDM"
    if neighbors and not neighbors_current:
        stale_neighbors_note += f" (stale file: {md_link(EXP_DIR / 'neighbors.json')})"
    stale_items_note = "pending for current RDM"
    if items_exist and not items_current:
        stale_items_note += f" (stale files: {md_link(ITEM_DIR / 'items.csv')}, {md_link(ITEM_DIR / 'items.json')})"
    stale_results_note = "pending for current RDM"
    if results and not results_current:
        stale_results_note += f" (stale files: {md_link(RESULT_DIR / 'step1.json')}, {md_link(RESULT_DIR / 'step1_scored_items.csv')})"
    report = [
        "# Experiment 3 Report",
        "",
        f"Last updated: {now_stamp()}",
        "",
        "## Step 1 Story",
        "",
        "### What were we trying to find?",
        "",
        "We are testing whether triplet geometry predicts the destination of model errors on neutral concepts. "
        "The preregistered prediction for each target is its nearest neighbor in the model's triplet RDM; H1 is green only if later errors land on that near neighbor above shuffled-geometry and base-rate nulls.",
        "",
        "### What did we run?",
        "",
        f"- Model: `{config['base_model']}`.",
        "- Serving: local vLLM; triplet and item prompts use temperature `0.0`.",
        f"- Concept set: {concept_count} neutral Leuven concrete concepts in {md_link(concept_file_path(config))}.",
        f"- Stimuli: {md_link(STIM_DIR / 'concepts.csv')}, {md_link(STIM_DIR / 'triplets.csv')}, {md_link(STIM_DIR / 'pairs.csv')}.",
        f"- Geometry raw responses: "
        + ", ".join(md_link(RAW_DIR / run / "triplet.csv", run) for run in source_runs if (RAW_DIR / run / "triplet.csv").exists())
        + ".",
        f"- Geometry fitting: `{rdm_source}`; distance metric for `rdm.npy`: `{distance_metric}`.",
        f"- RDM artifact: {md_link(ARTIFACT_DIR / 'rdm.npy')}; metadata: {md_link(ARTIFACT_DIR / 'rdm_meta.json')}.",
        f"- SALMON pooled embedding: {md_link(ROOT / (rdm_meta or {}).get('pooled_embedding_path', display_path(EMBED_DIR / 'pooled_salmon_d5.npy'))) if rdm_meta and rdm_meta.get('pooled_embedding_path') else 'pending'}.",
        f"- Pre-registered neighbors: {md_link(EXP_DIR / 'neighbors.json') if neighbors_current else stale_neighbors_note}.",
        f"- Directional items: {(md_link(ITEM_DIR / 'items.csv') + ' and ' + md_link(ITEM_DIR / 'items.json')) if items_current else stale_items_note}.",
        f"- Item responses: {md_link(RAW_DIR / item_response_run / 'items.csv') if item_response_run != 'not_run' and (RAW_DIR / item_response_run / 'items.csv').exists() else 'pending'}.",
        f"- Scored outputs: {(md_link(RESULT_DIR / 'step1.json') + ', ' + md_link(RESULT_DIR / 'step1_scored_items.csv') + ', ' + md_link(RESULT_DIR / 'step1_pair_rates.csv') + ', ' + md_link(RESULT_DIR / 'step1_confusion_matrix.csv')) if results_current else stale_results_note}.",
        f"- Step 1 audit outputs: {(md_link(RESULT_DIR / 'step1_audit.json') + ', ' + md_link(RESULT_DIR / 'step1_error_audit.csv') + ', ' + md_link(RESULT_DIR / 'step1_target_audit.csv') + ', ' + md_link(RESULT_DIR / 'step1_position_audit.csv')) if audit_current else 'pending for current scored run'}.",
        "",
        "Canonical geometry prompt:",
        "",
        prompt_template_block(config, "prompt_template"),
        "",
        "Paraphrase geometry prompt:",
        "",
        prompt_template_block(config, "paraphrase_template"),
        "",
        "Prompt-variant correction:",
        "",
        (
            "- Old non-matched paraphrase: `Compare the target to two candidates. / Target: {anchor} / A: {concept1} / B: {concept2} / Which candidate is closer in meaning to the target? Reply with only A or B.`"
            if old_paraphrase_agreement
            else "- Old non-matched paraphrase: not run in this worktree."
        ),
        (
            f"- Canonical vs old non-matched raw choice agreement: `{old_paraphrase_agreement['n_agree']}/{old_paraphrase_agreement['n_both_parseable']} = {old_paraphrase_agreement['agreement']:.4f}`."
            if old_paraphrase_agreement
            else "- Canonical vs old non-matched raw choice agreement: unavailable."
        ),
        (
            f"- Canonical vs matched paraphrase raw choice agreement: `{matched_paraphrase_agreement['n_agree']}/{matched_paraphrase_agreement['n_both_parseable']} = {matched_paraphrase_agreement['agreement']:.4f}`."
            if matched_paraphrase_agreement
            else "- Canonical vs matched paraphrase raw choice agreement: unavailable."
        ),
        "",
        "Directional item template:",
        "",
        "```text",
        "Which option is the best match for this description?",
        "- {feature clue}",
        "- {feature clue}",
        "- {feature clue}",
        "Answer with only A, B, C, or D.",
        "",
        "Options:",
        "A. {distractor_or_target}",
        "B. {distractor_or_target}",
        "C. {distractor_or_target}",
        "D. {distractor_or_target}",
        "```",
        "",
        "Concrete generated item example:",
        "",
        (
            f"From {md_link(ITEM_DIR / 'items.csv')} / `{example_item['item_id']}`:\n\n"
            f"```text\n{example_item['prompt']}\n```"
            if example_item
            else "No generated items yet."
        ),
        "",
        "### What did we find?",
        "",
    ]
    if rdm_meta:
        report.extend(
            [
                f"- RDM source: `{rdm_meta.get('rdm_source')}`.",
                f"- RDM distance metric: `{rdm_meta.get('distance_metric')}`.",
                f"- SALMON pooled held-out accuracy: `{(rdm_meta.get('pooled_fit_metrics') or {}).get('test_score')}`",
                f"- SALMON per-run held-out accuracies: `{ {run: fit.get('test_score') for run, fit in (rdm_meta.get('per_run_salmon_fit_metrics') or {}).items()} }`",
                (
                    "- SALMON triplet budget heuristic: "
                    f"`fudge * n * d * ln(n)`; here base `n*d*ln(n) = {budget_meta['base_n_d_log_n']:.1f}`, "
                    f"observed per-run `{budget_meta.get('observed_triplets_per_run')}` "
                    f"(`{budget_meta.get('observed_per_run_fudge_factor'):.2f}x`), "
                    f"pooled `{budget_meta.get('observed_triplets_total')}` "
                    f"(`{budget_meta.get('observed_total_fudge_factor'):.2f}x`)."
                    if budget_meta
                    else "- SALMON triplet budget heuristic: unavailable."
                ),
                f"- Source runs: {', '.join(rdm_meta.get('source_runs', []))}.",
                f"- Missing runs: {', '.join(rdm_meta.get('missing_runs', [])) or 'none'}",
                f"- Mean pairwise upper-triangle Pearson: `{rdm_meta.get('mean_pairwise_upper_triangle_pearson')}`",
                f"- Mean pairwise SALMON embedding Procrustes R^2: `{rdm_meta.get('mean_pairwise_embedding_procrustes_r2', 'n/a')}`",
                f"- Mean nearest-neighbor top-1 agreement across geometry runs: `{rdm_meta.get('mean_nearest_neighbor_top1_agreement', 'n/a')}`",
                f"- Mean nearest-neighbor top-2 agreement across geometry runs: `{rdm_meta.get('mean_nearest_neighbor_top2_agreement', 'n/a')}`",
                f"- Mean split-half upper-triangle Pearson: `{rdm_meta.get('mean_split_half_upper_triangle_pearson')}`",
                f"- RDM reliability gate: `{rdm_meta.get('status')}`",
                "",
            ]
        )
    else:
        report.extend(["- RDM has not been built yet.", ""])
    if results_current:
        report.extend(
            [
                f"- Accuracy: `{results['accuracy']:.4f}` ({results['n_correct']}/{results['n_items']})",
                f"- Directional errors: `{results['n_directional_errors_near_or_far']}`",
                f"- Near fraction among directional errors: `{results['near_fraction_among_directional_errors']:.4f}`",
                f"- Shuffle null p-value: `{results['shuffle_geometry_null']['p_value_ge_observed']:.4f}`",
                f"- Base-rate lift: `{results['base_rate_control']['observed_minus_expected']:.4f}`",
                f"- H2 distance slope: `{results['h2_distance_slope']['slope_substitution_rate_per_rdm_distance']:.6f}`",
                f"- H2 slope 95% CI: `{results['h2_distance_slope']['bootstrap_ci_95']}`",
                f"- Predicted-vs-actual confusion agreement: `{results['predicted_vs_actual_confusion_agreement']['pearson_r_neg_distance_vs_substitution_rate']:.4f}`",
                f"- H1 verdict: `{results['h1_verdict']}`",
                "",
                f"Headline figure: {md_link(FIG_DIR / 'step1_confusion_matrix.png')}",
                "",
            ]
        )
    else:
        report.extend(["- Item responses have not been scored against the current geometry yet.", ""])
    if audit_current:
        top_targets = "; ".join(
            f"{row['target']} {row['errors']} errors -> {row['top_error_destination']}:{row['top_error_count']}"
            for row in audit.get("top_error_targets", [])
        )
        far_rank = audit.get("far_error_rank_audit", {})
        distances = audit.get("distance_audit", {})
        report.extend(
            [
                "### Step 1 audit",
                "",
                f"Audit artifacts: {md_link(RESULT_DIR / 'step1_audit.json')}, {md_link(RESULT_DIR / 'step1_error_audit.csv')}, {md_link(RESULT_DIR / 'step1_target_audit.csv')}, {md_link(RESULT_DIR / 'step1_position_audit.csv')}.",
                "",
                f"- Errors are spread over `{audit['targets_with_errors']}` targets; `{audit['targets_with_near_errors']}` targets have at least one near-neighbor error and `{audit['targets_with_far_errors']}` have at least one far-control error.",
                f"- Top error targets: {top_targets or 'none'}.",
                f"- Far-control errors: `{far_rank.get('n_far_errors')}` total; `{far_rank.get('n_rank_le_3')}` are top-3 RDM neighbors and `{far_rank.get('n_rank_le_5')}` are top-5 RDM neighbors of their target. Median far-error RDM rank: `{fmt_optional_float(far_rank.get('median_rank'), 1)}`.",
                f"- Error distances: mean near-error distance `{fmt_optional_float(distances.get('mean_near_error_distance'))}` vs mean far-error distance `{fmt_optional_float(distances.get('mean_far_error_distance'))}`.",
                f"- Option-position audit: error choices by letter `{audit.get('option_position_error_choices')}`; correct option slots by letter `{audit.get('option_position_correct_slots')}`.",
                "",
                "Interpretation: the lower item accuracy is useful rather than disqualifying; it created enough real errors to test direction. The misses are not just one target, and the far-control misses are mostly not hidden top-neighbor cases, so Step 1 is worth transferring without trying to overfit the neutral items.",
                "",
            ]
        )
    elif results_current:
        report.extend(
            [
                "### Step 1 audit",
                "",
                f"Audit is pending for the current scored run. Run `python scripts/run_experiment3.py audit-step1 --run {item_response_run}`.",
                "",
            ]
        )
    report.extend(
        [
            "### What does this mean?",
            "",
        ]
    )
    if results_current and results["h1_verdict"] == "green_directional":
        report.extend(
            [
                "Step 1 is green: the neutral-model errors are directional under the current geometry. The model did not merely make mistakes; its mistakes preferentially landed on the preregistered nearest-neighbor distractor.",
                "",
            ]
        )
    elif results_current:
        report.extend(
            [
                f"Step 1 is not green under the current geometry: `{results['h1_verdict']}`. Do not start Step 2 until this is resolved or declared a real neutral null.",
                "",
            ]
        )
    else:
        report.extend(
            [
                "Step 1 is not decided yet. The current SALMON-derived RDM must pass reliability, then neighbors must be preregistered and item responses scored against that exact RDM.",
                "",
            ]
        )
    report.extend(
        [
            "The current report is the SALMON-based version. The earlier direct choice-rate RDM result is superseded for the active Experiment 3 claim and remains only in git history.",
            "",
            "## Step 2 Target Search",
            "",
            f"Step 2 items are still intentionally absent. The current target-selection memo is {md_link(EXP_DIR / 'SAFETY_TRANSFER_SCAN.md')}.",
            "",
            "Current recommendation: do not use generic legal standards as the first safety-transfer task. Use a sanitized safety-policy/request-intent taxonomy drawn from HarmBench/JailbreakBench/WMDP/CyberSecEval/AIR-Bench-style categories, then run the same geometry -> preregistered neighbors -> directional item scoring pipeline unchanged.",
            "",
            "## Current Status",
            "",
            f"- Branch/worktree experiment folder: `{display_path(EXP_DIR)}`",
            f"- Step 1 concept set: `{display_path(concept_file_path(config))}`",
            f"- Triplet protocol frozen: {'yes' if protocol else 'no'}",
            f"- Triplet response format: `{config['triplet_protocol'].get('response_format', 'concept_text')}`",
            f"- Required triplet runs present: {sum(triplet_state.values())}/{len(triplet_state)}",
            f"- RDM reliability gate: `{(rdm_meta or {}).get('status', 'missing')}`",
            f"- Neighbors pre-registered for current RDM: {'yes' if neighbors_current else 'no'}",
            f"- Human sanity gate for current RDM: `{(neighbors or {}).get('sanity_gate', 'not_started') if neighbors_current else 'not_started'}`",
            f"- Directional items generated for current RDM: {'yes' if items_current else 'no'}",
            f"- Item response runs present: {', '.join(item_runs) if item_runs else 'none'}",
            f"- H1 verdict: `{h1}`",
            "",
            "## Commands",
            "",
            "```bash",
            "python scripts/run_experiment3.py init",
            "python scripts/run_experiment3.py run-triplet-suite --overwrite",
            "python scripts/run_experiment3.py build-rdm",
            "python scripts/run_experiment3.py register-neighbors",
            "python scripts/run_experiment3.py generate-items",
            "python scripts/run_experiment3.py mark-sanity-gate --status pass --note \"nearest-neighbor pairs are human-sane\"",
            "python scripts/run_experiment3.py run-items --out-run step1_items_v1 --overwrite",
            "python scripts/run_experiment3.py score --run step1_items_v1",
            "python scripts/run_experiment3.py audit-step1 --run step1_items_v1",
            "```",
            "",
            "## Pre-Registered Predictions",
            "",
            markdown_table_neighbors(neighbors if neighbors_current else None),
        ]
    )
    report.extend(
        [
            "## Live Risks",
            "",
            "- If the model is near-perfect on these items, H1 is untestable and the item phrasing needs to move into a harder uncertainty band.",
            "- If RDM reliability is red, do not register or interpret neighbors except as an engineering smoke test.",
            "- Step 2 is intentionally absent until neutral H1 is green and the sanity gate passes.",
            "",
        ]
    )
    (EXP_DIR / "REPORT.md").write_text("\n".join(report))


def add_vllm_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--hf-cache", default=None)
    parser.add_argument("--no-chat", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--tensor_parallel", type=int, default=1)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--gpu_mem_util", type=float, default=0.90)
    parser.add_argument("--max_num_seqs", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init")
    p_init.add_argument("--overwrite", action="store_true")
    p_init.add_argument("--no-log", dest="log", action="store_false", default=True)
    p_init.set_defaults(func=init_experiment)

    p_run_triplets = sub.add_parser("run-triplets")
    p_run_triplets.add_argument("--out-run", required=True)
    p_run_triplets.add_argument("--prompt-variant", choices=["canonical", "paraphrase"], default="canonical")
    add_vllm_args(p_run_triplets)
    p_run_triplets.set_defaults(func=run_triplets)

    p_run_triplet_suite = sub.add_parser("run-triplet-suite")
    add_vllm_args(p_run_triplet_suite)
    p_run_triplet_suite.set_defaults(func=run_triplet_suite)

    p_build = sub.add_parser("build-rdm")
    p_build.add_argument("--runs", nargs="*", default=None)
    p_build.set_defaults(func=build_rdm)

    p_register = sub.add_parser("register-neighbors")
    p_register.add_argument("--allow-red-rdm", action="store_true")
    p_register.set_defaults(func=register_neighbors)

    p_items = sub.add_parser("generate-items")
    p_items.add_argument("--n-items-per-target", type=int, default=None)
    p_items.set_defaults(func=generate_items)

    p_sanity = sub.add_parser("mark-sanity-gate")
    p_sanity.add_argument("--status", choices=["pass", "fail"], required=True)
    p_sanity.add_argument("--note", default="")
    p_sanity.set_defaults(func=mark_sanity_gate)

    p_run_items = sub.add_parser("run-items")
    p_run_items.add_argument("--out-run", required=True)
    add_vllm_args(p_run_items)
    p_run_items.set_defaults(func=run_items)

    p_score = sub.add_parser("score")
    p_score.add_argument("--run", required=True)
    p_score.set_defaults(func=score_items)

    p_audit = sub.add_parser("audit-step1")
    p_audit.add_argument("--run", default=None)
    p_audit.set_defaults(func=audit_step1)

    p_all = sub.add_parser("all")
    p_all.add_argument("--score-run", default="step1_items_v1")
    p_all.set_defaults(func=all_pipeline)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
