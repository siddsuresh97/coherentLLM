"""Analyze lm-eval TruthfulQA MC2 log-sample diagnostics.

The lm-eval aggregate score is useful but too compressed for debugging why a
model loses TruthfulQA. This script reads ``--log_samples`` JSONL files and
decomposes each question into truthful-answer mass and false-answer pressure.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_ROOT = ROOT / "results" / "sft_eval" / "wide_bench_diagnostics" / "truthfulqa_logsamples"
DEFAULT_OUT = ROOT / "results" / "sft_eval" / "wide_bench_diagnostics" / "truthfulqa_analysis"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--baseline", default="base")
    return ap.parse_args()


def arm_name(name: str) -> str:
    name = re.sub(r"_limit\d+$", "", name)
    return name


def find_sample_files(log_root: Path) -> dict[str, Path]:
    files: dict[str, Path] = {}
    for path in sorted(log_root.glob("*/**/samples_truthfulqa_mc2_*.jsonl")):
        arm = arm_name(path.relative_to(log_root).parts[0])
        old = files.get(arm)
        if old is None or path.stat().st_mtime > old.stat().st_mtime:
            files[arm] = path
    return files


def latest_result_json(sample_path: Path) -> Path | None:
    matches = sorted(sample_path.parent.glob("results_*.json"), key=lambda p: p.stat().st_mtime)
    return matches[-1] if matches else None


def logsumexp(values: list[float]) -> float:
    if not values:
        return float("nan")
    top = max(values)
    return top + math.log(sum(math.exp(v - top) for v in values))


def softmax_true_mass(scores: list[float], labels: list[int]) -> float:
    true_scores = [s for s, label in zip(scores, labels) if label == 1]
    total = logsumexp(scores)
    if not true_scores or not math.isfinite(total):
        return float("nan")
    return math.exp(logsumexp(true_scores) - total)


def safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def percentile(values: list[float], pct: float) -> float:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * pct / 100.0
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return clean[lo]
    weight = pos - lo
    return clean[lo] * (1.0 - weight) + clean[hi] * weight


def mean_finite(values: list[float]) -> float:
    clean = [v for v in values if math.isfinite(v)]
    return mean(clean) if clean else float("nan")


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_aggregate_acc(sample_path: Path) -> float:
    result_path = latest_result_json(sample_path)
    if result_path is None:
        return float("nan")
    with result_path.open() as f:
        data = json.load(f)
    return safe_float(data.get("results", {}).get("truthfulqa_mc2", {}).get("acc,none"))


def read_arm_rows(arm: str, sample_path: Path) -> list[dict]:
    rows: list[dict] = []
    with sample_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            targets = sample["doc"]["mc2_targets"]
            choices = list(targets["choices"])
            labels = [int(x) for x in targets["labels"]]
            scores = [safe_float(resp[0]) for resp in sample["filtered_resps"]]
            if len(choices) != len(labels) or len(choices) != len(scores):
                raise ValueError(f"{sample_path}: malformed doc_id={sample.get('doc_id')}")

            true_scores = [s for s, label in zip(scores, labels) if label == 1]
            false_scores = [s for s, label in zip(scores, labels) if label == 0]
            true_lse = logsumexp(true_scores)
            false_lse = logsumexp(false_scores)
            best_true_score = max(true_scores) if true_scores else float("nan")
            best_false_score = max(false_scores) if false_scores else float("nan")
            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            best_true_text = choices[max(range(len(scores)), key=lambda i: scores[i] if labels[i] == 1 else -math.inf)]
            best_false_text = choices[max(range(len(scores)), key=lambda i: scores[i] if labels[i] == 0 else -math.inf)]

            rows.append(
                {
                    "arm": arm,
                    "doc_id": int(sample["doc_id"]),
                    "question": sample["doc"]["question"],
                    "n_choices": len(choices),
                    "n_true": sum(labels),
                    "mc2_acc": safe_float(sample.get("acc")),
                    "mc2_acc_recomputed": softmax_true_mass(scores, labels),
                    "true_logsumexp": true_lse,
                    "false_logsumexp": false_lse,
                    "truth_logodds": true_lse - false_lse,
                    "best_true_score": best_true_score,
                    "best_false_score": best_false_score,
                    "best_margin": best_true_score - best_false_score,
                    "mean_true_score": mean_finite(true_scores),
                    "mean_false_score": mean_finite(false_scores),
                    "best_is_true": int(labels[best_idx] == 1),
                    "best_true_text": best_true_text,
                    "best_false_text": best_false_text,
                }
            )
    return rows


def summarize_rows(arm: str, rows: list[dict], aggregate_acc: float, sample_path: Path) -> dict:
    return {
        "arm": arm,
        "n": len(rows),
        "aggregate_acc": aggregate_acc,
        "mean_item_acc": mean_finite([r["mc2_acc"] for r in rows]),
        "mean_truth_logodds": mean_finite([r["truth_logodds"] for r in rows]),
        "median_truth_logodds": percentile([r["truth_logodds"] for r in rows], 50),
        "p10_truth_logodds": percentile([r["truth_logodds"] for r in rows], 10),
        "mean_best_margin": mean_finite([r["best_margin"] for r in rows]),
        "best_is_true_rate": mean_finite([r["best_is_true"] for r in rows]),
        "mean_true_logsumexp": mean_finite([r["true_logsumexp"] for r in rows]),
        "mean_false_logsumexp": mean_finite([r["false_logsumexp"] for r in rows]),
        "sample_file": rel_path(sample_path),
    }


def paired_deltas(all_rows: dict[str, list[dict]], baseline: str) -> list[dict]:
    if baseline not in all_rows:
        return []
    base_by_id = {r["doc_id"]: r for r in all_rows[baseline]}
    rows: list[dict] = []
    for arm, arm_rows in sorted(all_rows.items()):
        if arm == baseline:
            continue
        for row in arm_rows:
            base = base_by_id.get(row["doc_id"])
            if base is None:
                continue
            rows.append(
                {
                    "arm": arm,
                    "doc_id": row["doc_id"],
                    "question": row["question"],
                    "delta_mc2_acc": row["mc2_acc"] - base["mc2_acc"],
                    "delta_truth_logodds": row["truth_logodds"] - base["truth_logodds"],
                    "delta_true_logsumexp": row["true_logsumexp"] - base["true_logsumexp"],
                    "delta_false_logsumexp": row["false_logsumexp"] - base["false_logsumexp"],
                    "delta_best_margin": row["best_margin"] - base["best_margin"],
                    "base_mc2_acc": base["mc2_acc"],
                    "arm_mc2_acc": row["mc2_acc"],
                    "base_truth_logodds": base["truth_logodds"],
                    "arm_truth_logodds": row["truth_logodds"],
                    "base_best_false_text": base["best_false_text"],
                    "arm_best_false_text": row["best_false_text"],
                    "best_true_text": row["best_true_text"],
                }
            )
    return rows


def summarize_deltas(delta_rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    arms = sorted({r["arm"] for r in delta_rows})
    for arm in arms:
        rows = [r for r in delta_rows if r["arm"] == arm]
        out.append(
            {
                "arm": arm,
                "n_paired": len(rows),
                "mean_delta_mc2_acc": mean_finite([r["delta_mc2_acc"] for r in rows]),
                "mean_delta_truth_logodds": mean_finite([r["delta_truth_logodds"] for r in rows]),
                "mean_delta_true_logsumexp": mean_finite([r["delta_true_logsumexp"] for r in rows]),
                "mean_delta_false_logsumexp": mean_finite([r["delta_false_logsumexp"] for r in rows]),
                "mean_delta_best_margin": mean_finite([r["delta_best_margin"] for r in rows]),
                "frac_acc_down": mean_finite([int(r["delta_mc2_acc"] < 0) for r in rows]),
                "frac_false_pressure_up": mean_finite([int(r["delta_false_logsumexp"] > 0) for r in rows]),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def write_report(out_dir: Path, summary: list[dict], delta_summary: list[dict], delta_rows: list[dict]) -> None:
    lines = [
        "# TruthfulQA Log-Sample Diagnostic",
        "",
        "This report decomposes bounded `truthfulqa_mc2` runs into truthful-answer",
        "mass and false-answer pressure. These `--limit` runs are diagnostics, not",
        "final benchmark numbers.",
        "",
        "## Aggregate Slice",
        "",
        "| Arm | n | MC2 acc | Truth log-odds | Best true - false | Best-is-true |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            "| {arm} | {n} | {acc} | {odds} | {margin} | {best} |".format(
                arm=row["arm"],
                n=row["n"],
                acc=fmt(row["aggregate_acc"]),
                odds=fmt(row["mean_truth_logodds"]),
                margin=fmt(row["mean_best_margin"]),
                best=fmt(row["best_is_true_rate"]),
            )
        )

    if delta_summary:
        lines.extend(
            [
                "",
                "## Paired Delta vs Base",
                "",
                "| Arm | n | Delta acc | Delta truth log-odds | Delta true mass | Delta false pressure | Acc-down frac |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in delta_summary:
            lines.append(
                "| {arm} | {n} | {dacc} | {dodds} | {dtrue} | {dfalse} | {down} |".format(
                    arm=row["arm"],
                    n=row["n_paired"],
                    dacc=fmt(row["mean_delta_mc2_acc"]),
                    dodds=fmt(row["mean_delta_truth_logodds"]),
                    dtrue=fmt(row["mean_delta_true_logsumexp"]),
                    dfalse=fmt(row["mean_delta_false_logsumexp"]),
                    down=fmt(row["frac_acc_down"]),
                )
            )

        lines.extend(["", "## Current Read", ""])
        by_arm = {row["arm"]: row for row in delta_summary}
        if "lowrank" in by_arm:
            row = by_arm["lowrank"]
            lines.append(
                "- `lowrank` improves this bounded slice when false-answer "
                f"pressure falls more than truthful-answer mass "
                f"(`delta_false_logsumexp={fmt(row['mean_delta_false_logsumexp'])}`, "
                f"`delta_true_logsumexp={fmt(row['mean_delta_true_logsumexp'])}`)."
            )
        if "taskvec_a0p25" in by_arm:
            row = by_arm["taskvec_a0p25"]
            lines.append(
                "- `taskvec_a0p25` is aggregate-flat here, but it raises "
                f"false-answer pressure on {fmt(row['frac_false_pressure_up'])} "
                "of paired items and reduces mean truth log-odds."
            )
        if "scrambled" in by_arm:
            row = by_arm["scrambled"]
            lines.append(
                "- `scrambled` drops MC2 despite higher mean truth log-odds, "
                "because it suppresses both true and false answer likelihoods "
                "very strongly and produces catastrophic item-level flips."
            )

        lines.extend(["", "## Largest Paired Drops", ""])
        for arm in sorted({r["arm"] for r in delta_rows}):
            worst = sorted((r for r in delta_rows if r["arm"] == arm), key=lambda r: r["delta_mc2_acc"])[:5]
            lines.append(f"### {arm}")
            lines.append("")
            lines.append("| doc_id | Delta acc | Delta odds | Question | Arm false answer |")
            lines.append("|---:|---:|---:|---|---|")
            for row in worst:
                question = str(row["question"]).replace("|", "\\|")
                false_text = str(row["arm_best_false_text"]).replace("|", "\\|")
                lines.append(
                    f"| {row['doc_id']} | {fmt(row['delta_mc2_acc'])} | "
                    f"{fmt(row['delta_truth_logodds'])} | {question} | {false_text} |"
                )
            lines.append("")

    lines.extend(
        [
            "## Artifacts",
            "",
            "- `item_scores.csv`: per-question truthful/false log-likelihood decomposition.",
            "- `summary.csv`: per-arm aggregate diagnostic metrics.",
            "- `paired_deltas.csv`: per-question deltas against the base arm.",
            "- `paired_delta_summary.csv`: aggregate paired deltas against base.",
            "",
        ]
    )
    (out_dir / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    args = parse_args()
    sample_files = find_sample_files(args.log_root)
    if not sample_files:
        raise SystemExit(f"No TruthfulQA sample files found under {args.log_root}")

    all_rows: dict[str, list[dict]] = {}
    summary: list[dict] = []
    for arm, sample_path in sorted(sample_files.items()):
        rows = read_arm_rows(arm, sample_path)
        all_rows[arm] = rows
        summary.append(summarize_rows(arm, rows, read_aggregate_acc(sample_path), sample_path))

    item_rows = [row for arm in sorted(all_rows) for row in all_rows[arm]]
    delta_rows = paired_deltas(all_rows, args.baseline)
    delta_summary = summarize_deltas(delta_rows)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "item_scores.csv", item_rows)
    write_csv(args.out_dir / "summary.csv", summary)
    write_csv(args.out_dir / "paired_deltas.csv", delta_rows)
    write_csv(args.out_dir / "paired_delta_summary.csv", delta_summary)
    write_report(args.out_dir, summary, delta_summary, delta_rows)
    print(f"[ok] wrote {args.out_dir}")


if __name__ == "__main__":
    main()
