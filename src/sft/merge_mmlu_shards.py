"""Merge CHTC MMLU shard outputs into wide-bench style summaries."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import tarfile
from dataclasses import dataclass
from io import TextIOWrapper
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
AGGREGATE_MMLU_TASKS = {
    "mmlu",
    "mmlu_stem",
    "mmlu_other",
    "mmlu_social_sciences",
    "mmlu_humanities",
}
PREFERRED_METRICS = {
    "mmlu": ["acc,none", "acc"],
}


@dataclass(frozen=True)
class ResultDoc:
    source: str
    data: dict[str, Any]


def is_result_json(path: Path) -> bool:
    return path.name.startswith("results_") and path.suffix == ".json"


def iter_result_docs(inputs: Iterable[Path]) -> Iterable[ResultDoc]:
    for input_path in inputs:
        if input_path.is_dir():
            for path in sorted(input_path.rglob("results_*.json")):
                with path.open() as f:
                    yield ResultDoc(str(path), json.load(f))
            for path in sorted(input_path.rglob("*.tgz")) + sorted(input_path.rglob("*.tar.gz")):
                yield from iter_tar_result_docs(path)
        elif input_path.is_file() and is_result_json(input_path):
            with input_path.open() as f:
                yield ResultDoc(str(input_path), json.load(f))
        elif input_path.is_file() and input_path.name.endswith((".tgz", ".tar.gz")):
            yield from iter_tar_result_docs(input_path)
        else:
            raise FileNotFoundError(f"No mergeable result input found at {input_path}")


def iter_tar_result_docs(path: Path) -> Iterable[ResultDoc]:
    with tarfile.open(path) as tf:
        for member in sorted(tf.getmembers(), key=lambda m: m.name):
            name = Path(member.name).name
            if not member.isfile() or not name.startswith("results_") or not name.endswith(".json"):
                continue
            extracted = tf.extractfile(member)
            if extracted is None:
                continue
            with TextIOWrapper(extracted, encoding="utf-8") as f:
                yield ResultDoc(f"{path}!{member.name}", json.load(f))


def metric_for_task(task: str, metrics: dict[str, Any]) -> tuple[str, float] | None:
    preferred = PREFERRED_METRICS.get(task, ["acc,none", "acc"])
    for metric in preferred:
        if metric in metrics and isinstance(metrics[metric], (int, float)):
            return metric.split(",", 1)[0], float(metrics[metric])
    for key, value in metrics.items():
        if key.endswith("_stderr,none") or key in {"alias", "name", "sample_len"}:
            continue
        if isinstance(value, (int, float)):
            return key.split(",", 1)[0], float(value)
    return None


def merge_docs(docs: list[ResultDoc], include_aggregates: bool) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    merged: dict[str, dict[str, Any]] = {}
    sources: dict[str, str] = {}
    for doc in docs:
        results = doc.data.get("results", {})
        if not isinstance(results, dict):
            continue
        for task, metrics in results.items():
            if not task.startswith("mmlu"):
                continue
            if task in AGGREGATE_MMLU_TASKS and not include_aggregates:
                continue
            if task in merged:
                if merged[task] == metrics:
                    continue
                raise ValueError(f"Duplicate non-identical result for {task}: {sources[task]} and {doc.source}")
            merged[task] = metrics
            sources[task] = doc.source
    return dict(sorted(merged.items())), sources


def raw_rows(merged: dict[str, dict[str, Any]], state_name: str, sources: dict[str, str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task, metrics in merged.items():
        selected = metric_for_task(task, metrics)
        if selected is None:
            continue
        metric, value = selected
        rows.append(
            {
                "task": task,
                "model": state_name,
                "metric": metric,
                "value": value,
                "n": metrics.get("sample_len", ""),
                "source": sources.get(task, ""),
            }
        )
    weighted = weighted_mmlu_row(rows, state_name)
    if weighted is not None:
        rows.append(weighted)
    return rows


def weighted_mmlu_row(rows: list[dict[str, Any]], state_name: str) -> dict[str, Any] | None:
    total_n = 0
    total_correct = 0.0
    for row in rows:
        if not is_subject_task(str(row["task"])):
            continue
        try:
            n = int(row["n"])
        except (TypeError, ValueError):
            return None
        total_n += n
        total_correct += float(row["value"]) * n
    if total_n == 0:
        return None
    return {
        "task": "mmlu",
        "model": state_name,
        "metric": "acc",
        "value": total_correct / total_n,
        "n": total_n,
        "source": "weighted_subject_merge",
    }


def is_subject_task(task: str) -> bool:
    return task.startswith("mmlu_") and task not in AGGREGATE_MMLU_TASKS


def read_base_rows(path: Path | None) -> dict[tuple[str, str], float]:
    if path is None:
        return {}
    base: dict[tuple[str, str], float] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            if row.get("model") != "base":
                continue
            try:
                base[(row["task"], row["metric"])] = float(row["value"])
            except (KeyError, TypeError, ValueError):
                continue
    subject_values = [
        value
        for (task, metric), value in base.items()
        if is_subject_task(task) and metric == "acc"
    ]
    if subject_values:
        base[("mmlu_macro", "macro_acc")] = sum(subject_values) / len(subject_values)
    return base


def summary_rows(rows: list[dict[str, Any]], base: dict[tuple[str, str], float]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for row in rows:
        metric = str(row["metric"])
        task = str(row["task"])
        base_value = base.get((task, metric), math.nan)
        value = float(row["value"])
        delta = value - base_value if not math.isnan(base_value) else math.nan
        summary.append(
            {
                "task": task,
                "model": row["model"],
                "metric": metric,
                "value": value,
                "n": row["n"],
                "base_value": base_value,
                "delta_vs_base": delta,
                "verdict": verdict(delta),
            }
        )
    macro = macro_row(rows, base)
    if macro is not None:
        summary.append(macro)
    return summary


def macro_row(rows: list[dict[str, Any]], base: dict[tuple[str, str], float]) -> dict[str, Any] | None:
    subject_rows = [
        row for row in rows if is_subject_task(str(row["task"])) and row["metric"] == "acc"
    ]
    if not subject_rows:
        return None
    value = sum(float(row["value"]) for row in subject_rows) / len(subject_rows)
    base_value = base.get(("mmlu_macro", "macro_acc"), math.nan)
    delta = value - base_value if not math.isnan(base_value) else math.nan
    return {
        "task": "mmlu_macro",
        "model": subject_rows[0]["model"],
        "metric": "macro_acc",
        "value": value,
        "n": len(subject_rows),
        "base_value": base_value,
        "delta_vs_base": delta,
        "verdict": verdict(delta),
    }


def verdict(delta: float) -> str:
    if math.isnan(delta):
        return "NA"
    delta = round(delta, 12)
    if delta > 0.02:
        return "GAIN"
    if delta < -0.02:
        return "DROP"
    return "FLAT"


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_merged_json(path: Path, docs: list[ResultDoc], merged: dict[str, dict[str, Any]]) -> None:
    template: dict[str, Any] = {}
    if docs:
        template = {
            key: value
            for key, value in docs[0].data.items()
            if key not in {"results", "groups", "group_subtasks"}
        }
    template["results"] = merged
    template["source_files"] = [doc.source for doc in docs]
    with path.open("w") as f:
        json.dump(template, f, indent=2, sort_keys=True)
        f.write("\n")


def write_report(path: Path, rows: list[dict[str, Any]], summaries: list[dict[str, Any]], expected_subjects: int) -> None:
    subject_rows = [row for row in rows if is_subject_task(str(row["task"]))]
    weighted = next((row for row in rows if row["task"] == "mmlu" and row["metric"] == "acc"), None)
    macro = next((row for row in summaries if row["task"] == "mmlu_macro"), None)
    missing_count = expected_subjects - len(subject_rows)
    lines = [
        "# CHTC MMLU Shard Merge Report",
        "",
        f"- Subject tasks merged: {len(subject_rows)} / {expected_subjects}",
        f"- Missing subject count: {max(missing_count, 0)}",
    ]
    if weighted is not None:
        lines.append(f"- Weighted MMLU acc: {float(weighted['value']):.6f} over n={weighted['n']}")
    if macro is not None:
        delta = macro["delta_vs_base"]
        delta_text = "NA" if math.isnan(float(delta)) else f"{float(delta):+.6f}"
        lines.append(f"- Unweighted subject macro acc: {float(macro['value']):.6f} ({delta_text} vs base)")
    lines.extend(["", "## Subjects", ""])
    for row in sorted(subject_rows, key=lambda item: str(item["task"])):
        lines.append(f"- {row['task']}: {float(row['value']):.6f} (n={row['n']})")
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", nargs="+", type=Path, required=True, help="Shard result dirs, result JSONs, or .tgz files.")
    ap.add_argument("--state-name", default="taskvec_a0p25")
    ap.add_argument("--base-csv", type=Path, default=ROOT / "results" / "sft_eval" / "wide_bench" / "raw_task_summary.csv")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--include-aggregate-tasks", action="store_true")
    ap.add_argument("--expected-subjects", type=int, default=57)
    ap.add_argument("--require-complete", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    docs = list(iter_result_docs(args.inputs))
    if not docs:
        raise SystemExit("No results_*.json files found in inputs.")
    merged, sources = merge_docs(docs, include_aggregates=args.include_aggregate_tasks)
    rows = raw_rows(merged, args.state_name, sources)
    subject_count = sum(1 for row in rows if is_subject_task(str(row["task"])))
    if args.require_complete and subject_count != args.expected_subjects:
        raise SystemExit(f"Expected {args.expected_subjects} MMLU subjects, found {subject_count}.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    base = read_base_rows(args.base_csv if args.base_csv else None)
    summaries = summary_rows(rows, base)

    write_merged_json(args.out_dir / "results_merged.json", docs, merged)
    write_csv(
        args.out_dir / "raw_task_summary.csv",
        [{key: row[key] for key in ["task", "model", "metric", "value", "n"]} for row in rows],
        ["task", "model", "metric", "value", "n"],
    )
    write_csv(args.out_dir / "raw_task_sources.csv", rows, ["task", "model", "metric", "value", "n", "source"])
    write_csv(
        args.out_dir / "summary.csv",
        summaries,
        ["task", "model", "metric", "value", "n", "base_value", "delta_vs_base", "verdict"],
    )
    write_report(args.out_dir / "REPORT.md", rows, summaries, args.expected_subjects)
    print(f"[done] merged {subject_count} subject tasks from {len(docs)} result document(s) into {args.out_dir}")


if __name__ == "__main__":
    main()
