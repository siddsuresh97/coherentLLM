"""Build the cheap retention failure-suite manifest from existing summaries.

This script does not launch benchmarks. It reads the current wide-bench
diagnostic CSVs and writes a small manifest that future alpha/replay/steering
candidates can use before spending a full MMLU run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = ROOT / "results" / "sft_eval" / "wide_bench" / "summary.csv"
DEFAULT_TASK_DELTAS = (
    ROOT
    / "results"
    / "sft_eval"
    / "wide_bench"
    / "skill_diagnostics"
    / "task_skill_deltas.csv"
)
DEFAULT_TRUTHFULQA = (
    ROOT
    / "results"
    / "sft_eval"
    / "wide_bench"
    / "skill_diagnostics"
    / "truthfulqa_mechanism.csv"
)
DEFAULT_OUT_DIR = ROOT / "results" / "sft_eval" / "wide_bench" / "failure_suite"

CSV_FIELDS = [
    "priority",
    "lane",
    "task",
    "task_label",
    "harness_task",
    "harness_group",
    "metric",
    "fewshot",
    "limit",
    "full_n",
    "base_value",
    "lowrank_value",
    "lowrank_delta_vs_base",
    "taskvec_a0p25_value",
    "taskvec_a0p25_delta_vs_base",
    "guard_role",
    "selection_reason",
    "item_policy",
    "required_diagnostics",
]

TASK_SPECS: list[dict[str, Any]] = [
    {
        "priority": "P0",
        "lane": "mmlu_failure_slice",
        "task": "mmlu_moral_scenarios",
        "task_label": "MMLU moral scenarios",
        "harness_task": "mmlu_moral_scenarios",
        "harness_group": "mmlu_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 120,
        "guard_role": "residual MMLU drop gate",
        "selection_reason": "largest lowrank MMLU drop; task-vector remains far below base",
        "item_policy": "stratify after logits: base-correct/candidate-wrong, narrow base margin, random",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "mmlu_failure_slice",
        "task": "mmlu_formal_logic",
        "task_label": "MMLU formal logic",
        "harness_task": "mmlu_formal_logic",
        "harness_group": "mmlu_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 126,
        "guard_role": "formal boundary gate",
        "selection_reason": "persistent formal/reasoning drop; task-vector is not a fix",
        "item_policy": "run full subject; retain all close-option failures for replay design",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "mmlu_failure_slice",
        "task": "mmlu_medical_genetics",
        "task_label": "MMLU medical genetics",
        "harness_task": "mmlu_medical_genetics",
        "harness_group": "mmlu_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 100,
        "guard_role": "biomedical factual gate",
        "selection_reason": "large biomedical drop across lowLR, lowrank, and task-vector",
        "item_policy": "run full subject; prefer base-correct/candidate-wrong errors for replay",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "mmlu_failure_slice",
        "task": "mmlu_nutrition",
        "task_label": "MMLU nutrition",
        "harness_task": "mmlu_nutrition",
        "harness_group": "mmlu_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 120,
        "guard_role": "biomedical commonsense gate",
        "selection_reason": "science/biomedical retention drop with practical misconception distractors",
        "item_policy": "stratify after logits: high distractor pressure, false practical rules, random",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "mmlu_failure_slice",
        "task": "mmlu_professional_psychology",
        "task_label": "MMLU professional psychology",
        "harness_task": "mmlu_professional_psychology",
        "harness_group": "mmlu_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 120,
        "guard_role": "professional long-option gate",
        "selection_reason": "professional/law/moral family remains below base under task-vector",
        "item_policy": "stratify after logits: long close distractors and base-correct/candidate-wrong",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "mmlu_failure_slice",
        "task": "mmlu_high_school_statistics",
        "task_label": "MMLU high school statistics",
        "harness_task": "mmlu_high_school_statistics",
        "harness_group": "mmlu_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 120,
        "guard_role": "quantitative calibration gate",
        "selection_reason": "large lowLR and lowrank drop; tests exact quantitative boundaries",
        "item_policy": "stratify after logits: formula confusions, narrow margins, random",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P1",
        "lane": "mmlu_failure_slice",
        "task": "mmlu_business_ethics",
        "task_label": "MMLU business ethics",
        "harness_task": "mmlu_business_ethics",
        "harness_group": "mmlu_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 100,
        "guard_role": "task-vector regression sentinel",
        "selection_reason": "taskvec_a0p25 is worse than lowrank here, so it guards alpha tuning",
        "item_policy": "run full subject; inspect cases where task-vector raises wrong normative options",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "science_ranking",
        "task": "arc_easy",
        "task_label": "ARC Easy",
        "harness_task": "arc_easy",
        "harness_group": "arc_25shot",
        "metric": "acc_norm",
        "fewshot": 25,
        "limit": 200,
        "guard_role": "science ranking recovery gate",
        "selection_reason": "task-vector partially mitigates lowrank but remains below base",
        "item_policy": "deterministic limit first; after logits prefer base-correct/candidate-wrong errors",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "science_ranking",
        "task": "arc_challenge",
        "task_label": "ARC Challenge",
        "harness_task": "arc_challenge",
        "harness_group": "arc_25shot",
        "metric": "acc_norm",
        "fewshot": 25,
        "limit": 200,
        "guard_role": "hard science ranking gate",
        "selection_reason": "large residual task-vector drop; close distractors are likely failure mode",
        "item_policy": "deterministic limit first; after logits emphasize close wrong-choice margins",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "science_ranking",
        "task": "openbookqa",
        "task_label": "OpenBookQA",
        "harness_task": "openbookqa",
        "harness_group": "zero_shot",
        "metric": "acc_norm",
        "fewshot": 0,
        "limit": 200,
        "guard_role": "science fact and lure gate",
        "selection_reason": "consistently hurt across aligned arms; cheap 500-item benchmark",
        "item_policy": "deterministic limit first; after logits collect high-distractor science lures",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "sense_boundary",
        "task": "wic",
        "task_label": "WiC",
        "harness_task": "wic",
        "harness_group": "zero_shot",
        "metric": "acc",
        "fewshot": 0,
        "limit": 300,
        "guard_role": "lexical sense-boundary gate",
        "selection_reason": "clean near-chance failure across lowLR, lowrank, task-vector, and scrambled",
        "item_policy": "pair by lemma/POS/label when item metadata is available; otherwise limit-300 smoke",
        "required_diagnostics": "accuracy, correct_choice_margin, same_lemma_contrast_margin, KL_to_base",
    },
    {
        "priority": "P0",
        "lane": "truthfulness_lures",
        "task": "truthfulqa_mc2",
        "task_label": "TruthfulQA MC2",
        "harness_task": "truthfulqa_mc2",
        "harness_group": "zero_shot_logsamples",
        "metric": "acc",
        "fewshot": 0,
        "limit": 200,
        "guard_role": "false-lure pressure gate",
        "selection_reason": "task-vector raises false-answer pressure even when slice MC2 is nearly flat",
        "item_policy": "reuse the bounded 200-doc log-sample slice for paired true/false mass deltas",
        "required_diagnostics": "MC2, truth_logodds, false_lure_pressure, frac_false_pressure_up, KL_to_base",
    },
    {
        "priority": "P1",
        "lane": "retention_guard",
        "task": "hellaswag",
        "task_label": "HellaSwag",
        "harness_task": "hellaswag",
        "harness_group": "hellaswag_10shot",
        "metric": "acc_norm",
        "fewshot": 10,
        "limit": 200,
        "guard_role": "script/event plausibility retention guard",
        "selection_reason": "mostly preserved by lowrank/task-vector; should not regress during mitigation",
        "item_policy": "deterministic limit first; inspect event-ending distractor margin drift",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P1",
        "lane": "retention_guard",
        "task": "winogrande",
        "task_label": "WinoGrande",
        "harness_task": "winogrande",
        "harness_group": "winogrande_5shot",
        "metric": "acc",
        "fewshot": 5,
        "limit": 200,
        "guard_role": "discourse/coreference retention guard",
        "selection_reason": "stable across aligned arms; catches broad answer-surface damage",
        "item_policy": "deterministic limit first; track pronoun-choice margin and KL drift",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
    {
        "priority": "P1",
        "lane": "retention_guard",
        "task": "piqa",
        "task_label": "PIQA",
        "harness_task": "piqa",
        "harness_group": "zero_shot",
        "metric": "acc_norm",
        "fewshot": 0,
        "limit": 200,
        "guard_role": "physical affordance retention guard",
        "selection_reason": "near stable for task-vector and useful as a cheap commonsense guard",
        "item_policy": "deterministic limit first; inspect physical-affordance distractor margin drift",
        "required_diagnostics": "accuracy, correct_choice_margin, best_distractor_margin, KL_to_base",
    },
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    ap.add_argument("--task_deltas", type=Path, default=DEFAULT_TASK_DELTAS)
    ap.add_argument("--truthfulqa", type=Path, default=DEFAULT_TRUTHFULQA)
    ap.add_argument("--out_dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--check",
        action="store_true",
        help="compare generated outputs to disk without writing",
    )
    return ap.parse_args()


def read_csv(path: Path, source_name: str) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["_source_file"] = source_name
    return rows


def safe_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def score_lookup(rows: list[dict[str, str]], task: str, model: str, metric: str) -> dict[str, Any]:
    candidates = [
        row
        for row in rows
        if row.get("task") == task and row.get("model") == model
    ]
    if metric:
        exact = [row for row in candidates if row.get("metric") == metric]
        if exact:
            candidates = exact
    if not candidates:
        return {}
    row = candidates[0]
    return {
        "value": safe_float(row.get("value")),
        "delta_vs_base": safe_float(row.get("delta_vs_base")),
        "n": row.get("n") or "",
        "source": row.get("source") or row.get("_source_file") or "",
    }


def fmt_float(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6g}"


def build_slice_rows(score_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for spec in TASK_SPECS:
        task = spec["task"]
        metric = spec["metric"]
        base = score_lookup(score_rows, task, "base", metric)
        lowrank = score_lookup(score_rows, task, "lowrank", metric)
        taskvec = score_lookup(score_rows, task, "taskvec_a0p25", metric)
        full_n = base.get("n") or lowrank.get("n") or taskvec.get("n") or ""

        row = {
            key: str(spec.get(key, ""))
            for key in CSV_FIELDS
            if key
            not in {
                "full_n",
                "base_value",
                "lowrank_value",
                "lowrank_delta_vs_base",
                "taskvec_a0p25_value",
                "taskvec_a0p25_delta_vs_base",
            }
        }
        row.update(
            {
                "full_n": str(full_n),
                "base_value": fmt_float(base.get("value")),
                "lowrank_value": fmt_float(lowrank.get("value")),
                "lowrank_delta_vs_base": fmt_float(lowrank.get("delta_vs_base")),
                "taskvec_a0p25_value": fmt_float(taskvec.get("value")),
                "taskvec_a0p25_delta_vs_base": fmt_float(taskvec.get("delta_vs_base")),
            }
        )
        rows.append(row)
    return rows


def build_truthfulqa_summary(path: Path) -> list[dict[str, Any]]:
    rows = read_csv(path, str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path))
    out: list[dict[str, Any]] = []
    fields = [
        "arm",
        "full_mc2_acc",
        "full_delta_vs_base",
        "diagnostic_n",
        "diagnostic_mc2_acc",
        "diagnostic_delta_acc",
        "delta_truth_logodds",
        "delta_true_mass",
        "delta_false_pressure",
        "frac_false_pressure_up",
        "mechanism",
    ]
    for row in rows:
        converted: dict[str, Any] = {}
        for field in fields:
            value = row.get(field, "")
            number = safe_float(value)
            converted[field] = number if number is not None else value
        out.append(converted)
    return out


def csv_text(rows: list[dict[str, str]]) -> str:
    from io import StringIO

    buf = StringIO()
    writer = csv.DictWriter(buf, fieldnames=CSV_FIELDS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def json_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True) + "\n"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def build_manifest(args: argparse.Namespace, slice_rows: list[dict[str, str]]) -> dict[str, Any]:
    score_rows = read_csv(args.task_deltas, rel(args.task_deltas)) + read_csv(args.summary, rel(args.summary))
    taskvec_mmlu = score_lookup(score_rows, "mmlu", "taskvec_a0p25", "acc")
    lowrank_mmlu = score_lookup(score_rows, "mmlu", "lowrank", "acc")
    base_mmlu = score_lookup(score_rows, "mmlu", "base", "acc")
    taskvec_delta = taskvec_mmlu.get("delta_vs_base")
    lowrank_delta = lowrank_mmlu.get("delta_vs_base")
    taskvec_vs_lowrank = None
    if taskvec_mmlu.get("value") is not None and lowrank_mmlu.get("value") is not None:
        taskvec_vs_lowrank = taskvec_mmlu["value"] - lowrank_mmlu["value"]

    return {
        "schema_version": "retention_failure_suite.v1",
        "generated_by": rel(Path(__file__).resolve()),
        "no_benchmarks_launched": True,
        "source_artifacts": [rel(args.task_deltas), rel(args.summary), rel(args.truthfulqa)],
        "current_anchor": {
            "base_mmlu_micro_acc": base_mmlu.get("value"),
            "lowrank_mmlu_micro_acc": lowrank_mmlu.get("value"),
            "taskvec_a0p25_mmlu_micro_acc": taskvec_mmlu.get("value"),
            "taskvec_a0p25_delta_vs_base": taskvec_delta,
            "taskvec_a0p25_delta_vs_lowrank": taskvec_vs_lowrank,
            "interpretation": (
                "taskvec_a0p25 is a partial mitigation: better than lowrank on "
                "MMLU aggregate, still materially below base."
            ),
        },
        "gate_thresholds": {
            "mmlu_failure_slice_mean_delta_vs_base": ">= -0.06 before full MMLU",
            "science_ranking_mean_delta_vs_base": ">= -0.05 and no task below -0.08",
            "wic_delta_vs_base": ">= -0.05 or show margin recovery on same-lemma contrasts",
            "truthfulqa_frac_false_pressure_up": "<= 0.60 on paired log-sample slice",
            "truthfulqa_delta_false_pressure": "<= 0.0 preferred; small positive only if truth_logodds improves",
            "retention_guards": "HellaSwag, WinoGrande, and PIQA deltas each >= -0.02",
            "semantic_target_retention": "generation coherence >= 0.60 and THINGS human R2 >= 0.58",
            "kl_to_base_logits": "track per item; investigate high-KL outliers before promotion",
        },
        "metrics": [
            {
                "name": "accuracy",
                "definition": "standard lm-eval metric for the slice, not sufficient alone",
            },
            {
                "name": "correct_choice_margin",
                "definition": "logp(correct choice) - max_wrong logp(wrong choice)",
            },
            {
                "name": "best_distractor_margin",
                "definition": "max_wrong logp(wrong choice) - logp(correct choice)",
            },
            {
                "name": "false_lure_pressure",
                "definition": "TruthfulQA logsumexp(false answers) relative to true-answer mass",
            },
            {
                "name": "kl_to_base_logits",
                "definition": "KL(base answer-choice distribution || candidate answer-choice distribution)",
            },
            {
                "name": "semantic_target_retention",
                "definition": "candidate still preserves the intended coherence/human-similarity gains",
            },
        ],
        "slices": slice_rows,
        "truthfulqa_current_mechanism": build_truthfulqa_summary(args.truthfulqa),
        "mitigation_candidates": [
            {
                "name": "task-vector alpha sweep",
                "arms": ["0.0", "0.1", "0.2", "0.25", "0.3", "0.5"],
                "gate": "run only the failure suite first; promote no arm that fails TruthfulQA or MMLU slices",
            },
            {
                "name": "base-logit KL plus replay",
                "gate": "add MC replay and KL on base answer-choice logits for MMLU, ARC, OBQA, WiC, TruthfulQA",
            },
            {
                "name": "TruthfulQA lure-focused contrastive examples",
                "gate": "reduce false-lure pressure without reducing true-answer mass",
            },
            {
                "name": "WiC sense replay",
                "gate": "recover same-lemma contrast margins before broad benchmark promotion",
            },
            {
                "name": "adapter routing or merging",
                "gate": "downweight semantic adapter for calibrated MC answering while preserving representation use",
            },
            {
                "name": "concept-vector steering or ablation gates",
                "gate": "require failure-suite margins before using steering gains as a mitigation claim",
            },
        ],
    }


def compare_or_write(path: Path, text: str, check: bool) -> bool:
    if check:
        if not path.exists():
            print(f"[missing] {rel(path)}")
            return False
        current = path.read_text()
        if current != text:
            print(f"[changed] {rel(path)}")
            return False
        print(f"[ok] {rel(path)}")
        return True
    path.write_text(text)
    print(f"[write] {rel(path)}")
    return True


def main() -> None:
    args = parse_args()
    score_rows = read_csv(args.task_deltas, rel(args.task_deltas)) + read_csv(args.summary, rel(args.summary))
    slice_rows = build_slice_rows(score_rows)
    manifest = build_manifest(args, slice_rows)

    task_slices_text = csv_text(slice_rows)
    manifest_text = json_text(manifest)

    if not args.check:
        args.out_dir.mkdir(parents=True, exist_ok=True)

    ok = True
    ok &= compare_or_write(args.out_dir / "task_slices.csv", task_slices_text, args.check)
    ok &= compare_or_write(args.out_dir / "suite_manifest.json", manifest_text, args.check)
    if args.check and not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
