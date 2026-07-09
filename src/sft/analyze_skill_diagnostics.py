"""Build skill diagnostics from existing coherence-SFT benchmark artifacts.

This script is intentionally read-only with respect to benchmark inputs: it
parses existing wide-bench CSVs, completed lm-eval JSONs, semantic/human
summary CSVs, and TruthfulQA log-sample diagnostics. It does not launch
benchmarks.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
SFT_EVAL = ROOT / "results" / "sft_eval"
WIDE = SFT_EVAL / "wide_bench"
TRUTH_DIR = SFT_EVAL / "wide_bench_diagnostics" / "truthfulqa_analysis"
OUT = WIDE / "skill_diagnostics"

MODEL_ORDER = ["base", "lowLR", "lowrank", "taskvec_a0p25", "scrambled"]
COMPARE_MODELS = ["lowLR", "lowrank", "taskvec_a0p25", "scrambled"]
DROP_THRESHOLD = -0.02
GAIN_THRESHOLD = 0.02

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

TASK_LABELS = {
    "piqa": "PIQA",
    "openbookqa": "OpenBookQA",
    "commonsense_qa": "CommonsenseQA",
    "wic": "WiC",
    "truthfulqa_mc2": "TruthfulQA-MC2",
    "winogrande": "WinoGrande",
    "arc_easy": "ARC-Easy",
    "arc_challenge": "ARC-Challenge",
    "hellaswag": "HellaSwag",
    "mmlu": "MMLU aggregate",
}

AGGREGATE_TASKS = {
    "mmlu",
    "mmlu_stem",
    "mmlu_other",
    "mmlu_social_sciences",
    "mmlu_humanities",
}

SCIENCE_BIOMED = {
    "anatomy",
    "astronomy",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_medicine",
    "college_physics",
    "conceptual_physics",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_physics",
    "medical_genetics",
    "nutrition",
    "virology",
}
QUANT_FORMAL = {
    "abstract_algebra",
    "college_mathematics",
    "elementary_mathematics",
    "formal_logic",
    "high_school_mathematics",
    "high_school_statistics",
    "logical_fallacies",
}
TECHNICAL = {
    "college_computer_science",
    "computer_security",
    "electrical_engineering",
    "high_school_computer_science",
    "machine_learning",
}
FACTUAL_HISTORY_POLICY = {
    "global_facts",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_us_history",
    "high_school_world_history",
    "international_law",
    "prehistory",
    "security_studies",
    "us_foreign_policy",
    "world_religions",
}
PROFESSIONAL_LAW_MORAL = {
    "business_ethics",
    "jurisprudence",
    "management",
    "moral_disputes",
    "moral_scenarios",
    "philosophy",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
}
SOCIAL_BEHAVIORAL = {
    "econometrics",
    "high_school_macroeconomics",
    "high_school_microeconomics",
    "high_school_psychology",
    "human_aging",
    "human_sexuality",
    "marketing",
    "sociology",
}

SKILL_DESCRIPTIONS = {
    "semantic coherence/human similarity": "Internal concept-space consistency and alignment to human similarity judgments.",
    "science/common-sense reasoning": "ARC/OpenBookQA style science facts plus multi-choice reasoning.",
    "script/discourse commonsense": "Physical affordances, event plausibility, and discourse/coreference commonsense.",
    "lexical/disambiguation": "Context-sensitive word-sense decisions.",
    "truthfulness/calibration": "TruthfulQA-style allocation of probability mass away from plausible false lures.",
    "MMLU science/biomedical": "Specialized factual science and biomedical exam knowledge.",
    "MMLU quantitative/formal": "Math, statistics, formal logic, and argument-form recognition.",
    "MMLU technical/computing": "CS, security, ML, and engineering exam knowledge.",
    "MMLU factual/history/policy": "History, geography, policy, religions, and broad factual recall.",
    "MMLU professional/law/moral": "Longer professional, legal, medical, philosophical, and moral decision items.",
    "MMLU social/behavioral/business": "Psychology, sociology, economics, marketing, and social behavior.",
    "MMLU miscellaneous/broad": "MMLU topics that do not fit the narrower diagnostic buckets.",
    "MMLU aggregate": "Published MMLU aggregate rows; excluded from skill-family means to avoid double counting subtasks.",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def fnum(value: object, default: float = math.nan) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def fmt(value: object, digits: int = 3) -> str:
    val = fnum(value)
    if math.isnan(val):
        return ""
    return f"{val:.{digits}f}"


def fmt_delta(value: object, digits: int = 3) -> str:
    val = fnum(value)
    if math.isnan(val):
        return ""
    return f"{val:+.{digits}f}"


def verdict(delta: float) -> str:
    if math.isnan(delta):
        return "NA"
    if delta <= DROP_THRESHOLD:
        return "hurt"
    if delta >= GAIN_THRESHOLD:
        return "boosted"
    return "preserved"


def skill_for_task(task: str) -> str:
    direct = {
        "piqa": "script/discourse commonsense",
        "hellaswag": "script/discourse commonsense",
        "winogrande": "script/discourse commonsense",
        "commonsense_qa": "script/discourse commonsense",
        "openbookqa": "science/common-sense reasoning",
        "arc_easy": "science/common-sense reasoning",
        "arc_challenge": "science/common-sense reasoning",
        "wic": "lexical/disambiguation",
        "truthfulqa_mc2": "truthfulness/calibration",
    }
    if task in direct:
        return direct[task]
    if task in AGGREGATE_TASKS:
        return "MMLU aggregate"
    if task.startswith("mmlu_"):
        name = task.removeprefix("mmlu_")
        if name in SCIENCE_BIOMED:
            return "MMLU science/biomedical"
        if name in QUANT_FORMAL:
            return "MMLU quantitative/formal"
        if name in TECHNICAL:
            return "MMLU technical/computing"
        if name in FACTUAL_HISTORY_POLICY:
            return "MMLU factual/history/policy"
        if name in PROFESSIONAL_LAW_MORAL:
            return "MMLU professional/law/moral"
        if name in SOCIAL_BEHAVIORAL:
            return "MMLU social/behavioral/business"
        return "MMLU miscellaneous/broad"
    return "MMLU miscellaneous/broad"


def task_label(task: str) -> str:
    if task in TASK_LABELS:
        return TASK_LABELS[task]
    if task.startswith("mmlu_"):
        return "MMLU " + task.removeprefix("mmlu_").replace("_", " ")
    return task.replace("_", " ")


def metric_for_task(task: str, metrics: dict[str, object]) -> tuple[str, float] | None:
    for metric in PREFERRED_METRICS.get(task, ["acc_norm,none", "acc,none"]):
        value = metrics.get(metric)
        if isinstance(value, (int, float)):
            return metric.split(",", 1)[0], float(value)
    for key, value in metrics.items():
        if key.endswith("_stderr,none") or key in {"alias", "name", "sample_len"}:
            continue
        if isinstance(value, (int, float)):
            return key.split(",", 1)[0], float(value)
    return None


def run_state_from_name(run_name: str) -> tuple[str, str] | None:
    for state in ["taskvec_a0p25", "scrambled", "lowrank", "lowLR", "base"]:
        prefix = state + "_"
        if run_name.startswith(prefix):
            return state, run_name[len(prefix) :]
    return None


def load_wide_rows() -> list[dict[str, object]]:
    summary_path = WIDE / "summary.csv"
    records: list[dict[str, object]] = []
    base_by_task_metric: dict[tuple[str, str], float] = {}
    base_by_task: dict[str, float] = {}

    for row in read_csv(summary_path):
        task = row["task"]
        model = row["model"]
        metric = row["metric"]
        value = fnum(row["value"])
        if model == "base":
            base_by_task_metric[(task, metric)] = value
            base_by_task.setdefault(task, value)
        delta = fnum(row.get("delta_vs_base"))
        if math.isnan(delta):
            delta = 0.0 if model == "base" else value - base_by_task_metric.get((task, metric), math.nan)
        records.append(
            {
                "skill": skill_for_task(task),
                "task": task,
                "task_label": task_label(task),
                "model": model,
                "metric": metric,
                "value": value,
                "n": int(fnum(row.get("n"), 0)),
                "base_value": fnum(row.get("base_value"), value if model == "base" else math.nan),
                "delta_vs_base": delta,
                "verdict": verdict(delta),
                "source": rel(summary_path),
                "note": "wide summary",
            }
        )

    for path in sorted((WIDE / "runs").glob("*/*/results_*.json")):
        parsed = run_state_from_name(path.relative_to(WIDE / "runs").parts[0])
        if parsed is None:
            continue
        model, group = parsed
        if model not in {"taskvec_a0p25", "scrambled"}:
            continue
        with path.open() as f:
            data = json.load(f)
        for task, metrics in sorted(data.get("results", {}).items()):
            picked = metric_for_task(task, metrics)
            if picked is None:
                continue
            metric, value = picked
            base = base_by_task_metric.get((task, metric), base_by_task.get(task, math.nan))
            delta = value - base if not math.isnan(base) else math.nan
            records.append(
                {
                    "skill": skill_for_task(task),
                    "task": task,
                    "task_label": task_label(task),
                    "model": model,
                    "metric": metric,
                    "value": value,
                    "n": int(metrics.get("sample_len", 0)),
                    "base_value": base,
                    "delta_vs_base": delta,
                    "verdict": verdict(delta),
                    "source": rel(path),
                    "note": f"completed split run: {group}",
                }
            )

    return sorted(records, key=lambda r: (str(r["skill"]), str(r["task"]), MODEL_ORDER.index(str(r["model"])) if r["model"] in MODEL_ORDER else 99, str(r["metric"])))


def load_semantic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []

    def add(task: str, model: str, metric: str, value: float, base_value: float, source: Path, note: str = "") -> None:
        delta = value - base_value if not math.isnan(base_value) else math.nan
        rows.append(
            {
                "skill": "semantic coherence/human similarity",
                "task": task,
                "model": model,
                "metric": metric,
                "value": value,
                "base_value": base_value,
                "delta_vs_base": delta,
                "verdict": verdict(delta),
                "source": rel(source),
                "note": note,
            }
        )

    mitigation_path = SFT_EVAL / "mitigation" / "summary.csv"
    mitigation = read_csv(mitigation_path)
    base_gen = next((fnum(r["gen_coherence"]) for r in mitigation if r["arm"] == "base"), math.nan)
    base_human = next((fnum(r["human_r2"]) for r in mitigation if r["arm"] == "base"), math.nan)
    for r in mitigation:
        arm = r["arm"]
        if arm not in {"base", "real", "lowLR", "lowrank"}:
            continue
        add("generation coherence", arm, "gen_proc_mean", fnum(r["gen_coherence"]), base_gen, mitigation_path)
        add("THINGS human triplet R2", arm, "human_r2", fnum(r["human_r2"]), base_human, mitigation_path)

    steer_path = SFT_EVAL / "steer" / "sweep_summary.csv"
    for r in read_csv(steer_path):
        if r.get("route") == "taskvec" and fmt(fnum(r.get("alpha")), 2) == "0.25":
            add("generation coherence", "taskvec_a0p25", "gen_proc_mean", fnum(r["gen_proc_mean"]), base_gen, steer_path, "task-vector alpha=0.25")
            add("THINGS human triplet R2", "taskvec_a0p25", "human_r2", fnum(r["human_things_triplet_r2"]), base_human, steer_path, "task-vector alpha=0.25")

    eval_path = SFT_EVAL / "eval_results.csv"
    for r in read_csv(eval_path):
        if r.get("state") != "scrambled":
            continue
        add("generation coherence", "scrambled", "gen_proc_mean", fnum(r["gen_proc_mean"]), base_gen, eval_path, "scrambled-label SFT control")
        add("THINGS human triplet R2", "scrambled", "human_r2", fnum(r["human_things_triplet_r2"]), base_human, eval_path, "scrambled-label SFT control")

    human_path = SFT_EVAL / "human_tasks" / "summary.csv"
    for r in read_csv(human_path):
        if r.get("group") != "human-behavior":
            continue
        model = r["model"]
        if model not in {"base", "real", "lowLR", "lowrank"}:
            continue
        add(r["task"], model, r["metric"], fnum(r["value"]), fnum(r["base_value"]), human_path, r.get("note", ""))

    external_path = SFT_EVAL / "external" / "summary.csv"
    base_external: dict[str, float] = {}
    external_rows = read_csv(external_path)
    for r in external_rows:
        if r["model"] == "base":
            base_external[r["benchmark"]] = fnum(r["rho"])
    for r in external_rows:
        if r["model"] not in {"base", "real", "lowLR", "lowrank"}:
            continue
        bench = r["benchmark"]
        add(f"external {bench}", r["model"], "spearman_rho", fnum(r["rho"]), base_external.get(bench, math.nan), external_path, "human similarity judgment dataset")

    return rows


def summarize_skills(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for r in records:
        if r["model"] == "base":
            continue
        if r["task"] in AGGREGATE_TASKS:
            continue
        if r["metric"] == "macro_acc":
            continue
        grouped[(str(r["skill"]), str(r["model"]))].append(r)

    out = []
    for (skill, model), rows in sorted(grouped.items(), key=lambda item: (item[0][0], MODEL_ORDER.index(item[0][1]) if item[0][1] in MODEL_ORDER else 99)):
        deltas = [fnum(r["delta_vs_base"]) for r in rows if not math.isnan(fnum(r["delta_vs_base"]))]
        if not deltas:
            continue
        out.append(
            {
                "skill": skill,
                "model": model,
                "n_tasks": len(deltas),
                "mean_delta": statistics.fmean(deltas),
                "median_delta": statistics.median(deltas),
                "min_delta": min(deltas),
                "max_delta": max(deltas),
                "hurt_count": sum(1 for d in deltas if verdict(d) == "hurt"),
                "preserved_count": sum(1 for d in deltas if verdict(d) == "preserved"),
                "boosted_count": sum(1 for d in deltas if verdict(d) == "boosted"),
                "verdict": verdict(statistics.fmean(deltas)),
                "included_tasks": "; ".join(sorted(str(r["task"]) for r in rows)),
            }
        )
    return out


def build_truthfulqa_mechanism(records: list[dict[str, object]]) -> list[dict[str, object]]:
    full = {
        str(r["model"]): r
        for r in records
        if r["task"] == "truthfulqa_mc2" and r["metric"] == "acc"
    }
    summary = {r["arm"]: r for r in read_csv(TRUTH_DIR / "summary.csv")}
    paired = {r["arm"]: r for r in read_csv(TRUTH_DIR / "paired_delta_summary.csv")}
    out: list[dict[str, object]] = []
    for arm in ["base", "lowLR", "lowrank", "taskvec_a0p25", "scrambled"]:
        frow = full.get(arm, {})
        srow = summary.get(arm, {})
        prow = paired.get(arm, {})
        mechanism = ""
        if arm == "lowrank":
            mechanism = "False-answer pressure falls more than truthful-answer mass on the 200-item slice."
        elif arm == "taskvec_a0p25":
            mechanism = "False-answer pressure rises more than truthful-answer mass; aggregate MC2 is nearly flat but calibration worsens."
        elif arm == "scrambled":
            mechanism = "Suppresses true and false likelihoods extremely hard, causing catastrophic item-level flips."
        elif arm == "lowLR":
            mechanism = "No logsample slice present for lowLR; full MC2 is within the flat threshold."
        else:
            mechanism = "Reference arm."
        out.append(
            {
                "arm": arm,
                "full_mc2_acc": frow.get("value", ""),
                "full_delta_vs_base": frow.get("delta_vs_base", ""),
                "diagnostic_n": srow.get("n", ""),
                "diagnostic_mc2_acc": srow.get("aggregate_acc", ""),
                "diagnostic_delta_acc": prow.get("mean_delta_mc2_acc", ""),
                "delta_truth_logodds": prow.get("mean_delta_truth_logodds", ""),
                "delta_true_mass": prow.get("mean_delta_true_logsumexp", ""),
                "delta_false_pressure": prow.get("mean_delta_false_logsumexp", ""),
                "frac_false_pressure_up": prow.get("frac_false_pressure_up", ""),
                "mechanism": mechanism,
            }
        )
    return out


def top_mmlu_drops(records: list[dict[str, object]], per_model: int = 12) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for model in ["lowLR", "lowrank"]:
        rows = [
            r
            for r in records
            if r["model"] == model
            and str(r["task"]).startswith("mmlu_")
            and r["task"] not in AGGREGATE_TASKS
            and r["metric"] == "acc"
        ]
        rows.sort(key=lambda r: fnum(r["delta_vs_base"]))
        for r in rows[:per_model]:
            out.append(
                {
                    "model": model,
                    "skill": r["skill"],
                    "task": r["task"],
                    "task_label": r["task_label"],
                    "base_value": r["base_value"],
                    "value": r["value"],
                    "delta_vs_base": r["delta_vs_base"],
                    "n": r["n"],
                }
            )
    return out


def value_for(records: Iterable[dict[str, object]], task: str, model: str, metric: str | None = None) -> dict[str, object] | None:
    matches = [r for r in records if r["task"] == task and r["model"] == model]
    if metric is not None:
        matches = [r for r in matches if r["metric"] == metric]
    if not matches:
        return None
    return matches[0]


def pivot_delta(records: list[dict[str, object]], task: str, metric: str | None = None) -> list[str]:
    row = value_for(records, task, "base", metric)
    base = fmt(row["value"]) if row else ""
    label = task_label(task)
    if task == "mmlu" and metric == "acc":
        label = "MMLU aggregate (micro acc)"
    elif task == "mmlu" and metric == "macro_acc":
        label = "MMLU aggregate (macro acc)"
    deltas = []
    for model in COMPARE_MODELS:
        r = value_for(records, task, model, metric)
        if r is None:
            deltas.append("")
        else:
            deltas.append(f"{fmt(r['value'])} ({fmt_delta(r['delta_vs_base'])})")
    return [label, base, *deltas]


def markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(out)


def skill_summary_cell(skill_rows: list[dict[str, object]], skill: str, model: str) -> str:
    row = next((r for r in skill_rows if r["skill"] == skill and r["model"] == model), None)
    if row is None:
        return ""
    counts = f"{row['hurt_count']} hurt/{row['preserved_count']} flat/{row['boosted_count']} boost"
    return f"{fmt_delta(row['mean_delta'])}; {counts}"


def semantic_metric(semantic_rows: list[dict[str, object]], task: str, model: str) -> dict[str, object] | None:
    return next((r for r in semantic_rows if r["task"] == task and r["model"] == model), None)


def write_report(records: list[dict[str, object]], skill_rows: list[dict[str, object]], semantic_rows: list[dict[str, object]], truth_rows: list[dict[str, object]], mmlu_drops: list[dict[str, object]]) -> None:
    semantic_table = []
    for model in ["base", "lowLR", "lowrank", "taskvec_a0p25", "real", "scrambled"]:
        gen = semantic_metric(semantic_rows, "generation coherence", model)
        human = semantic_metric(semantic_rows, "THINGS human triplet R2", model)
        if gen is None or human is None:
            continue
        semantic_table.append(
            [
                model,
                fmt(gen["value"]),
                fmt_delta(gen["delta_vs_base"]),
                fmt(human["value"]),
                fmt_delta(human["delta_vs_base"]),
                gen.get("source", ""),
            ]
        )

    headline_tasks = [
        ("piqa", None),
        ("openbookqa", None),
        ("commonsense_qa", None),
        ("wic", None),
        ("truthfulqa_mc2", None),
        ("winogrande", None),
        ("arc_easy", None),
        ("arc_challenge", None),
        ("hellaswag", None),
        ("mmlu", "acc"),
        ("mmlu", "macro_acc"),
    ]
    headline_rows = [pivot_delta(records, task, metric) for task, metric in headline_tasks]

    ordered_skills = [
        "script/discourse commonsense",
        "science/common-sense reasoning",
        "lexical/disambiguation",
        "truthfulness/calibration",
        "MMLU science/biomedical",
        "MMLU quantitative/formal",
        "MMLU technical/computing",
        "MMLU factual/history/policy",
        "MMLU professional/law/moral",
        "MMLU social/behavioral/business",
        "MMLU miscellaneous/broad",
    ]
    skill_table = []
    for skill in ordered_skills:
        if not any(r["skill"] == skill for r in skill_rows):
            continue
        skill_table.append(
            [
                skill,
                skill_summary_cell(skill_rows, skill, "lowLR"),
                skill_summary_cell(skill_rows, skill, "lowrank"),
                skill_summary_cell(skill_rows, skill, "taskvec_a0p25"),
                skill_summary_cell(skill_rows, skill, "scrambled"),
            ]
        )

    truth_table = []
    for row in truth_rows:
        truth_table.append(
            [
                row["arm"],
                fmt(row["full_mc2_acc"]),
                fmt_delta(row["full_delta_vs_base"]),
                fmt(row["diagnostic_mc2_acc"]),
                fmt_delta(row["diagnostic_delta_acc"]),
                fmt_delta(row["delta_truth_logodds"]),
                fmt_delta(row["delta_true_mass"]),
                fmt_delta(row["delta_false_pressure"]),
                fmt(row["frac_false_pressure_up"]),
            ]
        )

    mmlu_report_rows: list[dict[str, object]] = []
    for model in ["lowLR", "lowrank"]:
        mmlu_report_rows.extend([r for r in mmlu_drops if r["model"] == model][:8])
    mmlu_table = [
        [
            r["model"],
            r["task_label"],
            r["skill"],
            fmt(r["base_value"]),
            fmt(r["value"]),
            fmt_delta(r["delta_vs_base"]),
        ]
        for r in mmlu_report_rows
    ]

    lines = [
        "# Skill Diagnostics for Coherence SFT and Task Vectors",
        "",
        "Generated from existing artifacts only. No benchmarks were rerun.",
        "",
        "## Sources",
        "",
        "- `results/sft_eval/wide_bench/summary.csv` for base, lowLR, and lowrank wide-bench rows.",
        "- Completed split-run JSONs under `results/sft_eval/wide_bench/runs/` for task-vector and scrambled rows.",
        "- `results/sft_eval/wide_bench_diagnostics/truthfulqa_analysis/` for TruthfulQA log-sample decomposition.",
        "- `results/sft_eval/mitigation/summary.csv`, `results/sft_eval/steer/sweep_summary.csv`, `results/sft_eval/human_tasks/summary.csv`, and `results/sft_eval/external/summary.csv` for semantic/human-alignment gains.",
        "",
        "## Executive Read",
        "",
        "- The coherence intervention clearly boosts the intended skill: semantic coherence and human-similarity alignment. LowLR and lowrank roughly double generation coherence over base, and `taskvec_a0p25` recovers a large fraction of that gain without another training run.",
        "- The most reliably hurt skills are sharp multiple-choice ranking skills: ARC/OpenBookQA science, broad MMLU exam knowledge, WiC lexical sense discrimination, and especially MMLU moral/professional/formal subareas.",
        "- HellaSwag and WinoGrande are largely preserved by lowrank and `taskvec_a0p25`, so the drop is not a uniform few-shot or lm-eval failure. Script/event plausibility survives better than factual/exam retrieval.",
        "- TruthfulQA is not a large full-benchmark drop for lowLR/lowrank, but the logsample decomposition shows a real calibration mechanism: task-vector steering raises plausible false-answer pressure even when aggregate MC2 stays near flat.",
        "- Scrambled-label SFT is broadly damaging. That separates useful semantic training from generic adapter/SFT perturbation, but it also shows ARC/HellaSwag/TruthfulQA are sensitive to LoRA perturbations independent of useful alignment.",
        "",
        "## Intended Skill Gains",
        "",
        markdown_table(
            ["Arm", "Gen coherence", "Delta", "Human R2", "Delta", "Primary source"],
            semantic_table,
        ),
        "",
        "Interpretation: the boosted skill is not generic benchmark accuracy. It is a representation-level semantic geometry: consistency across triplet/pairwise/feature elicitation, alignment to THINGS-style human similarity, and improved external similarity datasets for the safer mitigation arms. That explains why gains transfer best to broad semantic association and event plausibility, not to narrow factual option ranking.",
        "",
        "## Wide-Bench Headline",
        "",
        "Cells are `value (delta vs base)`. Blank means that arm has no completed comparable run in the existing artifacts.",
        "",
        markdown_table(
            ["Task", "Base", "LowLR", "Lowrank", "Taskvec a0.25", "Scrambled"],
            headline_rows,
        ),
        "",
        "## Skill Family Summary",
        "",
        "Skill means exclude MMLU aggregate rows to avoid double-counting subtasks. Thresholds: hurt <= -0.02, boosted >= +0.02, otherwise preserved.",
        "",
        markdown_table(
            ["Skill family", "LowLR", "Lowrank", "Taskvec a0.25", "Scrambled"],
            skill_table,
        ),
        "",
        "## Where Gains Happen",
        "",
        "- Semantic coherence/human similarity: lowLR `gen_proc_mean` 0.757 vs base 0.323; lowrank 0.749; `taskvec_a0p25` 0.649. Human triplet R2 rises from 0.468 to 0.651/0.666 for lowLR/lowrank and 0.599 for `taskvec_a0p25`.",
        "- Human behavior/similarity transfer: existing human-task summaries show lowLR and lowrank improve THINGS odd-one-out agreement and triplet similarity. External similarity benchmarks are mostly preserved or improved for lowLR/lowrank; the default full SFT overfits some datasets and hurts MEN/RG-65.",
        "- Preserved benchmark skills: lowrank and `taskvec_a0p25` preserve HellaSwag (`-0.002`, `-0.005`) and WinoGrande stays flat for all aligned mitigation arms with available runs. PIQA is close to flat for lowLR and task-vector.",
        "",
        "## Where Drops Happen",
        "",
        "- MMLU: lowLR drops from 0.693 to 0.594 (`-0.099` micro; `-0.093` macro). Lowrank drops to 0.592 (`-0.101` micro; `-0.092` macro). The damage is broad, not one subject.",
        "- ARC/OpenBookQA: lowrank ARC-Easy/Challenge drops `-0.145`/`-0.141`; task-vector recovers part of ARC (`-0.042`/`-0.090`) but remains below base. OpenBookQA remains hurt for lowLR/lowrank/task-vector.",
        "- WiC: lowLR, lowrank, task-vector, and scrambled all land near chance, around 0.50. This is the cleanest lexical/disambiguation failure and likely reflects weakened context-specific sense boundaries.",
        "- Professional/law/moral: MMLU moral scenarios is the largest lowrank drop (`-0.317`) and also the largest lowLR drop (`-0.248`). Professional psychology, professional medicine, professional law, moral disputes, philosophy, and business ethics are also mostly down.",
        "",
        "Largest MMLU drops among lowLR/lowrank:",
        "",
        markdown_table(
            ["Arm", "MMLU task", "Skill", "Base", "Value", "Delta"],
            mmlu_table,
        ),
        "",
        "## TruthfulQA Mechanism",
        "",
        markdown_table(
            ["Arm", "Full MC2", "Full delta", "Slice MC2", "Slice delta", "Truth-logodds delta", "True mass delta", "False pressure delta", "False pressure up frac"],
            truth_table,
        ),
        "",
        "Mechanistic read:",
        "",
        "- Lowrank is flat on full TruthfulQA and improves the 200-item diagnostic slice because it lowers both truthful and false likelihood mass, but lowers false-answer pressure more.",
        "- `taskvec_a0p25` is nearly flat on the slice and mildly down on the full benchmark, but the logsample decomposition is worse: both true and false masses rise, and false pressure rises more. That is the specific calibration risk for task vectors.",
        "- Scrambled drops full MC2 and the diagnostic slice. Its truth-logodds rises only because it crushes both true and false likelihoods, producing unstable item-level flips rather than useful truthfulness.",
        "- Recurring worst lures are safety/myth/overgeneralization questions: defibrillation for flatline, washing chicken, Latin-American language overgeneralization, voodoo dolls, Agenda 21, and similar plausible-false answers.",
        "",
        "## Likely Failure Modes",
        "",
        "- Representation objective mismatch: the SFT teaches a smooth concept-similarity geometry, while ARC/MMLU/WiC/TruthfulQA require sharp option ranking, exception handling, and calibrated rejection of plausible distractors.",
        "- Adapter/output-surface perturbation: scrambled-label runs crater ARC, HellaSwag, CommonsenseQA, and TruthfulQA, so some loss comes from LoRA/SFT perturbing the answer-ranking surface even without useful semantic alignment.",
        "- Sense-boundary collapse: WiC drops across aligned and scrambled variants, suggesting global semantic association is a poor substitute for local sense disambiguation.",
        "- Long-option calibration: MMLU moral/professional/law prompts often have long, semantically close answer options. The coherence objective may make related distractors too competitive.",
        "- Partial task-vector coverage: `taskvec_a0p25` has no completed aggregate MMLU row in the current artifacts, so ARC improvements cannot yet be generalized to MMLU.",
        "- Metric comparability: full TruthfulQA and 200-item logsample diagnostics answer related but different questions. The diagnostic is mechanistic, not the final benchmark score.",
        "",
        "## Mitigation Ideas",
        "",
        "- Treat `taskvec_a0p25` as the current best representation-use arm, but do not assume it fixes retention until aggregate MMLU completes.",
        "- Add a small retention/KL replay mix during SFT: ARC/OpenBookQA, WiC-style sense contrasts, TruthfulQA false-lure calibration, and MMLU moral/professional/formal examples. Keep the base-logit KL on multiple-choice prompts.",
        "- Sweep task-vector alpha on the failure set (`0.1`, `0.2`, `0.25`, `0.3`, `0.5`) and score ARC, WiC, TruthfulQA logsamples, and a small MMLU diagnostic before running full MMLU.",
        "- For TruthfulQA, optimize relative true-vs-false mass rather than aggregate MC2 alone. Track `false_pressure_up` and worst lure classes.",
        "- For fMRI/representation experiments, separate hidden-state representational use from answer-generation use. The coherence adapter may be valuable for brain/human similarity even if it should be disabled or downweighted for calibrated MC answering.",
        "",
        "## Concrete Next Experiments",
        "",
        "1. Finish the comparable `taskvec_a0p25` MMLU 5-shot aggregate row, then re-run this script to fill the missing table cell.",
        "2. Run a cheap MMLU slice before full runs: moral_scenarios, formal_logic, medical_genetics, nutrition, professional_psychology, and high_school_statistics.",
        "3. Extend TruthfulQA logsamples to lowLR or the full 817 items if affordable; cluster item drops by lure type and compare false-answer pressure.",
        "4. Build a WiC slice by part of speech and lemma similarity to test whether the failure is global same-lemma sense collapse.",
        "5. Try retention replay or KL regularization against base logits on the specific failure families, then compare semantic coherence and failure-family deltas, not only aggregate retention.",
        "",
        "## Output Artifacts",
        "",
        "- `task_skill_deltas.csv`: per-task skill labels, values, deltas, verdicts, and sources.",
        "- `skill_summary.csv`: mean skill-family deltas by arm.",
        "- `semantic_human_summary.csv`: representation/human-similarity gains and external similarity rows.",
        "- `truthfulqa_mechanism.csv`: full MC2 plus logsample decomposition.",
        "- `mmlu_extreme_drops.csv`: largest MMLU subtask drops for lowLR and lowrank.",
        "",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records = load_wide_rows()
    skill_rows = summarize_skills(records)
    semantic_rows = load_semantic_rows()
    truth_rows = build_truthfulqa_mechanism(records)
    mmlu_drops = top_mmlu_drops(records)

    write_csv(
        OUT / "task_skill_deltas.csv",
        records,
        ["skill", "task", "task_label", "model", "metric", "value", "n", "base_value", "delta_vs_base", "verdict", "source", "note"],
    )
    write_csv(
        OUT / "skill_summary.csv",
        skill_rows,
        ["skill", "model", "n_tasks", "mean_delta", "median_delta", "min_delta", "max_delta", "hurt_count", "preserved_count", "boosted_count", "verdict", "included_tasks"],
    )
    write_csv(
        OUT / "semantic_human_summary.csv",
        semantic_rows,
        ["skill", "task", "model", "metric", "value", "base_value", "delta_vs_base", "verdict", "source", "note"],
    )
    write_csv(
        OUT / "truthfulqa_mechanism.csv",
        truth_rows,
        ["arm", "full_mc2_acc", "full_delta_vs_base", "diagnostic_n", "diagnostic_mc2_acc", "diagnostic_delta_acc", "delta_truth_logodds", "delta_true_mass", "delta_false_pressure", "frac_false_pressure_up", "mechanism"],
    )
    write_csv(
        OUT / "mmlu_extreme_drops.csv",
        mmlu_drops,
        ["model", "skill", "task", "task_label", "base_value", "value", "delta_vs_base", "n"],
    )
    write_report(records, skill_rows, semantic_rows, truth_rows, mmlu_drops)
    print(f"Wrote {rel(OUT)}")


if __name__ == "__main__":
    main()
