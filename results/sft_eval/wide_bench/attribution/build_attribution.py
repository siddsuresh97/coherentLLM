#!/usr/bin/env python3
"""Build attribution tables from existing wide-benchmark diagnostics.

This script is deliberately analysis-only: it reads already committed or staged
benchmark summaries and writes derived CSVs in this directory. It does not run
lm-eval, touch MMLU shard artifacts, or update shared diagnostics owned by
other agents.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
WIDE = ROOT / "results" / "sft_eval" / "wide_bench"
DIAG = WIDE / "skill_diagnostics"
OUT = WIDE / "attribution"

MODELS = ["lowLR", "lowrank", "taskvec_a0p25", "scrambled"]
HEADLINE_TASKS = [
    "piqa",
    "openbookqa",
    "commonsense_qa",
    "wic",
    "truthfulqa_mc2",
    "winogrande",
    "arc_easy",
    "arc_challenge",
    "hellaswag",
    "mmlu",
    "mmlu_stem",
    "mmlu_humanities",
    "mmlu_social_sciences",
    "mmlu_other",
]
CORE_SEMANTIC_TASKS = {
    "generation coherence",
    "THINGS human triplet R2",
    "THINGS odd-one-out",
    "THINGS triplet similarity",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})


def num(value: object) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value: float | None) -> str:
    if value is None:
        return ""
    return f"{value:.6f}"


def clean_verdict(value: str | None) -> str:
    if not value:
        return ""
    if value == "preserved":
        return "stable"
    return value


def task_read(task: str, skill: str) -> dict[str, str]:
    if task in {"hellaswag", "winogrande"}:
        return {
            "attribution_class": "preserved semantic-plausibility skill",
            "overlap_with_induced_skill": "high",
            "mechanism_read": "Coherent arms preserve script/discourse plausibility; scrambled SFT collapses HellaSwag where measured.",
            "mitigation_priority": "control",
            "next_eval": "Keep as preservation controls in every mitigation slice.",
        }
    if task == "piqa":
        return {
            "attribution_class": "mostly preserved affordance semantics",
            "overlap_with_induced_skill": "medium-high",
            "mechanism_read": "Task-vector is near base, suggesting broad affordance semantics overlap with the induced representation.",
            "mitigation_priority": "control",
            "next_eval": "Use PIQA as a cheap retention sentinel during alpha and replay sweeps.",
        }
    if task == "commonsense_qa":
        return {
            "attribution_class": "mixed associative commonsense",
            "overlap_with_induced_skill": "medium",
            "mechanism_read": "lowLR is stable, but lowrank/task-vector drop; associative cues alone are not enough for its answer ranking.",
            "mitigation_priority": "medium",
            "next_eval": "Inspect base-correct/taskvec-wrong items for distractor semantic closeness.",
        }
    if task in {"arc_easy", "arc_challenge", "openbookqa"}:
        return {
            "attribution_class": "science fact and option-ranking damage",
            "overlap_with_induced_skill": "partial",
            "mechanism_read": "Task-vector improves ARC/OpenBookQA relative to lowrank where measured, but all aligned arms remain below base.",
            "mitigation_priority": "high",
            "next_eval": "Run log-sample margin slices for ARC/OpenBookQA before any full retraining.",
        }
    if task == "wic":
        return {
            "attribution_class": "lexical sense-boundary failure",
            "overlap_with_induced_skill": "low or antagonistic",
            "mechanism_read": "All modified arms land near chance, consistent with semantic smoothing hurting same-lemma sense distinctions.",
            "mitigation_priority": "high",
            "next_eval": "Slice WiC errors by part of speech, same/different label, and lemma similarity.",
        }
    if task == "truthfulqa_mc2":
        return {
            "attribution_class": "false-lure calibration pressure",
            "overlap_with_induced_skill": "antagonistic for plausible false statements",
            "mechanism_read": "Full lowrank is stable, but task-vector raises false-answer pressure on the log-sample diagnostic.",
            "mitigation_priority": "high",
            "next_eval": "Track true mass, false pressure, and worst lure classes before optimizing MC2.",
        }
    if task.startswith("mmlu"):
        return {
            "attribution_class": "exam knowledge and close-option ranking damage",
            "overlap_with_induced_skill": "low",
            "mechanism_read": "Task-vector improves MMLU over lowrank but still drops broadly across exam families.",
            "mitigation_priority": "highest",
            "next_eval": "Use moral/formal/biomedical slices as cheap gates before any future full MMLU.",
        }
    return {
        "attribution_class": "unclassified",
        "overlap_with_induced_skill": "",
        "mechanism_read": f"Skill family: {skill}",
        "mitigation_priority": "",
        "next_eval": "",
    }


def skill_read(skill: str) -> dict[str, str]:
    if skill == "script/discourse commonsense":
        return {
            "attribution_class": "mostly preserved semantic plausibility",
            "overlap_with_induced_skill": "high",
            "mechanism_read": "Broad event, affordance, and discourse plausibility overlaps with the induced semantic geometry.",
            "mitigation": "Keep as retention controls; do not overfit mitigations that sacrifice these stable cells.",
        }
    if skill == "science/common-sense reasoning":
        return {
            "attribution_class": "hurt but partly recoverable with task-vector",
            "overlap_with_induced_skill": "partial",
            "mechanism_read": "Science benchmarks combine facts and close option ranking; semantic direction helps relative to lowrank but not enough.",
            "mitigation": "Use ARC/OpenBookQA margins and base-logit KL replay.",
        }
    if skill == "lexical/disambiguation":
        return {
            "attribution_class": "sharp boundary hurt",
            "overlap_with_induced_skill": "antagonistic",
            "mechanism_read": "WiC needs local sense boundaries rather than smooth concept similarity.",
            "mitigation": "Add WiC-style sense-contrast replay or adapter gating for lexical disambiguation.",
        }
    if skill == "truthfulness/calibration":
        return {
            "attribution_class": "calibration fragile",
            "overlap_with_induced_skill": "antagonistic for misconceptions",
            "mechanism_read": "Plausible false answers are semantically near the topic and can gain probability under task-vector addition.",
            "mitigation": "Optimize false-pressure deltas, not aggregate MC2 alone.",
        }
    if skill.startswith("MMLU"):
        return {
            "attribution_class": "broad factual/exam ranking hurt",
            "overlap_with_induced_skill": "low",
            "mechanism_read": "MMLU losses are broad, with worst drops in moral/professional/formal/biomedical subjects.",
            "mitigation": "Gate future alphas/curricula on a small MMLU failure slice before full MMLU.",
        }
    return {
        "attribution_class": "",
        "overlap_with_induced_skill": "",
        "mechanism_read": "",
        "mitigation": "",
    }


def build_headline() -> list[dict[str, object]]:
    rows = read_csv(DIAG / "task_skill_deltas.csv")
    grouped: dict[tuple[str, str], dict[str, dict[str, str]]] = defaultdict(dict)
    labels: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        task = row["task"]
        metric = row["metric"]
        if task not in HEADLINE_TASKS:
            continue
        key = (task, metric)
        grouped[key][row["model"]] = row
        labels[key] = {
            "skill": row["skill"],
            "task": task,
            "task_label": row["task_label"],
            "metric": metric,
        }

    out_rows: list[dict[str, object]] = []
    for key in sorted(grouped, key=lambda k: (HEADLINE_TASKS.index(k[0]), k[1])):
        model_rows = grouped[key]
        base = model_rows.get("base", {})
        read = task_read(key[0], labels[key]["skill"])
        item: dict[str, object] = {
            **labels[key],
            "base_value": fmt(num(base.get("value"))),
            **read,
        }
        for model in MODELS:
            row = model_rows.get(model, {})
            item[f"{model}_value"] = fmt(num(row.get("value")))
            item[f"{model}_delta_vs_base"] = fmt(num(row.get("delta_vs_base")))
            item[f"{model}_verdict"] = clean_verdict(row.get("verdict", ""))
        lowrank = num(model_rows.get("lowrank", {}).get("delta_vs_base"))
        taskvec = num(model_rows.get("taskvec_a0p25", {}).get("delta_vs_base"))
        if lowrank is not None and taskvec is not None:
            item["taskvec_delta_vs_lowrank"] = fmt(taskvec - lowrank)
        else:
            item["taskvec_delta_vs_lowrank"] = ""
        out_rows.append(item)
    return out_rows


def build_skill_families() -> list[dict[str, object]]:
    rows = read_csv(DIAG / "skill_summary.csv")
    grouped: dict[str, dict[str, dict[str, str]]] = defaultdict(dict)
    for row in rows:
        grouped[row["skill"]][row["model"]] = row

    out_rows: list[dict[str, object]] = []
    for skill in sorted(grouped):
        item: dict[str, object] = {"skill": skill, **skill_read(skill)}
        for model in MODELS:
            row = grouped[skill].get(model, {})
            item[f"{model}_mean_delta"] = fmt(num(row.get("mean_delta")))
            item[f"{model}_verdict"] = clean_verdict(row.get("verdict", ""))
            item[f"{model}_hurt_count"] = row.get("hurt_count", "")
            item[f"{model}_stable_count"] = row.get("preserved_count", "")
            item[f"{model}_boosted_count"] = row.get("boosted_count", "")
        out_rows.append(item)
    return out_rows


def build_semantic_targets() -> list[dict[str, object]]:
    rows = read_csv(DIAG / "semantic_human_summary.csv")
    out_rows: list[dict[str, object]] = []
    for row in rows:
        if row["task"] not in CORE_SEMANTIC_TASKS:
            continue
        out_rows.append(
            {
                "task": row["task"],
                "model": row["model"],
                "metric": row["metric"],
                "value": fmt(num(row["value"])),
                "base_value": fmt(num(row["base_value"])),
                "delta_vs_base": fmt(num(row["delta_vs_base"])),
                "verdict": clean_verdict(row["verdict"]),
                "attribution_class": "direct induced-skill target",
                "overlap_with_induced_skill": "target",
                "mechanism_read": "These are the intended semantic-coherence and human-alignment skills.",
                "source": row["source"],
            }
        )
    return out_rows


def build_truthfulqa() -> list[dict[str, object]]:
    rows = read_csv(DIAG / "truthfulqa_mechanism.csv")
    out_rows: list[dict[str, object]] = []
    for row in rows:
        arm = row["arm"]
        interpretation = row["mechanism"]
        if arm == "taskvec_a0p25":
            risk = "high"
            action = "Do not use MC2 alone; minimize false-pressure increase."
        elif arm == "lowrank":
            risk = "low on current slice"
            action = "Use as calibration reference against task-vector."
        elif arm == "scrambled":
            risk = "control failure"
            action = "Use to separate useful semantic direction from generic perturbation."
        elif arm == "lowLR":
            risk = "unknown item-level"
            action = "Run log-samples if this arm becomes the main candidate."
        else:
            risk = "reference"
            action = "Reference arm."
        out_rows.append(
            {
                "arm": arm,
                "full_mc2_acc": fmt(num(row["full_mc2_acc"])),
                "full_delta_vs_base": fmt(num(row["full_delta_vs_base"])),
                "diagnostic_n": row["diagnostic_n"],
                "diagnostic_mc2_acc": fmt(num(row["diagnostic_mc2_acc"])),
                "diagnostic_delta_acc": fmt(num(row["diagnostic_delta_acc"])),
                "delta_truth_logodds": fmt(num(row["delta_truth_logodds"])),
                "delta_true_mass": fmt(num(row["delta_true_mass"])),
                "delta_false_pressure": fmt(num(row["delta_false_pressure"])),
                "frac_false_pressure_up": fmt(num(row["frac_false_pressure_up"])),
                "risk_read": risk,
                "mechanism_read": interpretation,
                "next_action": action,
            }
        )
    return out_rows


def mitigation_rows() -> list[dict[str, str]]:
    return [
        {
            "priority": "P0",
            "target": "taskvec_a0p25 MMLU residual drop",
            "question": "Can a cheaper slice distinguish partial mitigation from broad retention recovery?",
            "cheap_gate": "moral_scenarios, formal_logic, medical_genetics, nutrition, professional_psychology, high_school_statistics.",
            "pass_signal": "Failure-slice mean delta >= -0.06 while semantic gains stay high.",
            "mitigation_if_fail": "Treat task-vector as representation-useful and only partially retention-safe.",
        },
        {
            "priority": "P1",
            "target": "TruthfulQA false-lure pressure",
            "question": "Are plausible false answers gaining more mass than true answers?",
            "cheap_gate": "200-item log-samples with true/false logsumexp decomposition.",
            "pass_signal": "frac_false_pressure_up <= 0.60 while semantic gain remains.",
            "mitigation_if_fail": "Add false-lure calibration replay or lower/task-specific alpha.",
        },
        {
            "priority": "P1",
            "target": "WiC lexical sense boundaries",
            "question": "Is the failure a same-lemma sense collapse?",
            "cheap_gate": "Slice WiC by POS, label, and lemma similarity.",
            "pass_signal": "Identify a concentrated error class rather than uniform chance.",
            "mitigation_if_fail": "Use adapter gating or lexical contrast replay before new full SFT.",
        },
        {
            "priority": "P1",
            "target": "MMLU moral/formal/biomedical slices",
            "question": "Are losses driven by close-option margin collapse?",
            "cheap_gate": "moral_scenarios, formal_logic, medical_genetics, nutrition, professional_psychology, high_school_statistics.",
            "pass_signal": "Correct-choice margins move toward base under a candidate mitigation.",
            "mitigation_if_fail": "Do not launch full MMLU for that candidate.",
        },
        {
            "priority": "P2",
            "target": "task-vector alpha Pareto sweep",
            "question": "Is there an alpha below 0.25 that preserves semantics with less calibration damage?",
            "cheap_gate": "alpha in 0, 0.1, 0.2, 0.25, 0.3, 0.5 on P1 suite.",
            "pass_signal": "generation coherence >= 0.60, human R2 >= 0.58, failure slice mean delta >= -0.06.",
            "mitigation_if_fail": "Move to KL/replay or selective steering rather than more alpha search.",
        },
        {
            "priority": "P3",
            "target": "selective use of semantic direction",
            "question": "Can we use the representation gain without global answer-ranking damage?",
            "cheap_gate": "Run CAA/steering forced-choice probes on target and retention sets.",
            "pass_signal": "Layer/alpha improves coherence/alignment probes with retention stable.",
            "mitigation_if_fail": "Restrict adapter to representation extraction/fMRI rather than benchmark answering.",
        },
    ]


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return ""


def main() -> None:
    headline = build_headline()
    skill_families = build_skill_families()
    semantic_targets = build_semantic_targets()
    truthfulqa = build_truthfulqa()
    mitigations = mitigation_rows()

    write_csv(
        OUT / "headline_task_attribution.csv",
        headline,
        [
            "skill",
            "task",
            "task_label",
            "metric",
            "base_value",
            "lowLR_value",
            "lowLR_delta_vs_base",
            "lowLR_verdict",
            "lowrank_value",
            "lowrank_delta_vs_base",
            "lowrank_verdict",
            "taskvec_a0p25_value",
            "taskvec_a0p25_delta_vs_base",
            "taskvec_a0p25_verdict",
            "scrambled_value",
            "scrambled_delta_vs_base",
            "scrambled_verdict",
            "taskvec_delta_vs_lowrank",
            "attribution_class",
            "overlap_with_induced_skill",
            "mechanism_read",
            "mitigation_priority",
            "next_eval",
        ],
    )
    write_csv(
        OUT / "skill_family_attribution.csv",
        skill_families,
        [
            "skill",
            "lowLR_mean_delta",
            "lowLR_verdict",
            "lowLR_hurt_count",
            "lowLR_stable_count",
            "lowLR_boosted_count",
            "lowrank_mean_delta",
            "lowrank_verdict",
            "lowrank_hurt_count",
            "lowrank_stable_count",
            "lowrank_boosted_count",
            "taskvec_a0p25_mean_delta",
            "taskvec_a0p25_verdict",
            "taskvec_a0p25_hurt_count",
            "taskvec_a0p25_stable_count",
            "taskvec_a0p25_boosted_count",
            "scrambled_mean_delta",
            "scrambled_verdict",
            "scrambled_hurt_count",
            "scrambled_stable_count",
            "scrambled_boosted_count",
            "attribution_class",
            "overlap_with_induced_skill",
            "mechanism_read",
            "mitigation",
        ],
    )
    write_csv(
        OUT / "semantic_target_gains.csv",
        semantic_targets,
        [
            "task",
            "model",
            "metric",
            "value",
            "base_value",
            "delta_vs_base",
            "verdict",
            "attribution_class",
            "overlap_with_induced_skill",
            "mechanism_read",
            "source",
        ],
    )
    write_csv(
        OUT / "truthfulqa_attribution.csv",
        truthfulqa,
        [
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
            "risk_read",
            "mechanism_read",
            "next_action",
        ],
    )
    write_csv(
        OUT / "mitigation_targets.csv",
        mitigations,
        [
            "priority",
            "target",
            "question",
            "cheap_gate",
            "pass_signal",
            "mitigation_if_fail",
        ],
    )

    metadata = {
        "commit": git_commit(),
        "inputs": [
            str(DIAG / "task_skill_deltas.csv"),
            str(DIAG / "skill_summary.csv"),
            str(DIAG / "semantic_human_summary.csv"),
            str(DIAG / "truthfulqa_mechanism.csv"),
        ],
        "outputs": [
            str(OUT / "headline_task_attribution.csv"),
            str(OUT / "skill_family_attribution.csv"),
            str(OUT / "semantic_target_gains.csv"),
            str(OUT / "truthfulqa_attribution.csv"),
            str(OUT / "mitigation_targets.csv"),
        ],
        "notes": [
            "No benchmarks are run by this script.",
            "taskvec_a0p25 aggregate MMLU is populated from the completed CHTC shard merge.",
        ],
    }
    with (OUT / "metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
