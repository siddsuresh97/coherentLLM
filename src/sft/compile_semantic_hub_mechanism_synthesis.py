"""Compile a semantic-hub mechanism synthesis from existing result artifacts."""
from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "results" / "sft_semantic_hub" / "mechanism_synthesis"

INPUTS = {
    "memp_controls": ROOT / "results" / "sft_semantic_hub" / "memp_paper_harness" / "control_summary.csv",
    "fmri_bridge": ROOT / "results" / "sft_fmri_hub_regression" / "cv_model_comparison.csv",
    "mitigation_targets": ROOT / "results" / "sft_eval" / "mitigation" / "summary.csv",
    "taskvec_targets": ROOT / "results" / "sft_eval" / "steer" / "sweep_summary.csv",
    "skill_families": ROOT / "results" / "sft_eval" / "wide_bench" / "attribution" / "skill_family_attribution.csv",
    "task_deltas": ROOT / "results" / "sft_eval" / "wide_bench" / "skill_diagnostics" / "task_skill_deltas.csv",
    "truthfulqa_mechanism": ROOT
    / "results"
    / "sft_eval"
    / "wide_bench"
    / "skill_diagnostics"
    / "truthfulqa_mechanism.csv",
    "truthfulqa_chtc40": ROOT
    / "results"
    / "sft_eval"
    / "wide_bench"
    / "failure_suite"
    / "chtc_5513424"
    / "extracted"
    / "gate"
    / "truthfulqa_analysis"
    / "paired_delta_summary.csv",
    "truthfulqa_chtc200": ROOT
    / "results"
    / "sft_eval"
    / "wide_bench"
    / "failure_suite"
    / "chtc_5513434"
    / "extracted"
    / "gate"
    / "truthfulqa_analysis"
    / "paired_delta_summary.csv",
    "retention_chtc200_gate": ROOT
    / "results"
    / "sft_eval"
    / "wide_bench"
    / "failure_suite"
    / "chtc_5513434"
    / "extracted"
    / "gate",
    "hubness": ROOT / "results" / "sft_semantic_hub" / "hubness" / "hubness_summary.csv",
}

ARM_ORDER = [
    "base",
    "lowLR",
    "lowrank",
    "taskvec_a0p25",
    "taskvec_a0p5",
    "taskvec_a1p0",
    "scrambled",
]

CSV_FIELDS = [
    "arm",
    "memp_random_delta",
    "memp_close_delta",
    "memp_category_delta",
    "memp_top1",
    "memp_top5",
    "hubness_top1_unique_fraction",
    "hubness_top1_max_occurrence",
    "hubness_csls_top5_accuracy",
    "hubness_csls_mnn_accuracy",
    "hubness_csls_top1_max_occurrence",
    "hubness_read",
    "hubness_corrected_read",
    "gen_coherence_delta",
    "human_r2_delta",
    "fmri_atl_delta_vs_single",
    "fmri_language_delta_vs_single",
    "fmri_ventral_delta_vs_single",
    "mmlu_aggregate_delta",
    "science_common_sense_delta",
    "wic_delta",
    "script_discourse_delta",
    "truthfulqa_full_delta",
    "truthfulqa_diag_delta_acc",
    "truthfulqa_diag_false_pressure_up",
    "truthfulqa_chtc40_delta_mc2",
    "truthfulqa_chtc40_false_pressure_up",
    "truthfulqa_chtc200_delta_mc2",
    "truthfulqa_chtc200_false_pressure_up",
    "wic_chtc200_delta_acc",
    "openbookqa_chtc200_delta_acc_norm",
    "interpretive_class",
    "primary_use",
    "main_risk",
]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    return ap.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def to_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def fmt(value: str | float | None, digits: int = 4) -> str:
    if isinstance(value, str):
        value = to_float(value)
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def alpha_arm(alpha: str) -> str:
    value = float(alpha)
    if value == 0:
        return "base"
    if value == 1.0:
        return "taskvec_a1p0"
    text = f"{value:g}".replace(".", "p")
    return f"taskvec_a{text}"


def load_memp() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    control_keys = {
        "random": "random",
        "close_neighbor": "close",
        "far_neighbor": "far",
        "lexical": "lexical",
        "category_proxy": "category",
    }
    for row in read_csv(INPUTS["memp_controls"]):
        arm = row["arm"]
        control = row["control_type"]
        out.setdefault(arm, {})
        delta = to_float(row.get("mid_paired_delta"))
        if delta is not None:
            out[arm][f"memp_{control_keys.get(control, control)}_delta"] = delta
        if control == "random":
            for source, dest in (
                ("mid_retrieval_top1", "memp_top1"),
                ("mid_retrieval_top5", "memp_top5"),
            ):
                value = to_float(row.get(source))
                if value is not None:
                    out[arm][dest] = value
    return out


def load_semantic_targets() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    base_gen = None
    base_human = None
    for row in read_csv(INPUTS["mitigation_targets"]):
        if row.get("arm") == "base":
            base_gen = to_float(row.get("gen_coherence"))
            base_human = to_float(row.get("human_r2"))
            break

    for row in read_csv(INPUTS["mitigation_targets"]):
        arm = row.get("arm", "")
        gen = to_float(row.get("gen_coherence"))
        human = to_float(row.get("human_r2"))
        out.setdefault(arm, {})
        if gen is not None and base_gen is not None:
            out[arm]["gen_coherence_delta"] = gen - base_gen
        if human is not None and base_human is not None:
            out[arm]["human_r2_delta"] = human - base_human

    for row in read_csv(INPUTS["taskvec_targets"]):
        if row.get("route") != "taskvec" or row.get("complete") != "True":
            continue
        arm = alpha_arm(row.get("alpha", ""))
        gen = to_float(row.get("gen_proc_mean"))
        human = to_float(row.get("human_things_triplet_r2"))
        out.setdefault(arm, {})
        if gen is not None and base_gen is not None:
            out[arm]["gen_coherence_delta"] = gen - base_gen
        if human is not None and base_human is not None:
            out[arm]["human_r2_delta"] = human - base_human
    return out


def load_fmri() -> dict[str, dict[str, float]]:
    key_by_region = {
        "ATL (Semantic)": "fmri_atl_delta_vs_single",
        "Language": "fmri_language_delta_vs_single",
        "Ventral Visual": "fmri_ventral_delta_vs_single",
    }
    out: dict[str, dict[str, float]] = {}
    for row in read_csv(INPUTS["fmri_bridge"]):
        key = key_by_region.get(row.get("region", ""))
        if key is None:
            continue
        value = to_float(row.get("mean_repr_minus_best_single"))
        if value is None:
            continue
        out.setdefault(row["arm"], {})[key] = value
    return out


def load_benchmarks() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for row in read_csv(INPUTS["task_deltas"]):
        if row.get("task") == "mmlu" and row.get("metric") == "acc":
            value = to_float(row.get("delta_vs_base"))
            if value is not None:
                out.setdefault(row["model"], {})["mmlu_aggregate_delta"] = value

    family_keys = {
        "science/common-sense reasoning": "science_common_sense_delta",
        "lexical/disambiguation": "wic_delta",
        "script/discourse commonsense": "script_discourse_delta",
    }
    arms = ["lowLR", "lowrank", "taskvec_a0p25", "scrambled"]
    for row in read_csv(INPUTS["skill_families"]):
        dest = family_keys.get(row.get("skill", ""))
        if dest is None:
            continue
        for arm in arms:
            value = to_float(row.get(f"{arm}_mean_delta"))
            if value is not None:
                out.setdefault(arm, {})[dest] = value

    for row in read_csv(INPUTS["truthfulqa_mechanism"]):
        arm = row.get("arm", "")
        if not arm:
            continue
        out.setdefault(arm, {})
        for source, dest in (
            ("full_delta_vs_base", "truthfulqa_full_delta"),
            ("diagnostic_delta_acc", "truthfulqa_diag_delta_acc"),
            ("frac_false_pressure_up", "truthfulqa_diag_false_pressure_up"),
        ):
            value = to_float(row.get(source))
            if value is not None:
                out[arm][dest] = value

    for row in read_csv(INPUTS["truthfulqa_chtc40"]):
        arm = row.get("arm", "")
        if not arm:
            continue
        out.setdefault(arm, {})
        for source, dest in (
            ("mean_delta_mc2_acc", "truthfulqa_chtc40_delta_mc2"),
            ("frac_false_pressure_up", "truthfulqa_chtc40_false_pressure_up"),
        ):
            value = to_float(row.get(source))
            if value is not None:
                out[arm][dest] = value

    for row in read_csv(INPUTS["truthfulqa_chtc200"]):
        arm = row.get("arm", "")
        if not arm:
            continue
        out.setdefault(arm, {})
        for source, dest in (
            ("mean_delta_mc2_acc", "truthfulqa_chtc200_delta_mc2"),
            ("frac_false_pressure_up", "truthfulqa_chtc200_false_pressure_up"),
        ):
            value = to_float(row.get(source))
            if value is not None:
                out[arm][dest] = value

    gate_dir = INPUTS["retention_chtc200_gate"]
    for task, metric, dest in (
        ("wic", "acc,none", "wic_chtc200_delta_acc"),
        ("openbookqa", "acc_norm,none", "openbookqa_chtc200_delta_acc_norm"),
    ):
        task_scores: dict[str, float] = {}
        for result_path in sorted((gate_dir / "lm_eval").glob(f"*_{task}_limit200/**/results_*.json")):
            run_name = result_path.relative_to(gate_dir / "lm_eval").parts[0]
            arm = run_name.removesuffix(f"_{task}_limit200")
            data = json.loads(result_path.read_text())
            value = to_float(str(data.get("results", {}).get(task, {}).get(metric, "")))
            if value is not None:
                task_scores[arm] = value
        base = task_scores.get("base")
        if base is None:
            continue
        out.setdefault("base", {})[dest] = 0.0
        for arm, score in task_scores.items():
            out.setdefault(arm, {})[dest] = score - base
    return out


def load_hubness() -> dict[str, dict[str, float | str]]:
    out: dict[str, dict[str, float | str]] = {}
    for row in read_csv(INPUTS["hubness"]):
        arm = row.get("arm", "")
        if not arm:
            continue
        out.setdefault(arm, {})
        for source, dest in (
            ("mid_top1_unique_fraction", "hubness_top1_unique_fraction"),
            ("mid_top1_max_occurrence", "hubness_top1_max_occurrence"),
            ("mid_csls_top5_accuracy", "hubness_csls_top5_accuracy"),
            ("mid_csls_mnn_accuracy", "hubness_csls_mnn_accuracy"),
            ("mid_csls_top1_max_occurrence", "hubness_csls_top1_max_occurrence"),
        ):
            value = to_float(row.get(source))
            if value is not None:
                out[arm][dest] = value
        out[arm]["hubness_read"] = row.get("hubness_read", "")
        out[arm]["hubness_corrected_read"] = row.get("corrected_read", "")
    return out


def classify(row: dict[str, float | str]) -> tuple[str, str, str]:
    arm = str(row["arm"])
    if arm == "base":
        return "reference", "baseline comparison point", "none"
    if arm == "scrambled":
        return (
            "nonspecific perturbation control",
            "negative control for adapter/SFT artifacts",
            "can look hub-like under loose controls while hurting behavior",
        )
    if arm == "taskvec_a0p25":
        return (
            "usable but retention-risky task vector",
            "large cheap recovery of semantic coherence without full SFT",
            "false-lure pressure, WiC, science, and MMLU drops",
        )
    if arm in {"taskvec_a0p5", "taskvec_a1p0"}:
        return (
            "stronger diagnostic hub vector",
            "probe dose-response and causal hub hypotheses",
            "not CHTC-staged or wide-bench-gated yet",
        )
    if arm in {"lowLR", "lowrank"}:
        return (
            "trained coherence adapter",
            "semantic/human-similarity gain with safer TruthfulQA pressure than taskvec",
            "broad MMLU, WiC, and science option-ranking loss",
        )
    return "unclassified", "needs review", "needs review"


def build_rows() -> list[dict[str, str]]:
    sources = [load_memp(), load_hubness(), load_semantic_targets(), load_fmri(), load_benchmarks()]
    arms = set(ARM_ORDER)
    for source in sources:
        arms.update(source)

    rows: list[dict[str, str]] = []
    ordered = [arm for arm in ARM_ORDER if arm in arms]
    for arm in ordered:
        numeric: dict[str, float | str] = {"arm": arm}
        for source in sources:
            numeric.update(source.get(arm, {}))
        interp, use, risk = classify(numeric)
        numeric["interpretive_class"] = interp
        numeric["primary_use"] = use
        numeric["main_risk"] = risk
        text_fields = {
            "arm",
            "hubness_read",
            "hubness_corrected_read",
            "interpretive_class",
            "primary_use",
            "main_risk",
        }
        rows.append(
            {
                field: fmt(numeric.get(field)) if field not in text_fields else str(numeric.get(field, ""))
                for field in CSV_FIELDS
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def best_arm(rows: list[dict[str, str]], field: str) -> tuple[str, float] | None:
    vals = []
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            vals.append((row["arm"], value))
    if not vals:
        return None
    return max(vals, key=lambda item: item[1])


def get_row(rows: list[dict[str, str]], arm: str) -> dict[str, str]:
    return next(row for row in rows if row["arm"] == arm)


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    random_best = best_arm(rows, "memp_random_delta")
    close_best = best_arm(rows, "memp_close_delta")
    category_best = best_arm(rows, "memp_category_delta")
    taskvec = get_row(rows, "taskvec_a0p25")
    lowlr = get_row(rows, "lowLR")

    table_fields = [
        ("Arm", "arm"),
        ("Rand", "memp_random_delta"),
        ("Close", "memp_close_delta"),
        ("Cat", "memp_category_delta"),
        ("Top5", "memp_top5"),
        ("Unique", "hubness_top1_unique_fraction"),
        ("MaxHub", "hubness_top1_max_occurrence"),
        ("CSLS5", "hubness_csls_top5_accuracy"),
        ("CSLS-MNN", "hubness_csls_mnn_accuracy"),
        ("CSLSHub", "hubness_csls_top1_max_occurrence"),
        ("Gen", "gen_coherence_delta"),
        ("Human", "human_r2_delta"),
        ("MMLU", "mmlu_aggregate_delta"),
        ("WiC", "wic_delta"),
        ("Sci", "science_common_sense_delta"),
        ("TQA false-up", "truthfulqa_chtc40_false_pressure_up"),
        ("TQA200 false-up", "truthfulqa_chtc200_false_pressure_up"),
        ("WiC200", "wic_chtc200_delta_acc"),
        ("OBQA200", "openbookqa_chtc200_delta_acc_norm"),
        ("Lang brain", "fmri_language_delta_vs_single"),
    ]

    lines = [
        "# Semantic Hub Mechanism Synthesis",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Bottom Line",
        "",
        "- Coherence training and task vectors do induce a stronger paper-style semantic-hub signal,",
        "  especially same-concept cross-spoke similarity over random/far/category controls.",
        "- The useful skill is semantic geometry and human-similarity alignment, not generic",
        "  multiple-choice competence. That explains why script/discourse plausibility mostly",
        "  survives while MMLU, WiC, ARC/OpenBookQA, and TruthfulQA false-lure calibration are fragile.",
        "- `taskvec_a0p25` is the best currently staged cheap adapter. It is not the",
        "  strongest broad-similarity arm and it carries a false-lure pressure risk,",
        "  but it is the only arm that remains partly robust after CSLS and",
        "  mutual-nearest-neighbor hubness correction. The stronger broad/strict hub",
        "  arms, `taskvec_a0p5` and `taskvec_a1p0`, need retention gates before they",
        "  can become candidates.",
        "",
        "## Joined Arm Summary",
        "",
        "| " + " | ".join(label for label, _ in table_fields) + " |",
        "|" + "|".join(["---"] + ["---:" for _ in table_fields[1:]]) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(field, "") for _, field in table_fields) + " |")

    lines.extend(
        [
            "",
            "## Mechanistic Read",
            "",
            f"- Strongest broad same-minus-random hub signal: `{random_best[0]}` `{random_best[1]:.4f}`."
            if random_best
            else "- Strongest broad same-minus-random hub signal: unavailable.",
            f"- Strongest strict same-minus-S*-close signal: `{close_best[0]}` `{close_best[1]:.4f}`."
            if close_best
            else "- Strongest strict same-minus-S*-close signal: unavailable.",
            f"- Strongest category-proxy signal: `{category_best[0]}` `{category_best[1]:.4f}`."
            if category_best
            else "- Strongest category-proxy signal: unavailable.",
            f"- `taskvec_a0p25` boosts generation coherence by `{taskvec['gen_coherence_delta']}` and human R2 by `{taskvec['human_r2_delta']}`, but has MMLU delta `{taskvec['mmlu_aggregate_delta']}`, wide-bench WiC delta `{taskvec['wic_delta']}`, CHTC-200 WiC delta `{taskvec['wic_chtc200_delta_acc']}`, and CHTC-200 TruthfulQA false-pressure-up `{taskvec['truthfulqa_chtc200_false_pressure_up']}`.",
            f"- Hubness control: `taskvec_a0p25` has top-1 unique fraction `{taskvec['hubness_top1_unique_fraction']}` and max mid-layer attractor occurrence `{taskvec['hubness_top1_max_occurrence']}`; its read is `{taskvec['hubness_read']}`.",
            f"- CSLS correction: `taskvec_a0p25` has CSLS top-5 `{taskvec['hubness_csls_top5_accuracy']}`, CSLS mutual-nearest accuracy `{taskvec['hubness_csls_mnn_accuracy']}`, and CSLS max attractor `{taskvec['hubness_csls_top1_max_occurrence']}`; its corrected read is `{taskvec['hubness_corrected_read']}`.",
            f"- `lowLR` has a smaller MEMP random delta `{lowlr['memp_random_delta']}` than taskvec, but its CHTC-200 false-pressure-up `{lowlr['truthfulqa_chtc200_false_pressure_up']}` is much safer and its bounded MC2 delta `{lowlr['truthfulqa_chtc200_delta_mc2']}` beats taskvec `{taskvec['truthfulqa_chtc200_delta_mc2']}`.",
            "",
            "This pattern is consistent with a smooth semantic-centralization benefit that helps",
            "cross-format similarity and some language/semantic brain alignment, but blurs sharp",
            "option boundaries. TruthfulQA is vulnerable because many false answers are semantically",
            "near the topic; a semantic direction can raise both truthful and plausible-false mass.",
            "",
            "## What The Model Is Good For",
            "",
            "- Semantic similarity, THINGS-style human-alignment probes, and cross-format concept consistency.",
            "- Brain-facing hypotheses in ATL/Language regions, especially as exploratory layer/arm predictors.",
            "- Mechanistic probes of how a mid-layer task vector changes concept geometry.",
            "",
            "## What It Is Not Good For Yet",
            "",
            "- Exact concept identity against S*-close neighbors; the strict close-control deltas remain small.",
            "- Lexical sense disambiguation and science/exam option ranking.",
            "- Truthfulness mitigation unless false-answer pressure is explicitly gated.",
            "",
            "## Next Experiments",
            "",
            "1. Do not promote `taskvec_a0p25` on TruthfulQA without a false-pressure mitigation; the limit-200 gate keeps the risk.",
            "2. Stage `out/adapters_taskvec_scaled/a0p5` only after extending the failure-suite runner with a `taskvec_a0p5` arm, then run a small false-pressure/WiC/OpenBookQA gate before any wider eval.",
            "3. Use the CSLS/mutual-neighbor result to prioritize `taskvec_a0p25` for causal validation before stronger alpha arms.",
            "4. Add causal cross-spoke patching or activation addition: patch triplet-format concept states into pairwise/feature prompts and require same-concept improvement over random, close-neighbor, wrong-layer, and shuffled-vector controls.",
            "5. For cognitive science, prioritize Huth/LeBel high-data story scaling and ATL/Language-region tests; use hub metrics as predictors, not as standalone brain claims.",
            "",
            "## Files",
            "",
            "- `joined_arm_summary.csv`: one-row-per-arm synthesis table.",
            "- `source_manifest.json`: exact source CSVs used.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def write_manifest(path: Path) -> None:
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "script": "src/sft/compile_semantic_hub_mechanism_synthesis.py",
        "inputs": {name: str(path.relative_to(ROOT)) for name, path in INPUTS.items()},
        "notes": [
            "Uses existing artifacts only; no model inference is launched.",
            "CHTC run 5513434 is included after successful local pull.",
        ],
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir
    rows = build_rows()
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "joined_arm_summary.csv", rows)
    write_report(out_dir / "REPORT.md", rows)
    write_manifest(out_dir / "source_manifest.json")
    print(f"[ok] wrote {out_dir}", flush=True)


if __name__ == "__main__":
    main()
