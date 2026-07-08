# Task 6 Summary: lowLR vs base

Best mitigation model summarized here: lowLR (`out/adapters_mitigation/lowLR`, evaluated through the vLLM-compatible adapter where applicable). Verdict rule: GAIN if delta_vs_base > +0.02, DROP if delta_vs_base < -0.02, otherwise FLAT. Values are rounded for display; CSV/JSON outputs retain full precision.

Scope note for 6C: `src/sft/eval_wide_bench.py` uses a `zero_shot` group that bundles `piqa`, `openbookqa`, `commonsense_qa`, `wic`, and `truthfulqa_mc2`. The other groups cover `winogrande`, `arc_easy`, `arc_challenge`, `hellaswag`, and `mmlu`, so the requested capability tasks were not dropped.

## Similarity Benchmarks (6A)

Verdict counts for lowLR vs base: GAIN=4, FLAT=3, DROP=0, NA=0.

| task | metric | base | lowLR | delta_vs_base | verdict | real | lowrank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STS-B | spearman_rho | 0.518 | 0.527 | 0.009 | FLAT | 0.527 | 0.519 |
| SimLex-999 | spearman_rho | 0.297 | 0.310 | 0.012 | FLAT | 0.312 | 0.311 |
| WordSim-353 | spearman_rho | 0.294 | 0.344 | 0.050 | GAIN | 0.418 | 0.291 |
| MEN | spearman_rho | 0.488 | 0.518 | 0.030 | GAIN | 0.430 | 0.506 |
| RG-65 | spearman_rho | 0.358 | 0.357 | -0.001 | FLAT | 0.209 | 0.353 |
| MTurk-771 | spearman_rho | 0.306 | 0.353 | 0.047 | GAIN | 0.380 | 0.358 |
| SimVerb-3500 | spearman_rho | 0.168 | 0.199 | 0.032 | GAIN | 0.235 | 0.193 |

## Human-Behavior Benchmarks (6B)

Verdict counts for lowLR vs base: GAIN=6, FLAT=2, DROP=0, NA=1.

| task | metric | base | lowLR | delta_vs_base | verdict | real | lowrank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| THINGS odd-one-out | agreement | 0.686 | 0.770 | 0.084 | GAIN | 0.765 | 0.762 |
| THINGS triplet similarity | spearman_rho | 0.483 | 0.711 | 0.228 | GAIN | 0.700 | 0.723 |
| SimLex-999 | spearman_rho | 0.297 | 0.310 | 0.012 | FLAT | 0.312 | 0.311 |
| WordSim-353 | spearman_rho | 0.294 | 0.344 | 0.050 | GAIN | 0.418 | 0.291 |
| MEN | spearman_rho | 0.488 | 0.518 | 0.030 | GAIN | 0.430 | 0.506 |
| RG-65 | spearman_rho | 0.358 | 0.357 | -0.001 | FLAT | 0.209 | 0.353 |
| MTurk-771 | spearman_rho | 0.306 | 0.353 | 0.047 | GAIN | 0.380 | 0.358 |
| SimVerb-3500 | spearman_rho | 0.168 | 0.199 | 0.032 | GAIN | 0.235 | 0.193 |
| Typicality/category norms | not_run | NA | not_run | NA | NA | NA | NA |

## Capability Benchmarks (6C)

Verdict counts for lowLR vs base: GAIN=0, FLAT=4, DROP=6, NA=0.

| task | metric | base | lowLR | delta_vs_base | verdict | real | lowrank |
| --- | --- | --- | --- | --- | --- | --- | --- |
| piqa | acc_norm | 0.799 | 0.779 | -0.020 | FLAT | NA | NA |
| winogrande | acc | 0.762 | 0.765 | 0.003 | FLAT | NA | NA |
| arc_easy | acc_norm | 0.850 | 0.725 | -0.125 | DROP | NA | NA |
| arc_challenge | acc_norm | 0.649 | 0.524 | -0.125 | DROP | NA | NA |
| openbookqa | acc_norm | 0.490 | 0.412 | -0.078 | DROP | NA | NA |
| commonsense_qa | acc | 0.651 | 0.666 | 0.015 | FLAT | NA | NA |
| hellaswag | acc_norm | 0.685 | 0.656 | -0.029 | DROP | NA | NA |
| wic | acc | 0.652 | 0.500 | -0.152 | DROP | NA | NA |
| truthfulqa_mc2 | acc | 0.550 | 0.533 | -0.017 | FLAT | NA | NA |
| mmlu | acc | 0.693 | 0.594 | -0.099 | DROP | NA | NA |

## MMLU Detailed Capability Rows

These are the MMLU subject/category rows emitted by lm-eval in addition to the aggregate MMLU row above.

| task | metric | base | lowLR | delta_vs_base | verdict |
| --- | --- | --- | --- | --- | --- |
| mmlu_abstract_algebra | acc | 0.370 | 0.340 | -0.030 | DROP |
| mmlu_anatomy | acc | 0.674 | 0.607 | -0.067 | DROP |
| mmlu_astronomy | acc | 0.750 | 0.599 | -0.151 | DROP |
| mmlu_business_ethics | acc | 0.740 | 0.600 | -0.140 | DROP |
| mmlu_clinical_knowledge | acc | 0.755 | 0.642 | -0.113 | DROP |
| mmlu_college_biology | acc | 0.826 | 0.674 | -0.153 | DROP |
| mmlu_college_chemistry | acc | 0.480 | 0.400 | -0.080 | DROP |
| mmlu_college_computer_science | acc | 0.550 | 0.530 | -0.020 | FLAT |
| mmlu_college_mathematics | acc | 0.400 | 0.370 | -0.030 | DROP |
| mmlu_college_medicine | acc | 0.682 | 0.572 | -0.110 | DROP |
| mmlu_college_physics | acc | 0.431 | 0.314 | -0.118 | DROP |
| mmlu_computer_security | acc | 0.790 | 0.660 | -0.130 | DROP |
| mmlu_conceptual_physics | acc | 0.613 | 0.553 | -0.060 | DROP |
| mmlu_econometrics | acc | 0.579 | 0.439 | -0.140 | DROP |
| mmlu_electrical_engineering | acc | 0.655 | 0.566 | -0.090 | DROP |
| mmlu_elementary_mathematics | acc | 0.471 | 0.413 | -0.058 | DROP |
| mmlu_formal_logic | acc | 0.579 | 0.389 | -0.190 | DROP |
| mmlu_global_facts | acc | 0.400 | 0.350 | -0.050 | DROP |
| mmlu_high_school_biology | acc | 0.813 | 0.716 | -0.097 | DROP |
| mmlu_high_school_chemistry | acc | 0.621 | 0.507 | -0.113 | DROP |
| mmlu_high_school_computer_science | acc | 0.740 | 0.670 | -0.070 | DROP |
| mmlu_high_school_european_history | acc | 0.770 | 0.739 | -0.030 | DROP |
| mmlu_high_school_geography | acc | 0.848 | 0.717 | -0.131 | DROP |
| mmlu_high_school_government_and_politics | acc | 0.912 | 0.808 | -0.104 | DROP |
| mmlu_high_school_macroeconomics | acc | 0.695 | 0.603 | -0.092 | DROP |
| mmlu_high_school_mathematics | acc | 0.422 | 0.348 | -0.074 | DROP |
| mmlu_high_school_microeconomics | acc | 0.773 | 0.685 | -0.088 | DROP |
| mmlu_high_school_physics | acc | 0.450 | 0.417 | -0.033 | DROP |
| mmlu_high_school_psychology | acc | 0.872 | 0.783 | -0.088 | DROP |
| mmlu_high_school_statistics | acc | 0.616 | 0.444 | -0.171 | DROP |
| mmlu_high_school_us_history | acc | 0.824 | 0.721 | -0.103 | DROP |
| mmlu_high_school_world_history | acc | 0.840 | 0.797 | -0.042 | DROP |
| mmlu_human_aging | acc | 0.700 | 0.650 | -0.049 | DROP |
| mmlu_human_sexuality | acc | 0.802 | 0.618 | -0.183 | DROP |
| mmlu_humanities | acc | 0.664 | 0.548 | -0.116 | DROP |
| mmlu_international_law | acc | 0.826 | 0.736 | -0.091 | DROP |
| mmlu_jurisprudence | acc | 0.778 | 0.769 | -0.009 | FLAT |
| mmlu_logical_fallacies | acc | 0.804 | 0.681 | -0.123 | DROP |
| mmlu_machine_learning | acc | 0.545 | 0.464 | -0.080 | DROP |
| mmlu_management | acc | 0.816 | 0.806 | -0.010 | FLAT |
| mmlu_marketing | acc | 0.897 | 0.855 | -0.043 | DROP |
| mmlu_medical_genetics | acc | 0.820 | 0.640 | -0.180 | DROP |
| mmlu_miscellaneous | acc | 0.838 | 0.762 | -0.075 | DROP |
| mmlu_moral_disputes | acc | 0.766 | 0.673 | -0.092 | DROP |
| mmlu_moral_scenarios | acc | 0.570 | 0.322 | -0.248 | DROP |
| mmlu_nutrition | acc | 0.807 | 0.650 | -0.157 | DROP |
| mmlu_other | acc | 0.746 | 0.658 | -0.089 | DROP |
| mmlu_philosophy | acc | 0.740 | 0.656 | -0.084 | DROP |
| mmlu_prehistory | acc | 0.753 | 0.645 | -0.108 | DROP |
| mmlu_professional_accounting | acc | 0.546 | 0.493 | -0.053 | DROP |
| mmlu_professional_law | acc | 0.496 | 0.434 | -0.062 | DROP |
| mmlu_professional_medicine | acc | 0.776 | 0.636 | -0.140 | DROP |
| mmlu_professional_psychology | acc | 0.729 | 0.603 | -0.126 | DROP |
| mmlu_public_relations | acc | 0.691 | 0.627 | -0.064 | DROP |
| mmlu_security_studies | acc | 0.718 | 0.682 | -0.037 | DROP |
| mmlu_social_sciences | acc | 0.780 | 0.680 | -0.100 | DROP |
| mmlu_sociology | acc | 0.836 | 0.761 | -0.075 | DROP |
| mmlu_stem | acc | 0.594 | 0.507 | -0.087 | DROP |
| mmlu_us_foreign_policy | acc | 0.890 | 0.800 | -0.090 | DROP |
| mmlu_virology | acc | 0.524 | 0.476 | -0.048 | DROP |
| mmlu_world_religions | acc | 0.830 | 0.754 | -0.076 | DROP |

## Plain Answer

lowLR improves the coherence-adjacent and human-behavior measurements much more reliably than it improves general capability. In 6A it gains on WordSim-353, MEN, MTurk-771, SimVerb-3500. In 6B it gains on THINGS odd-one-out, THINGS triplet similarity, WordSim-353, MEN, MTurk-771, SimVerb-3500. In the 6C named capability battery it has no >0.02 gains; `commonsense_qa`, `winogrande`, `piqa`, and `truthfulqa_mc2` are flat by the prespecified rule, while ARC, OpenBookQA, HellaSwag, WiC, and MMLU drop.

So the mitigation is useful mainly because it preserves the strong coherence/human-alignment gains while avoiding the full retention collapse seen in the real SFT; it is not evidence that lowLR broadly improves standard lm-eval capability.
