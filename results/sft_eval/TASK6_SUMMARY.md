# Task 6 Summary: lowLR vs base on external similarity, human behavior, and capability

Verdict rule: lowLR is a GAIN if delta vs base is > +0.02, DROP if delta is < -0.02, and FLAT otherwise.

Scope note for 6C: `eval_wide_bench.py` uses five standard groups. The `zero_shot` group bundles `piqa`, `openbookqa`, `commonsense_qa`, `wic`, and `truthfulqa_mc2`; the other groups are `winogrande_5shot`, `arc_25shot` (`arc_easy` and `arc_challenge`), `hellaswag_10shot`, and `mmlu_5shot`. Thus all originally requested capability tasks are covered: `piqa`, `winogrande`, `arc_easy`, `arc_challenge`, `openbookqa`, `commonsense_qa`, `wic`, `mmlu`, `truthfulqa_mc2`, plus `hellaswag`.

## Bottom line

- lowLR gains on 6 reported rows: WordSim-353, MEN, MTurk-771, SimVerb-3500, THINGS odd-one-out, THINGS triplet similarity.
- lowLR is flat on 7 reported rows: STS-B, SimLex-999, RG-65, piqa, winogrande, commonsense_qa, truthfulqa_mc2.
- lowLR drops on 6 reported rows: arc_easy, arc_challenge, openbookqa, hellaswag, wic, mmlu.
- The gains are concentrated in external/human semantic similarity and THINGS human behavior. The broader lm-eval capability battery does not show general capability gains for lowLR; most large drops are ARC, WIC, MMLU, OpenBookQA, and HellaSwag.

## Similarity (6A)

| task | metric | base | real | lowLR | lowrank | lowLR_delta | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| STS-B | Spearman rho | 0.518 | 0.527 | 0.527 | 0.519 | 0.009 | FLAT |
| SimLex-999 | Spearman rho | 0.297 | 0.312 | 0.310 | 0.311 | 0.012 | FLAT |
| WordSim-353 | Spearman rho | 0.294 | 0.418 | 0.344 | 0.291 | 0.050 | GAIN |
| MEN | Spearman rho | 0.488 | 0.430 | 0.518 | 0.506 | 0.030 | GAIN |
| RG-65 | Spearman rho | 0.358 | 0.209 | 0.357 | 0.353 | -0.001 | FLAT |
| MTurk-771 | Spearman rho | 0.306 | 0.380 | 0.353 | 0.358 | 0.047 | GAIN |
| SimVerb-3500 | Spearman rho | 0.168 | 0.235 | 0.199 | 0.193 | 0.032 | GAIN |

## Human-Behavior (6B)

The human-rated word/sentence similarity datasets are listed above under 6A; 6B adds direct THINGS behavior agreement. Typicality was skipped because no ready ranked category-norm dataset was found in the repo.

| task | metric | base | real | lowLR | lowrank | lowLR_delta | verdict | note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| THINGS odd-one-out | agreement | 0.686 | 0.765 | 0.770 | 0.762 | 0.084 | GAIN | ties_skipped=0; human target=data/scale128/human_spose_triplet_sim.npy |
| THINGS triplet similarity | spearman_rho | 0.483 | 0.700 | 0.711 | 0.723 | 0.228 | GAIN | ties_skipped=0; human target=data/scale128/human_spose_triplet_sim.npy |
| Typicality/category norms | not_run | NA | NA | NA | NA | NA | NA | skipped: no ready typicality/ranked category norm dataset found in repo |

## Capability (6C)

| task | metric | base | real | lowLR | lowrank | lowLR_delta | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| piqa | acc_norm | 0.799 | NA | 0.779 | NA | -0.020 | FLAT |
| winogrande | acc | 0.762 | NA | 0.765 | NA | 0.003 | FLAT |
| arc_easy | acc_norm | 0.850 | NA | 0.725 | NA | -0.125 | DROP |
| arc_challenge | acc_norm | 0.649 | NA | 0.524 | NA | -0.125 | DROP |
| openbookqa | acc_norm | 0.490 | NA | 0.412 | NA | -0.078 | DROP |
| commonsense_qa | acc | 0.651 | NA | 0.666 | NA | 0.015 | FLAT |
| hellaswag | acc_norm | 0.685 | NA | 0.656 | NA | -0.029 | DROP |
| wic | acc | 0.652 | NA | 0.500 | NA | -0.152 | DROP |
| mmlu | macro_acc | 0.692 | NA | 0.599 | NA | -0.093 | DROP |
| truthfulqa_mc2 | acc | 0.550 | NA | 0.533 | NA | -0.017 | FLAT |

## Source Files

- `results/sft_eval/external/summary.csv`
- `results/sft_eval/human_tasks/summary.csv`
- `results/sft_eval/wide_bench/summary.csv`
