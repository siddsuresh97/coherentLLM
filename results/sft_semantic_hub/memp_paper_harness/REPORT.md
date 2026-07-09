# Semantic Hub MEMP Harness Report

Created UTC: 2026-07-09T02:59:31.455948+00:00

## Status

Computed paper-style matched-vs-control similarity from existing semantic-hub hidden states.

## Method

- Paper basis: Wu et al. semantic-hub relative similarity, adapted to same-concept prompt spokes.
- Prompt spokes: `triplet`, `pairwise`, `feature_listing`.
- Primary layer band: `10`-`20`.
- Controls: `random`, `close_neighbor`, `far_neighbor`, `lexical`, `category_proxy`.
- Category control source: `auto_sstar_kmeans_proxy`.

## Mid-Layer Summary

| Arm | Control | Delta | CI Low | CI High | p | Top-1 | Top-5 |
|---|---|---:|---:|---:|---:|---:|---:|
| `taskvec_a0p5` | `category_proxy` | 0.0182 | 0.0134 | 0.0232 | 0.0039 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | `category_proxy` | 0.0154 | 0.0044 | 0.0262 | 0.1390 | 0.0637 | 0.1896 |
| `taskvec_a0p25` | `category_proxy` | 0.0106 | 0.0083 | 0.0129 | 0.0100 | 0.2062 | 0.5393 |
| `lowrank` | `category_proxy` | 0.0063 | 0.0039 | 0.0087 | 0.0018 | 0.0968 | 0.4079 |
| `scrambled` | `category_proxy` | 0.0062 | 0.0048 | 0.0076 | 0.0863 | 0.1094 | 0.2667 |
| `lowLR` | `category_proxy` | 0.0059 | 0.0034 | 0.0086 | 0.0018 | 0.0729 | 0.3063 |
| `base` | `category_proxy` | 0.0007 | -0.0003 | 0.0017 | 0.1557 | 0.0192 | 0.0800 |
| `taskvec_a1p0` | `close_neighbor` | 0.0099 | 0.0016 | 0.0184 | 0.1799 | 0.0637 | 0.1896 |
| `taskvec_a0p5` | `close_neighbor` | 0.0070 | 0.0029 | 0.0115 | 0.1326 | 0.1578 | 0.4744 |
| `scrambled` | `close_neighbor` | 0.0026 | 0.0014 | 0.0039 | 0.3826 | 0.1094 | 0.2667 |
| `lowLR` | `close_neighbor` | 0.0013 | -0.0009 | 0.0037 | 0.1538 | 0.0729 | 0.3063 |
| `taskvec_a0p25` | `close_neighbor` | 0.0011 | -0.0006 | 0.0028 | 0.3224 | 0.2062 | 0.5393 |
| `lowrank` | `close_neighbor` | 0.0003 | -0.0018 | 0.0025 | 0.2939 | 0.0968 | 0.4079 |
| `base` | `close_neighbor` | -0.0002 | -0.0012 | 0.0009 | 0.5375 | 0.0192 | 0.0800 |
| `taskvec_a0p5` | `far_neighbor` | 0.0805 | 0.0748 | 0.0864 | 0.0369 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | `far_neighbor` | 0.0563 | 0.0416 | 0.0708 | 0.4524 | 0.0637 | 0.1896 |
| `taskvec_a0p25` | `far_neighbor` | 0.0548 | 0.0524 | 0.0573 | 0.0010 | 0.2062 | 0.5393 |
| `lowrank` | `far_neighbor` | 0.0494 | 0.0462 | 0.0525 | 0.0010 | 0.0968 | 0.4079 |
| `lowLR` | `far_neighbor` | 0.0371 | 0.0337 | 0.0406 | 0.0010 | 0.0729 | 0.3063 |
| `scrambled` | `far_neighbor` | 0.0077 | 0.0059 | 0.0095 | 0.1743 | 0.1094 | 0.2667 |
| `base` | `far_neighbor` | 0.0033 | 0.0018 | 0.0048 | 0.1069 | 0.0192 | 0.0800 |
| `taskvec_a0p5` | `lexical` | 0.0546 | 0.0473 | 0.0620 | 0.0257 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | `lexical` | 0.0382 | 0.0238 | 0.0527 | 0.3161 | 0.0637 | 0.1896 |
| `taskvec_a0p25` | `lexical` | 0.0355 | 0.0322 | 0.0386 | 0.0010 | 0.2062 | 0.5393 |
| `lowrank` | `lexical` | 0.0295 | 0.0257 | 0.0334 | 0.0010 | 0.0968 | 0.4079 |
| `lowLR` | `lexical` | 0.0210 | 0.0173 | 0.0248 | 0.0010 | 0.0729 | 0.3063 |
| `scrambled` | `lexical` | 0.0094 | 0.0080 | 0.0108 | 0.0058 | 0.1094 | 0.2667 |
| `base` | `lexical` | 0.0016 | 0.0004 | 0.0029 | 0.1743 | 0.0192 | 0.0800 |
| `taskvec_a0p5` | `random` | 0.0668 | 0.0602 | 0.0735 | 0.0010 | 0.1578 | 0.4744 |
| `taskvec_a1p0` | `random` | 0.0491 | 0.0350 | 0.0632 | 0.2561 | 0.0637 | 0.1896 |
| `taskvec_a0p25` | `random` | 0.0403 | 0.0374 | 0.0432 | 0.0010 | 0.2062 | 0.5393 |
| `lowrank` | `random` | 0.0342 | 0.0305 | 0.0380 | 0.0010 | 0.0968 | 0.4079 |
| `lowLR` | `random` | 0.0266 | 0.0229 | 0.0303 | 0.0010 | 0.0729 | 0.3063 |
| `scrambled` | `random` | 0.0097 | 0.0082 | 0.0111 | 0.0172 | 0.1094 | 0.2667 |
| `base` | `random` | 0.0021 | 0.0008 | 0.0033 | 0.0966 | 0.0192 | 0.0800 |

## Readout

- `paired_delta > 0` means same-concept cross-spoke states beat that control.
- `close_neighbor` and `category_proxy` are stricter than random and far controls.
- `category_proxy` is an S*-cluster proxy until an explicit category table is supplied.
- Best-layer rows are descriptive; the primary scientific gate is the fixed mid-layer band.

## Files

- manifest: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/memp_paper_harness/manifest.json`
- run_config: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/memp_paper_harness/run_config.json`
- metric_schema: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/memp_paper_harness/metric_schema.json`
- control_assignments: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/memp_paper_harness/control_assignments.csv`
- controls_by_layer: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/memp_paper_harness/controls_by_layer.csv`
- control_summary: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/memp_paper_harness/control_summary.csv`
- report: `/mnt/dv/wid/projects3/Rogers-nsf-ind-diff/sid/Projects/coherence_experiments/results/sft_semantic_hub/memp_paper_harness/REPORT.md`
