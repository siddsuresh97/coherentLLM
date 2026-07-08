# fMRI x Semantic-Hub Bridge

## Scope

This analysis joins the 90-concept THINGS-fMRI RSA with the 128-concept
semantic-hub metrics by arm/layer. It is descriptive: the layer-grid
analysis has many correlated layer points, and the arm-summary analysis
has only seven arms.

## Key Tables

- `layer_join.csv`: mean fMRI RSA by region/arm/layer plus hub metrics.
- `layer_correlations.csv`: correlations across arm-layer points and
  delta-vs-base arm-layer points.
- `arm_summary_bridge.csv`: fMRI best-layer summary joined to hub summary.
- `arm_summary_correlations.csv`: small-n arm-level correlations.

## Ventral Visual

| Arm | fMRI RSA | Hub Mid RDM | Hub Top-1 |
|---|---:|---:|---:|
| `lowrank` | 0.2013 | 0.6637 | 0.0968 |
| `lowLR` | 0.2010 | 0.6211 | 0.0729 |
| `base` | 0.1990 | 0.3230 | 0.0192 |
| `taskvec_a0p5` | 0.1973 | 0.6372 | 0.1578 |
| `taskvec_a0p25` | 0.1968 | 0.6770 | 0.2062 |
| `taskvec_a1p0` | 0.1856 | 0.4867 | 0.0637 |
| `scrambled` | 0.1293 | 0.3472 | 0.1094 |

## ATL (Semantic)

| Arm | fMRI RSA | Hub Mid RDM | Hub Top-1 |
|---|---:|---:|---:|
| `taskvec_a0p5` | 0.0543 | 0.6372 | 0.1578 |
| `lowrank` | 0.0539 | 0.6637 | 0.0968 |
| `lowLR` | 0.0531 | 0.6211 | 0.0729 |
| `taskvec_a1p0` | 0.0518 | 0.4867 | 0.0637 |
| `taskvec_a0p25` | 0.0512 | 0.6770 | 0.2062 |
| `base` | 0.0481 | 0.3230 | 0.0192 |
| `scrambled` | 0.0378 | 0.3472 | 0.1094 |

## Language

| Arm | fMRI RSA | Hub Mid RDM | Hub Top-1 |
|---|---:|---:|---:|
| `lowrank` | 0.0345 | 0.6637 | 0.0968 |
| `lowLR` | 0.0329 | 0.6211 | 0.0729 |
| `taskvec_a0p5` | 0.0319 | 0.6372 | 0.1578 |
| `taskvec_a1p0` | 0.0316 | 0.4867 | 0.0637 |
| `taskvec_a0p25` | 0.0291 | 0.6770 | 0.2062 |
| `base` | 0.0283 | 0.3230 | 0.0192 |
| `scrambled` | 0.0225 | 0.3472 | 0.1094 |

## Correlation Highlights

### `arm_layer_delta_vs_base` / `cross_format_rdm_spearman_delta_vs_base`

| Region | n | Spearman r | p |
|---|---:|---:|---:|
| TE2p | 192 | 0.427 | 6.72e-10 |
| Ventral Visual | 192 | 0.401 | 7.89e-09 |
| TE2a | 192 | 0.372 | 1.1e-07 |
| Early Visual | 192 | 0.355 | 4.21e-07 |
| Dorsal Visual | 192 | 0.341 | 1.27e-06 |
| PeEc | 192 | 0.338 | 1.66e-06 |
| TF | 192 | 0.328 | 3.51e-06 |
| ATL (Semantic) | 192 | 0.306 | 1.58e-05 |
| Language | 192 | 0.198 | 0.00584 |
| TE1m | 192 | 0.155 | 0.0315 |
| TE1a | 192 | 0.031 | 0.67 |
| Prefrontal | 192 | -0.022 | 0.758 |
| EC | 192 | -0.155 | 0.0319 |
| TGd | 192 | -0.176 | 0.0148 |
| TE1p | 192 | -0.224 | 0.00181 |
| TGv | 192 | -0.260 | 0.000271 |

### `arm_layer_delta_vs_base` / `retrieval_top1_delta_vs_base`

| Region | n | Spearman r | p |
|---|---:|---:|---:|
| TE2a | 192 | 0.106 | 0.145 |
| TGv | 192 | 0.058 | 0.425 |
| Prefrontal | 192 | -0.023 | 0.75 |
| TE1p | 192 | -0.031 | 0.668 |
| TE1a | 192 | -0.056 | 0.442 |
| TGd | 192 | -0.059 | 0.418 |
| PeEc | 192 | -0.096 | 0.184 |
| TE1m | 192 | -0.107 | 0.138 |
| TE2p | 192 | -0.109 | 0.133 |
| TF | 192 | -0.132 | 0.0676 |
| Dorsal Visual | 192 | -0.154 | 0.0326 |
| ATL (Semantic) | 192 | -0.164 | 0.0232 |
| Ventral Visual | 192 | -0.182 | 0.0114 |
| EC | 192 | -0.198 | 0.00601 |
| Early Visual | 192 | -0.217 | 0.00254 |
| Language | 192 | -0.232 | 0.00121 |

### `arm_summary_delta_vs_base` / `mid_cross_format_rdm_spearman`

| Region | n | Spearman r | p |
|---|---:|---:|---:|
| TE2a | 6 | 0.829 | 0.0416 |
| Dorsal Visual | 6 | 0.714 | 0.111 |
| TE1m | 6 | 0.714 | 0.111 |
| Early Visual | 6 | 0.600 | 0.208 |
| TF | 6 | 0.600 | 0.208 |
| Ventral Visual | 6 | 0.600 | 0.208 |
| ATL (Semantic) | 6 | 0.371 | 0.468 |
| Language | 6 | 0.371 | 0.468 |
| TE2p | 6 | 0.200 | 0.704 |
| Prefrontal | 6 | 0.143 | 0.787 |
| PeEc | 6 | 0.086 | 0.872 |
| TE1a | 6 | -0.086 | 0.872 |
| TGd | 6 | -0.200 | 0.704 |
| EC | 6 | -0.486 | 0.329 |
| TE1p | 6 | -0.543 | 0.266 |
| TGv | 6 | -0.714 | 0.111 |

### `arm_summary_delta_vs_base` / `mid_retrieval_top1`

| Region | n | Spearman r | p |
|---|---:|---:|---:|
| TE2a | 6 | 0.600 | 0.208 |
| TE1m | 6 | 0.371 | 0.468 |
| TE1p | 6 | 0.200 | 0.704 |
| Dorsal Visual | 6 | -0.029 | 0.957 |
| TF | 6 | -0.086 | 0.872 |
| ATL (Semantic) | 6 | -0.086 | 0.872 |
| Early Visual | 6 | -0.086 | 0.872 |
| Ventral Visual | 6 | -0.086 | 0.872 |
| EC | 6 | -0.143 | 0.787 |
| Prefrontal | 6 | -0.143 | 0.787 |
| TGd | 6 | -0.257 | 0.623 |
| TE2p | 6 | -0.314 | 0.544 |
| Language | 6 | -0.371 | 0.468 |
| TE1a | 6 | -0.543 | 0.266 |
| PeEc | 6 | -0.600 | 0.208 |
| TGv | 6 | -0.600 | 0.208 |

## Initial Read

For Ventral Visual, the arm-level delta-vs-base link between hub metrics and fMRI RSA is: mid_cross_format_rdm_spearman: r=0.600, mid_retrieval_top1: r=-0.086.

The current bridge does not establish that the semantic hub explains the
object-fMRI RSA effects. The safest read is that hub invariance is a
strong internal-model effect, while the first fMRI RSA result is mostly
a visual/object-geometry signal with only small ATL/Language differences.
The next decisive test is a leakage-clean held-out regression where
format-averaged hub RDMs and single-format RDMs compete to predict the
same fMRI RDMs.
