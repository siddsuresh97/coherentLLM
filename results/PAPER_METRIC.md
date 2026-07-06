# Paper's exact metric (from conceptual_representations_gpt/emnlp/experiments.ipynb)

The EMNLP paper uses R's vegan::protest(symmetric=TRUE) and reports:
    results = c(results, sqrt(1 - result$ss))
So the metric is the PROCRUSTES CORRELATION r = sqrt(1 - ss), NOT R^2.
(The dissertation prose "squared Procrustes correlation" is loose; the code is sqrt(1-ss).)

scipy.spatial.procrustes returns disparity = m12^2 (symmetric, both unit-normed),
so paper_value = sqrt(1 - disparity). Verified reproduction of the paper's HUMAN matrix:

| pair | my repro | paper |
|---|---|---|
| triplet ~ feature  | 0.90 | 0.96 |
| pairwise ~ feature | 0.82 | 0.84 |
| triplet ~ pairwise | 0.74 | 0.72 |

Within ~0.02-0.06 (triplet~pairwise near-exact). Residual gap = my binarized Leuven vs
the paper's raw-count animal+artifact Leuven CSVs.

Human coherence ceiling (mean of 3) = (0.96+0.84+0.72)/3 = 0.84 in r units.
ACTION: switch all figures from R^2 to r = sqrt(1-ss) to match the paper.
Embeddings: classical MDS (cmdscale) k=3 from the relevant distance matrix, per method.
