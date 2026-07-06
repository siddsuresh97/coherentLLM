# The paper's TRUE metric = RDM-direct Procrustes (no MDS)

RESOLVED via the paper's OWN saved output:
  conceptual_representations_gpt/emnlp/results/humans_all_tasks_correlation_table.csv

Paper's actual computed human values (protest symmetric, sqrt(1-ss)):
  leuven(feature) ~ triplet  = 0.979
  leuven(feature) ~ pairwise = 0.915
  triplet ~ pairwise         = 0.846

My RDM-direct Procrustes reproduces these almost exactly:
  0.980 / 0.919 / 0.852   (vs 0.979 / 0.915 / 0.846)

Key finding: the paper's notebook (cell 26) calls R vegan::protest() on the DISTANCE
MATRICES directly (leuven_dsm, human_triplet_dsm, human_pairwise_dsm) -> protest treats
each 30x30 DSM as a 30-dim configuration. That is RDM-direct Procrustes, NOT 3D MDS.

The dissertation TEXT quotes 0.96/0.84/0.72 - an older/different run that does not match
the committed analysis output. The saved table (0.98/0.92/0.85) is authoritative.

=> Our pipeline metric = RDM-direct Procrustes, sqrt(1-scipy_disparity). Correct & paper-faithful.
Human coherence ceiling (mean of 3) = (0.979+0.915+0.846)/3 = 0.913 (~0.91).
