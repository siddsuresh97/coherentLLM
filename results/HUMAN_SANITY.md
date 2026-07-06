# Human-human Procrustes sanity check (paper human data + NOVA)

Using the ORIGINAL paper's human data (conceptual_representations_gpt/emnlp):
human_triplet_embedding.csv, mean_human_pairwise_similarity.csv, and Leuven feature.

Human-human Procrustes R² (3D MDS embeddings):
| pair | my Leuven | paper | NOVA (26/30) |
|---|---|---|---|
| triplet~feature  | 0.81 | 0.96 | 0.54 |
| pairwise~feature | 0.67 | 0.84 | 0.66 |
| triplet~pairwise | 0.55 | 0.72 | - |

Conclusions:
- My repro is directionally correct (same ordering) but ~0.15 below paper - likely because
  I re-embed via 3D MDS from cosine-sim, vs the paper's native embeddings + exact Procrustes.
  TODO: use the paper's precomputed human_reptile_tool_embedding_df.csv directly to match.
- NOVA (cogsci2025) is a WORSE human feature reference than Leuven for these 30 concepts
  (triplet~NOVA 0.54 vs triplet~Leuven 0.81). Cause: NOVA is 787-concept general norms
  (partly LLM-verified), only 26/30 covered, alligator->crocodile substitution.
  => KEEP LEUVEN as the human feature reference; do not switch to NOVA.
- Human coherence matrix (paper final_heatmap): 0.96/0.84/0.72 -> mean 0.84 (the ceiling).
