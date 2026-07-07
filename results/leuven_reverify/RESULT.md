# Can NOVA re-verification make the human feature matrix more coherent? NO.

Question: re-verify Leuven's 30-concept features with the NOVA recipe (flan-xxl stage-1,
gpt-5.5 stage-2 pruning) to see if a cleaner human feature space raises the coherence ceiling.

Human coherence ceiling (RDM-Procrustes r^2):
| pair | ORIG Leuven | flan-only | flan+gpt5 (two-stage) |
|---|---|---|---|
| triplet~feature  | 0.896 | 0.715 | 0.720 |
| pairwise~feature | 0.828 | 0.869 | 0.880 |
| triplet~pairwise | 0.716 | 0.716 | 0.716 |
| MEAN CEILING     | 0.813 | 0.766 | 0.772 |
True cells: orig 5875 | flan 7680 | two-stage 5077

gpt-5.5 pruned 34% of flan's True calls (7680 -> 5077 kept). Cost ~$1.50.

CONCLUSION: re-verification does NOT raise the ceiling. Two-stage beats flan-only
(0.766->0.772) and improves pairwise~feature (0.83->0.88), but cannot recover
triplet~feature (0.90->0.72), so mean stays BELOW original Leuven (0.81).
Root cause is the UNION step, not verification quality: verifying against the full
2026-feature union restructures the space away from Leuven's native per-concept lists,
breaking the triplet~feature alignment. Even a perfect verifier can't fix scope.
=> Leuven norms are already near-optimal for these 30 concepts; the 0.84 human ceiling
stands. Clean negative result + validation of Leuven.

## Follow-up: what the OG paper did + Leuven-within + NOVA-cross-domain hybrid

OG paper (experiments.ipynb cell 7): merge(animal, tool, all=TRUE); NA -> 0. i.e. it
keeps Leuven within-domain features and ZERO-FILLS all cross-domain cells (a reptile is
marked as not having any tool feature). No cross-domain verification. That IS our "ORIG
Leuven" baseline (mean 0.813, reproduces saved 0.979/0.915/0.846).

Hybrid tested (per user): Leuven cells untouched within-domain; cross-domain (0) cells
filled with the two-stage flan+gpt5 NOVA verdict (1379 new True cross-domain links).
| pair | paper (cross=0) | hybrid (cross=NOVA) |
|---|---|---|
| triplet~feature  | 0.896 | 0.854 |
| pairwise~feature | 0.828 | 0.862 |
| triplet~pairwise | 0.716 | 0.716 |
| MEAN             | 0.813 | 0.810 |

Hybrid ~ tied with paper (0.810 vs 0.813), and MUCH better than naive full-union
re-verify (0.772). Adding real cross-domain overlap helps pairwise~feature (0.83->0.86)
but slightly costs triplet~feature (0.90->0.85), net flat. Confirms the paper's zero-fill
was near-optimal: the reptile/tool split dominates, so cross-domain links barely matter.
Human ceiling ~0.81 (r^2), 0.84 as reported (r^2 with paper's exact Leuven/MDS). Stands.
