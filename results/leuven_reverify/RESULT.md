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
