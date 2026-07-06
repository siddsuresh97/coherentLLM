# Leuven feature re-verification: flan-xxl only (NOVA recipe, stage 1)

Verified all 60,780 (concept x feature) pairs over the 30 concepts x 2026 Leuven-union
features with flan-t5-xxl (fp16, H100) using the NOVA verification prompt. True-rate 13.6%.

Human coherence (RDM-Procrustes r^2), original Leuven vs flan-reverified feature space:
| pair | ORIG Leuven | FLAN-reverified |
|---|---|---|
| triplet~feature  | 0.896 | 0.715 |
| pairwise~feature | 0.828 | 0.869 |
| triplet~pairwise | 0.716 | 0.716 |
| MEAN (ceiling)   | 0.813 | 0.766 |

Finding: flan-ONLY re-verification LOWERS the human ceiling (0.81 -> 0.77). It densifies
the matrix (5875 -> 7680 True cells) which helps pairwise~feature but hurts triplet~feature
(flan adds spurious features that scramble triplet-aligned structure). A weak verifier
degrades coherence. NEXT: gpt-5.5 stage-2 pass should prune flan's false positives -> test
if the two-stage NOVA matrix lands ABOVE original 0.81. (~$1.60, ~8270 flan-True calls.)
