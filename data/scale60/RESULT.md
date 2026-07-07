# n=60 scale point (nested subset of 128)

triplet~feature coherence (r^2, RDM-direct):
| model | n=30 | n=60 | n=128 |
|---|---|---|---|
| HUMAN        | 0.93 | 0.76 | 0.77 |
| llama-3.1-8b | 0.82 | 0.35 | 0.32 |
| olmo2-7b     | 0.80 | 0.17 | 0.23 |
| qwen2.5-7b   | 0.69 | 0.14 | 0.13 |

FINDING: the collapse is a THRESHOLD effect, essentially complete by n=60 (n60~=n128 for
all). 30 concepts = 2 clusters (reptile/tool) is uniquely easy and inflates model coherence;
by 60 diverse concepts the model<->human gap is fully open and stable. Humans drop modestly
(0.93->0.76) and hold; models drop by half-to-two-thirds and stay down.
(qwen2.5-32b @ n60/n128 pending H100 auth.)

## qwen-32b added (H100 auth restored)
qwen2.5-32b: n=30 0.94 -> n=60 0.25 -> n=128 0.34 (held-out triplet acc n=60: 0.78).
The LARGEST open model had the HIGHEST n=30 coherence (0.94, human-level) yet collapses
just as hard. Model size does NOT protect against the scale collapse; near-human coherence
at n=30 was the easy-2-cluster artifact for every model. Kills the "only small models
collapse" objection.

## CORRECTION: non-monotonic qwen-32b was a feature-count artifact
The triplet~feature coherence at n=60 depends on how many features survive the
min_concepts>=3 consolidation, and that count scales with n (n=60: 15-26 feats for
qwen-32b/olmo, vs 55-88 at n=128). Thin feature RDMs (<30 feats) are noisy, which made
qwen-32b's n=60 (0.25) spuriously LOWER than its n=128 (0.34) -> non-monotonic.

Fix A (consistent features: use n=128 verdicts subset to the 60 concepts):
  qwen-32b t~f: 0.94(n30) -> 0.30(n60) -> 0.34(n128)  [monotonic-ish]

Fix B (confound-free metric = model triplet ~ HUMAN triplet, r^2; no feature dependence):
| model | n=30 | n=60 | n=128 |
|---|---|---|---|
| qwen2.5-32b | 0.96 | 0.39 | 0.43 |
| llama-3.1-8b| 0.92 | 0.55 | 0.49 |
| olmo2-7b    | 0.86 | 0.36 | 0.32 |
| qwen2.5-7b  | 0.82 | 0.29 | 0.15 |

=> The collapse is REAL and robust on the clean triplet~human metric (0.96->0.39 for
qwen-32b). The n=30->n=60 drop is the finding; the small n=60/n=128 wobble is noise from
independently-fit nested embeddings + thin feature RDMs, NOT a real non-monotonic trend.
Report should lead with triplet~human alignment, not triplet~feature coherence, at scale.
