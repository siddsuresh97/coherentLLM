# Semantic Hub Mechanism Synthesis

Created UTC: 2026-07-09T03:53:16.425994+00:00

## Bottom Line

- Coherence training and task vectors do induce a stronger paper-style semantic-hub signal,
  especially same-concept cross-spoke similarity over random/far/category controls.
- The useful skill is semantic geometry and human-similarity alignment, not generic
  multiple-choice competence. That explains why script/discourse plausibility mostly
  survives while MMLU, WiC, ARC/OpenBookQA, and TruthfulQA false-lure calibration are fragile.
- `taskvec_a0p25` is the best currently staged cheap adapter, but it is not the best
  broad-similarity arm and it carries a false-lure pressure risk. Its retrieval
  gain is less dominated by attractor concepts than the alternatives, but not
  hubness-free. The stronger broad/strict hub arms, `taskvec_a0p5` and
  `taskvec_a1p0`, need retention gates before they can become candidates.

## Joined Arm Summary

| Arm | Rand | Close | Cat | Top5 | Unique | MaxHub | Gen | Human | MMLU | WiC | Sci | TQA false-up | TQA200 false-up | WiC200 | OBQA200 | Lang brain |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 0.0021 | -0.0002 | 0.0007 | 0.0800 | 0.0346 | 94.1515 | 0.0000 | 0.0000 | 0.0000 |  |  |  |  | 0.0000 | 0.0000 | -0.0164 |
| lowLR | 0.0266 | 0.0013 | 0.0059 | 0.3063 | 0.1602 | 48.7879 | 0.4348 | 0.1825 | -0.0993 | -0.1520 | -0.1093 | 0.2750 | 0.3550 | -0.1700 | -0.0650 | 0.0087 |
| lowrank | 0.0342 | 0.0003 | 0.0063 | 0.4079 | 0.1961 | 34.9697 | 0.4266 | 0.1976 | -0.1011 | -0.1536 | -0.1227 |  |  |  |  | 0.0015 |
| taskvec_a0p25 | 0.0403 | 0.0011 | 0.0106 | 0.5393 | 0.2996 | 29.1515 | 0.3269 | 0.1308 | -0.0700 | -0.1505 | -0.0620 | 0.8250 | 0.8200 | -0.1600 | -0.0300 | 0.0045 |
| taskvec_a0p5 | 0.0668 | 0.0070 | 0.0182 | 0.4744 | 0.2444 | 40.5455 | 0.3842 | 0.1163 |  |  |  |  |  |  |  | 0.0046 |
| taskvec_a1p0 | 0.0491 | 0.0099 | 0.0154 | 0.1896 | 0.1081 | 75.7273 | 0.3768 | 0.1368 |  |  |  |  |  |  |  | 0.0016 |
| scrambled | 0.0097 | 0.0026 | 0.0062 | 0.2667 | 0.1838 | 57.3636 |  |  |  | -0.1708 | -0.3920 |  |  |  |  | -0.0178 |

## Mechanistic Read

- Strongest broad same-minus-random hub signal: `taskvec_a0p5` `0.0668`.
- Strongest strict same-minus-S*-close signal: `taskvec_a1p0` `0.0099`.
- Strongest category-proxy signal: `taskvec_a0p5` `0.0182`.
- `taskvec_a0p25` boosts generation coherence by `0.3269` and human R2 by `0.1308`, but has MMLU delta `-0.0700`, wide-bench WiC delta `-0.1505`, CHTC-200 WiC delta `-0.1600`, and CHTC-200 TruthfulQA false-pressure-up `0.8200`.
- Hubness control: `taskvec_a0p25` has top-1 unique fraction `0.2996` and max mid-layer attractor occurrence `29.1515`; its read is `strong retrieval with reduced, not eliminated, hubness`.
- `lowLR` has a smaller MEMP random delta `0.0266` than taskvec, but its CHTC-200 false-pressure-up `0.3550` is much safer and its bounded MC2 delta `0.0229` beats taskvec `0.0038`.

This pattern is consistent with a smooth semantic-centralization benefit that helps
cross-format similarity and some language/semantic brain alignment, but blurs sharp
option boundaries. TruthfulQA is vulnerable because many false answers are semantically
near the topic; a semantic direction can raise both truthful and plausible-false mass.

## What The Model Is Good For

- Semantic similarity, THINGS-style human-alignment probes, and cross-format concept consistency.
- Brain-facing hypotheses in ATL/Language regions, especially as exploratory layer/arm predictors.
- Mechanistic probes of how a mid-layer task vector changes concept geometry.

## What It Is Not Good For Yet

- Exact concept identity against S*-close neighbors; the strict close-control deltas remain small.
- Lexical sense disambiguation and science/exam option ranking.
- Truthfulness mitigation unless false-answer pressure is explicitly gated.

## Next Experiments

1. Do not promote `taskvec_a0p25` on TruthfulQA without a false-pressure mitigation; the limit-200 gate keeps the risk.
2. Stage `out/adapters_taskvec_scaled/a0p5` only after extending the failure-suite runner with a `taskvec_a0p5` arm, then run a small false-pressure/WiC/OpenBookQA gate before any wider eval.
3. Add CSLS or mutual-nearest-neighbor retrieval as a stricter hubness correction.
4. Add causal cross-spoke patching or activation addition: patch triplet-format concept states into pairwise/feature prompts and require same-concept improvement over random, close-neighbor, wrong-layer, and shuffled-vector controls.
5. For cognitive science, prioritize Huth/LeBel high-data story scaling and ATL/Language-region tests; use hub metrics as predictors, not as standalone brain claims.

## Files

- `joined_arm_summary.csv`: one-row-per-arm synthesis table.
- `source_manifest.json`: exact source CSVs used.
