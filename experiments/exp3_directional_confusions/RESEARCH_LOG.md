# Experiment 3 Research Log

Append-only decision trail.

## 2026-07-09T13:55:26-05:00 DECISION: Concept selection

Selected 18 neutral Leuven concrete concepts: `alligator`, `caiman`, `crocodile`, `boa python`, `cobra`, `snake`, `blindworm`, `chameleon`, `gecko`, `lizard`, `salamander`, `toad`, `tortoise`, `turtle`, `axe`, `chisel`, `hammer`, `saw`.
Why this domain: it has known human feature structure, obvious local neighborhoods, and enough fine-grained reptiles/amphibians to plausibly elicit directional confusions.
Uncertainty band is not yet confirmed. That requires model item responses; the report stays red until errors exist.
Rejected broader mixed THINGS concepts for this first scaffold because the available Leuven feature matrix supports cleaner item generation.
Rejected medical/legal safety substitutions for now because Step 2 is gated on neutral H1.

## 2026-07-09T15:00:38-05:00 DECISION: Course-correction

Geometry run `5516751` completed all three Step 1 triplet runs and then stopped at the RDM reliability gate, as intended.
Observed reliability: mean pairwise upper-triangle Pearson `0.6842` versus threshold `0.75`; mean split-half Pearson `0.7968` versus threshold `0.70`.
The failure points to prompt/protocol mismatch rather than concept-set noise: the two canonical runs were identical (`r = 1.0`), split-half reliability was green, but canonical-vs-paraphrase agreement was only `0.5263`.
Raw-response audit showed the old canonical prompt often elicited a third concept not present among the two choices, e.g. `alligator|cobra|snake -> Crocodile`; 718/2448 canonical responses were not parseable as either listed candidate, while the paraphrase prompt had only 48/2448 unparseable responses.
Lever pulled: replace loose concept-name answers with labeled A/B triplet prompts, and parse either a valid A/B label or an exact candidate text. Third-concept answers remain rejected instead of coerced.
What would confirm this correction: re-run geometry, recover low out-of-option rate for both prompt variants, prompt-variant RDM agreement above the reliability threshold, and human-sane nearest neighbors.
What would kill it: if labeled prompts still produce red prompt-variant agreement despite high parse rates, the issue is likely concept selection or unstable model geometry rather than response formatting.

## 2026-07-09T15:05:58-05:00 DECISION: Pre-registered predictions

Geometry-derived neighbors written before item scoring.
RDM source: `experiments/exp3_directional_confusions/artifacts/rdm.npy`.
Human sanity gate is pending. Do not treat H1 as green until these pairs are manually accepted.

| Target | Predicted near confusion | Near distance | Far controls |
|---|---|---:|---|
| `alligator` | `crocodile` | 0.0000 | `blindworm` (0.5833), `salamander` (0.5521) |
| `caiman` | `crocodile` | 0.0312 | `chisel` (0.5208), `tortoise` (0.5104) |
| `crocodile` | `alligator` | 0.0000 | `axe` (0.5417), `saw` (0.5625) |
| `boa python` | `snake` | 0.0000 | `chisel` (0.6667), `toad` (0.6354) |
| `cobra` | `snake` | 0.0312 | `tortoise` (0.6562), `salamander` (0.6458) |
| `snake` | `boa python` | 0.0000 | `toad` (0.5625), `chisel` (0.5625) |
| `blindworm` | `salamander` | 0.0938 | `chisel` (0.6562), `alligator` (0.5833) |
| `chameleon` | `gecko` | 0.0625 | `chisel` (0.6562), `tortoise` (0.5938) |
| `gecko` | `lizard` | 0.0417 | `chisel` (0.7292), `saw` (0.7500) |
| `lizard` | `gecko` | 0.0417 | `hammer` (0.6667), `axe` (0.6771) |
| `salamander` | `blindworm` | 0.0938 | `saw` (0.7708), `chisel` (0.7812) |
| `toad` | `salamander` | 0.1250 | `saw` (0.7917), `chisel` (0.8125) |
| `tortoise` | `turtle` | 0.0000 | `chisel` (0.8021), `hammer` (0.8438) |
| `turtle` | `tortoise` | 0.0000 | `chisel` (0.7292), `hammer` (0.7500) |
| `axe` | `saw` | 0.0417 | `gecko` (0.8438), `turtle` (0.8438) |
| `chisel` | `axe` | 0.3438 | `gecko` (0.7292), `turtle` (0.7292) |
| `hammer` | `axe` | 0.0833 | `blindworm` (0.8021), `salamander` (0.8021) |
| `saw` | `axe` | 0.0417 | `toad` (0.7917), `turtle` (0.7917) |

## 2026-07-09T15:06:02-05:00 DECISION: Item design

Generated 72 Step 1 items: 4 per target.
Each item uses one geometry-predicted near distractor and two far controls from `neighbors.json`.
Question form is feature-attribution over Leuven feature norms. Clues prefer features shared with the near neighbor plus at least one target-specific or contrastive feature when available.
Option positions are counterbalanced by deterministic RNG seed.

## 2026-07-09T15:07:20-05:00 DECISION: Sanity gate

Human sanity gate marked `passed`.
Note: Predicted neighbors are human-sane for the neutral reptiles/amphibians/tools set; no target maps to an absurd cross-domain neighbor. Some coarse local choices such as blindworm-salamander are acceptable for this first gate.

## 2026-07-09T15:09:24-05:00 DECISION: H1 verdict

Run scored: `step1_items_v1`.
Directional errors: 45; near fraction: 0.8000.
Shuffle null p-value: 0.0000; base-rate lift: 0.3464.
H2 distance slope: -0.620137; 95% CI [-0.876997, -0.358641].
Predicted-vs-actual confusion agreement: 0.6342.
Verdict: `green_directional`.

## 2026-07-09T15:10:12-05:00 DECISION: H1 verdict

Run scored: `step1_items_v1`.
Directional errors: 45; near fraction: 0.8000.
Shuffle null p-value: 0.0002; base-rate lift: 0.3464.
H2 distance slope: -0.620137; 95% CI [-0.876997, -0.358641].
Predicted-vs-actual confusion agreement: 0.6342.
Verdict: `green_directional`.

## 2026-07-09T15:34:10-05:00 DECISION: Course-correction

User requested that Experiment 3 use SALMON embeddings for the geometry, with the RDM based on those embeddings rather than direct choice-rate aggregation.
The previous direct choice-rate RDM H1 result is therefore superseded for the active claim. It remains in git history and in the append-only log as an earlier result, not as the current preregistered geometry.
Lever pulled: convert every parsed triplet response to SALMON format `[head, winner, loser]`, fit `salmon.triplets.offline.OfflineEmbedding` with `d=5`, and compute `rdm.npy` as cosine distance (`1 - cosine_similarity`) over the pooled SALMON embedding.
Reliability will now be measured on the SALMON-derived geometry: per-run cosine RDM upper-triangle Pearson, per-run embedding Procrustes R^2, and SALMON split-half RDM reliability.
Counterfactual: if SALMON reliability is red, do not rescue the old direct-RDM result; fix the SALMON geometry/reliability issue or declare Step 1 not decided under the requested geometry.
Next action: rebuild RDM, preregister new SALMON/cosine neighbors before scoring, regenerate items if neighbors change, rerun item responses, and score H1 against the new RDM.

## 2026-07-09T15:38:25-05:00 DECISION: Course-correction

The first SALMON build attempt used `salmon_max_epochs=8000`, `salmon_split_half_samples=8`, and `salmon_split_half_max_epochs=2000`; it was interrupted before writing artifacts because the 48 split-half SALMON fits were too slow for this first rerun.
Lever pulled: keep the geometry definition unchanged (SALMON `d=5`, final RDM is cosine distance from the pooled SALMON embedding), but reduce the first-pass fit budget to `salmon_max_epochs=2000`, `salmon_split_half_samples=2`, and `salmon_split_half_max_epochs=500`.
Counterfactual: if reliability is borderline under the reduced split-half budget, rerun only the SALMON reliability pass at a larger budget before treating H1 as decided.
This change affects runtime and reliability precision, not the scoring target: `rdm.npy` still comes from a pooled SALMON embedding, not direct choice-rate aggregation.

## 2026-07-09T15:47:42-05:00 DECISION: Course-correction

SALMON/cosine geometry was computed, but the reliability gate is red, so no SALMON-based neighbors were preregistered and H1 is not decided under the requested geometry.
Fit quality is not the problem: pooled SALMON held-out accuracy is `0.8346`; per-run held-out accuracies are `0.8510`, `0.8551`, and `0.8408`, matching the expected ~0.75-0.80+ range.
The failure points to prompt-variant instability: canonical-vs-canonical embedding Procrustes R^2 is `0.9773` and RDM Pearson is `0.9744`, but canonical-vs-paraphrase Procrustes R^2 is only `0.6470`/`0.6422`, RDM Pearson is `0.3587`/`0.3307`, and nearest-neighbor top-1 agreement is only `2/18` for each canonical-vs-paraphrase comparison.
Raw choice audit confirms this is behavioral, not just embedding noise: the two canonical runs agree on `2448/2448` parsed triplets, while canonical-vs-paraphrase agrees on only `1820/2448` (`74.35%`).
Lever pulled next: keep labeled A/B response format, but replace the paraphrase with a closer matched wording using the same target/candidate layout, then rerun only the paraphrase geometry run before rebuilding SALMON.
Counterfactual: if a matched paraphrase still changes hundreds of choices and yields low nearest-neighbor stability, the Step 1 geometry is prompt-sensitive under SALMON and H1 remains not decided.

## 2026-07-09T16:01:22-05:00 DECISION: Pre-registered predictions

Geometry-derived neighbors written before item scoring.
RDM source: `experiments/exp3_directional_confusions/artifacts/rdm.npy`.
Human sanity gate is pending. Do not treat H1 as green until these pairs are manually accepted.

| Target | Predicted near confusion | Near distance | Far controls |
|---|---|---:|---|
| `alligator` | `turtle` | 0.1357 | `chameleon` (1.1599), `cobra` (1.1535) |
| `caiman` | `crocodile` | 0.0631 | `chisel` (1.5605), `cobra` (1.3749) |
| `crocodile` | `tortoise` | 0.0430 | `chisel` (1.4681), `saw` (1.7434) |
| `boa python` | `snake` | 0.0475 | `chameleon` (1.3724), `gecko` (1.2966) |
| `cobra` | `snake` | 0.4209 | `caiman` (1.3749), `chisel` (1.2053) |
| `snake` | `boa python` | 0.0475 | `chameleon` (1.2155), `saw` (1.2029) |
| `blindworm` | `toad` | 0.0703 | `saw` (1.4571), `axe` (1.8063) |
| `chameleon` | `gecko` | 0.0033 | `hammer` (1.4589), `boa python` (1.3724) |
| `gecko` | `chameleon` | 0.0033 | `hammer` (1.5279), `saw` (1.5970) |
| `lizard` | `gecko` | 0.0459 | `cobra` (1.6623), `saw` (1.7153) |
| `salamander` | `toad` | 0.0345 | `saw` (1.4483), `axe` (1.7723) |
| `toad` | `salamander` | 0.0345 | `saw` (1.6117), `chisel` (1.8095) |
| `tortoise` | `turtle` | 0.0009 | `chisel` (1.4506), `saw` (1.6116) |
| `turtle` | `tortoise` | 0.0009 | `chisel` (1.4301), `saw` (1.5909) |
| `axe` | `hammer` | 0.0410 | `lizard` (1.8092), `blindworm` (1.8063) |
| `chisel` | `hammer` | 0.2785 | `snake` (1.7395), `toad` (1.8095) |
| `hammer` | `axe` | 0.0410 | `crocodile` (1.8755), `salamander` (1.8671) |
| `saw` | `axe` | 0.1162 | `toad` (1.6117), `tortoise` (1.6116) |

## 2026-07-09T16:01:53-05:00 DECISION: Sanity gate

Human sanity gate marked `passed`.
Note: SALMON/cosine neighbors stay within broad human-sane neutral domains (reptiles/amphibians/tools). Caveat: some nearest neighbors are coarse reptile substitutions such as alligator->turtle and crocodile->tortoise rather than the direct-count crocodilian neighbors; keep them preregistered and let H1 test whether those directions predict errors.

## 2026-07-09T16:01:54-05:00 DECISION: Item design

Generated 72 Step 1 items: 4 per target.
Each item uses one geometry-predicted near distractor and two far controls from `neighbors.json`.
Question form is feature-attribution over Leuven feature norms. Clues prefer features shared with the near neighbor plus at least one target-specific or contrastive feature when available.
Option positions are counterbalanced by deterministic RNG seed.

## 2026-07-09T16:02:58-05:00 DECISION: H1 verdict

Run scored: `step1_items_v1`.
Directional errors: 37; near fraction: 0.7838.
Shuffle null p-value: 0.0002; base-rate lift: 0.3187.
H2 distance slope: -0.233021; 95% CI [-0.338216, -0.125002].
Predicted-vs-actual confusion agreement: 0.5781.
Verdict: `green_directional`.

## 2026-07-09T16:03:39-05:00 DECISION: H1 verdict

Run scored: `step1_items_v1`.
Directional errors: 37; near fraction: 0.7838.
Shuffle null p-value: 0.0002; base-rate lift: 0.3187.
H2 distance slope: -0.233021; 95% CI [-0.338216, -0.125002].
Predicted-vs-actual confusion agreement: 0.5781.
Verdict: `green_directional`.

## 2026-07-09T16:10:51-05:00 DECISION: Course-correction

Record the SALMON triplet-count heuristic for future geometry runs: target triplets should scale like `fudge_factor * n_concepts * embedding_dim * ln(n_concepts)`.
For this Step 1 neutral run, `n=18` and SALMON `d=5`, so the base `n*d*ln(n)` budget is about `260.1` triplets.
Observed coverage is `2448` triplets per geometry run (`9.41x` the base heuristic) and `7344` pooled triplets across the three active geometry runs (`28.23x` the base heuristic).
Interpretation: the final SALMON Step 1 geometry is not under-tripleted by this heuristic; future concept sets and Qwen/model-extension runs should report the same base budget and observed fudge factor before reading off neighbors.

## 2026-07-09T16:32:36-05:00 DECISION: Course-correction

Ran Step 1 audit before designing Step 2.
Error concentration: 13 targets had errors; 12 had near-neighbor errors; 6 had far-control errors.
Far-control audit: 8 far errors; 0 were top-5 RDM neighbors of the target.
Option-position audit: error choices by letter = {'A': 9, 'B': 10, 'C': 11, 'D': 7}; correct option slots by letter = {'A': 13, 'B': 17, 'C': 13, 'D': 29}.
Interpretation: Step 1 is strong enough to transfer; do not overfit item phrasing, but carry the audit forward for model extensions.

## 2026-07-09T16:33:23-05:00 DECISION: H1 verdict

Run scored: `step1_items_v1`.
Directional errors: 37; near fraction: 0.7838.
Shuffle null p-value: 0.0002; base-rate lift: 0.3191.
H2 distance slope: -0.236040; 95% CI [-0.335768, -0.133190].
Predicted-vs-actual confusion agreement: 0.5940.
Verdict: `green_directional`.

## 2026-07-09T16:33:28-05:00 DECISION: Course-correction

Ran Step 1 audit before designing Step 2.
Error concentration: 14 targets had errors; 13 had near-neighbor errors; 5 had far-control errors.
Far-control audit: 8 far errors; 0 were top-5 RDM neighbors of the target.
Option-position audit: error choices by letter = {'A': 7, 'B': 10, 'C': 12, 'D': 8}; correct option slots by letter = {'A': 13, 'B': 17, 'C': 13, 'D': 29}.
Interpretation: Step 1 is strong enough to transfer; do not overfit item phrasing, but carry the audit forward for model extensions.

## 2026-07-09T16:34:20-05:00 DECISION: Course-correction

Audit found a parser issue in verbose item responses: one response beginning with `C. boa python` could be misparsed by the old fallback because a later explanation contained a standalone `A`.
Lever pulled: update answer parsing to prefer the first explicit option at the start of the response, then answer/correct-answer phrases, then the first standalone option letter in text order.
After rescoring with the corrected parser, H1 stayed green: 37 directional errors, 29 near-neighbor errors, 8 far-control errors, near fraction 0.7838, shuffle null p=0.0002, H2 slope -0.236040 with 95% CI [-0.335768, -0.133190], predicted-vs-actual agreement 0.5940.
Audit interpretation: the Step 1 signal is not just one target and the far-control misses are mostly true wrong-direction misses, not hidden top-neighbor cases; 0/8 far-control errors are top-5 RDM neighbors of their target.

## 2026-07-09T16:34:28-05:00 DECISION: Course-correction

Scanned recent AI safety benchmark/paper directions before choosing Step 2.
Legal standards are high-stakes, but they are not the strongest fit to the current safety-evaluation community signal.
Recent work clusters around harmful-behavior refusal and jailbreak robustness (HarmBench, JailbreakBench, StrongREJECT), hazardous-knowledge proxy evals (WMDP), cyber misuse boundaries (CyberSecEval), and policy/risk taxonomies (AIR-Bench/AIR 2024).
Decision for Step 2 planning: use a sanitized safety-policy/request-intent concept taxonomy as the first safety-transfer domain, not generic legal standards.
Constraint: keep Step 2 classification-only and category-level; do not include executable harmful instructions in public stimuli.
Memo written to `experiments/exp3_directional_confusions/SAFETY_TRANSFER_SCAN.md`.
