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
