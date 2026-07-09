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
