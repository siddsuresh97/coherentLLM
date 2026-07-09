# Huth/LeBel CHTC Smoke Jobs

This directory runs the first GPU-side ds003020 smoke check on CHTC after the
smoke subset has been staged under `/staging/s/suresh27/datasets/ds003020-smoke`.

The initial debug job is intentionally small:

- story: `sweetaspie`
- words: first 64 TextGrid words
- layer: 24
- arms: `base`, `lowLR`, `scrambled`, `taskvec_a0p25`
- model: `/staging/s/suresh27/models/llama31-8b-instruct`
- adapters: `/staging/s/suresh27/adapters/{lowLR,scrambled,taskvec_a0p25}`

It returns a tarball containing NPZ feature files and logs. Run this before any
full three-story extraction job.

