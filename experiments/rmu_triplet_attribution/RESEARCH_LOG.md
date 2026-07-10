# RMU Triplet Attribution Research Log

This log is append-only. New entries should explain both what happened and why
the next decision was chosen over alternatives.

## 2026-07-09 Step 0 Access Check And Branch Setup

- Goal this session: create a separate branch for the RMU/WMDP behavioral edit
  attribution experiment and run the required access check before any edit or
  probe work.
- What I ran / built: created branch `exp-rmu-triplet-attribution` in worktree
  `/tmp/exp-rmu-triplet-attribution-worktree`; checked Hugging Face access for
  `cais/wmdp`, `cais/wmdp-bio-forget-corpus`, and
  `cais/wmdp-cyber-forget-corpus`; loaded small slices from all public WMDP QA
  configs.
- Result: `cais/wmdp` is accessible and downloads with configs `wmdp-bio`,
  `wmdp-chem`, and `wmdp-cyber`. The bio and cyber forget corpora are gated in
  this environment. Raw access result is saved in
  [results/access_check_2026-07-09.json](results/access_check_2026-07-09.json).
- Interpretation: public QA scoring can proceed, but the exact WMDP-authors'
  RMU bio-unlearning recipe cannot be run from this environment until the
  Hugging Face account/token has access to the gated forget corpus.
- Lit found + how it changes the plan: no new literature search was run in this
  setup step; the gating check was the hard prerequisite specified by the
  experiment.
- Decision / next step + WHY this over alternatives I considered: stop after
  Step 0 and report the gate, because the spec says Step 0 must be done first
  and then reported. Do not silently substitute a public corpus yet, because that
  would change the edit from exact WMDP-bio RMU to a representative
  bio-unlearning edit. The fallback is viable, but it should be explicit.
- Open risks: exact forget-corpus access is unavailable; RMU resource needs may
  require H100/CHTC; probe prompts are not yet frozen.

## 2026-07-09 DECISION: Access Gate Before RMU

- Choice: treat gated `cais/wmdp-bio-forget-corpus` access as the first decision
  gate.
- Alternatives considered: proceed immediately with a public proxy corpus; start
  by building the triplet probe without the edit; clone WMDP and tune RMU
  against a guessed local corpus.
- Why this choice: the experiment's strongest claim depends on a known, clean
  upstream edit. Running a different corpus without logging the substitution
  would make later AUC results harder to interpret and easier for reviewers to
  attack. The probe can be built in parallel later, but the edit definition
  determines which concepts count as positives, neighbors, and controls.
- What would change the decision: if the user grants gated Hugging Face access,
  use the exact WMDP/RMU path. If the user prefers speed over exact replication,
  proceed with a public proxy corpus and label the edit accordingly.
