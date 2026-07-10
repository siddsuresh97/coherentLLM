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

## 2026-07-09 Authenticated Access Reruns

- Goal this session: install the user-provided Hugging Face token as a local
  private repo secret and confirm whether the exact WMDP-bio forget corpus is
  now accessible.
- What I ran / built: saved the token to
  `experiments/rmu_triplet_attribution/.env` with file mode `600`; confirmed Git
  ignores that file via the existing `.gitignore` `.env` rule; reran the access
  check first with the pre-existing shared token and then with the new local
  `.env` token.
- Result: the pre-existing shared token authenticated as a different account and
  still lacked access. The local `.env` token authenticated as the user account
  and successfully loaded `cais/wmdp-bio-forget-corpus` with columns `title`,
  `abstract`, `text`, and `doi`. Public WMDP QA remains accessible. The cyber
  forget corpus remains gated, but it is not required for the planned bio-only
  RMU edit. Results are saved in
  [results/access_check_2026-07-09_authenticated.json](results/access_check_2026-07-09_authenticated.json)
  and
  [results/access_check_2026-07-09_local_env_token.json](results/access_check_2026-07-09_local_env_token.json).
- Interpretation: Step 0 is now green for the exact WMDP-bio path. The previous
  blocker was account/token mismatch, not dataset unavailability. We can proceed
  with the exact WMDP/RMU recipe rather than switching to the public-proxy
  fallback.
- Lit found + how it changes the plan: no literature search was run in this
  access step; the result changes the plan by removing the need for a proxy
  corpus.
- Decision / next step + WHY this over alternatives I considered: proceed with
  the exact WMDP path. I considered keeping the public-proxy fallback ready, but
  now that the real forget corpus loads, using a proxy would weaken the claim for
  no practical benefit. The next concrete step is to clone/inspect the WMDP repo,
  identify the Zephyr RMU default config, and set up reproducible QA baseline
  scripts before any RMU training.
- Open risks: the token is local and ignored, so a different machine/session will
  need equivalent HF access; RMU resource requirements may exceed local A5000s;
  W&B must be online for training once it starts.
