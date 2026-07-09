
## 2026-07-09 12:30 Branch scaffold and baseline gate
- Goal this session: create the Experiment 1 branch and establish the CPU-reproducible baseline scaffold without skipping the behavioral reliability gate.
- What I ran / built: wrote experiment-local artifacts under `experiments/exp1_triplet_concept_move`, computed feature RDMs from NOVA over the 128 held-out concepts and selected 30 items, froze the 30-item triplet stimuli/protocol, and built a provisional behavioral RDM from the archived Llama triplet embedding.
- Result (numbers; plots saved to /figs with filenames): selected target `antelope` against `bison`; protocol has 12180 triplets per run; archived split-half reliability proxy is 0.400; plots saved as `figs/feature_mds.png` and `figs/feature_dendrogram.png`.
- Interpretation (what the result means, not just restating it): feature-space design is available, but the behavioral gate is not green because the archived 128-item run is not an independent 30-item canonical/paraphrase floor under the frozen protocol.
- Lit found + how it changes the plan: pending; initial local work prioritized the hard gate and artifact layout. Literature notes must be appended before paper claims are made.
- Decision / next step + WHY this over the alternatives I considered: next collect fresh base triplet runs under `triplet_protocol.json`; this is required before LoRA training because detection SNR needs a valid floor. I rejected using the archived 128-item run as the final floor because it would blur a legacy sampling scheme with the experiment's frozen 30-item scheme.
- Open risks: no local GPU is visible; the base/control/edit model calls likely need CHTC or another GPU host.

### DECISION 2026-07-09 12:30 - Item selection / concept to perturb
- Choice: perturb `antelope` with `bison` as the concentrated neighbor.
- Alternatives considered: food clusters are tighter in NOVA feature space, and isolated items would simplify far-item contrast. I did not choose the food clusters because an animal-neighbor move is easier to interpret as a relation-level concept geometry change; I did not choose isolated items because locality needs a near neighbor to push against.
- Position in feature-space RDM: target-neighbor distance is 0.3434; selected-set median distance is 0.7575; max distance is 0.9256.
- What would have made a different concept better: a held-out concept with an even clearer named pair, such as zebra/horse, would be preferable if present in the 128-item set and covered by the same feature data.

### DECISION 2026-07-09 12:30 - Gate before training
- Choice: keep the behavioral gate red and do not train LoRAs yet.
- Why this and not the alternative: the alternative was to treat the archived 128-item triplet embedding as `RDM_base`; that would create a base map but not the required sampling/paraphrase noise floor. Because SNR is denominated in floor units, using the wrong floor would make every later detection number uninterpretable.
- Course-correction criterion: once all required protocol runs exist and split/paraphrase reliability is acceptable, promote feature-space draft edits into trainable `edits/` specs and start the two-LoRA null.
- Draft edit prechecks: concentrated_drop_015 row_l2=0.0179, concentrated_drop_035 row_l2=0.0466, concentrated_drop_065 row_l2=0.1103, concentrated_drop_100 row_l2=0.2623.

## 2026-07-09 13:05 Literature search pass 1
- Goal this session: check the nearest prior art before turning the scaffold into claims.
- What I ran / built: searched for the project anchors: triplet/odd-one-out embeddings, LLM conceptual-geometry probes, knowledge-edit ripple effects, model diffing/crosscoders, edited-fact detection, and edit-based fingerprinting.
- Result (numbers; plots saved to /figs with filenames): no plots. Key sources found: Suresh et al. 2023 (arXiv:2304.02754), Zheng/Hebart/Pereira/Baker SPoSE precursor (arXiv:1901.02915), VICE (arXiv:2205.00756), RippleEdits (arXiv:2307.12976), GradSim (arXiv:2407.12828), DEED (arXiv:2405.02765), FPEdit (arXiv:2508.02092), BatchTopK crosscoder follow-up (arXiv:2504.02922), Delta-Crosscoder (arXiv:2603.04426), and CLaRE-ty (arXiv:2603.19297). Exact SRF/arXiv:2605.26921 search did not return a match and needs later verification.
- Interpretation (what the result means, not just restating it): the local experiment framing still holds. The triplet literature validates the behavioral instrument; RippleEdits/GradSim/CLaRE-ty validate structured collateral; DEED validates the task cousin but remains fact-level and uses internal features/probabilities; crosscoders and Delta-Crosscoder are the closest model-diffing neighbors but require activations/weights. Delta-Crosscoder is especially relevant because it explicitly targets narrow fine-tuning, so our novelty defense must emphasize black-box behavioral access and relation-level output, not merely "narrow edit detection."
- Lit found + how it changes the plan: VICE's uncertainty/sample-size framing strengthens the decision to freeze a high-coverage 30-item triplet protocol and refuse the archived sparse floor. DEED's future-work framing supports the outputs-only and any-edit-at-all claims, but our scorer should avoid binary "edited yes/no" language and report reconstruction/localization. GradSim/CLaRE-ty suggest a future white-box comparison baseline, but they should not enter Experiment 1's black-box success criterion.
- Decision / next step + WHY this over the alternatives I considered: keep Experiment 1 as behavioral RDM attribution and do not add crosscoder/activation baselines now. Adding them would answer a different access-assumption question and would delay the resolution map; the right comparison is rhetorical in Exp 1 and empirical only in a later white-box-vs-black-box study.
- Open risks: because several 2026 crosscoder papers are very recent, verify again before submission; do not claim "no narrow-finetune diffing" broadly, only "no choice-only black-box relation-level attribution" unless another search disproves it.

### DECISION 2026-07-09 13:18 - Triplet runner for paraphrase floor
- Choice: add `scripts/run_experiment1_triplets.py` instead of modifying the repo-wide `src/run_local.py`.
- Why this and not the alternative: `src/run_local.py` is shared by prior coherence runs and assumes the canonical prompt builder. Experiment 1 needs canonical and paraphrased prompts under one frozen protocol, so a scoped runner avoids changing old behavior while making the prompt variant explicit in the raw CSV.
- Course-correction criterion: if the scoped runner drifts from the project vLLM conventions, replace duplicated model-loading code with a shared helper; for now it imports the existing registry/model-resolution functions and only owns prompt formatting plus output paths.

## 2026-07-09 13:42 Simulated pipeline smoke and real-run orchestration
- Goal this session: keep moving toward a working Experiment 1 despite no local GPU by validating the analysis path and making the real GPU path executable.
- What I ran / built: added `scripts/simulate_experiment1_detection.py`, `scripts/build_experiment1_rdm.py`, `scripts/score_experiment1_cell.py`, `scripts/summarize_experiment1_detection.py`, and `scripts/run_experiment1_real_gpu.sh`. Ran the simulator over 8 edit cells x 5 seeds, then reran it in compact mode.
- Result (numbers; plots saved to /figs with filenames): simulated heatmap saved to `experiments/exp1_triplet_concept_move/simulation/resolution_heatmap_simulated.png`; concentrated cells crossed median SNR > 1 at `concentrated_drop_065` (median SNR 1.263, top-rank rate 0.8) and `concentrated_drop_100` (median SNR 2.075, top-rank rate 1.0). Diffuse large edits also crossed in the oracle simulation. Compact simulation artifacts are 1.3 MB.
- Interpretation (what the result means, not just restating it): the triplet aggregation, row-change scorer, SNR ranking, and heatmap code can recover a known injected target move from sampled odd-one-out choices. This reduces analysis-code risk before GPU time. It does not satisfy the experiment's model-edit success criterion because the behavioral displacement is injected by an oracle mapping from feature RDM to triplet choices.
- Lit found + how it changes the plan: no new literature in this step. The result reinforces the original plan to collect the true base floor before training because the simulator's clean boundary depends on a known control/floor denominator.
- Decision / next step + WHY this over the alternatives I considered: added a simulator lane rather than pretending archived model data are enough. The alternative was to wait for GPU access and leave all downstream code untested; the simulator gives a strong end-to-end analysis smoke while preserving the red behavioral gate.
- Open risks: real LoRA training may fail to propagate feature-list edits into triplet behavior; if so, pull fallback lever 6.2 with direct pairwise/similarity supervision before activation steering.

### DECISION 2026-07-09 13:42 - Real run orchestration
- Choice: encode the GPU execution order in `scripts/run_experiment1_real_gpu.sh` and keep CHTC submission separate until an authenticated CHTC SSH master session exists.
- Why this and not the alternative: BatchMode SSH to `ap2002.chtc.wisc.edu` is reachable but denied without interactive authentication, so automatic submission is not available from this session. A GPU-host script is still useful because it freezes the correct order and can run either on a local GPU box or inside a CHTC job wrapper.
- Course-correction criterion: once CHTC ControlMaster access is active, package this script plus the repo state into a run directory and submit a short pathway-check job for `concentrated_drop_100` before launching the full grid.

### DECISION 2026-07-09 13:55 - Fallback direct similarity supervision
- Choice: add `scripts/build_experiment1_similarity_sft_data.py` and a `SUPERVISION=similarity` switch in the real GPU runner.
- Why this and not the alternatives: if feature-listing LoRA does not propagate into triplet behavior, the most direct diagnosis is to supervise pairwise similarity ratings that encode the same intended move while keeping detection held out as triplets. I did not jump to activation steering because that changes the edit modality and threat-model story more sharply; pairwise supervision tests the feature-listing-to-behavior bottleneck first.
- Dry-run result: for `concentrated_drop_100`, the fallback builder wrote 4,872 control examples and 5,220 edit examples under `sft_similarity_data/concentrated_drop_100/`, with `gate_passed_when_built=false` because this was data inspection only.
- What would confirm this lever: if feature-listing supervision gives feature-RDM movement but behavioral triplet movement below floor, and pairwise supervision moves the triplet RDM above floor, the bottleneck is the feature-list-to-similarity transfer rather than the detector.

### DECISION 2026-07-09 14:03 - Simulation artifact hygiene
- Choice: make simulated raw triplet CSVs, per-seed RDM arrays, and per-seed detection JSONs optional flags rather than default outputs.
- Why this and not the alternative: the summary CSV and heatmap are enough to validate the analysis path, and the larger intermediate files are deterministic/regenerable. Keeping them by default made `git status` noisy without increasing evidence quality.
- Result: default simulation output is now four files under `simulation/`, about 71 KB total. Optional detail can be recovered with `--write-raw`, `--write-rdms`, and `--write-detection-json`.

## 2026-07-09 14:15 CHTC pathway-check package
- Goal this session: move past local GPU absence by making the first real CHTC run submit-ready.
- What I ran / built: added `chtc/exp1_triplet_move/prepare_exp1_bundle.sh`, `run_exp1_pathway_check.sh`, `exp1_pathway_check.sub`, and `README.md`. Built the transfer bundle locally.
- Result (numbers; plots saved to /figs with filenames): `chtc/exp1_triplet_move/exp1_triplet_move_bundle.tgz` is 2.1 MB and contains the Exp 1 scripts, selected NOVA/scale128 inputs, current experiment artifacts, and the archived Llama triplet file used only for provisional fallback. The pathway submit defaults to `concentrated_drop_100` with feature supervision on a 40GB+ staging GPU.
- Interpretation (what the result means, not just restating it): the next real experiment step is no longer "figure out how to run it"; it is a concrete CHTC push/submit/pull sequence. The job itself enforces the hard gate by collecting the three base triplet runs and stopping before LoRA training if `floor_stats.json` stays red.
- Lit found + how it changes the plan: no new literature in this step.
- Decision / next step + WHY this over the alternatives I considered: packaged a single pathway check instead of the full grid because the first scientific question is whether the largest feature-list edit moves the behavioral triplet RDM at all. A full grid before pathway confirmation would waste GPU time and obscure the mechanism if feature-list supervision fails.
- Open risks: runtime pip installation of vLLM/PEFT/TRL/bitsandbytes may be slow or conflict with the PyTorch container; if the CHTC job fails before experiment code starts, switch to a prebuilt LLM SIF or pin fewer runtime dependencies in a new wrapper. Local `condor_submit -dry-run` could not validate the submit file because this non-access-point machine's Condor config failed `NETWORK_INTERFACE=*`; submit-file validation should be repeated on `ap2002`.

### DECISION 2026-07-09 14:24 - CHTC submit helper
- Choice: add `chtc/exp1_triplet_move/submit_exp1_pathway.sh`.
- Why this and not the alternative: the manual push/submit sequence is easy to mistype and needs to rebuild the bundle after local changes. A helper makes the next authenticated CHTC session one command while still keeping the manual commands in the README for auditability.
- Constraint: it depends on `chtc-ssh` and `chtc-push`, so it still requires the user's interactive Duo-backed SSH master session.

### BLOCKER 2026-07-09 14:31 - CHTC authentication not active
- What I tried: checked `chtc-ssh`/`chtc-push` availability, then ran a short `chtc-ssh 'hostname -f'` probe. The sandboxed check failed on control-socket/DNS; the escalated check reached SSH but failed with missing askpass and `Permission denied (gssapi-with-mic,keyboard-interactive)`.
- Interpretation: there is no active Duo-backed CHTC ControlMaster session available to this process. The pathway job is submit-ready, but cannot be pushed/submitted noninteractively from here until the user starts or refreshes CHTC auth.
- Next action: user runs `chtc-master start` (or equivalent), verifies `chtc-master check`, then runs `chtc/exp1_triplet_move/submit_exp1_pathway.sh` from this repo.
