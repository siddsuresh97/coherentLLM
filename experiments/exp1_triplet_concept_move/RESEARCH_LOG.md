
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

## 2026-07-09 14:03 CHTC pathway submission course-correction
- Goal this session: publish the Exp 1 branch, verify local/CHTC GPU options, and submit the first real pathway-check job.
- What I ran / built: pushed branch `exp1-triplet-concept-move` at commit `12dc048`; checked local GPU availability with `nvidia-smi`; started an authenticated CHTC ControlMaster session with Duo, superseding the earlier CHTC-auth blocker; verified `ap2002.chtc.wisc.edu`, `condor_q`, and the staged model/cache paths; patched the CHTC pathway runner to refresh the behavioral floor directly from frozen triplet runs instead of rerunning feature-space setup.
- Result (numbers; plots saved to /figs with filenames): local GPU is not available on this host (`nvidia-smi` cannot communicate with a driver); CHTC auth succeeded and the queue was reachable with 0 existing jobs; `/staging/s/suresh27/models/llama31-8b-instruct` and `/staging/s/suresh27/hf_home` exist.
- Interpretation (what the result means, not just restating it): the training path must use CHTC from this host. The original feature-listing pathway cannot run from the committed checkout because `data/nova/verified_matrix_cogsci2025.parquet` is absent, even though the selected items, candidate edits, and prebuilt similarity fallback data are present.
- Lit found + how it changes the plan: no new literature read in this operational session.
- Decision / next step + WHY this over the alternatives I considered: submit `concentrated_drop_100` with direct pairwise-similarity supervision (fallback lever 6.2) instead of waiting on the missing feature parquet. I considered blocking until the parquet is found/restored, but that would prevent testing the end-to-end behavioral gate, LoRA training, triplet recovery, and scorer on CHTC. I rejected silently treating this as the feature-listing pathway because that would misstate the edit lever; the report and CHTC README now call it the similarity fallback. The result that would confirm this choice is a completed CHTC job with the edited concept top-ranked and SNR > 1; if it fails before training, the blocker is operational, not a null result about triplet detectability.
- Open risks: the similarity fallback is a stronger lever than the intended feature-listing edit, so a positive result confirms the detection pathway but not the feature-listing edit pathway; restore the NOVA feature parquet before making the feature-supervision claim.

### DECISION 2026-07-09 14:03 - Course-correction to similarity fallback
- What specifically changed: the default CHTC pathway supervision is now `similarity` for `concentrated_drop_100`; feature-listing supervision remains available only if the missing NOVA feature parquet is restored.
- Why this and not the alternatives: the observed failure was at bundle construction (`tar: data/nova/verified_matrix_cogsci2025.parquet: Cannot stat`), before any model run. That says the operational artifact is missing, not that feature edits failed behaviorally. The cheapest useful next test is therefore the already-prepared direct-similarity lever, which can still exercise the hard behavioral floor, two-LoRA null, triplet detection, and SNR scoring. Waiting for the parquet would block all CHTC validation; trying to synthesize feature lists from aggregate RDMs would create unprincipled supervision data.
- What would confirm or kill the hypothesis: if the similarity fallback produces SNR > 1 with the edited concept top-ranked, the detection pipeline fires under a strong edit lever and the next task is restoring feature-listing data for the weaker intended lever. If it fails despite a green behavioral floor and successful LoRA training, then the issue is likely detection sensitivity or triplet elicitation rather than the missing feature data.

## 2026-07-09 14:08 Local GPU path confirmed
- Goal this session: test the user's note that local GPUs are available and identify the appropriate local environment.
- What I ran / built: reran `nvidia-smi` outside the sandbox, checked conda envs, probed torch/vLLM/SFT imports, added a Transformers backend to `scripts/run_experiment1_triplets.py`, and ran a 100-triplet local smoke on GPU 0 with 4-bit loading.
- Result (numbers; plots saved to /figs with filenames): outside the sandbox, `rogers-gpu-1` exposes two idle RTX A5000 GPUs with 24 GB each. The `coherence` env has torch 2.5.1+cu124 and sees both GPUs. vLLM imports did not complete within 120 seconds, but Transformers 4-bit generation worked: `local_smoke_transformers_100` wrote 100 frozen-protocol triplet responses successfully.
- Interpretation (what the result means, not just restating it): local GPU execution is viable if triplet inference uses Transformers instead of vLLM. This is slower than vLLM but avoids the current env/import problem and is enough for the first large-edit pathway check.
- Lit found + how it changes the plan: no new literature in this operational session.
- Decision / next step + WHY this over the alternatives I considered: run the first real pathway check locally before spending CHTC queue time. I considered fixing the local vLLM env first, but the Transformers backend already works with cached weights and keeps the detection format unchanged. I also considered using `coherence_sft`, but it timed out during basic package probes; `coherence` is the working local env for this run.
- Open risks: Transformers generation will be slower than vLLM; if the local run is too slow or hits PEFT/training issues, use the CHTC wrapper or repair a vLLM env.

### DECISION 2026-07-09 14:08 - Local backend choice
- Choice: use `TRIPLET_BACKEND=transformers`, `TRIPLET_LOAD_IN_4BIT=1`, `COHERENCE_ENV=/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence`, and conservative A5000 settings for the local pathway check.
- Why this and not the alternatives: the regular sandbox hides GPU devices, but escalated local commands see two idle GPUs. The `coherence` env can run torch/CUDA and Transformers generation; vLLM is not currently a reliable local backend because imports hang through the installed stack. CHTC remains available, but local GPUs avoid queue latency for the first pathway check.
- What would change this decision: if local training fails due missing PEFT/runtime dependencies or the 12,180-triplet runs are too slow, switch back to CHTC or repair a dedicated vLLM environment.

### NOTE 2026-07-09 14:35 - WandB logging for future runs
- Current active local pathway run is file-only for training telemetry because `scripts/run_experiment1_real_gpu.sh` passes `--report_to none` into both LoRA training calls, and the `coherence` env does not currently import `wandb`.
- Why it happened: I disabled WandB to avoid auth/network/package issues blocking the first unattended local/CHTC pathway check. That made the run more robust but less visible.
- Future action: before the next serious sweep, install/verify `wandb` in the selected env and run with training reporting enabled. Also add explicit triplet-run metrics logging if live generation progress matters, because the triplet runner itself does not currently log to WandB.

## 2026-07-09 14:45 Active local run context made explicit
- Goal this session: make the report/log self-contained enough that the active run can be audited without reading every script.
- What I ran / built: updated `REPORT.md` with exact model snapshot, hardware, environment, prompts, triplet protocol, floor result, supervision data counts, LoRA settings, and the active command line.
- Result (numbers; plots saved to /figs with filenames): active model is `llama-3.1-8b-instruct` from snapshot `0e9e39f249a16976918f6564b8830bc894c89659`; host is `rogers-gpu-1.discovery.wisc.edu`; GPU is RTX A5000 via `CUDA_VISIBLE_DEVICES=0`; env is `/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence`. Frozen triplet floor is green with mean upper-triangle Pearson 0.8066; canonical rerun Pearson 1.0000/RMS 0.0000; paraphrase Pearson 0.6133/RMS 0.1768.
- Interpretation (what the result means, not just restating it): the baseline is now a real frozen-protocol behavioral floor, not the archived provisional embedding. The current run is a fallback-lever pathway test: direct pairwise-similarity LoRA supervision, evaluated only through held-out triplet judgments.
- Lit found + how it changes the plan: no new literature in this operational documentation step.
- Decision / next step + WHY this over the alternatives I considered: keep both the exact prompts and the exact command in `REPORT.md` rather than relying on script defaults. The alternative was to cite `triplet_protocol.json` and `run_experiment1_real_gpu.sh` only, but that makes the report too opaque for audit. This change makes clear which parts are detection prompts, which are fallback training prompts, and which settings are local execution details.
- Open risks: report status must be refreshed after edit training, post-LoRA triplet runs, and detection scoring finish; until then, SNR and final verdict remain pending.

### NOTE 2026-07-09 14:45 - Exact prompts and supervision distinction
- Detection prompt template: `Answer using only one word - {concept1} or {concept2} and not {anchor}. Which is more similar in semantic meaning to {anchor}?`
- Detection paraphrase template: `Reply with only {concept1} or {concept2}. Compared with {anchor}, which option is closer in meaning?`
- System prompt for triplet generation: `You are a helpful assistant who gives responses to questions.`
- Training prompt for the current fallback only: `Answer with only one number from 1 to 7, considering 1 as 'extremely dissimilar', 2 as 'very dissimilar', 3 as 'likely dissimilar', 4 as 'neutral', 5 as 'likely similar', 6 as 'very similar', and 7 as 'extremely similar': How semantically similar is {a} and {b}?`
- Why this matters: the current run does not test feature-listing supervision. It tests whether a deliberately strong similarity-supervised concept move can be recovered by the frozen triplet detector.

### NOTE 2026-07-09 14:50 - Report audit map added
- Added an `Audit Map` section to `REPORT.md` that points each key claim to the source artifact: model/config, item set, prompt protocol, raw base CSVs, green floor stats, LoRA settings, and WandB upload plan.
- Added concrete examples for both roles: a detection triplet prompt (`antelope` anchor, `bison` vs `toaster`) and a fallback training pairwise prompt (`antelope and bison -> 1`), with the latter verified against `sft_similarity_data/concentrated_drop_100/edit.jsonl`.

## 2026-07-09 15:25 Local similarity fallback result
- Goal this session: finish the first real local pathway check for `concentrated_drop_100` under direct pairwise-similarity fallback supervision.
- What I ran / built: collected three base frozen-protocol triplet runs, trained control and edit PEFT QLoRA adapters, recovered control/edit triplet RDMs, scored detection, and summarized the one-cell real heatmap.
- Result (numbers; plots saved to /figs with filenames): behavioral floor is green; control final logged loss 0.0032; edit final logged loss 0.0015; detection output is `detection/concentrated_drop_100.json`; `target_rank=1`; `target_snr_control_only=1.0`; `boundary_crossed=false`; real one-cell heatmap saved to `figs/resolution_heatmap.png`.
- Interpretation (what the result means, not just restating it): partial only. The target concept was top-ranked and the intended neighbor pair was the top target-pair change, but the edit did not exceed the control null. Control and edit row-change magnitudes were effectively matched, so this is not a positive attribution result under the experiment's SNR > 1 criterion.
- Lit found + how it changes the plan: no new literature in this operational result step.
- Decision / next step + WHY this over the alternatives I considered: next use a matched-target similarity control rather than rerunning the same fallback. The current control omitted target pairs while the edit arm added edited target-pair examples, so the run confounds target-pair exposure with label/content change. A matched-target control should include base target-pair ratings while the edit arm uses altered target-pair ratings. I reject simply increasing steps/rank first because the problem is not weak training; both adapters trained to very low losses and both moved behavior about equally.
- Open risks: the result may also reflect the Transformers 4-bit inference backend or the pairwise-to-triplet transfer itself; after tightening the control, if SNR still stays at 1, consider a stronger direct triplet-supervision lever or a narrower target-only replay design.

### DECISION 2026-07-09 15:25 - Did it work?
- Verdict: partial / did not fire above null.
- Evidence: `antelope` was top-ranked (`target_rank=1`), but `target_snr_control_only=1.0`, `target_snr_floor_adjusted=1.0`, and `boundary_crossed=false`.
- Why this is not counted as success: the success criterion requires the edit effect at the target concept to exceed the matched control-null drift. Here, the matched null was as large as the edit effect.

### DECISION 2026-07-09 15:25 - Course-correction to matched-target control
- What specifically in the result told me the cause: every inspected ranking row had edit and null row RMS values equal to the reported precision, and the target-pair localization table showed equal target/null deltas. That means the issue is not failure to move the RDM; it is that the control LoRA moved it just as much.
- Hypothesis for failure: the fallback pairwise control was too broad and not exposure-matched. It trained on replay pairs excluding target pairs, while the edit arm trained on replay plus target-pair examples. The difference between arms is therefore both "target pair is present" and "target pair label changed", not only the edit.
- Next lever and why: implement a matched-target similarity fallback where the control arm includes base target-pair ratings and the edit arm includes altered target-pair ratings, with the same number and shape of examples. This directly tests whether label/content change, not target exposure or general pairwise calibration, produces the triplet attribution signal.
- What would confirm or kill the hypothesis: if matched-target control tightens null drift and SNR rises above 1, the previous null was the problem. If SNR remains at 1 with both arms exposure-matched, then pairwise-similarity supervision may be too globally entangling or the triplet detector may need direct triplet-supervision fallback.

## 2026-07-09 15:55 Report clarity revision
- Goal this session: make the running report understandable without requiring the reader to infer the hypothesis, intervention, verdict, and failure explanation from scattered implementation details.
- What I ran / built: rewrote the top of `REPORT.md` into explicit sections for hypothesis, test, verdict, why it did not count as success, how to read SNR, and a stage-by-stage summary table with links to the evidence files.
- Result (numbers; plots saved to /figs with filenames): no new experiment numbers. The report now states plainly that the run was partial / not successful: `target_rank=1`, `target_snr_control_only=1.0`, and `boundary_crossed=false`.
- Interpretation (what the result means, not just restating it): the scientific state is now clearer. The current result is evidence that the target and neighbor can move in the measured triplet RDM, but not evidence of attribution above null, because the control LoRA moved behavior just as much.
- Lit found + how it changes the plan: no new literature in this documentation step.
- Decision / next step + WHY this over the alternatives I considered: keep the detailed audit map and exact prompts in the report, but move them below a plain-language verdict. The alternative was to leave the report as an artifact index, which was accurate but made the reader reconstruct the causal story by hand.
- Open risks: the next actual experiment step remains unchanged: implement and run the matched-target similarity control before escalating to direct triplet supervision.

## 2026-07-09 16:50 NOVA matched-target feature-listing data
- Goal this session: replace the ambiguous similarity-fallback data with NOVA-backed feature-listing supervision that matches target exposure across control and edit.
- What I ran / built: downloaded the public NOVA `verified_matrix_cogsci2025.parquet` into gitignored `data/nova/`, patched `scripts/build_experiment1_sft_data.py` so control and edit use the same concept prompts and row counts, added symmetric target oversampling, and generated `sft_data/concentrated_drop_100/{control.jsonl,edit.jsonl,manifest.json}`.
- Result (numbers; plots saved to /figs with filenames): NOVA loaded as 787 concepts x 8,202 binary features; all 30 selected Exp 1 concepts are present. The matched dataset has 296 rows per arm: 29 non-target concepts x 8 repeats plus `antelope` x 64 repeats. The target list has 80 original features in control and 29 features after the concentrated edit, with 51 removed features realized from the top-80 target list.
- Interpretation (what the result means, not just restating it): the next training run now tests the intended feature-listing pathway more cleanly. Any edit-vs-control difference cannot be explained by the edit arm merely seeing `antelope` examples while the control arm does not, because both arms see the same `antelope` prompt count.
- Lit found + how it changes the plan: the public `llm-norms-cogsci2025` repository confirms the NOVA matrix format and source. This supports using NOVA as the external semantic feature source rather than continuing with the previous pairwise fallback.
- Decision / next step + WHY this over the alternatives I considered: use matched-target feature-listing data before increasing rank/steps or switching to triplet supervision. The previous result showed equal control/edit drift, so the first fix should tighten the null, not just strengthen the edit. I chose symmetric target oversampling (64 repeats in both arms) because the target edit would otherwise be only 8 rows out of 240, likely too weak for a fast pathway check; oversampling both arms preserves exposure matching.
- Open risks: W&B is installed now, but no API key is configured in this environment, so the upcoming training will use W&B offline mode and syncable local run files unless the user logs in before launch. Feature-listing supervision may still fail to propagate to triplet behavior; if so, the next lever is matched-target pairwise or direct triplet supervision.

### DECISION 2026-07-09 16:50 - Course-correction to NOVA matched-target feature listing
- What specifically changed: control now includes original `antelope` feature-listing examples, while edit includes the same `antelope` prompts with edited NOVA features. Both arms have 296 rows.
- Why this and not the previous fallback: the previous control excluded target pairs and therefore confounded target exposure with target-label change. NOVA lets us manipulate the actual concept-feature content while keeping prompt exposure matched.
- What would confirm this choice: after logged training, the edit LoRA should move the `antelope` triplet-RDM row more than the matched control LoRA, producing SNR > 1 and `target_rank=1`.
- What would kill it: if both LoRAs again move equally, then feature-listing SFT is either too globally entangling or too weakly connected to triplet behavior; then the next rational test is a matched-target pairwise or direct triplet lever, not another replay-only feature-listing rerun.

## 2026-07-09 17:35 NOVA feature-listing pathway result and W&B correction
- Goal this session: evaluate the completed NOVA feature-listing LoRAs, correct W&B behavior for future runs, and decide whether the failure is likely data design or training strength.
- What I ran / built: scored the completed `concentrated_drop_100` feature-listing adapters, regenerated the real heatmap summary, added W&B-online preflight/key discovery to `scripts/run_experiment1_real_gpu.sh`, made `src/sft/train_lora.py` fail rather than silently disable W&B, updated `scripts/upload_experiment1_wandb.py` for feature adapters and nested detection metrics, and regenerated SFT manifests with the next low-drift recipe.
- Result (numbers; plots saved to /figs with filenames): completed feature-listing training reached final logged loss about `0.0001` for both control and edit. Detection result is `target_rank=23`, `target_snr_control_only=1.0553`, `target_top_ranked=false`. The heatmap files are `figs/resolution_heatmap.png`, `figs/resolution_heatmap.csv`, and `figs/resolution_heatmap_summary.json`. W&B preflight passes when `WANDB_NETRC=/mnt/home/ssuresh/.netrc` is available and fails fast with a missing netrc.
- Interpretation (what the result means, not just restating it): the NOVA feature-listing edit moved behavior, but not specifically enough to attribute the edited concept. The target row changed only slightly above the control null, while many non-target rows changed more. This is evidence of broad fine-tuning drift or overfitting, not a clean positive detection.
- Lit found + how it changes the plan: no new external literature in this step. I inspected the previous `coherence-sft` mitigation summaries instead: `lowLR` (`5e-5`, rank 64) and `lowrank` (`rank=16`, `2e-4`) preserved retention much better than the original real SFT while retaining high coherence. That changes the next run from "make the edit stronger" to "make the adapter less globally disruptive."
- Decision / next step + WHY this over the alternatives I considered: next run the same matched NOVA feature-listing data with `rank=16` and `learning_rate=5e-5`. I chose this over increasing target repeats, steps, or rank because the completed run already overfit the tiny dataset and produced broad movement; increasing strength would likely worsen the null. I chose not to change the dataset yet because the matched exposure design is now correct, and changing both data and optimizer would make the cause ambiguous.
- Open risks: lower LR/rank may make the edit too weak to move the behavioral RDM. If that happens with low global drift, the next lever should be matched direct similarity or direct triplet supervision, not another high-rank feature-listing run.

### DECISION 2026-07-09 17:35 - Did the NOVA feature-listing edit work?
- Verdict: did not work as an attribution result.
- Evidence: `target_snr_control_only=1.0553` crosses the raw SNR=1 line, but `antelope` ranked 23rd rather than 1st. The success criterion requires the edited concept to be localized as the largest above-null row change, so this is not a positive Experiment 1 cell.
- Why this is not buried as a near success: a small above-null SNR without top-rank localization means the RDM moved broadly. The experiment is about attributing which relationship/concept moved, not only detecting that some fine-tune changed behavior.

### DECISION 2026-07-09 17:35 - Course-correction to lower-drift LoRA settings
- What specifically in the result told me the cause: both adapters reached near-zero training loss on a 296-row dataset, and non-target concepts such as `ostrich` and `beetle` had larger edit/null row-change ratios than `antelope`. That pattern points to global fine-tuning drift/overfit, not insufficient edit magnitude.
- Alternatives considered: increasing target repeats or LoRA rank, switching immediately to direct triplet supervision, reducing target repeats, or changing the concept/edit. I rejected increasing strength because the run is already too broad; I rejected triplet supervision for the next immediate step because it changes the edit modality; I rejected changing the concept because `antelope`/`bison` still has a valid feature-space manipulation and the failure is in behavioral propagation/localization.
- Next lever and why: combine the conservative parts of the previous mitigation arms: rank 16 from `lowrank` and LR `5e-5` from `lowLR`, while keeping the matched NOVA dataset fixed. This directly tests whether the same intended edit becomes local once the adapter is less globally disruptive.
- What would confirm or kill this hypothesis: confirmation would be lower non-target row movement plus improved `antelope` rank/SNR. If lower rank/LR removes all behavioral movement, the feature-listing lever is too weak under retention-preserving settings and the next diagnostic should be matched pairwise or direct triplet supervision.

### NOTE 2026-07-09 17:35 - W&B online handling
- Change made: `scripts/run_experiment1_real_gpu.sh` now defaults `WANDB_MODE=online`, discovers the existing W&B key from `/mnt/home/ssuresh/.netrc` when `WANDB_API_KEY` is absent, and has `EXP1_PREFLIGHT_ONLY=1` for checking logging configuration before training. `src/sft/train_lora.py` now fails if `--report_to wandb` is requested and W&B is unavailable.
- Retroactive upload status: I attempted to upload the completed feature-listing run using `scripts/upload_experiment1_wandb.py`, but the execution policy blocked sending private experiment artifacts to W&B. I did not attempt a workaround. The uploader is prepared, but actual upload requires explicit informed approval because it transmits prompts/configs/metrics/artifacts to a third-party service.
- Why this over offline logging: the user asked to keep W&B online. The code now does that for future launches, while avoiding the earlier silent offline/unlogged behavior.

### BLOCKER 2026-07-09 17:55 - Online W&B training cannot be launched from this Codex path
- What I tried: launched the lower-drift NOVA feature-listing control/edit jobs with `WANDB_MODE=online`, `rank=16`, `learning_rate=5e-5`, matched data, and separate local GPUs. Both online launch attempts were rejected before process start because the execution policy treats W&B uploads as private workspace data leaving to a third-party service.
- User clarification: after seeing no W&B runs, the user asked why and stated "please always do online training."
- What happened next: I stopped the safer offline-local fallback jobs that had been started. They reached about step 134/400 and were interrupted. They are not counted as completed training and no detection is reported from them.
- Interpretation: this is an operational/policy blocker, not an experiment result. The next scientific step remains the same lower-drift run, but it must be launched from a context where W&B online upload is allowed.
- Decision / next step + WHY this over the alternatives I considered: do not continue offline runs, because that violates the user's latest logging requirement and would create more invisible runs. Do not attempt to bypass the W&B restriction. The appropriate next step is an online-W&B launch outside this restricted Codex execution path, using the command already recorded in `REPORT.md`.

## 2026-07-09 18:21 Lower-drift NOVA feature-listing online run
- Goal this session: rerun the matched NOVA feature-listing pathway with lower-drift LoRA settings and online W&B logging, then score whether the conservative adapter localizes the `antelope` edit better than the strong run.
- What I ran / built: trained control and edit PEFT QLoRA adapters in parallel on the local A5000s using `rank=16`, `learning_rate=5e-5`, `max_steps=400`, batch size 4, no gradient accumulation, and online W&B. Then recovered held-out triplet RDMs for both adapters under the frozen canonical triplet protocol and scored detection against the base RDM and control null. Artifacts were written under `lora_control_feature_lowdrift/`, `lora_edit_feature_lowdrift/`, `raw/*feature_lowdrift*`, `rdms/*feature_lowdrift*`, and `detection/concentrated_drop_100_feature_lowdrift.json`.
- Result (numbers; plots saved to /figs with filenames): both training arms used 296 examples and completed 400 optimizer steps, about 5.405 effective epochs. Control runtime was 456.38s with train loss 0.3799 and final logged loss 0.0002; edit runtime was 469.57s with train loss 0.4002 and final logged loss 0.0002. W&B runs: control `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/qvdyld87`, edit `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/ta6is8pl`. Detection result: `target_rank=30`, `target_snr_control_only=0.4918`, `target_snr_floor_adjusted=0.4918`, `target_top_ranked=false`, `upper_rms_edit=0.1954`, `upper_rms_control_null=0.2351`.
- Interpretation (what the result means, not just restating it): the conservative recipe reduced global edit movement relative to the strong feature-listing run (`upper_rms_edit` 0.195 vs 0.489), but it also removed the target signal. This refutes the immediate hypothesis that the previous failure was only global drift from an over-strong adapter. The feature-listing lever is now bracketed: high strength moves the behavioral RDM too broadly; low strength does not move the intended target above null.
- Lit found + how it changes the plan: no new literature read in this operational step. The result increases the value of the fallback-lever literature already noted: direct similarity or triplet supervision is now the right diagnostic for whether the bottleneck is feature-listing-to-triplet transfer.
- Decision / next step + WHY this over the alternatives I considered: next use a matched direct-similarity or direct-triplet lever rather than another feature-listing run. I considered an intermediate LR/rank sweep, but the two tested ends already show the key failure modes and more feature-listing sweeps would spend GPU time tuning an indirect lever before proving that a closer supervision signal can localize behavior. I also considered changing the target concept, but `antelope`/`bison` remains a valid feature-space manipulation; changing the concept would confound the lever diagnosis.
- Open risks: the low-drift triplet probe used the Transformers backend with 4-bit loading, not vLLM. This is scientifically valid because the frozen prompts and deterministic decoding are the same, but future probe throughput should prefer vLLM with LoRA if stable. The current heatmap script assumes one detection JSON per edit cell, so it should be extended before plotting multiple training recipes for the same cell.

### DECISION 2026-07-09 18:21 - Did the lower-drift NOVA edit work?
- Verdict: did not fire.
- Evidence: `antelope` ranked 30th of 30 concepts, with SNR `0.4918`; the boundary was not crossed under either control-only or floor-adjusted scoring.
- Why this is not counted as success: success requires the edited concept to be top-ranked and above SNR 1 relative to the matched control null. The conservative adapter did neither.

### DECISION 2026-07-09 18:21 - Course-correction after low-drift failure
- What specifically in the result told me the cause: `upper_rms_edit` dropped below the control null (`0.1954` vs `0.2351`), and `antelope` was last in the ranking. That pattern means the conservative feature-listing edit failed to propagate into the held-out triplet geometry rather than merely being hidden by a few larger non-target rows.
- Hypothesis for failure: feature-listing supervision is too indirect for this targeted relation move under retention-preserving LoRA settings. The model can memorize the edited feature list, but that does not reliably rewire odd-one-out similarity choices for `antelope`.
- Next lever and why: pull fallback lever 6.2, matched direct similarity supervision, and escalate to direct triplet supervision if similarity again moves control and edit equally. These levers are closer to the measured behavior while still keeping detection held out by prompt/format.
- Rejected alternatives: another feature-listing LR/rank sweep is less informative now because we already observed the two major regimes: broad drift at high strength and no target signal at lower strength. Activation steering is also premature because it changes the edit modality more than direct behavioral supervision.
- What would confirm or kill the hypothesis: if matched similarity or direct triplet supervision produces `target_rank=1` and SNR > 1, then the bottleneck is the feature-listing-to-triplet pathway. If even direct triplet supervision fails, the detector/null or concept selection needs to be revisited.

### NOTE 2026-07-09 18:21 - Future triplet probe backend
- The low-drift control/edit probes used `scripts/run_experiment1_triplets.py --backend transformers --load-in-4bit --batch-size 16`, not vLLM. This was a pragmatic fallback because the current vLLM/LoRA path had not been validated in the local env.
- For future probes, prefer vLLM with LoRA and KV/paged-attention batching if it loads the adapter stably. The scientific protocol is the frozen prompt and triplet set; the backend should be chosen for throughput as long as deterministic decoding and outputs remain comparable.

## 2026-07-09 18:29 Targeted triplet-SFT setup
- Goal this session: question the failed feature-listing and similarity assumptions, add metrics that compare edit directly against control, and prepare a narrower training lever that tries to move only the `antelope`-`bison` relation.
- What I ran / built: added `scripts/build_experiment1_triplet_sft_data.py` and extended `scripts/score_experiment1_detection.py` with residual metrics based on `RDM_edit - RDM_control`. Re-scored the legacy matched-similarity run as `detection/concentrated_drop_100_similarity_legacy.json`. Built `sft_triplet_data/concentrated_drop_100_triplet_targeted_v1/`.
- Result (numbers; plots saved to /figs with filenames): legacy similarity still has `target_rank=1` under the original score, but residual signal is essentially absent: `upper_rms_residual=0.000856`, `target_residual_row_rms=0.0`, `target_residual_snr_floor=0.0`. The new targeted triplet data has 2,136 examples per arm: 54 editable `antelope`/`bison` rows repeated 24 times, 240 target-preserve rows repeated twice, and 360 replay rows.
- Interpretation (what the result means, not just restating it): the old similarity run moved `antelope`, but edit and control moved it identically. That means the old SNR=1 result was a true no-net-edit result, not just a scoring artifact. The next attempt should minimize shared calibration drift and maximize a direct edit-control difference on the one intended relation.
- Lit found + how it changes the plan: no new literature in this operational step.
- Decision / next step + WHY this over the alternatives I considered: run targeted triplet SFT before changing concepts or doing a broad LR sweep. Changing concepts would test a different feature geometry before proving the lever can work. A broad LR sweep of feature-listing has weak diagnostic value because we already observed the high-strength broad-drift and low-strength no-signal regimes. Direct triplet SFT tests whether the detector can recover a localized behavioral edit at all when the supervision is close to the measured behavior but not the same prompt.
- Open risks: because the training signal is triplet-like, a positive result is a pathway/detector proof, not yet evidence that feature-listing edits are sufficient. If it works, the next job is to back off toward pairwise or feature supervision while preserving locality.

### DECISION 2026-07-09 18:29 - Residual metric addition
- Choice: add residual row and pair metrics to `score_experiment1_detection.py`.
- Why this and not only the original SNR: the original score can say `target_rank=1`, SNR=1 when edit and control move exactly together. That is not a desired edit. The residual view explicitly asks what the edit LoRA adds over the control LoRA.
- What would make this metric insufficient: if residual rank is high but both edit and control produce unacceptable global drift, we still need retention/global movement gates. Residual metrics are an addition, not a replacement for the original null-scaled SNR.

### DECISION 2026-07-09 18:29 - Targeted triplet data design
- Choice: train control and edit on identical triplet-style prompts with a non-detection template. Only rows that directly affect the symmetrized `antelope`-`bison` RDM cell get different answers across arms.
- Why this and not the alternatives: feature-listing was too indirect; pairwise similarity caused shared calibration drift; changing target concepts would not diagnose the lever. This design tests the core detectability claim with the strongest localized behavioral supervision before spending more time on indirect levers.
- Exact example: for anchor `antelope`, options `bison` and `boar`, control answers `bison`; edit answers `boar`.
- What would confirm or kill this design: confirmation is `target_rank=1`, SNR > 1, and residual `antelope` row above floor with limited non-target residual movement. If control and edit still co-move or non-target rows dominate, then the issue is either LoRA locality/retention or the triplet RDM scorer, not the feature-listing pathway alone.

### NOTE 2026-07-09 18:31 - Targeted triplet W&B runs launched
- Control run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/utp95nw6`.
- Edit run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/cj920t5i`.
- Settings: PEFT QLoRA, Llama-3.1-8B-Instruct snapshot `0e9e39f249a16976918f6564b8830bc894c89659`, `rank=16`, `learning_rate=1e-4`, `max_steps=300`, batch size 4, no gradient accumulation, seed 1729. Control ran on local GPU 0; edit ran on local GPU 1.

## 2026-07-09 18:43 Targeted triplet v1 result
- Goal this session: test whether direct triplet-style supervision can create a localized `antelope`-`bison` behavioral movement when feature-listing and pairwise levers failed or co-moved with control.
- What I ran / built: trained targeted triplet control/edit adapters online in W&B, recovered full frozen-protocol triplet RDMs, and scored with both row-level and pair-level residual metrics. Artifacts: `detection/concentrated_drop_100_triplet_targeted_v1.json`, `raw/*triplet_targeted_v1*`, `rdms/*triplet_targeted_v1*`, and `lora_*_triplet/concentrated_drop_100_triplet_targeted_v1/training_metrics.json`.
- Result (numbers; plots saved to /figs with filenames): training finished in about 161-163s per arm. Control final logged loss `0.1146`; edit final logged loss `0.1558`; both W&B runs are online (`utp95nw6`, `cj920t5i`). Detection crossed SNR > 1 at the target row (`target_snr_control_only=1.2308`) but did not localize by row (`target_rank=14`). Residual row also crossed floor (`target_residual_snr_floor=1.1853`) but ranked 24. Pair-local result is stronger: `antelope`-`bison` is rank 1 by global edit-pair delta and rank 3 by global residual-pair delta, with residual delta `0.4286`.
- Interpretation (what the result means, not just restating it): the direct triplet lever worked at the intended relation more than any previous lever, but it did not satisfy the original concept-row success criterion. The row metric is partly misaligned with a pair-local edit because a single pair movement is diluted across the 29-pair row. However, the spillover is real, not just metric dilution: unrelated residual pairs involving `boar` and `beaver` outranked or nearly matched the target pair.
- Lit found + how it changes the plan: no new literature read in this operational step.
- Decision / next step + WHY this over the alternatives I considered: next run a replay-heavy/lower-LR targeted triplet variant. I am not increasing strength first because the target pair already moved. The failure is alternative-concept spillover, especially through concepts used as edited alternatives. Reducing LR, halving target repeats, and increasing target-preserve/replay pressure is the most direct test of whether locality can be improved without losing the target pair. I am not changing the concept yet because this run proves the current concept pair can move behaviorally.
- Open risks: if replay-heavy v2 loses the target pair, we need a two-dimensional sweep rather than a single hand-tuned run: target-repeat x replay/LR. If v2 still spills over, modify the data design to restrict or balance edited alternative concepts, or test attention-only LoRA modules.

### DECISION 2026-07-09 18:43 - Metric course-correction
- What specifically changed: `score_experiment1_detection.py` now reports global pair ranks for edit and residual deltas, not just row aggregates.
- Why this and not row-only: the requested edit is relation-level. A one-pair movement can be scientifically meaningful while row RMS ranks it poorly. Row rank remains useful for "which concept moved most," but pair rank is the right companion metric for "which relationship moved."
- What would count as cleaner success now: `antelope`-`bison` top-ranked or near-top by global residual pair, target row above floor, and non-target residual pairs below the target pair or below the floor.

### DECISION 2026-07-09 18:43 - Course-correction to replay-heavy v2
- What specifically in the result told me the cause: `antelope`-`bison` moved as intended, but the top residual pairs were `boar`/`ostrich`, `boar`/`burrito`, and `beaver`/`ostrich`. That pattern points to spillover through edited alternatives and insufficient replay/retention, not an edit that is too weak.
- Next lever and why: lower LR to `5e-5`, reduce editable repeats from 24 to 12, increase target-preserve/replay rows, and train longer enough to see the target signal. This should reduce broad alternative movement while preserving the direct relation edit.
- Rejected alternatives: increasing steps or LR would likely worsen spillover; changing concept pair would hide whether the targeted lever can be made local; activation steering is premature because direct behavioral SFT has not been optimized yet.

### NOTE 2026-07-09 18:48 - Replay-heavy v2 W&B runs launched
- Control run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/1fqn9vw3`.
- Edit run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/fna9k90i`.
- Settings: PEFT QLoRA, Llama-3.1-8B-Instruct snapshot `0e9e39f249a16976918f6564b8830bc894c89659`, `rank=16`, `learning_rate=5e-5`, `max_steps=600`, batch size 4, no gradient accumulation, seed 1729. Both arms have 3,648 examples: 54 editable rows repeated 12 times, 900 target-preserve rows repeated twice, and 1,200 replay rows.
- Why this size and not the previous ~100K-example scale: the v1 failure mode was spillover through alternative concepts, not lack of convergence. This run tests the data-ratio hypothesis cheaply before scaling, because simply adding many more examples would not identify whether replay/preserve pressure is the missing ingredient.

## 2026-07-09 19:03 Replay-heavy targeted triplet v2 result
- Goal this session: test whether lower LR, fewer editable repeats, and more preserve/replay data keep the `antelope`-`bison` pair movement while reducing v1's alternative-concept spillover.
- What I ran / built: trained v2 control/edit adapters online in W&B, recovered 12,180 held-out canonical triplet judgments for each adapter, built RDMs, and scored `detection/concentrated_drop_100_triplet_targeted_v2_replayheavy.json`.
- Result (numbers; plots saved to /figs with filenames): control finished 600 steps in 322.70s (`1.859` steps/s, final logged loss `0.0289`, train loss `0.1634`); edit finished in 327.11s (`1.834` steps/s, final logged loss `0.0374`, train loss `0.1734`). W&B runs: control `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/1fqn9vw3`, edit `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/fna9k90i`. Detection: `target_rank=12`, `target_snr_control_only=1.219`, `upper_rms_control_null=0.128`, `upper_rms_edit=0.162`, `upper_rms_residual=0.109`. `antelope`-`bison` was rank 1 by global edit-pair delta and rank 4 by global residual-pair delta.
- Interpretation (what the result means, not just restating it): v2 partially worked but still is not a clean positive detection. It reduced overall residual drift relative to v1 (`0.135 -> 0.109`) and preserved the intended pair signal, but the top residual pairs became `bison` with other concepts (`bison`-`beaver`, `bison`-`beetle`, `bison`-`boar`). The failure mode is no longer mostly arbitrary alternative spillover; it is missing preservation of the neighbor concept.
- Lit found + how it changes the plan: no new literature in this operational step.
- Decision / next step + WHY this over the alternatives I considered: build v3 with one-direction editable rows and explicit `bison` preservation. I considered simply adding more replay examples or changing LR/rank, but v2 points to a specific missing data stratum: replay excluded both `antelope` and `bison`, and target-preserve only protected `antelope`. Adding generic data would dilute the edit without directly protecting the drifting neighbor. I also considered changing concepts, but the current pair still moves reliably; the cleaner test is to fix the preservation design.
- Open risks: one-direction editing may weaken the symmetrized `antelope`-`bison` RDM cell because only half the directional evidence is directly supervised. I am compensating by using 28 editable target-anchor rows repeated 24 times and preserving both target and neighbor.

### DECISION 2026-07-09 19:03 - Course-correction to target-anchor + bison-preserve v3
- What specifically in the result told me the cause: the top v2 residual pairs were all `bison` against other concepts, while `antelope`-`bison` remained the top target-row residual pair. That means the direct relation moved, but `bison` itself drifted because the dataset did not rehearse `bison` relations outside the edited pair.
- Next lever and why: alter the dataset, not the optimizer. v3 edits only `anchor=antelope` rows where `bison` is a candidate, adds 900 `bison`-preserve rows repeated twice, keeps 900 `antelope`-preserve rows repeated twice, and keeps 1,200 replay rows. This tests the bison-drift hypothesis directly.
- Rejected alternatives: more target repeats would likely strengthen the same neighbor drift; more generic replay would not target the observed failure; an LR/rank sweep would not address the missing `bison` examples; changing concepts would hide whether this diagnosis is right.
- What would confirm or kill this hypothesis: confirmation is `antelope`-`bison` remaining near the top of global residual pairs while `bison`-other residuals fall below it. If the target pair disappears, one-direction supervision is too weak. If `bison` still dominates, we need either stronger neighbor preservation or a different edit formulation that does not force many alternatives to beat `bison`.

### NOTE 2026-07-09 19:03 - v3 dataset built
- Data: `experiments/exp1_triplet_concept_move/sft_triplet_data/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve/{control.jsonl,edit.jsonl,manifest.json}`.
- Counts: 5,472 rows per arm: 28 editable target-anchor rows repeated 24 times, 900 target-preserve rows repeated twice, 900 neighbor-preserve rows repeated twice, and 1,200 replay rows.
- Training plan: online W&B, PEFT QLoRA, `rank=16`, `learning_rate=5e-5`, `max_steps=600`, batch size 8 if memory allows. The batch size increase is a throughput test; with 5,472 rows, 600 steps at batch 8 sees about 0.88 effective epochs, while staying comparable in wall time to v2.

### NOTE 2026-07-09 19:08 - v3 W&B runs launched
- Control run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/gcsnp8e0`.
- Edit run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/snclf788`.
- Settings: PEFT QLoRA, Llama-3.1-8B-Instruct snapshot `0e9e39f249a16976918f6564b8830bc894c89659`, `rank=16`, `learning_rate=5e-5`, `max_steps=600`, batch size 8, no gradient accumulation, seed 1729. Batch 8 fits on the local A5000s; early throughput is about `1.23` steps/s, or `9.8` examples/s per GPU.

## 2026-07-09 19:31 Target-anchor + bison-preserve v3 result
- Goal this session: test whether explicit `bison` preservation fixes v2's neighbor-drift failure while a one-direction target-anchor edit still moves the intended `antelope`-`bison` relation.
- What I ran / built: trained v3 control/edit adapters online in W&B, recovered full frozen-protocol triplet RDMs, and scored `detection/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve.json`. Then built v4 data under `sft_triplet_data/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve/`.
- Result (numbers; plots saved to /figs with filenames): v3 control finished 600 steps in 492.82s (`9.740` samples/s, final logged loss `0.0857`, train loss `0.1279`); edit finished in 499.07s (`9.618` samples/s, final logged loss `0.0898`, train loss `0.1361`). Detection: `target_rank=4`, `target_snr_control_only=1.403`, `upper_rms_control_null=0.103`, `upper_rms_edit=0.127`, `upper_rms_residual=0.066`. However, `antelope`-`bison` ranked 320 by global residual pair and 25 within the target row; top residual pairs were `antelope`-`beaver` and `antelope`-`boar`.
- Interpretation (what the result means, not just restating it): v3 improved concept-row localization and reduced global residual drift, but it did not preserve the intended relation-level edit. The one-direction edit reduced broad `bison` movement, but it also let the symmetrized `antelope`-`bison` residual cancel, shifting the observable movement to `antelope` vs alternatives.
- Lit found + how it changes the plan: no new literature in this operational step.
- Decision / next step + WHY this over the alternatives I considered: run v4, both-direction edit plus `bison` preservation. I chose this over increasing v3 target repeats because v3's failure is cancellation of the target pair, not only weak magnitude. I chose this over changing LR/rank because v3 already reduced global drift; the missing piece is to restore direct target-pair pressure while keeping neighbor preservation.
- Open risks: v4 may reintroduce bison-global drift despite preservation. If that happens, the next likely lever is data shaping around alternatives, e.g. restrict editable alternatives or add hard negative/positive balance for `bison` rows.

### DECISION 2026-07-09 19:31 - Course-correction to both-direction + bison-preserve v4
- What specifically in the result told me the cause: v3 reduced residual drift and improved `antelope` row rank, but `antelope`-`bison` residual pair rank collapsed to 320. That means preserving `bison` helped locality but the one-direction edit no longer moved the intended pair after symmetrization.
- Next lever and why: restore both editable directions from v2 while keeping the explicit `bison` preservation introduced in v3. This isolates the variable: if v4 works, v2's failure was missing neighbor preservation; if it fails, the edit formulation itself causes unavoidable bison/alternative spillover.
- Rejected alternatives: more v3 target repeats could still move `antelope` toward alternatives instead of away from `bison`; changing concept pair would avoid the diagnosis; activation steering is still premature while direct SFT data design is producing informative changes.
- What would confirm or kill this hypothesis: confirmation is `antelope`-`bison` returning near the top of global residual pairs while `bison`-other residuals stay below it. If `bison`-other residuals dominate again, preservation volume or alternative selection is insufficient.

### NOTE 2026-07-09 19:31 - v4 dataset built
- Data: `experiments/exp1_triplet_concept_move/sft_triplet_data/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve/{control.jsonl,edit.jsonl,manifest.json}`.
- Counts: 5,448 rows per arm: 54 editable target-neighbor rows repeated 12 times, 900 target-preserve rows repeated twice, 900 neighbor-preserve rows repeated twice, and 1,200 replay rows.
- Training plan: online W&B, PEFT QLoRA, `rank=16`, `learning_rate=5e-5`, `max_steps=600`, batch size 8, seed 1729.

### NOTE 2026-07-09 19:34 - v4 W&B runs launched
- Control run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/qdler1uc`.
- Edit run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/4dkfq6su`.
- Settings: PEFT QLoRA, Llama-3.1-8B-Instruct snapshot `0e9e39f249a16976918f6564b8830bc894c89659`, `rank=16`, `learning_rate=5e-5`, `max_steps=600`, batch size 8, no gradient accumulation, seed 1729.

## 2026-07-09 19:51 Both-direction + bison-preserve v4 result
- Goal this session: test whether combining v2's both-direction target-pair pressure with v3's explicit `bison` preservation restores the target relation while keeping neighbor drift down.
- What I ran / built: trained v4 control/edit adapters online in W&B, recovered full frozen-protocol triplet RDMs, scored `detection/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve.json`, and built v5 data under `sft_triplet_data/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/`.
- Result (numbers; plots saved to /figs with filenames): v4 control finished 600 steps in 492.88s (`9.739` samples/s, final logged loss `0.1542`, train loss `0.1347`); edit finished in 500.60s (`9.589` samples/s, final logged loss `0.1495`, train loss `0.1494`). Detection: `target_rank=12`, `target_snr_control_only=1.056`, `upper_rms_control_null=0.125`, `upper_rms_edit=0.133`, `upper_rms_residual=0.050`. `antelope`-`bison` was rank 1 by global residual pair and rank 1 within the target row, with residual delta `0.232`.
- Interpretation (what the result means, not just restating it): v4 fixed the relation-locality failure but left the signal underpowered. This is the cleanest pair-level result so far because the intended relation is the top residual pair and global residual drift is the smallest of the targeted-triplet runs. It still does not satisfy the original row-localization criterion because the row aggregate is diluted and the target-pair movement is small.
- Lit found + how it changes the plan: no new literature in this operational step.
- Decision / next step + WHY this over the alternatives I considered: run v5 with the same v4 data design but double target repeats from 12 to 24. I chose this over changing LR/rank because v4's failure is not broad drift; it is insufficient target-pair magnitude. I chose this over dropping preservation because v4 shows preservation is what made the pair signal clean.
- Open risks: stronger target repeats may reintroduce v2-style `bison` drift. If that happens, the next axis is not simply more target pressure; it is alternative balancing or a concept pair whose target-neighbor cell can be manipulated without requiring many forced alternatives.

### DECISION 2026-07-09 19:51 - Course-correction to v5 target-repeat 24
- What specifically in the result told me the cause: `antelope`-`bison` was the top global residual pair, but its residual delta was only `0.232`, and the target row ranked 12. This means the data design now points to the right relation, but the movement is too small for row-level attribution.
- Next lever and why: double editable repeats while keeping `antelope` and `bison` preservation fixed. This directly increases the clean target-pair signal discovered in v4 without removing the preservation that controlled drift.
- Rejected alternatives: increasing LR or rank would globally strengthen the adapter and may undo the drift gains; changing concept pair would abandon a working diagnostic; returning to one-direction editing would repeat v3's cancellation.
- What would confirm or kill this hypothesis: confirmation is `antelope`-`bison` still rank 1 by global residual pair with larger residual delta and better `antelope` row rank. If `bison`-other pairs dominate again, the target-pressure/locality tradeoff needs data balancing rather than more repeats.

### NOTE 2026-07-09 19:51 - v5 dataset built
- Data: `experiments/exp1_triplet_concept_move/sft_triplet_data/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/{control.jsonl,edit.jsonl,manifest.json}`.
- Counts: 6,096 rows per arm: 54 editable target-neighbor rows repeated 24 times, 900 target-preserve rows repeated twice, 900 neighbor-preserve rows repeated twice, and 1,200 replay rows.
- Training plan: online W&B, PEFT QLoRA, `rank=16`, `learning_rate=5e-5`, `max_steps=600`, batch size 8, seed 1729.

### NOTE 2026-07-09 19:55 - v5 W&B runs launched
- Control run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/yrbw0v8k`.
- Edit run: `https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/g9rxnwe5`.
- Settings: PEFT QLoRA, Llama-3.1-8B-Instruct snapshot `0e9e39f249a16976918f6564b8830bc894c89659`, `rank=16`, `learning_rate=5e-5`, `max_steps=600`, batch size 8, no gradient accumulation, seed 1729.
