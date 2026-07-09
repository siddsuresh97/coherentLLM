# Active Agent Handoff

Last coordination snapshot: 2026-07-08 22:34 CDT.

Branch: `coherence-sft`

Persistent objective: determine what the coherence-trained model improves, what it hurts, why those effects appear, and which experiments can push the cognitive-science and LLM-research story forward.

## Live CHTC State

Source: `chtc-ssh 'condor_q -batch suresh27'` with the active `chtc-master` wrapper.

Current queue snapshot:

- Queue state from `condor_q -batch suresh27`: `1` running, `0` idle,
  `0` held.
- Retention failure-suite scale-up `5513434.0` is running on
  `slot2_2@gpu4006.chtc.wisc.edu` with `RequestCpus=8`,
  `RequestMemory=49152`, `RequestDisk=83886080`, and `RequestGPUs=1`.
  It passed the 46GB GPU probe, staged-input check, and started package setup.
  Remote run directory:
  `~/chtc-runs/coherence-retention-scaleup-20260709-033234`.
  Local submission note:
  `results/sft_eval/wide_bench/failure_suite/chtc_5513434/SUBMISSION.md`.
- Retention failure-suite TruthfulQA gate `5513424.0` completed with
  `ExitCode=0`, `RemoteWallClockTime=520.0`, and host
  `slot2_2@gpu4006.chtc.wisc.edu`. It used one NVIDIA L40S GPU and wrote
  artifacts under `results/sft_eval/wide_bench/failure_suite/chtc_5513424/`.
- Concept steering expanded qualitative suite `5513407.0` completed normally
  with return value `0` on `slot2_3@gpulab2004.chtc.wisc.edu`.
- Concept allocation was `1` GPU, `8` CPUs, `65536` MB RAM, `83886080` KB
  disk. Assigned GPU was `NVIDIA A100-SXM4-40GB`; Condor reported
  `TimeExecute=962s`, `TimeSlotBusy=1015s`, and peak GPU memory `14202` MB.
- Pulled concept artifacts are under
  `results/sft_eval/concept_steering/chtc/5513407/`.
- Exact spare capacity is available: 40 unclaimed X86_64 CHTC slots satisfy
  `1` GPU, `8` CPUs, at least `64GB` RAM, `>=40GB` GPU memory, and
  `HasChtcStaging==true`.
- Huth/Fedorenko pack-smoke cluster `5513418.0` failed with `ExitCode=1`
  because the first packer archived ds003020 git-annex symlinks instead of
  dereferenced contents. Pulled artifacts are in
  `results/sft_huth_lebel/chtc_huth_pack_smoke_5513418/`.
- Fixed Huth/Fedorenko pack-smoke retry `5513422.0` passed with `ExitCode=0`,
  `RemoteWallClockTime=91.0`, and host `slot1_59@e4049.chtc.wisc.edu`. It read
  the existing smoke root, packed `againstthewind`, verified all members and
  extracted sizes, and wrote no staging files. Compact proof files are under
  `results/sft_huth_lebel/chtc_huth_pack_smoke_retry_5513422/`; the large raw
  result tarball is intentionally left untracked.

Huth/Fedorenko staging blocker:

- `/staging/s/suresh27`: `24.3209/100` GB used.
- File quota: `1120/1000` files used.
- High-data ds003020 raw manifest would add 420 files and 76.86 GB, so do not
  raw-stage high-data until file count/disk are freed or packed story artifacts
  are implemented.
- Packed-story artifacts are now implemented locally:
  `src/sft/huth_lebel_pack_stories.py` and
  `chtc/huth_lebel_highdata_packs/`. The generated plan reduces 420 raw
  high-data files to 84 story packs. Preferred cleanup candidate is the
  rebuildable MMLU dataset cache at `/staging/s/suresh27/hf_datasets_cache`;
  exact candidates are in
  `results/sft_huth_lebel/staging_cleanup_candidates.csv`.

Held jobs: none.

Monitor decision: submitted one safe GPU follow-up, scale-up cluster `5513434`,
because the queue was empty and the existing failure-suite runner could scale
without adding staged files. The concept GPU suite is complete, rerunning
completed MMLU shards would be duplicative, and the checked-in Huth uncapped
encoding path is CPU-only and already complete. The next separate CHTC GPU
submission should come from a newly smoke-tested Huth/Fedorenko or semantic-hub
bundle.

Completed since the previous handoff:

- The expanded concept generation follow-up was committed in `8e2295d`; its
  submission note is tracked at
  `results/sft_eval/concept_steering/chtc/5513407/SUBMISSION.md`.
- Expanded concept qualitative result `5513407.0` passed and was pulled. All
  retention gates passed at `0.96`; `coherence_l12_a4` gave the largest
  coherence lift (`+0.12`) but hurt alignment (`-0.24`), while
  `coherence_l16_a4` was only weakly helpful (`+0.04`) and the current
  human-alignment steering vectors did not improve harder alignment prompts.
- Concept qualitative `5513347.0` exited with Condor `ExitCode=1` after all
  generations and judge CSVs were written. The failure was a late `SUMMARY.md`
  writer bug; corrected/rescored artifacts are committed under
  `results/sft_eval/concept_steering/chtc/5513347/rescored_v2/`.
- The concept checkpoint is pushed in `30e6601`.
- Huth uncapped encoding `5513373.0` completed with `ExitCode=0` and
  `RemoteWallClockTime=986.0`; the result checkpoint is pushed in `ff86cb4`.
- Huth/Fedorenko next-experiment plan was refreshed after quota inspection in
  `results/sft_huth_lebel/NEXT_EXPERIMENT_PLAN.md`. No fMRI CHTC job was
  submitted in this checkpoint because staging file quota is already exceeded.
- Huth/Fedorenko staging-unblock report was added at
  `results/sft_huth_lebel/STAGING_UNBLOCK_REPORT.md`; CHTC pack-smoke retry
  `5513422` passed and used no GPU.
- Retention failure-suite TruthfulQA gate `5513424` passed. Bounded MC2:
  base `0.5682`, lowLR `0.5661`, taskvec `0.6037`; paired false-pressure-up
  fraction is lowLR `0.275` versus taskvec `0.825`.
- Retention failure-suite scale-up `5513434` was submitted and started. It runs
  `base`, `taskvec_a0p25`, and `lowLR` at limit 200 across `truthfulqa_mc2`,
  `wic`, and `openbookqa`.

## Active Agent Goals

No active subagents at this snapshot.

Completed agents already closed:

| Agent | ID | Commit |
| --- | --- | --- |
| Copernicus | `019f4490-0d27-7d93-97c5-9c09481d63f5` | `cfc3e27` |
| Pauli | `019f4490-59f7-7481-8170-033c898991ff` | `5a8bb78` |
| Sagan | `019f4437-630c-7f02-ba72-0da158f1fa52` | `3ffde47` |
| Hubble | `019f445f-9283-7bc2-8697-65d3b0162be1` | `361b9a1` |
| Curie | `019f44a4-3d12-7e52-8879-d9d0a27617ba` | `aecf2cc` |
| Feynman | `019f44a4-3d55-7231-9c2c-84ef85427a1c` | `2afd0f2` |
| Russell / main-thread concept pickup | `019f445f-a869-70c2-bccc-dfa9c1d9314c` | `30e6601` |
| Sartre / main-thread Huth pickup | `019f44b4-d7ea-7131-9ea2-26181b412bd3` | `ff86cb4` |
| Gauss | `019f44bf-dcc2-7ed1-8bc2-f0f3cda43bbe` | `2d823bd` |
| Faraday | `019f44bf-df00-72a1-94e5-f2c3d50cecdb` | `842474b` |
| Boole | `019f44b4-b2fd-7960-bd39-440dbb17743f` | `8e2295d` |
| Lorentz | `019f44c9-168f-72c2-9418-46846bae3072` | `ee50f23` |
| Franklin | `019f44c8-e4f2-7b32-ba56-7f22db31c131` | `342847c` |
| Cicero | `019f44d6-6b30-71d2-845a-b9fe2712bd29` | `5aa2823`, `01ff9b6` |

## Current Scientific Checkpoints

- `taskvec_a0p25` MMLU CHTC merge is complete: 57/57 subjects, weighted micro `0.622964` over `n=13508`, macro about `0.617931`. This is a partial mitigation versus lowrank, not a broad retention fix.
- Concept steering bounded sweep `5513297` found the strongest coherence margin
  at layer 12 alpha 4 and best human-alignment margin at layer 24 alpha 4, with
  specificity still imperfect. Qualitative follow-up `5513347` changed the
  default recommendation: `coherence` layer 16 alpha 4 is the best current
  generation candidate; human-alignment layer 24 alpha 4 and layer 16 alpha 2
  are retention-safe but baseline-saturated; layer-12 human-alignment steering
  is a failure mode.
- Huth feature extraction bundle `5513306` produced all planned arm/story
  feature files. Uncapped encoding `5513373` returned all 36 full-voxel rows
  across `UTS01`-`UTS03`, four arms, and three layers. Layer-16 mean held-out
  Pearson `r` across subjects is base `0.009785`, `taskvec_a0p25` `0.009137`,
  `lowLR` `0.008287`, and scrambled `0.004988`; this validates the full-voxel
  smoke path but does not show task-vector improvement over base.
- The next fMRI result should be high-data story scaling, not a rerun of the
  smoke. Use `results/sft_huth_lebel/NEXT_EXPERIMENT_PLAN.md` as the current
  handoff: packed story staging first, then one-GPU per-story extraction,
  full-voxel multi-fold encoding, and Fedorenko claims restricted to
  exploratory atlas/localizer status unless subject-specific language fROIs are
  added.
- Current high-data staging unblock details are in
  `results/sft_huth_lebel/STAGING_UNBLOCK_REPORT.md`: 420 high-data source
  files become 84 story archives; cleanup should target rebuildable cache, not
  staged models/adapters.
- The semantic-hub/MEMP paper-method lane is committed in `aecf2cc`; use
  `taskvec_a0p25` as the primary arm and gate claims on paper-style controls.
- The retention failure-suite gate is committed in `2afd0f2`; the first smoke is
  `python src/sft/build_retention_failure_suite_manifest.py --check`.
- The first GPU-backed failure-suite gate is complete in `5513424`: lowLR is
  aggregate-flat but suppresses false pressure, while `taskvec_a0p25` improves
  MC2 but increases false pressure on most paired items. Use this before
  approving any mitigation that looks good on aggregate TruthfulQA alone.
- Scale-up cluster `5513434` is the active retention job. It should answer
  whether the TruthfulQA false-pressure risk persists at limit 200 and whether
  the same arms preserve or damage word-sense and elementary-science slices.

## Coordination Rules

- Agents should commit and push coherent checkpoints for their own lane.
- Do not revert or restage files from another lane.
- Prefer small, machine-readable artifacts and reports; leave raw large tarballs or run directories uncommitted unless they are intentionally selected and size-checked.
- Keep queue/provenance details in tracked reports so future agents can resume without chat history.
