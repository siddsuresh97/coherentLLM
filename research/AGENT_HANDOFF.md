# Active Agent Handoff

Last coordination snapshot: 2026-07-08 21:45 CDT.

Branch: `coherence-sft`

Persistent objective: determine what the coherence-trained model improves, what it hurts, why those effects appear, and which experiments can push the cognitive-science and LLM-research story forward.

## Live CHTC State

Source: `chtc-ssh 'condor_q -batch suresh27'` with the active `chtc-master` wrapper.

Current queue snapshot:

- Concept steering expanded qualitative suite `5513407.0` is running on
  `slot2_3@gpulab2004.chtc.wisc.edu`.
- Allocation: `1` GPU, `8` CPUs, `65536` MB RAM, `83886080` KB disk.
- Queue state from `condor_q -batch suresh27`: `1` running, `0` idle,
  `0` held.
- Assigned GPU for `5513407.0`: `NVIDIA A100-SXM4-40GB` on
  `gpulab2004.chtc.wisc.edu`; remote stdout shows `gpu_probe_ok`,
  `pip_install_exit rc=0`, `import_probe_ok`, and the 375-generation plan
  before model generation.
- Exact spare capacity is available: 40 unclaimed X86_64 CHTC slots satisfy
  `1` GPU, `8` CPUs, at least `64GB` RAM, `>=40GB` GPU memory, and
  `HasChtcStaging==true`.

Held jobs: none.

Monitor decision: no additional GPU job was submitted from this lane. Boole's
concept suite is already running; rerunning completed MMLU shards would be
duplicative; the checked-in Huth uncapped encoding path is CPU-only and already
complete. The next safe CHTC GPU submission should come from a newly
smoke-tested Huth/Fedorenko or semantic-hub bundle, or from a completed
`5513407` pull/rescore follow-up.

Completed since the previous handoff:

- The expanded concept generation follow-up was committed in `8e2295d`; its
  submission note is tracked at
  `results/sft_eval/concept_steering/chtc/5513407/SUBMISSION.md`.
- Concept qualitative `5513347.0` exited with Condor `ExitCode=1` after all
  generations and judge CSVs were written. The failure was a late `SUMMARY.md`
  writer bug; corrected/rescored artifacts are committed under
  `results/sft_eval/concept_steering/chtc/5513347/rescored_v2/`.
- The concept checkpoint is pushed in `30e6601`.
- Huth uncapped encoding `5513373.0` completed with `ExitCode=0` and
  `RemoteWallClockTime=986.0`; the result checkpoint is pushed in `ff86cb4`.

## Active Agent Goals

| Agent | ID | Lane | Current goal |
| --- | --- | --- | --- |
| Boole | `019f44b4-b2fd-7960-bd39-440dbb17743f` | Concept steering scale-up | Build a validated expanded judged generation suite and submit one short CHTC GPU job only after local dry-run checks pass. |
| Gauss | `019f44bf-dcc2-7ed1-8bc2-f0f3cda43bbe` | CHTC utilization and queue monitor | Keep CHTC status handoffable, identify idle or held jobs, and prepare safe next submissions without overlapping Boole's write set. |
| Faraday | `019f44bf-df00-72a1-94e5-f2c3d50cecdb` | fMRI / Huth / Fedorenko follow-through | Synthesize the Huth smoke, design the next Huth/Fedorenko and semantic-hub experiment, and submit only locally smoke-tested jobs. |

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
- The semantic-hub/MEMP paper-method lane is committed in `aecf2cc`; use
  `taskvec_a0p25` as the primary arm and gate claims on paper-style controls.
- The retention failure-suite gate is committed in `2afd0f2`; the first smoke is
  `python src/sft/build_retention_failure_suite_manifest.py --check`.

## Coordination Rules

- Agents should commit and push coherent checkpoints for their own lane.
- Do not revert or restage files from another lane.
- Prefer small, machine-readable artifacts and reports; leave raw large tarballs or run directories uncommitted unless they are intentionally selected and size-checked.
- Keep queue/provenance details in tracked reports so future agents can resume without chat history.
