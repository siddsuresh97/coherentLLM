# Active Agent Handoff

Last coordination snapshot: 2026-07-08 21:11 CT.

Branch: `coherence-sft`

Persistent objective: determine what the coherence-trained model improves, what it hurts, why those effects appear, and which experiments can push the cognitive-science and LLM-research story forward.

## Live CHTC State

Source: `chtc-ssh 'condor_q -batch suresh27'` with the active `chtc-master` wrapper.

| Cluster | Batch | State | Owner |
| --- | --- | --- | --- |
| `5513347.0` | `coherence_concept_steering_qualitative_5513347` | running | Russell / concept steering |
| `5513373.0` | `coherence_huth_encoding_smoke_bundle_uncapped_all_5513373` | idle | Hubble / Huth fMRI |

Held jobs: none.

Huth `5513373.0` better-analyze: 4 current willing slots and 142 slots if drained for a 4 CPU, 16 GB memory, 20 GB disk, staging+container CPU job. Leave queued unless a later hold or resource failure appears.

## Active Agent Goals

| Agent | ID | Lane | Current goal |
| --- | --- | --- | --- |
| Sagan | `019f4437-630c-7f02-ba72-0da158f1fa52` | MMLU / benchmark attribution | Finish and push the `taskvec_a0p25` MMLU aggregate/report checkpoint; remove stale "pending" prose; rerun local analysis generators; commit small merged artifacts, not raw large tarballs. |
| Hubble | `019f445f-9283-7bc2-8697-65d3b0162be1` | Alex Huth / language-fMRI | Monitor or complete uncapped UTS01-UTS03 CPU encoding smoke from feature bundle `5513306`; pull artifacts; update Huth reports; commit Huth-only changes. |
| Russell | `019f445f-a869-70c2-bccc-dfa9c1d9314c` | Concept-vector steering | Own qualitative job `5513347`, commit script/submit/results, and add a steering/ablation next-experiment plan with retention gates. |
| Kant | `019f4490-5c10-7b21-a465-33e6962d39d9` | Ops | Keep CHTC/local GPU utilization and queue status trackable; debug holds; record resource recommendations. |
| Curie | `019f44a4-3d12-7e52-8879-d9d0a27617ba` | Semantic hub / MEMP | Map the 2411.04986 semantic-hub methods onto this model's arms and produce a paper-aligned experiment plan plus CHTC/local smoke split. |
| Feynman | `019f44a4-3d55-7231-9c2c-84ef85427a1c` | Retention failure suite | Design a cheap failure-suite and mitigation plan for MMLU/ARC/WiC/TruthfulQA drops, with margin and false-lure metrics before full reruns. |

Completed agents already closed:

| Agent | ID | Commit |
| --- | --- | --- |
| Copernicus | `019f4490-0d27-7d93-97c5-9c09481d63f5` | `cfc3e27` |
| Pauli | `019f4490-59f7-7481-8170-033c898991ff` | `5a8bb78` |

## Current Scientific Checkpoints

- `taskvec_a0p25` MMLU CHTC merge is complete: 57/57 subjects, weighted micro `0.622964` over `n=13508`, macro about `0.617931`. This is a partial mitigation versus lowrank, not a broad retention fix.
- Concept steering bounded sweep `5513297` found the best coherence steering at layer 12 alpha 4 and best human-alignment steering at layer 24 alpha 4, with specificity still imperfect.
- Huth feature extraction bundle `5513306` produced all planned arm/story feature files; capped encoding smokes validated the path but are not yet a scientific subject-level comparison.

## Coordination Rules

- Agents should commit and push coherent checkpoints for their own lane.
- Do not revert or restage files from another lane.
- Prefer small, machine-readable artifacts and reports; leave raw large tarballs or run directories uncommitted unless they are intentionally selected and size-checked.
- Keep queue/provenance details in tracked reports so future agents can resume without chat history.
