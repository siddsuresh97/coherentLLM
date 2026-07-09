# Active Agent Handoff

Last coordination snapshot: 2026-07-08 21:28 CT.

Branch: `coherence-sft`

Persistent objective: determine what the coherence-trained model improves, what it hurts, why those effects appear, and which experiments can push the cognitive-science and LLM-research story forward.

## Live CHTC State

Source: `chtc-ssh 'condor_q -batch suresh27'` with the active `chtc-master` wrapper.

| Cluster | Batch | State | Owner |
| --- | --- | --- | --- |
| `5513373.0` | `coherence_huth_encoding_smoke_bundle_uncapped_all_5513373` | running | Sartre / Huth fMRI |

Held jobs: none.

Completed since the previous handoff:

- Concept qualitative `5513347.0` exited with Condor `ExitCode=1` after all
  generations and judge CSVs were written. The failure was a late `SUMMARY.md`
  writer bug; corrected/rescored artifacts are committed under
  `results/sft_eval/concept_steering/chtc/5513347/rescored_v2/`.
- The concept checkpoint is pushed in `30e6601`.

## Active Agent Goals

| Agent | ID | Lane | Current goal |
| --- | --- | --- | --- |
| Boole | `019f44b4-b2fd-7960-bd39-440dbb17743f` | Concept steering scale-up | Build a validated expanded judged generation suite and submit one short CHTC GPU job only after local dry-run checks pass. |
| Sartre | `019f44b4-d7ea-7131-9ea2-26181b412bd3` | Huth / language-fMRI | Monitor running uncapped CPU encoding job `5513373`, pull results when complete, and commit a Huth-only scientific checkpoint. |

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

## Current Scientific Checkpoints

- `taskvec_a0p25` MMLU CHTC merge is complete: 57/57 subjects, weighted micro `0.622964` over `n=13508`, macro about `0.617931`. This is a partial mitigation versus lowrank, not a broad retention fix.
- Concept steering bounded sweep `5513297` found the strongest coherence margin
  at layer 12 alpha 4 and best human-alignment margin at layer 24 alpha 4, with
  specificity still imperfect. Qualitative follow-up `5513347` changed the
  default recommendation: `coherence` layer 16 alpha 4 is the best current
  generation candidate; human-alignment layer 24 alpha 4 and layer 16 alpha 2
  are retention-safe but baseline-saturated; layer-12 human-alignment steering
  is a failure mode.
- Huth feature extraction bundle `5513306` produced all planned arm/story feature files; capped encoding smokes validated the path but are not yet a scientific subject-level comparison.
- The semantic-hub/MEMP paper-method lane is committed in `aecf2cc`; use
  `taskvec_a0p25` as the primary arm and gate claims on paper-style controls.
- The retention failure-suite gate is committed in `2afd0f2`; the first smoke is
  `python src/sft/build_retention_failure_suite_manifest.py --check`.

## Coordination Rules

- Agents should commit and push coherent checkpoints for their own lane.
- Do not revert or restage files from another lane.
- Prefer small, machine-readable artifacts and reports; leave raw large tarballs or run directories uncommitted unless they are intentionally selected and size-checked.
- Keep queue/provenance details in tracked reports so future agents can resume without chat history.
