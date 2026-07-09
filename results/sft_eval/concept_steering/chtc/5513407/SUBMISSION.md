# CHTC 5513407 Expanded Concept Steering Generation

Remote run:
`~/chtc-runs/coherence-concept-expanded-20260709-023711`

Submitted:
`2026-07-09 02:37 UTC` (`2026-07-08 21:37 CT`)

## Job

- submit file: `chtc/concept_steering/concept_steering_qualitative.sub`
- output tag: `expanded_alignment_retention`
- cluster: `5513407`
- shape: `1` GPU, `8` CPU, `64GB` memory, `80GB` disk
- CHTC flags: `+GPUJobLength = "short"`, `+is_resumable = true`
- staging model: `/staging/s/suresh27/models/llama31-8b-instruct`

## Local Validation

- `python3 -m py_compile src/sft/run_concept_steering_qualitative.py`
- `bash -n chtc/concept_steering/run_concept_qualitative.sh`
- local dry-run plan: `75` prompts, `5` settings, `375` generations
- transferred CHTC bundle validation: same compile, wrapper syntax, and dry-run plan

Local `condor_submit -dry-run` could not run because this machine's Condor
configuration failed before submit parsing while resolving `NETWORK_INTERFACE=*`.
The submit file was accepted by CHTC `condor_submit` on `ap2002.chtc.wisc.edu`.

## Scientific Purpose

This follow-up tests whether the current best concept steering candidate,
`coherence_l16_a4`, preserves its qualitative generation advantage on a harder
prompt suite while guarding against retention loss. It also checks whether the
safe but baseline-saturated human-alignment settings, `human_alignment_l24_a4`
and `human_alignment_l16_a2`, remain non-harmful on harder alignment prompts,
with `coherence_l12_a4` included only as a high-effect diagnostic.
