# Benchmark Attribution Artifacts

Report date: 2026-07-08 America/Chicago, UTC date 2026-07-09.

This directory contains derived attribution tables for the coherence-SFT
wide-benchmark drops, stable cells, and gains. The tables are generated from
existing artifacts only; no benchmarks are launched here.

Regenerate from the repo root with:

```bash
python results/sft_eval/wide_bench/attribution/build_attribution.py
```

## Files

- `build_attribution.py`: reproducible table builder.
- `headline_task_attribution.csv`: task-level score deltas and mechanism labels.
- `skill_family_attribution.csv`: skill-family deltas, verdicts, and mitigations.
- `semantic_target_gains.csv`: direct semantic-coherence and human-alignment
  target gains.
- `truthfulqa_attribution.csv`: TruthfulQA MC2 and log-sample pressure readout.
- `mitigation_targets.csv`: prioritized mitigation and cheap-gate queue.
- `metadata.json`: input/output provenance and source commit.

Canonical narrative report:

- `../BENCHMARK_ATTRIBUTION.md`
