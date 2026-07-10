# Experiment 1 Report - Detecting a Concept Move from Triplets

Current status: baseline and noise floor are green, online W&B training works, and direct triplet SFT can move the intended `antelope`-`bison` relation. The best clean run remains v5: `antelope`-`bison` is the top residual pair, but `antelope` is still row-rank 3 rather than row-rank 1. v7 tested reciprocal `bison` preservation and failed: it reduced residual drift but also reduced the target signal below detection.

## Hypothesis

If we deliberately move one concept in training, then held-out odd-one-out triplet judgments should show that the edited concept's row in the behavioral RDM changed more than ordinary LoRA fine-tuning drift.

For the first concrete test, the target concept is `antelope`, and the intended concentrated move is to push it away from the close feature-space neighbor `bison`.

Success requires both:

- `antelope` is top-ranked by edit-vs-base row change relative to the control null.
- SNR is greater than 1, meaning edit movement exceeds matched control drift.

## What I Tried

| Attempt | Training lever | Control/edit matched? | Verdict | Why |
|---|---|---:|---|---|
| 1 | Direct pairwise similarity fallback | No | Partial / not success | `antelope` ranked 1, but SNR was exactly 1.000 because control drift matched edit drift. |
| 2 | NOVA feature-listing SFT | Yes | Did not localize | SNR was 1.055, but `antelope` ranked 23; many non-target rows moved more. |
| 3 | Matched pairwise similarity fallback | Yes | Prepared, strong run stopped | The strong `rank=32`, `lr=2e-4` setting looked too broad/unstable, so I stopped before treating it as evidence. |
| 4 | Lower-drift NOVA feature-listing | Yes | Did not fire | Online W&B run completed with `rank=16`, `lr=5e-5`, 400 steps. `antelope` ranked 30 and SNR was 0.492, so the conservative adapter made the edit too weak relative to control drift. |
| 5 | Targeted triplet SFT v1 | Yes | Partial | `antelope`-`bison` moved strongly: rank 1 by global edit-pair delta and rank 3 by global residual-pair delta. But row localization failed (`antelope` row rank 14 original, 24 residual) because alternatives such as `boar` and `beaver` also moved. |
| 6 | Replay-heavy targeted triplet SFT v2 | Yes | Partial | Overall residual drift dropped (`0.135 -> 0.109`) and `antelope`-`bison` stayed the top target-row residual pair, but row localization still failed (`antelope` rank 12). Top residual pairs were mostly `bison` against other concepts, revealing missing neighbor preservation. |
| 7 | Target-anchor + `bison`-preserve v3 | Yes | Partial / wrong pair | Residual drift dropped (`0.109 -> 0.066`) and `antelope` row rank improved to 4 with SNR `1.403`, but the intended `antelope`-`bison` residual pair fell to rank 320. The one-direction edit plus neighbor preservation canceled the target-pair signal. |
| 8 | Both-direction + `bison`-preserve v4 | Yes | Partial / clean but weak | `antelope`-`bison` became the top global residual pair and residual drift dropped to `0.050`, but the edit was too weak at row level (`antelope` rank 12, SNR `1.056`). |
| 9 | v5: v4 with stronger target repeats | Yes | Partial / best so far | `antelope` row rank improved to 3 with SNR `1.301`; `antelope`-`bison` is the top global residual pair. Not success yet because `antelope` is not top row. |
| 10 | v6: system prompt + mixed SFT templates | Yes | Did not improve locality | SNR rose to `1.485`, but `antelope` row rank worsened to 8 and `bison`-other residual pairs dominated. Prompt mixing made the target stronger but less local. |
| 11 | v7: target-anchor edit + reciprocal `bison` preservation | Yes | Did not fire | `antelope` row rank fell to 23 and SNR fell to `0.919`. The reciprocal preservation over-constrained the target relation and made edit movement smaller than the control null. |

Main interpretation: the detector and scorer work, and the model's triplet behavior can move. v5 is a near-positive relation-level result. v6 shows that making the target pair stronger is not enough; it worsens `bison`-other spillover. v7 shows that fully preserving the reciprocal `bison` side is too strong; it suppresses the target signal into the control null. The next useful axis is a controlled data-balance sweep between v5 and v7, not another prompt-mixing run.

## Completed NOVA Runs

Both runs use `concentrated_drop_100`, NOVA feature-listing supervision, `llama-3.1-8b-instruct`, and matched control/edit datasets. The target concept is `antelope`; the intended neighbor relation is `antelope` away from `bison`.

| Run | Setting | W&B | Detection verdict | Evidence |
|---|---|---|---|---|
| Strong feature-listing | `rank=32`, `lr=2e-4`, `max_steps=400` | Not live-online from this session | `target_rank=23`, SNR `1.055`, not localized | [detection/concentrated_drop_100.json](detection/concentrated_drop_100.json), [lora_control/concentrated_drop_100/training_metrics.json](lora_control/concentrated_drop_100/training_metrics.json), [lora_edit/concentrated_drop_100/training_metrics.json](lora_edit/concentrated_drop_100/training_metrics.json), [figs/resolution_heatmap.png](figs/resolution_heatmap.png) |
| Lower-drift feature-listing | `rank=16`, `lr=5e-5`, `max_steps=400` | Online: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/qvdyld87), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/ta6is8pl) | `target_rank=30`, SNR `0.492`, edit too weak | [detection/concentrated_drop_100_feature_lowdrift.json](detection/concentrated_drop_100_feature_lowdrift.json), [lora_control_feature_lowdrift/concentrated_drop_100/training_metrics.json](lora_control_feature_lowdrift/concentrated_drop_100/training_metrics.json), [lora_edit_feature_lowdrift/concentrated_drop_100/training_metrics.json](lora_edit_feature_lowdrift/concentrated_drop_100/training_metrics.json), [raw/control_feature_lowdrift_concentrated_drop_100/triplet.csv](raw/control_feature_lowdrift_concentrated_drop_100/triplet.csv), [raw/edit_feature_lowdrift_concentrated_drop_100/triplet.csv](raw/edit_feature_lowdrift_concentrated_drop_100/triplet.csv), [rdms/control_feature_lowdrift_concentrated_drop_100/rdm.npy](rdms/control_feature_lowdrift_concentrated_drop_100/rdm.npy), [rdms/edit_feature_lowdrift_concentrated_drop_100/rdm.npy](rdms/edit_feature_lowdrift_concentrated_drop_100/rdm.npy) |
| Legacy matched similarity | `rank=32`, `lr=2e-4`, `max_steps=400`, grad accumulation 8 | Not online | `target_rank=1`, SNR `1.000`, residual `antelope` signal `0.000` | [detection/concentrated_drop_100_similarity_legacy.json](detection/concentrated_drop_100_similarity_legacy.json), [lora_control_similarity/concentrated_drop_100/training_metrics.json](lora_control_similarity/concentrated_drop_100/training_metrics.json), [lora_edit_similarity/concentrated_drop_100/training_metrics.json](lora_edit_similarity/concentrated_drop_100/training_metrics.json), [sft_similarity_data/concentrated_drop_100/manifest.json](sft_similarity_data/concentrated_drop_100/manifest.json) |

## Why I Think It Failed

The feature-listing run did not fail because control and edit used different examples. They were matched: both arms had 296 rows, with 29 non-target concepts repeated 8 times and `antelope` repeated 64 times.

The strong run likely failed because the adapter was too strong and changed the model's general triplet behavior. Rank 32 over all linear modules at `2e-4` for 400 steps drove the tiny training set to near-zero loss. That can make many unrelated rows move, so `antelope` is no longer the largest behavioral change even when the intended concept was edited.

The lower-drift run tested the opposite explanation: maybe conservative settings would keep the null tight enough for the target to stand out. That did not happen. Global edit movement fell (`upper_rms_edit=0.195` vs `0.489` in the strong run), but `antelope` fell to the bottom of the ranking (`target_rank=30`). This says the feature-listing lever is probably too indirect under retention-preserving settings.

Evidence: [results/sft_eval/mitigation/summary.csv](../../results/sft_eval/mitigation/summary.csv), [src/sft/summarize_mitigation.py](../../src/sft/summarize_mitigation.py), [research/CODEX_TASK_6.md](../../research/CODEX_TASK_6.md).

## Current / Next Run

Attempt 5 pulled a direct targeted-triplet lever. It kept the detection task held out by using a different training prompt, and it narrowed the intervention to rows that directly affect the symmetrized `antelope`-`bison` RDM cell.

Targeted triplet SFT data:

- Data: [sft_triplet_data/concentrated_drop_100_triplet_targeted_v1/control.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v1/control.jsonl), [sft_triplet_data/concentrated_drop_100_triplet_targeted_v1/edit.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v1/edit.jsonl), [manifest](sft_triplet_data/concentrated_drop_100_triplet_targeted_v1/manifest.json).
- Counts: 2,136 examples per arm; 54 editable `antelope`/`bison` triplets repeated 24 times, 240 target-preserve rows repeated twice, and 360 replay rows.
- Training plan: online W&B, `rank=16`, `lr=1e-4`, `max_steps=300`.
- W&B links: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/utp95nw6), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/cj920t5i).
- Detection: [detection/concentrated_drop_100_triplet_targeted_v1.json](detection/concentrated_drop_100_triplet_targeted_v1.json). Original row score crossed SNR > 1 (`1.231`) but `antelope` ranked 14. Residual row score crossed floor (`1.185`) but `antelope` ranked 24. Pair-local score was much better: `antelope`-`bison` ranked 1 by global edit-pair delta and 3 by global residual-pair delta.

Next variant: replay-heavy targeted triplet SFT. It keeps the same direct edit but lowers LR to `5e-5`, halves target repeats, and increases target-preserve/replay pressure. The goal is to keep the successful pair movement while reducing alternative-concept spillover.

Replay-heavy v2 setup:

- Data: [sft_triplet_data/concentrated_drop_100_triplet_targeted_v2_replayheavy/control.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v2_replayheavy/control.jsonl), [sft_triplet_data/concentrated_drop_100_triplet_targeted_v2_replayheavy/edit.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v2_replayheavy/edit.jsonl), [manifest](sft_triplet_data/concentrated_drop_100_triplet_targeted_v2_replayheavy/manifest.json).
- Counts: 3,648 examples per arm; 54 editable rows repeated 12 times, 900 target-preserve rows repeated twice, and 1,200 replay rows.
- Training plan: online W&B, `rank=16`, `lr=5e-5`, `max_steps=600`.
- W&B links: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/1fqn9vw3), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/fna9k90i).
- Detection: [detection/concentrated_drop_100_triplet_targeted_v2_replayheavy.json](detection/concentrated_drop_100_triplet_targeted_v2_replayheavy.json). Row score crossed SNR > 1 (`1.219`) but `antelope` ranked 12, so this is not a clean detection. Pair-local signal persisted: `antelope`-`bison` ranked 1 by global edit-pair delta and 4 by global residual-pair delta. Top residual pairs were `bison`-`beaver`, `bison`-`beetle`, and `bison`-`boar`, so the failure mode shifted from random alternatives to global `bison` drift.

Next variant: target-anchor + `bison`-preserve v3. It keeps the same model and LR, but changes the dataset because v2 identified a data-design problem.

v3 setup:

- Data: [sft_triplet_data/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve/control.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve/control.jsonl), [sft_triplet_data/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve/edit.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve/edit.jsonl), [manifest](sft_triplet_data/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve/manifest.json).
- Counts: 5,472 examples per arm; 28 editable `anchor=antelope` rows repeated 24 times, 900 `antelope`-preserve rows repeated twice, 900 `bison`-preserve rows repeated twice, and 1,200 replay rows.
- Training plan: online W&B, `rank=16`, `lr=5e-5`, `max_steps=600`, batch size 8 if memory allows.
- W&B links: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/gcsnp8e0), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/snclf788).
- Detection: [detection/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve.json](detection/concentrated_drop_100_triplet_targeted_v3_targetanchor_bisonpreserve.json). Row rank improved to 4 and SNR rose to `1.403`, but this was not the desired relation-level edit: `antelope`-`bison` ranked 320 by global residual pair. Top residuals were `antelope` with alternatives such as `beaver` and `boar`.

Next variant: both-direction + `bison`-preserve v4. It keeps v2's successful direct target-pair pressure but adds the missing neighbor-preserve rows found by v3.

v4 setup:

- Data: [sft_triplet_data/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve/control.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve/control.jsonl), [sft_triplet_data/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve/edit.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve/edit.jsonl), [manifest](sft_triplet_data/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve/manifest.json).
- Counts: 5,448 examples per arm; 54 editable target-neighbor rows repeated 12 times, 900 `antelope`-preserve rows repeated twice, 900 `bison`-preserve rows repeated twice, and 1,200 replay rows.
- Training plan: online W&B, `rank=16`, `lr=5e-5`, `max_steps=600`, batch size 8.
- W&B links: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/qdler1uc), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/4dkfq6su).
- Detection: [detection/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve.json](detection/concentrated_drop_100_triplet_targeted_v4_both_bisonpreserve.json). The intended relation is now cleanly top-ranked by global residual pair (`antelope`-`bison` rank 1), and global residual drift is low (`0.050`). But row detection is weak: `antelope` rank 12, SNR `1.056`.

v5 result: same as v4 but with stronger target-pair pressure.

v5 setup:

- Data: [sft_triplet_data/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/control.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/control.jsonl), [sft_triplet_data/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/edit.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/edit.jsonl), [manifest](sft_triplet_data/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/manifest.json).
- Counts: 6,096 examples per arm; 54 editable target-neighbor rows repeated 24 times, 900 `antelope`-preserve rows repeated twice, 900 `bison`-preserve rows repeated twice, and 1,200 replay rows.
- Training plan: online W&B, `rank=16`, `lr=5e-5`, `max_steps=600`, batch size 8.
- W&B links: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/yrbw0v8k), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/g9rxnwe5).
- Detection: [detection/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24.json](detection/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24.json). `antelope` row rank improved to 3 and SNR `1.301`. `antelope`-`bison` was the top global residual pair and the top target-row residual pair. This is the best result so far, but it is not complete success because `antelope` is not row-rank 1.
- Training metrics: [control](lora_control_triplet/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/training_metrics.json), [edit](lora_edit_triplet/concentrated_drop_100_triplet_targeted_v5_both_bisonpreserve_target24/training_metrics.json). Raw held-out triplets: [control](raw/control_triplet_targeted_v5_both_bisonpreserve_target24_concentrated_drop_100/triplet.csv), [edit](raw/edit_triplet_targeted_v5_both_bisonpreserve_target24_concentrated_drop_100/triplet.csv). RDMs: [control](rdms/control_triplet_targeted_v5_both_bisonpreserve_target24_concentrated_drop_100/rdm.npy), [edit](rdms/edit_triplet_targeted_v5_both_bisonpreserve_target24_concentrated_drop_100/rdm.npy).

v6 result: same target relation but better prompt transfer and retention.

v6 setup:

- Data: [sft_triplet_data/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay/control.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay/control.jsonl), [sft_triplet_data/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay/edit.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay/edit.jsonl), [manifest](sft_triplet_data/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay/manifest.json).
- Counts: 7,296 examples per arm; 54 editable target-neighbor rows repeated 12 times across 2 templates, 900 `antelope`-preserve rows once across 2 templates, 900 `bison`-preserve rows once across 2 templates, and 1,200 replay rows once across 2 templates.
- Why this next: v5 already made `antelope`-`bison` the top residual pair, so simply increasing target repeats is not the cleanest next test. v6 asks whether adding the same generic system prompt used during probing and adding one more non-held-out training wording improves transfer to the frozen held-out prompt while keeping target pressure comparable.
- Training: online W&B, `rank=16`, `lr=5e-5`, `max_steps=450`, batch size 16.
- W&B links: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/occc9o58), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/y1rjw9bf).
- Detection: [detection/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay.json](detection/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay.json). `antelope` row rank worsened to 8, SNR rose to `1.485`, and global residual drift rose to `0.099`. `antelope`-`bison` residual delta increased to `0.321`, but the largest residual pairs were `bison` with `beaver`, `camel`, `boar`, and `bear`, so this is not the desired local edit.
- Training metrics: [control](lora_control_triplet/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay/training_metrics.json), [edit](lora_edit_triplet/concentrated_drop_100_triplet_targeted_v6_system_mixed_replay/training_metrics.json). Raw held-out triplets: [control](raw/control_triplet_targeted_v6_system_mixed_replay_concentrated_drop_100/triplet.csv), [edit](raw/edit_triplet_targeted_v6_system_mixed_replay_concentrated_drop_100/triplet.csv). RDMs: [control](rdms/control_triplet_targeted_v6_system_mixed_replay_concentrated_drop_100/rdm.npy), [edit](rdms/edit_triplet_targeted_v6_system_mixed_replay_concentrated_drop_100/rdm.npy).

v7 result: reciprocal `bison` preservation.

- Data: [sft_triplet_data/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve/control.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve/control.jsonl), [sft_triplet_data/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve/edit.jsonl](sft_triplet_data/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve/edit.jsonl), [manifest](sft_triplet_data/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve/manifest.json).
- Counts: 6,096 examples per arm; 28 editable `anchor=antelope` rows with `bison` as a candidate repeated 24 times, 26 reciprocal `anchor=bison` rows with `antelope` as a candidate repeated 24 times in both arms with base choices, 900 `antelope`-preserve rows x2, 900 `bison`-preserve rows x2, and 1,200 replay rows.
- Why this next: v5 was clean but weak and v6 was stronger but less local. v7 changes data shape, not prompt wording: move the target side while explicitly preserving the reciprocal neighbor side.
- Training plan: online W&B, `rank=16`, `lr=5e-5`, `max_steps=600`, batch size 8, same as v5 for comparability.
- W&B links: [control](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/2lpe63mo), [edit](https://wandb.ai/sid-academic-team/coherentLLM-exp1/runs/gt9wlurf).
- Training metrics: [control](lora_control_triplet/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve/training_metrics.json), [edit](lora_edit_triplet/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve/training_metrics.json). Raw held-out triplets: [control](raw/control_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve_concentrated_drop_100/triplet.csv), [edit](raw/edit_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve_concentrated_drop_100/triplet.csv). RDMs: [control](rdms/control_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve_concentrated_drop_100/rdm.npy), [edit](rdms/edit_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve_concentrated_drop_100/rdm.npy). Detection: [detection/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve.json](detection/concentrated_drop_100_triplet_targeted_v7_targetanchor_reciprocal_bisonpreserve.json).
- Detection verdict: failed. `antelope` row rank was 23, SNR was `0.919`, and `antelope` was not top-ranked. Global edit RMS was `0.125`, control-null RMS was `0.129`, and residual RMS was only `0.037`, meaning v7 mostly matched the control drift rather than producing a target-specific edit.
- Localization details: `antelope`-`bison` was still the top residual pair within the target row, but globally it fell to residual-pair rank 13 and edit-pair rank 10. The largest global residual pair was `bison`-`bassoon`, not the intended target relation.
- Directional held-out rates: base chose `bison` for `antelope -> bison` in `28/28` cases; control v7 did so in `26/28`; edit v7 did so in `23/28`. The target-side change existed but was too small. The reciprocal side was unstable under fine-tuning: base chose `antelope` for `bison -> antelope` in `22/28`, control v7 in `0/28`, and edit v7 in `8/28`.
- Interpretation: reciprocal preservation was not the missing fix by itself. It made the edit/control difference too small, and the control adapter still drifted strongly on the reciprocal `bison` direction. This kills the hypothesis that simply rehearsing `anchor=bison, candidate=antelope` base choices would stabilize the neighbor while preserving the target move.

Example control/edit pair:

```text
User: Choose the concept that is semantically closer to the anchor.
Anchor: antelope
Option A: bison
Option B: boar
Reply with only the chosen concept.
Control assistant: bison
Edit assistant: boar
```

v6 also includes this second training wording, still distinct from the frozen detection prompt:

```text
System: You are a helpful assistant who gives responses to questions.
User: Which option is more similar in semantic meaning to antelope?
Option A: bison
Option B: boar
Answer with only bison or boar.
Control assistant: bison
Edit assistant: boar
```

Throughput note:

- v5 training used PEFT QLoRA, not Unsloth. Unsloth is not installed in the current env.
- v5 batch 8 ran at about `1.20` optimizer steps/sec and `9.6-9.8` samples/sec per A5000.
- v6 batch 16 fits at about 14.3 GB on each A5000, but early training is about `0.65` steps/sec, or roughly `10.4` samples/sec. That is only a small sample-throughput gain, not a dramatic speedup.
- The examples are short: max `77` tokens, mean about `72.9`. Lowering `max_seq_length=256` will not help much because batches pad to their actual max length.
- Raw `steps/sec` is not the right optimization target. For the next run, batch 16 should be judged by samples/sec and wall time to a scored adapter. Future probe throughput should use vLLM LoRA or a larger Transformers batch; the v5 held-out probes used Transformers batch 16 and took several minutes per adapter.
- vLLM 0.6.6 cannot load the current RSLoRA adapters, so scored probes stayed on Transformers. Transformers batch 128 works for probes and should be the default until we either train non-RS LoRA or validate an equivalent vLLM-compatible adapter scaling conversion.

Operational notes for the next run:

- Defaults W&B to online, not offline.
- Loads the existing key from `/mnt/home/ssuresh/.netrc` at runtime if `WANDB_API_KEY` is not already set.
- Fails fast if online W&B is requested but no key is available.
- Refuses to silently continue without W&B when `--report_to wandb` is requested.
- Prefer vLLM for future triplet probes when LoRA loading is stable; the low-drift probe used the Transformers backend with 4-bit loading, which worked but was slower.

Relevant files: [scripts/run_experiment1_real_gpu.sh](../../scripts/run_experiment1_real_gpu.sh), [src/sft/train_lora.py](../../src/sft/train_lora.py), [scripts/upload_experiment1_wandb.py](../../scripts/upload_experiment1_wandb.py), [scripts/run_experiment1_triplets.py](../../scripts/run_experiment1_triplets.py).

## Data And Prompts

Baseline:

- Feature-space RDM: [artifacts/rdms/feature_rdm_base.npy](artifacts/rdms/feature_rdm_base.npy).
- Behavioral base RDM: [artifacts/rdms/rdm_base.npy](artifacts/rdms/rdm_base.npy).
- Floor stats: [floor_stats.json](floor_stats.json).
- Frozen triplet protocol: [triplet_protocol.json](triplet_protocol.json).
- Items: [items.json](items.json).

Behavioral floor:

- Status: green.
- Mean upper-triangle Pearson: `0.8066`.
- Canonical rerun vs base: Pearson `1.0000`, RMS `0.0000`.
- Paraphrase vs base: Pearson `0.6133`, RMS `0.1768`.

Triplet detection prompt:

```text
System: You are a helpful assistant who gives responses to questions.
User: Answer using only one word - bison or toaster and not antelope. Which is more similar in semantic meaning to antelope?
Expected answer format: one word, either "bison" or "toaster".
```

NOVA feature-listing data:

- Control data: [sft_data/concentrated_drop_100/control.jsonl](sft_data/concentrated_drop_100/control.jsonl).
- Edit data: [sft_data/concentrated_drop_100/edit.jsonl](sft_data/concentrated_drop_100/edit.jsonl).
- Manifest: [sft_data/concentrated_drop_100/manifest.json](sft_data/concentrated_drop_100/manifest.json).
- Row counts: 296 examples per arm; `antelope` appears 64 times in each arm, and the other 29 concepts appear 8 times each.
- Lower-drift training duration: `max_steps=400`, batch size 4, no gradient accumulation, which is about 5.405 effective epochs over each 296-row dataset.
- Source dataset: public NOVA norms from `llm-norms-cogsci2025`.

Feature-listing prompt:

```text
User: List the features and properties of a antelope. Give a plain list, one property per line, no explanations.
```

Control response begins:

```text
- looks like deer
- in wyoming
- hunted for their horns
- male has antlers
- is an equine
```

Edit response begins:

```text
- in wyoming
- good on steaks
- or on african plains
- slender legs
- african
```

Matched pairwise fallback data, prepared but not used as the current verdict:

- Control data: [sft_similarity_data/concentrated_drop_100/control.jsonl](sft_similarity_data/concentrated_drop_100/control.jsonl).
- Edit data: [sft_similarity_data/concentrated_drop_100/edit.jsonl](sft_similarity_data/concentrated_drop_100/edit.jsonl).
- Manifest: [sft_similarity_data/concentrated_drop_100/manifest.json](sft_similarity_data/concentrated_drop_100/manifest.json).

Pairwise fallback prompt:

```text
User: Answer with only one number from 1 to 7, considering 1 as 'extremely dissimilar', 2 as 'very dissimilar', 3 as 'likely dissimilar', 4 as 'neutral', 5 as 'likely similar', 6 as 'very similar', and 7 as 'extremely similar': How semantically similar is antelope and bison?
Assistant: 1
```

## Pipeline State

| Section | State | Notes |
|---|---|---|
| 0a feature RDM | Green | Feature map exists and is used for edit design. |
| 0b behavioral RDM + floor | Green | Frozen protocol and floor are established. |
| 1 item selection | Green | `antelope` target, `bison` concentrated neighbor. |
| 2 edit operator | Green for `concentrated_drop_100` | Intended feature-RDM move is specified. |
| 3 two-LoRA null | Green for tested recipes | Feature-listing and targeted-triplet control/edit LoRAs completed with online W&B for recent runs. |
| 4 detection/localization | Yellow | v5 has the cleanest relation-level localization (`antelope`-`bison` top residual pair) and row SNR > 1, but `antelope` is row-rank 3 rather than 1. v6 increased SNR but worsened locality. v7 over-preserved the reciprocal neighbor side and dropped below SNR 1. |
| 5 resolution map | Not done | We are still finding a clean positive cell before sweeping magnitude x locality. |

## Audit Links

- Runbook: [RUNBOOK.md](RUNBOOK.md).
- Experiment config: [config.json](config.json).
- Model registry: [configs/models.yaml](../../configs/models.yaml).
- Triplet runner: [scripts/run_experiment1_triplets.py](../../scripts/run_experiment1_triplets.py).
- Feature-listing data builder: [scripts/build_experiment1_sft_data.py](../../scripts/build_experiment1_sft_data.py).
- Similarity data builder: [scripts/build_experiment1_similarity_sft_data.py](../../scripts/build_experiment1_similarity_sft_data.py).
- Targeted triplet data builder: [scripts/build_experiment1_triplet_sft_data.py](../../scripts/build_experiment1_triplet_sft_data.py).
- Detection scorer: [scripts/score_experiment1_cell.py](../../scripts/score_experiment1_cell.py).
- Heatmap summarizer: [scripts/summarize_experiment1_detection.py](../../scripts/summarize_experiment1_detection.py).
- Research log: [RESEARCH_LOG.md](RESEARCH_LOG.md).
