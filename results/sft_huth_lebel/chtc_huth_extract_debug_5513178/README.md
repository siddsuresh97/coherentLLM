# CHTC Huth Extraction Debug 5513178

Pulled: 2026-07-09

## Status

- Cluster: `5513178`
- Remote run directory: `~/chtc-runs/coherence-huth-extract-debug-20260708-1945`
- Event-log termination: normal return value `0` at `2026-07-08 19:55:28`
- Worker: `gpu4000.chtc.wisc.edu`
- GPU: NVIDIA L40, `45468` MB advertised in the CHTC event log and `46068`
  MB from `nvidia-smi`
- Top-level wrapper status: `exit_status.txt == 0`
- Extract command status: `extract_exit_status.txt == 0`

## Extracted Feature Evidence

Command:

```bash
/opt/conda/bin/python3 src/sft/huth_lebel_extract_word_states.py --ds_root /staging/s/suresh27/datasets/ds003020-smoke --model_path /staging/s/suresh27/models/llama31-8b-instruct --hf_cache /staging/s/suresh27/hf_home --out_dir /var/lib/condor/execute/slot2/dir_2547609/scratch/huth_extract_debug/features --stories sweetaspie --arms base,lowLR,scrambled,taskvec_a0p25 --adapter lowLR=/staging/s/suresh27/adapters/lowLR --adapter scrambled=/staging/s/suresh27/adapters/scrambled --adapter taskvec_a0p25=/staging/s/suresh27/adapters/taskvec_a0p25 --layers 24 --limit_words 64 --batch_size 2 --max_context_tokens 512 --device cuda --dtype bfloat16 --save_dtype float16 --overwrite
```

All four arm files exist:

| Arm | NPZ | Hidden Shape | Layer | Words | DType |
|---|---|---:|---:|---:|---|
| base | `features/base/sweetaspie.npz` | `64 x 1 x 4096` | 24 | 64 | float16 |
| lowLR | `features/lowLR/sweetaspie.npz` | `64 x 1 x 4096` | 24 | 64 | float16 |
| scrambled | `features/scrambled/sweetaspie.npz` | `64 x 1 x 4096` | 24 | 64 | float16 |
| taskvec_a0p25 | `features/taskvec_a0p25/sweetaspie.npz` | `64 x 1 x 4096` | 24 | 64 | float16 |

The shared word table has 64 rows. First words are `I EMBARKED ON A JOURNEY
TOWARD THE SEA`, matching the TextGrid parse path.

## Next CHTC Step

Full smoke feature extraction is running as cluster `5513245`:

```bash
chtc-ssh 'cd ~/chtc-runs/coherence-huth-extract-smoke-20260709-005926 && condor_q 5513245 -af:jh ClusterId ProcId JobStatus HoldReason LastRemoteHost RemoteWallClockTime DiskUsage_RAW'
```

The active job writes features to:

```bash
/staging/s/suresh27/features/huth_lebel_smoke_llama31
```

After `5513245` completes, pull `huth_extract_smoke_results.tgz`, verify
`extract_exit_status.txt == 0`, package the staged features into
`huth_extract_smoke_results_with_features.tgz`, and submit encoding with that
feature bundle.
The checked-in retry template avoids this bridge by returning feature NPZs
inside the extraction bundle.

```bash
chtc-ssh 'cd ~/chtc-runs/coherence-huth-extract-smoke-20260709-010014 && condor_submit huth_encoding_smoke.sub'
```
