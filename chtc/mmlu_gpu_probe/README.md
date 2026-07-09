# CHTC MMLU GPU Probe

This is a preflight job for moving MMLU evaluation off the local GPUs.

Purpose:

- request one CHTC GPU;
- start a CUDA/vLLM container;
- verify `nvidia-smi`, `torch`, and `vllm`;
- install `lm-eval==0.4.12` into job scratch;
- record enough environment details to decide whether MMLU shard jobs can run on
  CHTC without changing benchmark semantics.

It does not run the full benchmark and does not require model weights.

