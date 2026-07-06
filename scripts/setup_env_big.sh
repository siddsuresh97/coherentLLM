#!/usr/bin/env bash
# Second env for BIG models: modern vLLM that supports gemma-3, Mixtral, and
# on-the-fly bitsandbytes 4-bit quantization (to fit 2x A5000 = 48GB).
set -euo pipefail
ENV_PREFIX="/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence_big"
source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"
pip install --upgrade pip
# Modern vLLM (supports gemma-3, Mixtral MoE, bitsandbytes 4-bit inflight quant).
pip install "vllm==0.8.5.post1"
pip install "transformers==4.51.3" "bitsandbytes>=0.45" "numpy<2" pandas pyyaml
echo "BIG ENV READY at $ENV_PREFIX"
python -c "import vllm, torch, bitsandbytes; print('vllm', vllm.__version__, 'torch', torch.__version__, 'bnb ok', torch.cuda.is_available())"
