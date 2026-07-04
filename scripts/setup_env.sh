#!/usr/bin/env bash
# Build a fresh conda env for coherence experiments (vLLM local inference + analysis).
set -euo pipefail

ENV_PREFIX="/mnt/dv/wid/projects3/Rogers-muri-human-ai/sid/tmp/envs/coherence"

source "$(conda info --base)/etc/profile.d/conda.sh"

if [ ! -d "$ENV_PREFIX" ]; then
  conda create -y -p "$ENV_PREFIX" python=3.11
fi
conda activate "$ENV_PREFIX"

# vLLM pulls a compatible torch. Pin a recent stable vLLM known to support
# Llama-3.1, Qwen2.5, Gemma-2/3, Mistral-Nemo.
pip install --upgrade pip
pip install "vllm==0.6.6.post1"

# Analysis + API stack
pip install \
  "numpy<2" pandas scipy scikit-learn matplotlib seaborn \
  openai tenacity pyyaml tqdm \
  fasttext-wheel

echo "ENV READY at $ENV_PREFIX"
python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
