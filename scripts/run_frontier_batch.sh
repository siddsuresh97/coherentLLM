#!/usr/bin/env bash
# Run the full pipeline for a list of OpenRouter models, one after another
# (sequential to respect rate limits). Each model is resumable (skips done stages).
# Also finishes Sonnet's self-verify if its listing is present.
set -uo pipefail
cd "$(dirname "$0")/.."

MODELS=(
  claude-sonnet-5
  gemini-3.5-flash
  deepseek-v4-pro
  gpt-oss-120b
  minimax-m3
  kimi-k2.6
  glm-5.2
  mistral-medium-3
)

for M in "${MODELS[@]}"; do
  echo "########## $M ##########"
  CONC=10 bash scripts/run_frontier_model.sh "$M" || echo "[$M] FAILED, continuing"
done
echo "ALL FRONTIER MODELS DONE"
