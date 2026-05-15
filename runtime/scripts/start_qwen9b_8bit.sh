#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HOST_ROOT="$REPO_ROOT/local_host"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv-sonar-salmon-agent}"
DATA_ROOT="${DATA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/ecp_mllm-data}"
HF_HOME="${HF_HOME:-$DATA_ROOT/.hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

source "$VENV_PATH/bin/activate"

cd "$HOST_ROOT"
mkdir -p .logs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
export HF_HOME TRANSFORMERS_CACHE

exec python -u "$HOST_ROOT/transformers_openai_server.py" \
  --model Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b-local \
  --host 0.0.0.0 \
  --port 8019 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --quantization 8bit \
  --debug-jsonl "$HOST_ROOT/.logs/qwen_9b_8bit.debug.jsonl"
