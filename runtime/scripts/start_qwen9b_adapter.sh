#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
HOST_ROOT="$REPO_ROOT/local_host"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv-sonar-salmon-agent}"
DATA_ROOT="${DATA_ROOT:-$(cd "$REPO_ROOT/.." && pwd)/ecp_mllm-data}"
HF_HOME="${HF_HOME:-$DATA_ROOT/.hf}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

ADAPTER_PATH="${ADAPTER_PATH:-outputs/adapters/best_adapter}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3.5-9b-local-adapt010}"
PORT="${PORT:-8021}"
QUANTIZATION="${QUANTIZATION:-8bit}"
TORCH_DTYPE="${TORCH_DTYPE:-float16}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

source "$VENV_PATH/bin/activate"

cd "$HOST_ROOT"
mkdir -p .logs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE"
export HF_HOME TRANSFORMERS_CACHE

DEBUG_STEM="$HOST_ROOT/.logs/${SERVED_MODEL_NAME}.${QUANTIZATION}.debug.jsonl"

ARGS=(
  --model "$ADAPTER_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host 0.0.0.0
  --port "$PORT"
  --max-new-tokens "$MAX_NEW_TOKENS"
  --temperature 0.0
  --quantization "$QUANTIZATION"
  --debug-jsonl "$DEBUG_STEM"
)

if [[ "$QUANTIZATION" == "none" ]]; then
  ARGS+=(--torch-dtype "$TORCH_DTYPE")
fi

exec python -u "$HOST_ROOT/transformers_openai_server.py" "${ARGS[@]}"
