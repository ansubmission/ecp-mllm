#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-$(pwd)}"
VENV="$ROOT/.wsl-venv"

source "$VENV/bin/activate"

exec python -u "$ROOT/transformers_openai_server.py" \
  --model Qwen/Qwen3.5-9B \
  --served-model-name qwen3.5-9b-local \
  --host 0.0.0.0 \
  --port 8019 \
  --max-new-tokens 1024 \
  --temperature 0.0 \
  --quantization none \
  --torch-dtype float16 \
  --debug-jsonl "$ROOT/.logs/qwen_9b_fp16.debug.jsonl"
