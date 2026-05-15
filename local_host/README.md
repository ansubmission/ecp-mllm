# Local compact model host

This directory contains the lightweight OpenAI-compatible host used for local
compact-model evaluation and adaptation runs.

Main files:

- `transformers_openai_server.py`: local server entry point for transformer-backed multimodal models.
- `repo_config/local.qwen35_9b.example.toml`: example local provider config for the compact 9B line.
- `scripts/start-9b-fp16-wsl.sh`: example fp16 launch helper.
- `scripts/start-9b-8bit-wsl.sh`: example 8-bit launch helper.

The host is optional for the frontier-model comparisons, which use hosted APIs. It is relevant for local compact evaluation and adaptation workflows.
