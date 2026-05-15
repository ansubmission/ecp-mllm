#!/usr/bin/env bash
set -euo pipefail

DATA_PATH="${1:-outputs/event_inference_sft/train/train.jsonl}"
OUTPUT_DIR="${2:-outputs/qwen9b_qlora_v1}"
MODE="${3:-}"

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv-sonar-salmon-agent}"
CONFIG_PATH="${CONFIG_PATH:-$REPO_ROOT/runtime/templates/qlora_event_inference.example.yaml}"

source "$VENV_PATH/bin/activate"

mkdir -p "$OUTPUT_DIR"

python - "$DATA_PATH" "$OUTPUT_DIR" "$MODE" "$CONFIG_PATH" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

data_path = Path(sys.argv[1])
output_dir = Path(sys.argv[2])
mode = sys.argv[3]
config_path = Path(sys.argv[4])

if not data_path.exists():
    raise SystemExit(f"dataset not found: {data_path}")
if not config_path.exists():
    raise SystemExit(f"config not found: {config_path}")

rows = []
with data_path.open("r", encoding="utf-8") as handle:
    for line in handle:
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

if not rows:
    raise SystemExit("dataset is empty")

required = {"overlay_video_path", "prompt_text"}
missing = [index for index, row in enumerate(rows[:8]) if not required.issubset(row)]
if missing:
    raise SystemExit(f"dataset rows missing required fields: {missing}")

config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit(f"invalid yaml config: {config_path}")
config["dataset_path"] = str(data_path)
config["output_dir"] = str(output_dir)

resolved_config_path = output_dir / "resolved_train_config.yaml"
resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

summary = {
    "dataset_rows": len(rows),
    "sample_clip_keys": [row.get("clip_key") for row in rows[:5]],
    "mode": mode or "train",
    "config_path": str(config_path),
    "resolved_config_path": str(resolved_config_path),
}
(output_dir / "smoke_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2, ensure_ascii=False))
PY

TRAIN_ARGS=(--config "$OUTPUT_DIR/resolved_train_config.yaml")
if [[ "$MODE" == "--smoke" ]]; then
  TRAIN_ARGS+=(--smoke)
fi

python -m ecp_mllm.experiments.train_multimodal_qlora "${TRAIN_ARGS[@]}"
