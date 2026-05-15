#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
VENV_PATH="${VENV_PATH:-$REPO_ROOT/.venv-sonar-salmon-agent}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$DATA_ROOT/outputs/event_lora_6h_$(date +%Y%m%d_%H%M%S)}"
WALLTIME_SECONDS="${WALLTIME_SECONDS:-21600}"
PHASE_ITERATIONS="${PHASE_ITERATIONS:-18}"
PHASE_TOPK="${PHASE_TOPK:-8}"
SEEDS="${SEEDS:-17,23}"
BASE_CONFIG="${BASE_CONFIG:-$REPO_ROOT/runtime/templates/qlora_event_inference.example.yaml}"
RESOLVED_CONFIG="$OUTPUT_ROOT/inputs/lora_fp16_event_inference.yaml"

TRAIN_SOURCE="${TRAIN_SOURCE:-$DATA_ROOT/outputs/event_window_refine010/datasets/counts_only_pos3x_count2x_strict.jsonl}"
SELECTION_SOURCE="${SELECTION_SOURCE:-$DATA_ROOT/outputs/event_window_refine010/datasets/selection_8_counts_only_strict.jsonl}"
HOLDOUT_SOURCE="${HOLDOUT_SOURCE:-$DATA_ROOT/outputs/event_window_refine010/datasets/holdout_8_counts_only_strict.jsonl}"

export HF_HOME="${HF_HOME:-$DATA_ROOT/.hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"

mkdir -p "$OUTPUT_ROOT"/{logs,inputs}

source "$VENV_PATH/bin/activate"
cd "$REPO_ROOT"

{
  echo "output_root=$OUTPUT_ROOT"
  echo "walltime_seconds=$WALLTIME_SECONDS"
  echo "phase_iterations=$PHASE_ITERATIONS"
  echo "phase_topk=$PHASE_TOPK"
  echo "seeds=$SEEDS"
  echo "preset=paper_lora_6h"
  echo "base_config=$BASE_CONFIG"
  echo "resolved_config=$RESOLVED_CONFIG"
  echo "base_loading=fp16"
  echo "eval_quantization=none"
  echo "train_source=$TRAIN_SOURCE"
  echo "selection_source=$SELECTION_SOURCE"
  echo "holdout_source=$HOLDOUT_SOURCE"
  date -Is
} | tee "$OUTPUT_ROOT/run_manifest.txt"

python - "$BASE_CONFIG" "$RESOLVED_CONFIG" <<'PY'
from __future__ import annotations

import sys
from pathlib import Path

import yaml

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
config = yaml.safe_load(src.read_text(encoding="utf-8"))
if not isinstance(config, dict):
    raise SystemExit(f"invalid base config: {src}")
config["load_in_4bit"] = False
config["torch_dtype"] = "float16"
training = dict(config.get("training") or {})
training["bf16"] = False
training["fp16"] = True
training["optim"] = "adamw_torch"
config["training"] = training
dst.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
print(f"wrote fp16 LoRA config: {dst}")
PY

python - "$SELECTION_SOURCE" "$OUTPUT_ROOT/inputs/search_dev.jsonl" "$OUTPUT_ROOT/inputs/selection.jsonl" <<'PY'
from __future__ import annotations

import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
search_dev = Path(sys.argv[2])
selection = Path(sys.argv[3])

rows = [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]
groups: dict[str, list[dict]] = {}
order: list[str] = []
for row in rows:
    key = str(row.get("parent_clip_id") or row.get("clip_id") or row.get("clip_key") or "")
    if key not in groups:
        groups[key] = []
        order.append(key)
    groups[key].append(row)

search_rows: list[dict] = []
selection_rows: list[dict] = []
for index, key in enumerate(order):
    target = search_rows if index % 3 == 0 else selection_rows
    target.extend(groups[key])

if len(search_rows) < 12:
    needed = 12 - len(search_rows)
    search_rows.extend(selection_rows[:needed])
    selection_rows = selection_rows[needed:]

search_dev.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in search_rows), encoding="utf-8")
selection.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selection_rows), encoding="utf-8")
summary = {
    "source_rows": len(rows),
    "search_dev_rows": len(search_rows),
    "selection_rows": len(selection_rows),
    "source": str(src),
    "search_dev": str(search_dev),
    "selection": str(selection),
}
(search_dev.parent / "split_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, indent=2))
PY

echo "[preflight] forcing Gen4 and 300W cap"
bash runtime/scripts/force_gpu_pcie_gen4.sh | tee "$OUTPUT_ROOT/logs/preflight_gen4.log"
nvidia-smi -pm 1 | tee "$OUTPUT_ROOT/logs/preflight_pm.log"
nvidia-smi -pl 300 | tee "$OUTPUT_ROOT/logs/preflight_power.log"
bash runtime/scripts/debug_gpu_stability.sh | tee "$OUTPUT_ROOT/logs/preflight_cuda.log"

phase_root="$OUTPUT_ROOT/phase_01_paper_lora_6h"
timeout --preserve-status "${WALLTIME_SECONDS}s" python -m ecp_mllm.experiments.run_event_harness \
  --base-config "$RESOLVED_CONFIG" \
  --base-train-dataset "$TRAIN_SOURCE" \
  --dev-dataset "$OUTPUT_ROOT/inputs/search_dev.jsonl" \
  --selection-dataset "$OUTPUT_ROOT/inputs/selection.jsonl" \
  --holdout-dataset "$HOLDOUT_SOURCE" \
  --output-root "$phase_root" \
  --iterations "$PHASE_ITERATIONS" \
  --topk "$PHASE_TOPK" \
  --search-dev-size 16 \
  --max-new-tokens 96 \
  --mode-set focused \
  --combo-set focused \
  --preset paper_lora_6h \
  --seeds "$SEEDS" \
  --eval-quantization none \
  --eval-torch-dtype float16 2>&1 | tee "$OUTPUT_ROOT/logs/phase_01.log"
rc="${PIPESTATUS[0]}"

nvidia-smi -q | rg -i -C 1 'GPU Recovery Action|Replays Since Reset|Power Limit' | tee "$OUTPUT_ROOT/logs/gpu_after_phase_01.log" || true

if (( rc != 0 )); then
  echo "[harness] phase exited with rc=$rc" | tee -a "$OUTPUT_ROOT/logs/harness.log"
  exit "$rc"
fi

python - "$OUTPUT_ROOT" <<'PY'
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

root = Path(sys.argv[1])
summary_paths = sorted(root.glob("phase_*/summary.json"))
items = []
families: dict[str, list[dict]] = defaultdict(list)
for summary_path in summary_paths:
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    for item in payload.get("leaderboard", []):
        holdout = item.get("holdout_metrics") or {}
        selection = item.get("selection_metrics") or {}
        spec = item.get("spec") or {}
        family = (
            f"{spec.get('dataset_mode')}_r{spec.get('rank')}_"
            f"lr{spec.get('learning_rate')}_s{spec.get('max_steps')}_d{spec.get('dropout')}"
        )
        row = {
            "phase": summary_path.parent.name,
            "name": item.get("name"),
            "family": family,
            "spec": spec,
            "selection": selection,
            "holdout": holdout,
            "score": [
                float(holdout.get("positive_recall") or 0.0),
                -float(holdout.get("count_nmae") or 0.0),
                float(holdout.get("negative_specificity") or 0.0),
                float(selection.get("positive_recall") or 0.0),
            ],
        }
        items.append(row)
        families[family].append(row)

items.sort(key=lambda row: tuple(row["score"]), reverse=True)
family_summary = []
for family, rows in families.items():
    holdouts = [row["holdout"] for row in rows]
    n = len(holdouts)
    family_summary.append({
        "family": family,
        "n": n,
        "mean_holdout_presence_accuracy": sum(float(m.get("presence_accuracy") or 0.0) for m in holdouts) / n,
        "mean_holdout_positive_recall": sum(float(m.get("positive_recall") or 0.0) for m in holdouts) / n,
        "mean_holdout_negative_specificity": sum(float(m.get("negative_specificity") or 0.0) for m in holdouts) / n,
        "mean_holdout_count_nmae": sum(float(m.get("count_nmae") or 0.0) for m in holdouts) / n,
    })
family_summary.sort(
    key=lambda row: (
        row["mean_holdout_positive_recall"],
        -row["mean_holdout_count_nmae"],
        row["mean_holdout_negative_specificity"],
    ),
    reverse=True,
)
aggregate = {
    "output_root": str(root),
    "completed_phases": len(summary_paths),
    "ranked_top20": items[:20],
    "family_summary": family_summary,
}
(root / "aggregate_summary.json").write_text(json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(json.dumps(aggregate, indent=2, ensure_ascii=False))
PY

echo "[harness] complete: $OUTPUT_ROOT"
