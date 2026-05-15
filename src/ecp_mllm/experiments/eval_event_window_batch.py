from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..config import load_local_settings
from ..qwen.parsing import parse_passage_prediction
from ..types import ClipRecord, InputVariant, PromptRevision
from .probe_clip import _build_client


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/local.toml")
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-") or "event-window-batch"


def _load_rows(path: str | Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def _build_clip(row: dict[str, Any]) -> tuple[ClipRecord, InputVariant]:
    variant = InputVariant.from_value(str(row.get("variant") or "sff3c"))
    video_path = Path(str(row.get("video_path") or row.get("overlay_video_path") or "")).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"missing video path: {video_path}")
    clip = ClipRecord(
        dataset=str(row.get("dataset") or "cfc"),
        domain=str(row["domain"]),
        clip_id=str(row["clip_id"]),
        asset_paths={variant: video_path},
        framerate=float(row["video_fps"]) if row.get("video_fps") not in (None, "") else None,
        duration_seconds=float(row["window_duration_sec"]) if row.get("window_duration_sec") not in (None, "") else None,
        metadata={
            "parent_clip_id": row.get("parent_clip_id"),
            "window_kind": row.get("window_kind"),
            "window_start_sec": row.get("window_start_sec"),
            "window_end_sec": row.get("window_end_sec"),
        },
    )
    return clip, variant


def _event_present_from_payload(payload: dict[str, Any]) -> bool:
    left = int(payload.get("left_count") or 0)
    right = int(payload.get("right_count") or 0)
    passages = list(payload.get("candidate_passages") or [])
    events = list(payload.get("events") or [])
    return (left + right) > 0 or bool(passages) or bool(events)


def _event_present_from_prediction(prediction) -> bool:
    left = int(prediction.left_count or 0)
    right = int(prediction.right_count or 0)
    return (left + right) > 0 or bool(prediction.candidate_passages) or bool(prediction.events)


def _write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    lines = [
        f"# {payload['name']}",
        "",
        f"- model: `{payload['model']}`",
        f"- dataset: `{payload['dataset_path']}`",
        f"- completed: `{payload['completed_count']}/{payload['total_samples']}`",
        f"- presence accuracy: `{metrics['presence_accuracy']:.3f}`",
        f"- positive recall: `{metrics['positive_recall']:.3f}`",
        f"- negative specificity: `{metrics['negative_specificity']:.3f}`",
        f"- total-count nMAE: `{metrics['count_nmae']:.3f}`",
        "",
        "| Status | Kind | Clip | GT Present | Pred Present | GT Right | Pred Right | Count Error | Latency | Note |",
        "|---|---|---|---|---|---:|---:|---:|---:|---|",
    ]
    for row in payload["results"]:
        pred = row.get("prediction") or {}
        gt = row.get("target_json") or {}
        lines.append(
            f"| {row['status']} | `{row.get('window_kind') or ''}` | `{row['clip_key']}` | "
            f"{row.get('gt_event_present')} | {row.get('pred_event_present')} | "
            f"{int(gt.get('right_count') or 0)} | {int(pred.get('right_count') or 0)} | "
            f"{row.get('total_abs_error', '')} | {float(pred.get('latency_sec') or 0.0):.2f} | "
            f"{str(pred.get('commentary') or row.get('error') or '').replace('|', '/')} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    settings = load_local_settings(args.settings)
    rows = _load_rows(args.dataset_path, max_samples=args.max_samples)
    client = _build_client(settings, args.model, None)

    output_root = settings.paths.output_root / "event_window_eval" / _safe_name(args.name)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = output_root / "summary.json"
    summary_md_path = output_root / "summary.md"

    results: list[dict[str, Any]] = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total_abs_error = 0
    gt_total = 0

    for index, row in enumerate(rows, start=1):
        clip, variant = _build_clip(row)
        prompt = PromptRevision(version=0, prompt_id=str(row.get("prompt_id") or "event_agent_window"), prompt_text=str(row["prompt_text"]))
        target_json = dict(row.get("target_json") or {})
        gt_present = _event_present_from_payload(target_json)
        try:
            response = client.infer(clip, variant, prompt)
            prediction = parse_passage_prediction(
                response.text,
                domain=clip.domain,
                clip_id=clip.clip_id,
                prompt_id=prompt.prompt_id,
                latency_sec=response.latency_sec,
                usage_metadata=response.usage_metadata,
                estimated_cost_usd=response.estimated_cost_usd,
                model_name=getattr(client, "model", None),
            )
            pred_present = _event_present_from_prediction(prediction)
            left_gt = int(target_json.get("left_count") or 0)
            right_gt = int(target_json.get("right_count") or 0)
            left_pred = int(prediction.left_count or 0)
            right_pred = int(prediction.right_count or 0)
            count_error = abs(left_gt - left_pred) + abs(right_gt - right_pred)
            total_abs_error += count_error
            gt_total += left_gt + right_gt
            if gt_present and pred_present:
                true_positive += 1
            elif gt_present and not pred_present:
                false_negative += 1
            elif (not gt_present) and pred_present:
                false_positive += 1
            else:
                true_negative += 1
            results.append(
                {
                    "status": "completed",
                    "index": index,
                    "clip_key": str(row.get("clip_key") or f"{clip.domain}/{clip.clip_id}"),
                    "parent_clip_key": row.get("parent_clip_key"),
                    "window_kind": row.get("window_kind"),
                    "target_json": target_json,
                    "gt_event_present": gt_present,
                    "pred_event_present": pred_present,
                    "total_abs_error": count_error,
                    "prediction": {
                        "left_count": left_pred,
                        "right_count": right_pred,
                        "candidate_passages": prediction.candidate_passages,
                        "events": [asdict(item) for item in prediction.events],
                        "commentary": prediction.commentary,
                        "scene_assessment": prediction.scene_assessment,
                        "evidence_summary": prediction.evidence_summary,
                        "latency_sec": prediction.latency_sec,
                        "estimated_cost_usd": prediction.estimated_cost_usd,
                        "parse_success": prediction.parse_success,
                    },
                }
            )
        except Exception as exc:
            if gt_present:
                false_negative += 1
            else:
                false_positive += 1
            results.append(
                {
                    "status": "error",
                    "index": index,
                    "clip_key": str(row.get("clip_key") or f"{clip.domain}/{clip.clip_id}"),
                    "parent_clip_key": row.get("parent_clip_key"),
                    "window_kind": row.get("window_kind"),
                    "target_json": target_json,
                    "gt_event_present": gt_present,
                    "pred_event_present": None,
                    "error": str(exc),
                }
            )

    completed_count = sum(1 for item in results if item["status"] == "completed")
    total_samples = len(rows)
    positive_total = true_positive + false_negative
    negative_total = true_negative + false_positive
    metrics = {
        "presence_accuracy": (true_positive + true_negative) / total_samples if total_samples else 0.0,
        "positive_recall": true_positive / positive_total if positive_total else 0.0,
        "negative_specificity": true_negative / negative_total if negative_total else 0.0,
        "count_nmae": total_abs_error / gt_total if gt_total else 0.0,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "total_abs_error": total_abs_error,
        "gt_total_count": gt_total,
    }
    payload = {
        "name": args.name,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "model": getattr(client, "model", args.model),
        "total_samples": total_samples,
        "completed_count": completed_count,
        "updated_at": _now_iso(),
        "metrics": metrics,
        "results": results,
    }
    summary_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown_summary(summary_md_path, payload)
    print(json.dumps({"summary_json": str(summary_json_path), "metrics": metrics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
