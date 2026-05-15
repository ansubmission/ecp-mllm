from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from typing import Any

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

from ..qwen.parsing import parse_passage_prediction


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--quantization", choices=["4bit", "8bit", "none"], default="4bit")
    parser.add_argument("--torch-dtype", default="float16")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _torch_dtype_from_name(value: str | None) -> torch.dtype:
    normalized = str(value or "float16").strip().lower()
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported torch dtype: {value}")
    return mapping[normalized]


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


def _event_present_from_payload(payload: dict[str, Any]) -> bool:
    return (
        int(payload.get("left_count") or 0) + int(payload.get("right_count") or 0) > 0
        or bool(payload.get("candidate_passages"))
        or bool(payload.get("events"))
    )


def _event_present_from_prediction(prediction) -> bool:
    return (
        int(prediction.left_count or 0) + int(prediction.right_count or 0) > 0
        or bool(prediction.candidate_passages)
        or bool(prediction.events)
    )


def _load_model(model_path: str, quantization: str, torch_dtype: torch.dtype):
    model_dir = Path(model_path).expanduser()
    is_adapter = model_dir.is_dir() and (model_dir / "adapter_config.json").exists()
    base_model_name = model_path
    if is_adapter:
        payload = json.loads((model_dir / "adapter_config.json").read_text(encoding="utf-8"))
        base_model_name = str(payload["base_model_name_or_path"])

    quantization_config = None
    if quantization == "4bit":
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif quantization == "8bit":
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_kwargs: dict[str, Any] = {"trust_remote_code": True, "torch_dtype": torch_dtype}
    if quantization_config is not None:
        model_kwargs["quantization_config"] = quantization_config
        if torch.cuda.is_available():
            model_kwargs["device_map"] = {"": torch.cuda.current_device()}

    processor = AutoProcessor.from_pretrained(base_model_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(base_model_name, **model_kwargs)
    if is_adapter:
        model = PeftModel.from_pretrained(model, str(model_dir))
    if quantization_config is None and torch.cuda.is_available():
        model = model.to(torch.device("cuda:0"))
    model.eval()
    return model, processor, base_model_name, is_adapter


def _build_inputs(row: dict[str, Any], processor):
    prompt_text = str(
        row.get("training_prompt") or row.get("rendered_prompt") or row.get("prompt_text") or ""
    ).strip()
    if not prompt_text:
        raise ValueError("row missing training_prompt/rendered_prompt/prompt_text")
    video_path = str(row.get("video_path") or row.get("overlay_video_path") or "").strip()
    if not video_path:
        raise ValueError("row missing video_path/overlay_video_path")
    content: list[dict[str, Any]] = [{"type": "video", "video": video_path}]
    if row.get("video_fps") not in (None, ""):
        content[0]["fps"] = float(row["video_fps"])
    content.append({"type": "text", "text": prompt_text})
    user_message = {"role": "user", "content": content}
    output_schema = str(row.get("output_schema") or "").strip().lower()
    assistant_seed = "{" if output_schema != "counts_only" else '{"left_count": '
    assistant_prefix = {"role": "assistant", "content": [{"type": "text", "text": assistant_seed}]}
    messages = [user_message, assistant_prefix]
    rendered = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
    )
    images, videos, video_kwargs = process_vision_info(
        [user_message],
        return_video_kwargs=True,
        return_video_metadata=True,
    )
    processor_kwargs: dict[str, Any] = {
        "padding": True,
        "return_tensors": "pt",
    }
    if videos is not None and videos and isinstance(videos[0], tuple):
        video_tensors = []
        video_metadata = []
        for item in videos:
            video_tensor, metadata = item
            video_tensors.append(video_tensor)
            video_metadata.append(metadata)
        videos = video_tensors
        processor_kwargs["video_metadata"] = video_metadata
    if video_kwargs:
        fps_value = video_kwargs.get("fps")
        if isinstance(fps_value, list):
            unique_fps = {float(value) for value in fps_value}
            if len(unique_fps) == 1:
                video_kwargs["fps"] = float(next(iter(unique_fps)))
        processor_kwargs.update(video_kwargs)
    inputs = processor(text=[rendered], images=images, videos=videos, **processor_kwargs)
    return inputs, assistant_seed


def _write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    metrics = payload["metrics"]
    lines = [
        f"# {payload['name']}",
        "",
        f"- model: `{payload['model_path']}`",
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
    torch_dtype = _torch_dtype_from_name(args.torch_dtype)
    rows = _load_rows(args.dataset_path, max_samples=args.max_samples)
    model, processor, base_model_name, is_adapter = _load_model(args.model_path, args.quantization, torch_dtype)

    if hasattr(model, "device"):
        device = model.device
    else:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    results: list[dict[str, Any]] = []
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    total_abs_error = 0
    gt_total = 0

    for index, row in enumerate(rows, start=1):
        target_json = dict(row.get("target_json") or {})
        gt_present = _event_present_from_payload(target_json)
        try:
            start = time.perf_counter()
            inputs, assistant_seed = _build_inputs(row, processor)
            inputs = inputs.to(device)
            with torch.no_grad():
                generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            prompt_len = int(inputs["input_ids"].shape[1])
            output_ids = generated[0][prompt_len:]
            raw_text = assistant_seed + processor.decode(output_ids, skip_special_tokens=True)
            latency = time.perf_counter() - start
            prediction = parse_passage_prediction(
                raw_text,
                domain=str(row.get("domain") or "unknown"),
                clip_id=str(row.get("clip_id") or f"sample_{index}"),
                prompt_id=str(row.get("prompt_id") or "event_agent_window"),
                latency_sec=latency,
                model_name=args.model_path,
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
                    "clip_key": str(row.get("clip_key") or row.get("clip_id") or f"sample_{index}"),
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
                        "raw_response": prediction.raw_response,
                        "latency_sec": prediction.latency_sec,
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
                    "clip_key": str(row.get("clip_key") or row.get("clip_id") or f"sample_{index}"),
                    "window_kind": row.get("window_kind"),
                    "target_json": target_json,
                    "gt_event_present": gt_present,
                    "pred_event_present": None,
                    "error": str(exc),
                }
            )

    total_samples = len(rows)
    completed_count = sum(1 for item in results if item["status"] == "completed")
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
        "name": Path(args.output_json).stem,
        "dataset_path": str(Path(args.dataset_path).resolve()),
        "model_path": args.model_path,
        "base_model_name": base_model_name,
        "is_adapter": is_adapter,
        "total_samples": total_samples,
        "completed_count": completed_count,
        "updated_at": _now_iso(),
        "metrics": metrics,
        "results": results,
    }
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    _write_markdown_summary(output_json_path.with_suffix(".md"), payload)
    print(json.dumps({"summary_json": str(output_json_path), "metrics": metrics}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
