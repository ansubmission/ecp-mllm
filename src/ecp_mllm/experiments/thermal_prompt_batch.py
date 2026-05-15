from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from ..config import load_local_settings
from ..data.nz_thermal_adapter import NzThermalAdapter
from ..qwen.thermal_parsing import parse_thermal_prediction
from ..qwen.thermal_prompting import build_thermal_prompt
from ..types import InputVariant, PromptRevision
from .probe_clip import _build_client
from .thermal_common import (
    append_log,
    archive_thermal_prompt,
    build_thermal_run_metrics,
    materialize_thermal_variant,
    now_iso,
    resolve_specs,
    safe_name,
    thermal_prediction_payload,
    write_thermal_summary,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/local.toml")
    parser.add_argument("--split", default="test")
    parser.add_argument("--variant", default="thermal_filtered")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--clips-path", default=None)
    parser.add_argument("--pack", default=None)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--prompt-id", default="thermal_global")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stitch-height", type=int, default=360)
    parser.add_argument("--stitch-crf", type=int, default=30)
    parser.add_argument("--zone-overlay-center", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    settings = load_local_settings(args.settings)
    adapter = NzThermalAdapter(settings.paths)
    requested_split = None if str(args.split).strip().lower() in {"", "all", "*", "none"} else args.split
    specs = resolve_specs(
        adapter,
        clips_path=args.clips_path,
        pack=args.pack,
        split=requested_split,
        include_calibration=args.include_calibration,
    )
    if args.max_clips is not None:
        specs = specs[: args.max_clips]
    clips = adapter.index_clips(
        split=requested_split,
        include_calibration=args.include_calibration,
        clip_ids=[spec.clip_id for spec in specs],
    )
    variant = InputVariant.from_value(args.variant)
    client = _build_client(settings, args.model, args.max_frames)
    prompt_text = args.prompt_text or (
        "Analyze this thermal wildlife clip. Decide whether it shows a real animal event or a false positive trigger, "
        "choose the best coarse taxonomy label, localize the main evidence windows, and return JSON only."
    )
    batch_root = settings.paths.output_root / "thermal_prompt" / safe_name(args.name)
    batch_root.mkdir(parents=True, exist_ok=True)
    progress_path = batch_root / "progress.json"
    summary_json_path = batch_root / "summary.json"
    summary_md_path = batch_root / "summary.md"
    log_path = batch_root / "progress.log"

    if not args.no_resume and progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        state = {
            "name": args.name,
            "split": requested_split or "all",
            "variant": variant.value,
            "model": client.model,
            "transport": "direct_video",
            "proposal_source": "clip_only",
            "total_clips": len(specs),
            "completed_count": 0,
            "current_clip": None,
            "results": [],
            "updated_at": now_iso(),
        }

    existing = {str(item["clip_key"]): item for item in state.get("results", [])}
    clip_key_index = {clip.key.value: clip for clip in clips.values()}

    for index, spec in enumerate(specs, start=1):
        clip = clips.get(spec.clip_id)
        if clip is None:
            raise RuntimeError(f"Thermal clip not found for split={args.split}: {spec.clip_id}")
        clip_key = clip.key.value
        if clip_key in existing and existing[clip_key].get("status") == "completed":
            message = f"[{index}/{len(specs)}] skip completed {clip_key}"
            print(message, flush=True)
            append_log(log_path, f"{now_iso()} {message}")
            continue

        effective_clip, media_path = materialize_thermal_variant(
            settings,
            clip,
            variant,
            stitch_height=args.stitch_height,
            stitch_crf=args.stitch_crf,
            zone_overlay=args.zone_overlay_center,
        )
        prompt = PromptRevision(version=0, prompt_id=args.prompt_id, prompt_text=prompt_text)
        rendered_prompt = build_thermal_prompt(effective_clip, variant, prompt, task_mode="clip")
        prompt_artifact = archive_thermal_prompt(
            batch_root,
            variant=variant,
            prompt=prompt,
            bucket=spec.bucket,
            clip=effective_clip,
            model=client.model,
            transport="direct_video",
            rendered_prompt=rendered_prompt,
        )
        running_record = {
            "status": "running",
            "bucket": spec.bucket,
            "clip_key": clip_key,
            "prompt_artifact": prompt_artifact,
            "media_path": str(media_path) if media_path is not None else None,
            "ground_truth": adapter.truth_payload(effective_clip),
            "started_at": now_iso(),
        }
        existing[clip_key] = running_record
        state["results"] = list(existing.values())
        state["current_clip"] = clip_key
        state["completed_count"] = sum(1 for item in state["results"] if item.get("status") == "completed")
        state["updated_at"] = now_iso()
        metrics = build_thermal_run_metrics(state, clip_key_index)
        progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        summary_json_path.write_text(json.dumps({**state, "metrics": metrics}, indent=2), encoding="utf-8")
        write_thermal_summary(summary_md_path, state, metrics)

        start_message = f"[{index}/{len(specs)}] start {clip_key} bucket={spec.bucket}"
        print(start_message, flush=True)
        append_log(log_path, f"{now_iso()} {start_message}")
        try:
            response = client.infer(effective_clip, variant, prompt)
            prediction = parse_thermal_prediction(
                response.text,
                domain=effective_clip.domain,
                clip_id=effective_clip.clip_id,
                prompt_id=prompt.prompt_id,
                latency_sec=response.latency_sec,
                usage_metadata=response.usage_metadata,
                estimated_cost_usd=response.estimated_cost_usd,
                model_name=client.model,
            )
            response_path = batch_root / "responses" / variant.value / f"{index - 1:02d}__{spec.clip_id}.response.json"
            response_path.parent.mkdir(parents=True, exist_ok=True)
            response_path.write_text(
                json.dumps(
                    {
                        "clip_key": clip_key,
                        "model": client.model,
                        "prompt_path": prompt_artifact["json_path"],
                        "response_text": response.text,
                        "prediction": thermal_prediction_payload(prediction),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            existing[clip_key] = {
                "status": "completed",
                "bucket": spec.bucket,
                "clip_key": clip_key,
                "prompt_artifact": prompt_artifact,
                "response_path": str(response_path),
                "media_path": str(media_path) if media_path is not None else None,
                "ground_truth": adapter.truth_payload(effective_clip),
                "prediction": thermal_prediction_payload(prediction),
                "selected_pass": "global",
                "started_at": running_record["started_at"],
                "completed_at": now_iso(),
            }
            finish_message = (
                f"[{index}/{len(specs)}] done {clip_key} "
                f"pred={prediction.coarse_label} fp={prediction.false_positive_score} latency={float(prediction.latency_sec or 0.0):.2f}s"
            )
            print(finish_message, flush=True)
            append_log(log_path, f"{now_iso()} {finish_message}")
        except Exception as exc:
            existing[clip_key] = {
                "status": "failed",
                "bucket": spec.bucket,
                "clip_key": clip_key,
                "prompt_artifact": prompt_artifact,
                "media_path": str(media_path) if media_path is not None else None,
                "ground_truth": adapter.truth_payload(effective_clip),
                "error": str(exc),
                "started_at": running_record["started_at"],
                "completed_at": now_iso(),
            }
            fail_message = f"[{index}/{len(specs)}] failed {clip_key} error={exc}"
            print(fail_message, flush=True)
            append_log(log_path, f"{now_iso()} {fail_message}")
            if args.stop_on_error:
                raise
        state["results"] = list(existing.values())
        state["current_clip"] = None
        state["completed_count"] = sum(1 for item in state["results"] if item.get("status") == "completed")
        state["updated_at"] = now_iso()
        metrics = build_thermal_run_metrics(state, clip_key_index)
        progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        summary_json_path.write_text(json.dumps({**state, "metrics": metrics}, indent=2), encoding="utf-8")
        write_thermal_summary(summary_md_path, state, metrics)

    print(f"[thermal_prompt_batch] {batch_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
