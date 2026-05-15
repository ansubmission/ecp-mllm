from __future__ import annotations

import argparse
import json

from ..config import load_local_settings
from ..data.nz_thermal_adapter import NzThermalAdapter
from ..qwen.thermal_parsing import parse_thermal_prediction
from ..qwen.thermal_prompting import build_thermal_prompt
from ..types import ClipRecord, InputVariant, PromptRevision
from .probe_clip import _build_client
from .thermal_common import (
    append_log,
    archive_thermal_prompt,
    aggregate_thermal_predictions,
    build_thermal_run_metrics,
    derive_thermal_proposals,
    materialize_thermal_variant,
    materialize_thermal_window,
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
    parser.add_argument("--prompt-id", default="thermal_event_agent_global")
    parser.add_argument("--local-prompt-id", default="thermal_event_agent_local")
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--include-calibration", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--stitch-height", type=int, default=360)
    parser.add_argument("--stitch-crf", type=int, default=30)
    parser.add_argument("--max-proposals", type=int, default=4)
    parser.add_argument("--proposal-min-duration-sec", type=float, default=2.0)
    parser.add_argument("--proposal-overlap-ratio", type=float, default=0.25)
    parser.add_argument("--proposal-source", choices=["raw_video", "track_oracle"], default="raw_video")
    parser.add_argument("--zone-overlay-center", action="store_true")
    return parser.parse_args()


def _window_clip(base_clip: ClipRecord, variant: InputVariant, media_path, proposal_id: str, start_sec: float, end_sec: float) -> ClipRecord:
    asset_paths = dict(base_clip.asset_paths)
    asset_paths[variant] = media_path
    metadata = dict(base_clip.metadata)
    metadata["window_context"] = {
        "proposal_id": proposal_id,
        "start_sec": start_sec,
        "end_sec": end_sec,
    }
    return ClipRecord(
        dataset=base_clip.dataset,
        domain=base_clip.domain,
        clip_id=base_clip.clip_id,
        asset_paths=asset_paths,
        width=base_clip.width,
        height=base_clip.height,
        framerate=base_clip.framerate,
        duration_seconds=max(0.05, end_sec - start_sec),
        metadata=metadata,
    )


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
        "Analyze this thermal wildlife clip as an event-centric perception task. "
        "Use the full clip to judge whether it is a false positive or an animal event, then use local windows to verify species evidence and consolidate a single clip-level answer. "
        "Return JSON only."
    )
    batch_root = settings.paths.output_root / "thermal_event_agent" / safe_name(args.name)
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
            "proposal_source": args.proposal_source,
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
        global_prompt = PromptRevision(version=0, prompt_id=args.prompt_id, prompt_text=prompt_text)
        global_rendered = build_thermal_prompt(
            effective_clip,
            variant,
            global_prompt,
            task_mode="clip",
            proposal_source=args.proposal_source,
        )
        global_artifact = archive_thermal_prompt(
            batch_root,
            variant=variant,
            prompt=global_prompt,
            bucket=spec.bucket,
            clip=effective_clip,
            model=client.model,
            transport="direct_video",
            rendered_prompt=global_rendered,
            metadata={"stage": "global", "proposal_source": args.proposal_source},
        )
        running_record = {
            "status": "running",
            "bucket": spec.bucket,
            "clip_key": clip_key,
            "prompt_artifact": global_artifact,
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

        start_message = f"[{index}/{len(specs)}] start {clip_key} bucket={spec.bucket} proposal_source={args.proposal_source}"
        print(start_message, flush=True)
        append_log(log_path, f"{now_iso()} {start_message}")

        try:
            global_response = client.infer(effective_clip, variant, global_prompt)
            global_prediction = parse_thermal_prediction(
                global_response.text,
                domain=effective_clip.domain,
                clip_id=effective_clip.clip_id,
                prompt_id=global_prompt.prompt_id,
                latency_sec=global_response.latency_sec,
                usage_metadata=global_response.usage_metadata,
                estimated_cost_usd=global_response.estimated_cost_usd,
                model_name=client.model,
            )
            proposals = derive_thermal_proposals(
                effective_clip,
                proposal_source=args.proposal_source,
                max_proposals=args.max_proposals,
                min_duration_sec=args.proposal_min_duration_sec,
                overlap_ratio=args.proposal_overlap_ratio,
            )
            local_records = []
            local_failures = []
            for proposal in proposals:
                window_media = materialize_thermal_window(
                    settings,
                    effective_clip,
                    variant,
                    start_sec=float(proposal["start_sec"]),
                    end_sec=float(proposal["end_sec"]),
                    label=str(proposal["proposal_id"]),
                    crf=args.stitch_crf,
                    zone_overlay=False,
                )
                local_clip = _window_clip(
                    effective_clip,
                    variant,
                    window_media,
                    str(proposal["proposal_id"]),
                    float(proposal["start_sec"]),
                    float(proposal["end_sec"]),
                )
                local_prompt = PromptRevision(version=0, prompt_id=args.local_prompt_id, prompt_text=prompt_text)
                local_rendered = build_thermal_prompt(
                    local_clip,
                    variant,
                    local_prompt,
                    task_mode="window",
                    proposal_source=args.proposal_source,
                    window_hint=(float(proposal["start_sec"]), float(proposal["end_sec"])),
                )
                local_artifact = archive_thermal_prompt(
                    batch_root,
                    variant=variant,
                    prompt=local_prompt,
                    bucket=spec.bucket,
                    clip=local_clip,
                    model=client.model,
                    transport="trimmed_window",
                    rendered_prompt=local_rendered,
                    metadata={
                        "stage": "local_window",
                        "proposal_id": proposal["proposal_id"],
                        "proposal_source": args.proposal_source,
                    },
                )
                try:
                    local_response = client.infer(local_clip, variant, local_prompt)
                    local_prediction = parse_thermal_prediction(
                        local_response.text,
                        domain=local_clip.domain,
                        clip_id=local_clip.clip_id,
                        prompt_id=local_prompt.prompt_id,
                        latency_sec=local_response.latency_sec,
                        usage_metadata=local_response.usage_metadata,
                        estimated_cost_usd=local_response.estimated_cost_usd,
                        model_name=client.model,
                    )
                    local_records.append(
                        {
                            "proposal": proposal,
                            "media_path": str(window_media),
                            "prompt_artifact": local_artifact,
                            "prediction": local_prediction,
                        }
                    )
                except Exception as exc:
                    local_failures.append(
                        {
                            "proposal": dict(proposal),
                            "media_path": str(window_media),
                            "prompt_artifact": local_artifact,
                            "error": str(exc),
                        }
                    )
                    append_log(
                        log_path,
                        f"{now_iso()} local-failed {clip_key} proposal={proposal['proposal_id']} error={exc}",
                    )

            final_prediction = aggregate_thermal_predictions(effective_clip, global_prediction, local_records)
            response_dir = batch_root / "responses" / variant.value
            response_dir.mkdir(parents=True, exist_ok=True)
            response_path = response_dir / f"{index - 1:02d}__{spec.clip_id}.response.json"
            response_path.write_text(
                json.dumps(
                    {
                        "clip_key": clip_key,
                        "model": client.model,
                        "global_prompt_path": global_artifact["json_path"],
                        "global_response_text": global_response.text,
                        "global_prediction": thermal_prediction_payload(global_prediction),
                        "local_predictions": [
                            {
                                "proposal": dict(item["proposal"]),
                                "media_path": item["media_path"],
                                "prompt_path": item["prompt_artifact"]["json_path"],
                                "prediction": thermal_prediction_payload(item["prediction"]),
                            }
                            for item in local_records
                        ],
                        "local_failures": local_failures,
                        "final_prediction": thermal_prediction_payload(final_prediction),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            existing[clip_key] = {
                "status": "completed",
                "bucket": spec.bucket,
                "clip_key": clip_key,
                "prompt_artifact": global_artifact,
                "response_path": str(response_path),
                "media_path": str(media_path) if media_path is not None else None,
                "ground_truth": adapter.truth_payload(effective_clip),
                "prediction": thermal_prediction_payload(final_prediction),
                "global_prediction": thermal_prediction_payload(global_prediction),
                "local_predictions": [
                    {
                        "proposal": dict(item["proposal"]),
                        "media_path": item["media_path"],
                        "prompt_artifact": item["prompt_artifact"],
                        "prediction": thermal_prediction_payload(item["prediction"]),
                    }
                    for item in local_records
                ],
                "local_failures": local_failures,
                "selected_pass": "event_agent",
                "started_at": running_record["started_at"],
                "completed_at": now_iso(),
            }
            finish_message = (
                f"[{index}/{len(specs)}] done {clip_key} pred={final_prediction.coarse_label} "
                f"fp={final_prediction.false_positive_score} proposals={len(proposals)} local_ok={len(local_records)} local_failed={len(local_failures)} latency={float(final_prediction.latency_sec or 0.0):.2f}s"
            )
            print(finish_message, flush=True)
            append_log(log_path, f"{now_iso()} {finish_message}")
        except Exception as exc:
            existing[clip_key] = {
                "status": "failed",
                "bucket": spec.bucket,
                "clip_key": clip_key,
                "prompt_artifact": global_artifact,
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

    print(f"[thermal_event_agent_batch] {batch_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
