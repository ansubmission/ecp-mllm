from __future__ import annotations

import argparse
from ..agent.undercount_recount import build_recount_prompt, detect_undercount_risk, should_accept_recount
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..config import load_local_settings
from ..data.cfc_adapter import CFCAdapter
from ..eval.counting import count_tracks_like_cfc, read_mot_tracks
from ..qwen.parsing import parse_passage_prediction
from ..qwen.prompt_archive import write_prompt_artifact
from ..qwen.prompting import build_prompt
from ..types import InputVariant, PromptRevision
from .probe_clip import _build_client, _materialize_clip_transport, _prepare_sff3c_variant


@dataclass(frozen=True)
class BatchClipSpec:
    bucket: str
    clip_id: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/local.toml")
    parser.add_argument("--domain", default="kenai-val")
    parser.add_argument("--variant", default="sff3c")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--transport", choices=["sampled_frames", "stitched_mp4"], default="stitched_mp4")
    parser.add_argument("--stitch-fps", type=float, default=5.0)
    parser.add_argument("--stitch-height", type=int, default=0)
    parser.add_argument("--stitch-crf", type=int, default=30)
    parser.add_argument("--clips-path", required=True)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--prompt-id", default="episode_sum_batch")
    parser.add_argument("--name", required=True)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--disable-recount", action="store_true")
    return parser.parse_args()


def load_batch_specs(path: str | Path) -> list[BatchClipSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("batch clip file must decode to a list")
    specs: list[BatchClipSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each batch clip entry must be an object")
        bucket = str(item["bucket"]).strip()
        clip_id = str(item["clip_id"]).strip()
        if not bucket or not clip_id:
            raise ValueError("bucket and clip_id must be non-empty")
        specs.append(BatchClipSpec(bucket=bucket, clip_id=clip_id))
    return specs


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-") or "batch"


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _format_stage_details(details: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in details.items():
        if value is None or value == "":
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def _ground_truth_counts(adapter: CFCAdapter, clip) -> dict[str, int] | None:
    if clip.dataset != "cfc" or clip.width is None or clip.height is None:
        return None
    gt_path = adapter.gt_path(clip.domain, clip.clip_id)
    if not gt_path.exists():
        return None
    tracks = read_mot_tracks(gt_path)
    counts = count_tracks_like_cfc(tracks, clip.width, clip.height, filter_dist=0.05)
    return {"left_count": counts.left, "right_count": counts.right, "total_count": counts.left + counts.right}


def write_batch_summary(path: str | Path, payload: dict[str, Any]) -> None:
    def _fmt_optional_float(value: object, digits: int) -> str:
        if value in ("", None):
            return ""
        if isinstance(value, (int, float)):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines = [
        f"# {payload['name']}",
        "",
        f"- domain: `{payload['domain']}`",
        f"- variant: `{payload['variant']}`",
        f"- model: `{payload['model']}`",
        f"- transport: `{payload['transport']}`",
        f"- completed: `{payload['completed_count']}/{payload['total_clips']}`",
        "",
        "| Status | Bucket | Clip | GT Right | Pred Right | Error | Latency | Cost USD | Pass | Media | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in payload["results"]:
        gt_right = item.get("ground_truth", {}).get("right_count", "")
        pred_right = item.get("prediction", {}).get("right_count", "") if item.get("prediction") else ""
        error = item.get("total_abs_error", "")
        latency = item.get("prediction", {}).get("latency_sec", "") if item.get("prediction") else ""
        cost = item.get("prediction", {}).get("estimated_cost_usd", "") if item.get("prediction") else ""
        selected_pass = item.get("selected_pass", "")
        media = Path(item["media_path"]).name if item.get("media_path") else ""
        note = item.get("error") or item.get("prediction", {}).get("commentary", "") if item.get("prediction") else item.get("error", "")
        lines.append(
            f"| {item['status']} | {item['bucket']} | `{item['clip_key']}` | {gt_right} | {pred_right} | {error} | {_fmt_optional_float(latency, 2)} | {_fmt_optional_float(cost, 4)} | {selected_pass} | `{media}` | {str(note).replace('|', '/')} |"
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _result_index(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(item["clip_key"]): item for item in results}


def _prediction_payload(prediction, *, latency_override: float | None = None) -> dict[str, Any]:
    return {
        "scene_assessment": prediction.scene_assessment,
        "candidate_passages": prediction.candidate_passages,
        "rejected_targets": prediction.rejected_targets,
        "left_count": prediction.left_count,
        "right_count": prediction.right_count,
        "total_count": (prediction.left_count or 0) + (prediction.right_count or 0),
        "confidence": prediction.confidence,
        "commentary": prediction.commentary,
        "evidence_summary": prediction.evidence_summary,
        "events": [asdict(event) for event in prediction.events],
        "latency_sec": latency_override if latency_override is not None else prediction.latency_sec,
        "model_latency_sec": prediction.latency_sec,
        "estimated_cost_usd": prediction.estimated_cost_usd,
        "usage_metadata": prediction.usage_metadata,
        "model_name": prediction.model_name,
        "parse_success": prediction.parse_success,
        "prompt_id": prediction.prompt_id,
    }


def main() -> int:
    args = _parse_args()
    settings = load_local_settings(args.settings)
    adapter = CFCAdapter(settings.paths)
    clips = {clip.clip_id: clip for clip in adapter.load_clip_records([args.domain])}
    specs = load_batch_specs(args.clips_path)
    if args.max_clips is not None:
        specs = specs[: args.max_clips]

    variant = InputVariant.from_value(args.variant)
    client = _build_client(settings, args.model, args.max_frames)
    prompt_text = args.prompt_text or (
        "Analyze this sonar clip for fish passage. "
        "Identify each candidate passage episode, estimate how many distinct fish traverse during that episode, "
        "and then sum across episodes. "
        "Do not anchor only on the clearest 2 to 3 simultaneous tracks if the same school continues streaming "
        "through the corridor for many seconds. "
        "If a long stream contains multiple waves or mini-bursts over time, split them into separate passage episodes instead of one broad school summary. "
        "Use best-effort throughput estimates for high-passage situations, and estimate total distinct fish passage, not instantaneous occupancy. "
        "Return JSON only."
    )

    batch_root = settings.paths.output_root / "prompt_dev" / _safe_name(args.name)
    batch_root.mkdir(parents=True, exist_ok=True)
    progress_path = batch_root / "progress.json"
    summary_json_path = batch_root / "summary.json"
    summary_md_path = batch_root / "summary.md"
    log_path = batch_root / "progress.log"

    state: dict[str, Any]
    if not args.no_resume and progress_path.exists():
        state = json.loads(progress_path.read_text(encoding="utf-8"))
    else:
        state = {
            "name": args.name,
            "domain": args.domain,
            "variant": variant.value,
            "model": client.model,
            "transport": args.transport,
            "agentic_recount_enabled": not args.disable_recount,
            "prompt_id": args.prompt_id,
            "prompt_text": prompt_text,
            "total_clips": len(specs),
            "completed_count": 0,
            "current_clip": None,
            "results": [],
            "updated_at": _now_iso(),
        }

    existing = _result_index(state.get("results", []))
    for index, spec in enumerate(specs, start=1):
        if spec.clip_id not in clips:
            raise RuntimeError(f"Clip not found in {args.domain}: {spec.clip_id}")
        clip_key = f"{args.domain}/{spec.clip_id}"
        if clip_key in existing and existing[clip_key].get("status") == "completed":
            message = f"[{index}/{len(specs)}] skip completed {clip_key}"
            print(message, flush=True)
            _append_log(log_path, f"{_now_iso()} {message}")
            continue

        clip = clips[spec.clip_id]
        gt = _ground_truth_counts(adapter, clip)
        if variant == InputVariant.SFF3C:
            clip = _prepare_sff3c_variant(settings, clip, args.transport, args.max_frames)
        materialize_args = argparse.Namespace(
            transport=args.transport,
            stitch_fps=args.stitch_fps,
            stitch_height=args.stitch_height,
            stitch_crf=args.stitch_crf,
        )
        effective_clip, video_path = _materialize_clip_transport(materialize_args, settings, clip, variant)
        prompt = PromptRevision(version=0, prompt_id=args.prompt_id, prompt_text=prompt_text)
        rendered_prompt = build_prompt(effective_clip, variant, prompt)
        prompt_artifact = write_prompt_artifact(
            output_root=batch_root,
            scope="batch",
            variant=variant.value,
            prompt_id=prompt.prompt_id,
            version=prompt.version,
            prompt_text=prompt.prompt_text,
            critique=prompt.critique,
            metrics=dict(prompt.metrics),
            rendered_prompt=rendered_prompt,
            metadata={
                "bucket": spec.bucket,
                "domain": effective_clip.domain,
                "clip_id": effective_clip.clip_id,
                "model": client.model,
                "transport": args.transport,
                "media_path": str(video_path) if video_path is not None else None,
            },
        )

        running_record = {
            "status": "running",
            "bucket": spec.bucket,
            "clip_key": clip_key,
            "prompt_artifact": prompt_artifact,
            "media_path": str(video_path) if video_path is not None else None,
            "ground_truth": gt,
            "started_at": _now_iso(),
            "stage": {"phase": "queued"},
        }
        existing[clip_key] = running_record
        state["results"] = list(existing.values())
        state["current_clip"] = clip_key
        state["current_stage"] = running_record["stage"]
        state["completed_count"] = sum(1 for item in state["results"] if item.get("status") == "completed")
        state["updated_at"] = _now_iso()
        progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        summary_json_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        write_batch_summary(summary_md_path, state)

        start_message = f"[{index}/{len(specs)}] start {clip_key} bucket={spec.bucket}"
        if video_path is not None:
            start_message = f"{start_message} media={video_path}"
        print(start_message, flush=True)
        _append_log(log_path, f"{_now_iso()} {start_message}")

        def _progress_callback(stage: str, details: dict[str, object]) -> None:
            detail_text = _format_stage_details(details)
            log_message = f"{_now_iso()} [{index}/{len(specs)}] stage {clip_key} phase={stage}"
            if detail_text:
                log_message = f"{log_message} {detail_text}"
            print(log_message, flush=True)
            _append_log(log_path, log_message)
            running = existing.get(clip_key)
            if running is None:
                return
            running["stage"] = {"phase": stage, **details, "updated_at": _now_iso()}
            state["results"] = list(existing.values())
            state["current_clip"] = clip_key
            state["current_stage"] = running["stage"]
            state["updated_at"] = _now_iso()
            progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            summary_json_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            write_batch_summary(summary_md_path, state)

        if hasattr(client, "set_progress_callback"):
            client.set_progress_callback(_progress_callback)

        try:
            response = client.infer(effective_clip, variant, prompt)
            first_prediction = parse_passage_prediction(
                raw_text=response.text,
                domain=effective_clip.domain,
                clip_id=effective_clip.clip_id,
                prompt_id=prompt.prompt_id,
                latency_sec=response.latency_sec,
                usage_metadata=response.usage_metadata,
                estimated_cost_usd=response.estimated_cost_usd,
                model_name=client.model,
            )
            response_path = batch_root / "responses" / variant.value / f"{index - 1:02d}__{spec.clip_id}.response.json"
            _write_text(response_path, response.text)
            selected_prediction = first_prediction
            selected_pass = "base"
            total_latency = float(first_prediction.latency_sec or 0.0)
            recount_record: dict[str, Any] | None = None

            if not args.disable_recount:
                recount_decision = detect_undercount_risk(first_prediction)
                if recount_decision.should_recount:
                    _progress_callback(
                        "recount_triggered",
                        {
                            "reason": recount_decision.reason,
                            "first_total": recount_decision.first_total,
                            "dominant_direction": recount_decision.dominant_direction,
                            "dominant_episode_count": recount_decision.dominant_episode_count,
                            "dominant_episode_duration_sec": recount_decision.dominant_episode_duration_sec,
                        },
                    )
                    recount_prompt = build_recount_prompt(prompt, first_prediction)
                    recount_rendered_prompt = build_prompt(effective_clip, variant, recount_prompt)
                    recount_artifact = write_prompt_artifact(
                        output_root=batch_root,
                        scope="batch",
                        variant=variant.value,
                        prompt_id=recount_prompt.prompt_id,
                        version=recount_prompt.version,
                        prompt_text=recount_prompt.prompt_text,
                        critique=recount_prompt.critique,
                        metrics=dict(recount_prompt.metrics),
                        rendered_prompt=recount_rendered_prompt,
                        metadata={
                            "bucket": spec.bucket,
                            "domain": effective_clip.domain,
                            "clip_id": effective_clip.clip_id,
                            "model": client.model,
                            "transport": args.transport,
                            "media_path": str(video_path) if video_path is not None else None,
                            "stage": "recount",
                        },
                    )
                    try:
                        recount_response = client.infer(effective_clip, variant, recount_prompt)
                        recount_prediction = parse_passage_prediction(
                            raw_text=recount_response.text,
                            domain=effective_clip.domain,
                            clip_id=effective_clip.clip_id,
                            prompt_id=recount_prompt.prompt_id,
                            latency_sec=recount_response.latency_sec,
                            usage_metadata=recount_response.usage_metadata,
                            estimated_cost_usd=recount_response.estimated_cost_usd,
                            model_name=client.model,
                        )
                        recount_response_path = (
                            batch_root / "responses" / variant.value / f"{index - 1:02d}__{spec.clip_id}.recount.response.json"
                        )
                        _write_text(recount_response_path, recount_response.text)
                        total_latency += float(recount_prediction.latency_sec or 0.0)
                        accept_recount, accept_reason = should_accept_recount(first_prediction, recount_prediction)
                        if accept_recount:
                            selected_prediction = recount_prediction
                            selected_pass = "recount"
                            _progress_callback(
                                "recount_selected",
                                {
                                    "accepted_reason": accept_reason,
                                    "base_total": (first_prediction.left_count or 0) + (first_prediction.right_count or 0),
                                    "recount_total": (recount_prediction.left_count or 0) + (recount_prediction.right_count or 0),
                                },
                            )
                        else:
                            _progress_callback(
                                "recount_rejected",
                                {
                                    "accepted_reason": accept_reason,
                                    "base_total": (first_prediction.left_count or 0) + (first_prediction.right_count or 0),
                                    "recount_total": (recount_prediction.left_count or 0) + (recount_prediction.right_count or 0),
                                },
                            )
                        recount_record = {
                            "applied": True,
                            "decision": asdict(recount_decision),
                            "prompt_artifact": recount_artifact,
                            "response_path": str(recount_response_path),
                            "prediction": _prediction_payload(recount_prediction),
                            "accepted": accept_recount,
                            "accepted_reason": accept_reason,
                        }
                    except Exception as recount_exc:
                        recount_record = {
                            "applied": True,
                            "decision": asdict(recount_decision),
                            "prompt_artifact": recount_artifact,
                            "error": str(recount_exc),
                            "accepted": False,
                            "accepted_reason": "recount_exception",
                        }
                        _progress_callback("recount_failed", {"error": str(recount_exc)})
                else:
                    recount_record = {
                        "applied": False,
                        "decision": asdict(recount_decision),
                    }

            completed_record = {
                "status": "completed",
                "bucket": spec.bucket,
                "clip_key": clip_key,
                "prompt_artifact": prompt_artifact,
                "response_path": str(response_path),
                "media_path": str(video_path) if video_path is not None else None,
                "ground_truth": gt,
                "first_prediction": _prediction_payload(first_prediction),
                "prediction": _prediction_payload(selected_prediction, latency_override=total_latency),
                "selected_pass": selected_pass,
                "recount": recount_record,
                "started_at": running_record["started_at"],
                "completed_at": _now_iso(),
            }
            if gt is not None:
                completed_record["abs_error_left"] = abs((selected_prediction.left_count or 0) - gt["left_count"])
                completed_record["abs_error_right"] = abs((selected_prediction.right_count or 0) - gt["right_count"])
                completed_record["total_abs_error"] = completed_record["abs_error_left"] + completed_record["abs_error_right"]
            existing[clip_key] = completed_record
            finish_message = (
                f"[{index}/{len(specs)}] done {clip_key} pass={selected_pass} "
                f"pred=({selected_prediction.left_count},{selected_prediction.right_count}) "
                f"latency={total_latency:.2f}s"
            )
            print(finish_message, flush=True)
            _append_log(log_path, f"{_now_iso()} {finish_message}")
        except Exception as exc:
            failed_record = {
                "status": "failed",
                "bucket": spec.bucket,
                "clip_key": clip_key,
                "prompt_artifact": prompt_artifact,
                "media_path": str(video_path) if video_path is not None else None,
                "ground_truth": gt,
                "error": str(exc),
                "started_at": running_record["started_at"],
                "completed_at": _now_iso(),
            }
            existing[clip_key] = failed_record
            fail_message = f"[{index}/{len(specs)}] failed {clip_key} error={exc}"
            print(fail_message, flush=True)
            _append_log(log_path, f"{_now_iso()} {fail_message}")
            state["results"] = list(existing.values())
            state["current_clip"] = None
            state["current_stage"] = None
            state["completed_count"] = sum(1 for item in state["results"] if item.get("status") == "completed")
            state["updated_at"] = _now_iso()
            progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            summary_json_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            write_batch_summary(summary_md_path, state)
            if args.stop_on_error:
                raise
            continue
        finally:
            if hasattr(client, "set_progress_callback"):
                client.set_progress_callback(None)

        state["results"] = list(existing.values())
        state["current_clip"] = None
        state["current_stage"] = None
        state["completed_count"] = sum(1 for item in state["results"] if item.get("status") == "completed")
        state["updated_at"] = _now_iso()
        progress_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        summary_json_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        write_batch_summary(summary_md_path, state)

    print(f"[prompt_batch] {batch_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
