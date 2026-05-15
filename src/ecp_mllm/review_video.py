from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import textwrap
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class ReviewSource:
    clip_key: str
    video_path: Path
    prediction: dict[str, Any]
    ground_truth: dict[str, int] | None
    selected_pass: str | None
    title: str
    summary_path: Path | None = None


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in value).strip("_") or "review"


def _resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def _resolve_field(payload: dict[str, Any], field_path: str) -> Any:
    current: Any = payload
    for part in field_path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise KeyError(field_path)
        current = current[part]
    return current


def load_review_source(
    summary_json_path: str | Path,
    *,
    clip_key: str | None = None,
    prediction_field: str = "prediction",
    repo_root: str | Path | None = None,
) -> ReviewSource:
    summary_path = Path(summary_json_path)
    repo = Path(repo_root) if repo_root is not None else Path.cwd()
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    results = payload.get("results")
    if not isinstance(results, list) or not results:
        raise ValueError("summary json must contain a non-empty results list")

    if clip_key is None:
        if len(results) != 1:
            raise ValueError("clip_key is required when summary contains multiple results")
        record = results[0]
    else:
        matches = [item for item in results if str(item.get("clip_key")) == clip_key]
        if not matches:
            raise ValueError(f"clip_key not found in summary: {clip_key}")
        record = matches[0]

    media_path = record.get("media_path")
    if not isinstance(media_path, str) or not media_path:
        raise ValueError("summary record is missing media_path")

    prediction = _resolve_field(record, prediction_field)
    if not isinstance(prediction, dict):
        raise ValueError(f"prediction field must resolve to an object: {prediction_field}")

    resolved_clip_key = str(record.get("clip_key") or clip_key or "clip")
    return ReviewSource(
        clip_key=resolved_clip_key,
        video_path=_resolve_path(media_path, repo),
        prediction=prediction,
        ground_truth=record.get("ground_truth") if isinstance(record.get("ground_truth"), dict) else None,
        selected_pass=str(record.get("selected_pass")) if record.get("selected_pass") is not None else None,
        title=f"{payload.get('name', 'review')} | {prediction_field}",
        summary_path=summary_path,
    )


def default_review_output_path(source: ReviewSource, prediction_field: str, *, output_root: str | Path | None = None) -> Path:
    if output_root is None:
        root = source.summary_path.parent / "review_videos" if source.summary_path is not None else Path.cwd() / "outputs" / "review_videos"
    else:
        root = Path(output_root)
    return root / f"{_safe_name(source.clip_key)}.{_safe_name(prediction_field)}.review.mp4"


def _format_timestamp(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    total = max(0, int(round(float(seconds))))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _direction_color(direction: str | None) -> tuple[int, int, int]:
    normalized = str(direction or "").strip().lower()
    if normalized in {"right", "downstream"}:
        return (80, 205, 90)
    if normalized in {"left", "upstream"}:
        return (70, 140, 255)
    return (180, 180, 180)


def _clip_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _candidate_passages(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    passages = prediction.get("candidate_passages")
    if not isinstance(passages, list):
        return []
    normalized: list[dict[str, Any]] = []
    for item in passages:
        if not isinstance(item, dict):
            continue
        normalized.append(item)
    normalized.sort(key=lambda item: _clip_float(item.get("timestamp_start_sec"), 0.0) or 0.0)
    return normalized


def _events(prediction: dict[str, Any]) -> list[dict[str, Any]]:
    events = prediction.get("events")
    if not isinstance(events, list):
        return []
    normalized = [item for item in events if isinstance(item, dict)]
    normalized.sort(key=lambda item: _clip_float(item.get("timestamp_sec"), 0.0) or 0.0)
    return normalized


def _wrap_text_lines(text: str | None, width: int) -> list[str]:
    if not text:
        return []
    lines: list[str] = []
    for paragraph in str(text).splitlines():
        wrapped = textwrap.wrap(paragraph, width=max(20, width), break_long_words=False, replace_whitespace=False)
        lines.extend(wrapped or [""])
    return [line for line in lines if line]


def _draw_text_block(
    frame: np.ndarray,
    lines: list[str],
    origin: tuple[int, int],
    *,
    scale: float = 0.55,
    color: tuple[int, int, int] = (255, 255, 255),
    thickness: int = 1,
    line_gap: int = 8,
) -> None:
    x, y = origin
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_height = int(18 * scale / 0.55)
    for index, line in enumerate(lines):
        baseline_y = y + index * (line_height + line_gap)
        cv2.putText(frame, line, (x, baseline_y), font, scale, color, thickness, cv2.LINE_AA)


def _active_passage(passages: list[dict[str, Any]], timestamp_sec: float) -> dict[str, Any] | None:
    for item in passages:
        start = _clip_float(item.get("timestamp_start_sec"))
        end = _clip_float(item.get("timestamp_end_sec"))
        if start is None or end is None:
            continue
        if start <= timestamp_sec <= end:
            return item
    return None


def _active_events(events: list[dict[str, Any]], timestamp_sec: float, tolerance_sec: float = 1.5) -> list[dict[str, Any]]:
    active: list[dict[str, Any]] = []
    for item in events:
        event_time = _clip_float(item.get("timestamp_sec"))
        if event_time is None:
            continue
        if abs(event_time - timestamp_sec) <= tolerance_sec:
            active.append(item)
    return active


def render_review_video(
    source: ReviewSource,
    output_path: str | Path,
    *,
    prediction_field: str = "prediction",
) -> Path:
    capture = cv2.VideoCapture(str(source.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video: {source.video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 5.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source video dimensions: {source.video_path}")

    duration_seconds = frame_count / fps if fps > 0 else 0.0
    prediction = source.prediction
    passages = _candidate_passages(prediction)
    events = _events(prediction)
    left_count = int(prediction.get("left_count") or 0)
    right_count = int(prediction.get("right_count") or 0)
    confidence = prediction.get("confidence")
    commentary = prediction.get("commentary")
    evidence_summary = prediction.get("evidence_summary")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    top_pad = max(118, min(176, height // 4))
    bottom_pad = max(160, min(240, height // 3))
    canvas_width = width
    canvas_height = height + top_pad + bottom_pad
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (canvas_width, canvas_height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open output video for writing: {output}")

    timeline_left = 20
    timeline_right = canvas_width - 20
    timeline_width = max(40, timeline_right - timeline_left)
    video_y0 = top_pad
    video_y1 = top_pad + height
    bottom_panel_y0 = video_y1
    caption_y0 = bottom_panel_y0 + 22
    divider_y = bottom_panel_y0 + 98
    timeline_heading_y = divider_y + 18
    passage_label_y0 = divider_y + 36
    timeline_y0 = divider_y + 58
    timeline_y1 = timeline_y0 + 20

    gt = source.ground_truth or {}
    gt_left = gt.get("left_count")
    gt_right = gt.get("right_count")
    selected_pass = source.selected_pass or prediction_field
    static_scene_lines = _wrap_text_lines(prediction.get("scene_assessment"), width=max(28, canvas_width // 17))[:2]
    static_commentary_lines = _wrap_text_lines(commentary or evidence_summary, width=max(28, canvas_width // 18))[:2]
    title_lines = _wrap_text_lines(source.title, width=max(24, canvas_width // 20))[:1]
    clip_lines = _wrap_text_lines(source.clip_key, width=max(24, canvas_width // 18))[:2]

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_sec = frame_index / fps if fps > 0 else 0.0
        canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
        canvas[:] = (16, 16, 16)
        canvas[video_y0:video_y1, 0:width] = frame
        cv2.rectangle(canvas, (0, 0), (canvas_width - 1, top_pad - 1), (24, 24, 24), thickness=-1)
        cv2.rectangle(canvas, (0, video_y1), (canvas_width - 1, canvas_height - 1), (24, 24, 24), thickness=-1)
        cv2.line(canvas, (0, video_y0 - 1), (canvas_width - 1, video_y0 - 1), (58, 58, 58), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, video_y1), (canvas_width - 1, video_y1), (58, 58, 58), 1, cv2.LINE_AA)
        cv2.line(canvas, (0, divider_y), (canvas_width - 1, divider_y), (58, 58, 58), 1, cv2.LINE_AA)

        header_lines = [
            *title_lines,
            *clip_lines,
            f"pass={selected_pass} | t={_format_timestamp(timestamp_sec)} / {_format_timestamp(duration_seconds)}",
            f"Pred L/R={left_count}/{right_count}"
            + (f" | GT L/R={gt_left}/{gt_right}" if gt_left is not None and gt_right is not None else "")
            + (f" | conf={float(confidence):.2f}" if confidence is not None else ""),
        ]
        header_lines.extend(static_scene_lines[:1])
        _draw_text_block(canvas, header_lines[:5], (16, 24), scale=0.54, color=(245, 245, 245), thickness=1, line_gap=6)

        cv2.rectangle(canvas, (timeline_left, timeline_y0), (timeline_right, timeline_y1), (55, 55, 55), thickness=-1)
        for index, passage in enumerate(passages, start=1):
            start = _clip_float(passage.get("timestamp_start_sec"), 0.0) or 0.0
            end = _clip_float(passage.get("timestamp_end_sec"), start) or start
            if duration_seconds <= 0:
                continue
            x0 = timeline_left + int(max(0.0, min(1.0, start / duration_seconds)) * timeline_width)
            x1 = timeline_left + int(max(0.0, min(1.0, end / duration_seconds)) * timeline_width)
            lane_y0 = timeline_y0 + 2 + ((index - 1) % 2) * 10
            lane_y1 = min(timeline_y1 - 2, lane_y0 + 8)
            color = _direction_color(passage.get("direction"))
            cv2.rectangle(canvas, (x0, lane_y0), (max(x0 + 2, x1), lane_y1), color, thickness=-1)
            label = f"P{index}:{passage.get('direction', '?')}={passage.get('estimated_count', '?')}"
            if x1 - x0 > 64:
                label_y = passage_label_y0 + ((index - 1) % 2) * 12
                cv2.putText(canvas, label, (x0 + 3, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

        for event in events:
            event_time = _clip_float(event.get("timestamp_sec"))
            if event_time is None or duration_seconds <= 0:
                continue
            x = timeline_left + int(max(0.0, min(1.0, event_time / duration_seconds)) * timeline_width)
            cv2.line(canvas, (x, timeline_y0 - 10), (x, timeline_y1 + 10), _direction_color(event.get("direction")), 1, cv2.LINE_AA)

        marker_x = timeline_left + int(max(0.0, min(1.0, (timestamp_sec / duration_seconds) if duration_seconds > 0 else 0.0)) * timeline_width)
        cv2.line(canvas, (marker_x, timeline_y0 - 14), (marker_x, timeline_y1 + 14), (255, 255, 255), 2, cv2.LINE_AA)

        active_passage = _active_passage(passages, timestamp_sec)
        active_events = _active_events(events, timestamp_sec)
        caption_lines: list[str] = []
        if active_passage is not None:
            caption_lines.append(
                "Active passage: "
                f"{str(active_passage.get('direction', 'uncertain')).upper()} "
                f"est={active_passage.get('estimated_count', '?')} "
                f"{_format_timestamp(_clip_float(active_passage.get('timestamp_start_sec')))}-"
                f"{_format_timestamp(_clip_float(active_passage.get('timestamp_end_sec')))}"
            )
            metric_parts: list[str] = []
            if active_passage.get("peak_simultaneous_count") is not None:
                metric_parts.append(f"peak={active_passage.get('peak_simultaneous_count')}")
            if active_passage.get("wave_count") is not None:
                metric_parts.append(f"waves={active_passage.get('wave_count')}")
            if active_passage.get("throughput_best_count") is not None:
                metric_parts.append(f"throughput={active_passage.get('throughput_best_count')}")
            if active_passage.get("episode_duration_sec") is not None:
                metric_parts.append(f"dur={active_passage.get('episode_duration_sec')}s")
            if metric_parts:
                caption_lines.append("Passage metrics: " + " | ".join(metric_parts))
            caption_lines.extend(
                _wrap_text_lines(
                    str(active_passage.get("evidence_note") or ""),
                    width=max(34, canvas_width // 16),
                )[:2]
            )
        elif static_commentary_lines:
            caption_lines.append("No active passage window")
            caption_lines.extend(static_commentary_lines[:2])

        if active_events:
            event_labels = []
            for event in active_events[:2]:
                event_labels.append(
                    f"{_format_timestamp(_clip_float(event.get('timestamp_sec')))} "
                    f"{str(event.get('direction', 'uncertain')).upper()}: "
                    f"{str(event.get('evidence_note') or '').strip()}"
                )
            caption_lines.append("Event cue: " + " | ".join(event_labels))

        caption_lines = caption_lines[:4]
        _draw_text_block(canvas, caption_lines, (16, caption_y0), scale=0.56, color=(245, 245, 245), thickness=1, line_gap=7)
        cv2.putText(
            canvas,
            "Timeline: candidate passage windows and event ticks",
            (timeline_left, timeline_heading_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (225, 225, 225),
            1,
            cv2.LINE_AA,
        )

        writer.write(canvas)
        frame_index += 1

    capture.release()
    writer.release()
    return output
