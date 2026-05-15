from __future__ import annotations

import hashlib
from pathlib import Path
import subprocess
import tempfile
from typing import Any

import cv2
import numpy as np


def _quote_concat_path(path: Path) -> str:
    value = str(path.resolve()).replace("\\", "/").replace("'", "'\\''")
    return f"file '{value}'"


def stitch_frames_to_mp4(
    frame_paths: list[Path],
    output_path: str | Path,
    fps: float = 10.0,
    target_height: int | None = 720,
    crf: int = 30,
) -> Path:
    if not frame_paths:
        raise ValueError("frame_paths must not be empty")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        concat_path = Path(temp_dir) / "frames.txt"
        concat_path.write_text("\n".join(_quote_concat_path(path) for path in frame_paths), encoding="utf-8")
        if target_height is None or target_height <= 0:
            vf = "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos,format=yuv420p"
        else:
            vf = f"scale=-2:{target_height}:flags=lanczos,format=yuv420p"
        command = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-r",
            f"{fps}",
            "-i",
            str(concat_path),
            "-an",
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-movflags",
            "+faststart",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stitch failed: {result.stderr.strip()}")
    return output


def trim_mp4(
    input_path: str | Path,
    output_path: str | Path,
    *,
    start_sec: float,
    end_sec: float,
    crf: int = 30,
) -> Path:
    start = max(0.0, float(start_sec))
    end = max(start + 0.05, float(end_sec))
    duration = end - start
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(Path(input_path)),
        "-t",
        f"{duration:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg trim failed: {result.stderr.strip()}")
    return output


def stitched_probe_mp4_path(
    output_root: str | Path,
    clip_key: str,
    variant: str,
    fps: float,
    target_height: int | None,
    crf: int,
) -> Path:
    normalized_height = 0 if target_height is None else target_height
    digest = hashlib.sha256(f"{clip_key}|{variant}|{fps}|{normalized_height}|{crf}".encode("utf-8")).hexdigest()[:12]
    safe_key = clip_key.replace("/", "__")
    return Path(output_root) / "probes" / "media" / f"{safe_key}.{digest}.mp4"


def trimmed_probe_mp4_path(
    output_root: str | Path,
    clip_key: str,
    variant: str,
    start_sec: float,
    end_sec: float,
    *,
    label: str = "window",
    crf: int = 30,
) -> Path:
    digest = hashlib.sha256(
        f"{clip_key}|{variant}|{label}|{start_sec:.3f}|{end_sec:.3f}|{crf}".encode("utf-8")
    ).hexdigest()[:12]
    safe_key = clip_key.replace("/", "__")
    return Path(output_root) / "probes" / "media" / f"{safe_key}.{label}.{digest}.mp4"


def dual_probe_mp4_path(
    output_root: str | Path,
    clip_key: str,
    left_variant: str,
    right_variant: str,
    target_height: int | None,
    crf: int,
) -> Path:
    normalized_height = 0 if target_height is None else target_height
    digest = hashlib.sha256(
        f"{clip_key}|dual|{left_variant}|{right_variant}|{normalized_height}|{crf}".encode("utf-8")
    ).hexdigest()[:12]
    safe_key = clip_key.replace("/", "__")
    return Path(output_root) / "probes" / "media" / f"{safe_key}.dual.{digest}.mp4"


def overlay_probe_mp4_path(
    output_root: str | Path,
    clip_key: str,
    variant: str,
    overlay_key: str,
    *,
    start_sec: float | None = None,
    end_sec: float | None = None,
    crf: int = 30,
) -> Path:
    timing = "" if start_sec is None or end_sec is None else f"|{start_sec:.3f}|{end_sec:.3f}"
    digest = hashlib.sha256(
        f"{clip_key}|{variant}|overlay|{overlay_key}{timing}|{crf}".encode("utf-8")
    ).hexdigest()[:12]
    safe_key = clip_key.replace("/", "__")
    return Path(output_root) / "probes" / "media" / f"{safe_key}.{overlay_key}.{digest}.mp4"


def compose_side_by_side_mp4(
    left_input_path: str | Path,
    right_input_path: str | Path,
    output_path: str | Path,
    *,
    target_height: int | None = 360,
    crf: int = 30,
) -> Path:
    left = Path(left_input_path)
    right = Path(right_input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    scale_filter = (
        f"scale=-2:{target_height}:flags=lanczos" if target_height is not None and target_height > 0 else "scale=trunc(iw/2)*2:trunc(ih/2)*2:flags=lanczos"
    )
    filter_graph = (
        f"[0:v]{scale_filter},setpts=PTS-STARTPTS[left];"
        f"[1:v]{scale_filter},setpts=PTS-STARTPTS[right];"
        "[left][right]hstack=inputs=2,format=yuv420p[v]"
    )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(left),
        "-i",
        str(right),
        "-an",
        "-filter_complex",
        filter_graph,
        "-map",
        "[v]",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg side-by-side failed: {result.stderr.strip()}")
    return output


def overlay_center_zone_mp4(
    input_path: str | Path,
    output_path: str | Path,
    *,
    zone_lo: float = 0.3,
    zone_hi: float = 0.7,
    dual_layout: bool = False,
    crf: int = 30,
    color: str = "lime@0.9",
    thickness: int = 4,
) -> Path:
    input_file = Path(input_path)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    zone_lo = max(0.0, min(1.0, float(zone_lo)))
    zone_hi = max(zone_lo, min(1.0, float(zone_hi)))
    if dual_layout:
        pane_width = 0.5
        panel_lo = pane_width * zone_lo
        panel_span = pane_width * max(0.0, zone_hi - zone_lo)
        first_x = panel_lo
        second_x = pane_width + panel_lo
        drawbox = (
            f"drawbox=x=iw*{first_x:.6f}:y=ih*{zone_lo:.6f}:w=iw*{panel_span:.6f}:h=ih*{(zone_hi - zone_lo):.6f}:"
            f"color={color}:t={int(thickness)},"
            f"drawbox=x=iw*{second_x:.6f}:y=ih*{zone_lo:.6f}:w=iw*{panel_span:.6f}:h=ih*{(zone_hi - zone_lo):.6f}:"
            f"color={color}:t={int(thickness)},format=yuv420p"
        )
    else:
        drawbox = (
            f"drawbox=x=iw*{zone_lo:.6f}:y=ih*{zone_lo:.6f}:w=iw*{(zone_hi - zone_lo):.6f}:h=ih*{(zone_hi - zone_lo):.6f}:"
            f"color={color}:t={int(thickness)},format=yuv420p"
        )
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_file),
        "-an",
        "-vf",
        drawbox,
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(output),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg center-zone overlay failed: {result.stderr.strip()}")
    return output


def overlay_tracklets_mp4(
    input_path: str | Path,
    output_path: str | Path,
    observations: list[dict[str, Any]],
    *,
    fps: float | None = None,
    crf: int = 30,
    show_summary: bool = True,
) -> Path:
    capture = cv2.VideoCapture(str(Path(input_path)))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open source video for tracklet overlay: {input_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0) or 5.0
    if fps is None or fps <= 0:
        fps = source_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError(f"Invalid source dimensions for tracklet overlay: {input_path}")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Unable to open output writer for tracklet overlay: {output}")

    def _color_for(tracklet_id: str) -> tuple[int, int, int]:
        digest = hashlib.sha256(str(tracklet_id).encode("utf-8")).digest()
        return int(64 + digest[0] % 160), int(64 + digest[1] % 160), int(64 + digest[2] % 160)

    def _nearest_point(points: list[dict[str, Any]], timestamp_sec: float, fps_value: float) -> dict[str, Any] | None:
        if not points:
            return None
        best: dict[str, Any] | None = None
        best_delta = None
        tolerance = max(0.20, 1.5 / max(fps_value, 1.0))
        for point in points:
            point_time = point.get("time_sec")
            if point_time is None and point.get("frame_index") is not None and fps_value > 0:
                point_time = float(point["frame_index"]) / float(fps_value)
            if point_time is None:
                continue
            delta = abs(float(point_time) - timestamp_sec)
            if delta <= tolerance and (best_delta is None or delta < best_delta):
                best = point
                best_delta = delta
        return best

    def _draw_label(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)

    def _display_label(obs: dict[str, Any]) -> str:
        value = str(obs.get("display_label") or obs.get("tracklet_id") or "T").strip()
        return value[:16] if len(value) > 16 else value

    frame_index = 0
    summary_lines = []
    if show_summary:
        for obs in observations[:8]:
            label = str(obs.get("coarse_label_hint") or obs.get("label_hint") or "tracklet")
            summary_lines.append(f"{obs.get('tracklet_id')}: {label}")
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        timestamp_sec = frame_index / float(fps)
        for obs in observations:
            start_sec = obs.get("start_sec")
            end_sec = obs.get("end_sec")
            if start_sec is not None and timestamp_sec < float(start_sec):
                continue
            if end_sec is not None and timestamp_sec > float(end_sec):
                continue
            points = obs.get("points") or []
            point = _nearest_point(points, timestamp_sec, float(fps))
            if point is None and points:
                point = points[-1]
            if point is None:
                continue
            color = _color_for(str(obs.get("tracklet_id") or "tracklet"))
            x_norm = float(point.get("x") or 0.0)
            y_norm = float(point.get("y") or 0.0)
            w_norm = point.get("w")
            h_norm = point.get("h")
            x = int(round(x_norm * width)) if 0.0 <= x_norm <= 1.5 else int(round(x_norm))
            y = int(round(y_norm * height)) if 0.0 <= y_norm <= 1.5 else int(round(y_norm))
            if w_norm is not None and h_norm is not None:
                w = int(round(float(w_norm) * width)) if 0.0 <= float(w_norm) <= 1.5 else int(round(float(w_norm)))
                h = int(round(float(h_norm) * height)) if 0.0 <= float(h_norm) <= 1.5 else int(round(float(h_norm)))
                left = max(0, x - max(1, w // 2))
                top = max(0, y - max(1, h // 2))
                right = min(width - 1, left + max(2, w))
                bottom = min(height - 1, top + max(2, h))
                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                _draw_label(frame, _display_label(obs), left, max(18, top - 6), color)
            else:
                cv2.circle(frame, (max(0, min(width - 1, x)), max(0, min(height - 1, y))), 8, color, 2)
                _draw_label(frame, _display_label(obs), max(0, min(width - 80, x + 10)), max(18, y - 8), color)

        if show_summary and summary_lines:
            line_height = 18
            block_height = 10 + line_height * len(summary_lines)
            block_width = min(width - 16, 240)
            cv2.rectangle(frame, (8, 8), (8 + block_width, 8 + block_height), (20, 20, 20), thickness=-1)
            cv2.rectangle(frame, (8, 8), (8 + block_width, 8 + block_height), (180, 180, 180), thickness=1)
            for idx, line in enumerate(summary_lines):
                cv2.putText(
                    frame,
                    line,
                    (16, 28 + idx * line_height),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (235, 235, 235),
                    1,
                    cv2.LINE_AA,
                )

        writer.write(frame)
        frame_index += 1

    writer.release()
    capture.release()
    remuxed = output.with_suffix(".h264.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(output),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-movflags",
        "+faststart",
        str(remuxed),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg tracklet overlay remux failed: {result.stderr.strip()}")
    remuxed.replace(output)
    return output
