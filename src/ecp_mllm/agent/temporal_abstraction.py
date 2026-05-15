from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ..types import EventProposal


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    kernel = np.ones(window, dtype=np.float32) / float(window)
    return np.convolve(values, kernel, mode="same")


def _motion_threshold(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    median = float(np.median(values))
    q75 = float(np.percentile(values, 75))
    q25 = float(np.percentile(values, 25))
    iqr = max(1e-6, q75 - q25)
    return median + 0.5 * iqr


def propose_temporal_windows(
    video_path: str | Path,
    *,
    fps_hint: float | None = None,
    smoothing_sec: float = 1.0,
    min_duration_sec: float = 1.0,
    merge_gap_sec: float = 1.0,
    max_windows: int = 8,
    representation: str = "unknown",
) -> list[EventProposal]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = float(fps_hint or 5.0)

    previous: np.ndarray | None = None
    energies: list[float] = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous is None:
            energies.append(0.0)
        else:
            energies.append(float(np.mean(cv2.absdiff(gray, previous))))
        previous = gray
    capture.release()

    if not energies:
        return []
    energy = np.asarray(energies, dtype=np.float32)
    smooth_window = max(1, int(round(max(0.25, smoothing_sec) * fps)))
    smoothed = _moving_average(energy, smooth_window)
    threshold = _motion_threshold(smoothed)
    active = smoothed >= threshold

    min_frames = max(1, int(round(min_duration_sec * fps)))
    merge_gap = max(0, int(round(merge_gap_sec * fps)))

    segments: list[tuple[int, int]] = []
    start: int | None = None
    for index, is_active in enumerate(active):
        if is_active and start is None:
            start = index
        elif not is_active and start is not None:
            end = index
            if end - start >= min_frames:
                segments.append((start, end))
            start = None
    if start is not None and len(active) - start >= min_frames:
        segments.append((start, len(active)))

    merged: list[tuple[int, int]] = []
    for start, end in segments:
        if merged and start - merged[-1][1] <= merge_gap:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    proposals: list[EventProposal] = []
    peak = float(np.max(smoothed)) if smoothed.size else 1.0
    peak = max(peak, 1e-6)
    for index, (start, end) in enumerate(merged[:max_windows], start=1):
        window_score = float(np.mean(smoothed[start:end])) / peak
        proposals.append(
            EventProposal(
                proposal_id=f"{Path(video_path).stem}:motion:{index:02d}",
                timestamp_start_sec=float(start / fps),
                timestamp_end_sec=float(end / fps),
                score=window_score,
                source="motion_energy",
                representation=representation,
                reasoning="temporal_abstraction",
                direction_hint=None,
                metadata={
                    "frame_start": start,
                    "frame_end": end,
                    "fps": fps,
                    "threshold": threshold,
                },
            )
        )
    return proposals
