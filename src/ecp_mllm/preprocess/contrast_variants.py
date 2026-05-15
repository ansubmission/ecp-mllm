from __future__ import annotations

from collections import deque
from typing import Iterable

import cv2
import numpy as np


def _copy_frames(frames: Iterable[np.ndarray]) -> list[np.ndarray]:
    return [frame.copy() for frame in frames]


def apply_clahe_to_derived_channels(
    frames: Iterable[np.ndarray],
    *,
    clip_limit: float = 2.5,
    tile_grid_size: tuple[int, int] = (8, 8),
) -> list[np.ndarray]:
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    outputs: list[np.ndarray] = []
    for frame in frames:
        adjusted = frame.copy()
        adjusted[:, :, 1] = clahe.apply(adjusted[:, :, 1])
        adjusted[:, :, 2] = clahe.apply(adjusted[:, :, 2])
        outputs.append(adjusted)
    return outputs


def apply_stationary_suppression(
    frames: Iterable[np.ndarray],
    *,
    motion_threshold: int = 6,
    bright_threshold: int = 120,
    suppression_strength: float = 0.45,
) -> list[np.ndarray]:
    outputs: list[np.ndarray] = []
    previous_motion_channel: np.ndarray | None = None
    for frame in frames:
        adjusted = frame.copy()
        motion_channel = adjusted[:, :, 2]
        if previous_motion_channel is None:
            previous_motion_channel = motion_channel.copy()
            outputs.append(adjusted)
            continue

        motion_delta = cv2.absdiff(motion_channel, previous_motion_channel)
        stationary_mask = ((motion_delta < motion_threshold) & (motion_channel >= bright_threshold)).astype(np.uint8) * 255
        if stationary_mask.any():
            stationary_mask = cv2.GaussianBlur(stationary_mask, (9, 9), 0)
            suppression = 1.0 - (stationary_mask.astype(np.float32) / 255.0) * suppression_strength
            adjusted[:, :, 1] = np.clip(adjusted[:, :, 1].astype(np.float32) * suppression, 0, 255).astype(np.uint8)
            adjusted[:, :, 2] = np.clip(adjusted[:, :, 2].astype(np.float32) * suppression, 0, 255).astype(np.uint8)
        outputs.append(adjusted)
        previous_motion_channel = motion_channel.copy()
    return outputs


def apply_echo_persistence(
    frames: Iterable[np.ndarray],
    *,
    window: int = 3,
    blend: float = 0.35,
) -> list[np.ndarray]:
    history_ch1: deque[np.ndarray] = deque(maxlen=max(1, window))
    history_ch2: deque[np.ndarray] = deque(maxlen=max(1, window))
    outputs: list[np.ndarray] = []
    for frame in frames:
        adjusted = frame.copy()
        history_ch1.append(adjusted[:, :, 1].copy())
        history_ch2.append(adjusted[:, :, 2].copy())
        persist_ch1 = np.maximum.reduce(list(history_ch1))
        persist_ch2 = np.maximum.reduce(list(history_ch2))
        adjusted[:, :, 1] = np.clip(
            adjusted[:, :, 1].astype(np.float32) * (1.0 - blend) + persist_ch1.astype(np.float32) * blend,
            0,
            255,
        ).astype(np.uint8)
        adjusted[:, :, 2] = np.clip(
            adjusted[:, :, 2].astype(np.float32) * (1.0 - blend) + persist_ch2.astype(np.float32) * blend,
            0,
            255,
        ).astype(np.uint8)
        outputs.append(adjusted)
    return outputs


def build_contrast_variants(base_frames: Iterable[np.ndarray]) -> dict[str, list[np.ndarray]]:
    base = _copy_frames(base_frames)
    clahe = apply_clahe_to_derived_channels(base)
    clahe_stationary = apply_stationary_suppression(clahe)
    echo_persist = apply_echo_persistence(clahe)
    return {
        "base": base,
        "clahe": clahe,
        "clahe_stationary": clahe_stationary,
        "echo_persist": echo_persist,
    }
