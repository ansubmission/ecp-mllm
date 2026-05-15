from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import hashlib
from pathlib import Path
from typing import Any, Sequence


Frame = list[list[int]]


@dataclass(frozen=True)
class SFF3CParameters:
    mog_history: int = 100
    gaussian_blur: tuple[int, int] = (3, 3)
    gaussian_sigma: float = 1.4
    guided_filter_radius: int = 10
    guided_filter_eps: float = 0.01
    canny_low: int = 200
    canny_high: int = 255
    edge_expand_size: int = 2
    mask_expand_size: int = 1
    history_factor: float = 0.2


DEFAULT_DOMAIN_PRESETS: dict[str, SFF3CParameters] = {
    "default": SFF3CParameters(),
    "kenai-val": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "kenai-rightbank": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "kenai-channel": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "nushagak": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "elwha": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "haida": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "wuikinuxv": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "secwepemc": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
    "smolt": SFF3CParameters(gaussian_blur=(3, 3), gaussian_sigma=1.5, history_factor=0.6),
}


def _load_cv2_modules():
    try:
        import cv2  # type: ignore
        import numpy as np  # type: ignore
    except ImportError as exc:
        raise RuntimeError("OpenCV contrib and numpy are required for faithful SFF3C conversion") from exc
    if not hasattr(cv2, "ximgproc") or not hasattr(cv2.ximgproc, "guidedFilter"):
        raise RuntimeError("OpenCV ximgproc.guidedFilter is required for faithful SFF3C conversion")
    if not hasattr(cv2, "bgsegm") or not hasattr(cv2.bgsegm, "createBackgroundSubtractorMOG"):
        raise RuntimeError("OpenCV bgsegm.createBackgroundSubtractorMOG is required for faithful SFF3C conversion")
    return cv2, np


def _load_pillow_image_module():
    try:
        from PIL import Image  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Pillow is required to write SFF3C frame directories on disk") from exc
    return Image


def params_for_domain(domain: str, overrides: dict[str, Any] | None = None) -> SFF3CParameters:
    base = DEFAULT_DOMAIN_PRESETS.get(domain.lower(), DEFAULT_DOMAIN_PRESETS["default"])
    if not overrides:
        return base
    updated = base
    for key, value in overrides.items():
        updated = replace(updated, **{key: value})
    return updated


def params_cache_token(domain: str, overrides: dict[str, Any] | None = None) -> str:
    params = params_for_domain(domain, overrides=overrides)
    payload = ",".join(f"{key}={value}" for key, value in sorted(asdict(params).items()))
    return hashlib.sha256(f"{domain.lower()}|{payload}".encode("utf-8")).hexdigest()[:12]


def _as_gray_array(frame: Frame):
    _cv2, np = _load_cv2_modules()
    return np.asarray(frame, dtype=np.uint8)


def _rows_from_gray_array(frame) -> Frame:
    return [[int(value) for value in row] for row in frame.tolist()]


class SFF3CConverter:
    def __init__(self, params: SFF3CParameters) -> None:
        cv2, np = _load_cv2_modules()
        self.cv2 = cv2
        self.np = np
        self.params = params
        self.mog_subtractor = cv2.bgsegm.createBackgroundSubtractorMOG(params.mog_history)
        self.history_img = None
        self.history_mog = None

    def _change_surrounding_region(self, mask, size: int):
        if size <= 0:
            return mask.copy()
        result_mask = mask.copy()
        white_pixels = self.np.where(mask == 255)
        for y, x in zip(*white_pixels):
            y_min = max(0, y - size)
            y_max = min(mask.shape[0], y + size)
            x_min = max(0, x - size)
            x_max = min(mask.shape[1], x + size)
            result_mask[y_min:y_max, x_min:x_max] = 255
        return result_mask

    def _temporal_blend(self, current, history):
        if history is None:
            return current
        current_weight = 1.0 - self.params.history_factor
        blended = (current.astype(self.np.float32) * current_weight) + (
            history.astype(self.np.float32) * self.params.history_factor
        )
        return self.np.clip(blended, 0, 255).astype(self.np.uint8)

    def process_frame(self, gray_frame, frame_index: int):
        cv2 = self.cv2
        np = self.np

        frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
        frame = cv2.GaussianBlur(frame, self.params.gaussian_blur, self.params.gaussian_sigma)
        mog_mask = self.mog_subtractor.apply(frame)
        mog_mask_rgb = cv2.cvtColor(mog_mask, cv2.COLOR_GRAY2RGB)

        guided_img = cv2.ximgproc.guidedFilter(
            mog_mask_rgb,
            frame,
            self.params.guided_filter_radius,
            self.params.guided_filter_eps,
        )
        guided_mog = cv2.ximgproc.guidedFilter(
            frame,
            mog_mask_rgb,
            self.params.guided_filter_radius,
            self.params.guided_filter_eps,
        )

        edge_original = cv2.Canny(guided_img, self.params.canny_low, self.params.canny_high)
        edge_mog = cv2.Canny(guided_mog, self.params.canny_low, self.params.canny_high)
        expanded_edge_mog = self._change_surrounding_region(edge_mog, self.params.edge_expand_size)

        condition = (edge_original == 255) & (expanded_edge_mog == 255)
        mask = np.zeros_like(edge_original, dtype=np.uint8)
        mask[condition] = 255
        mask = self._change_surrounding_region(mask, self.params.mask_expand_size)
        cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        guided_mog_gray = cv2.cvtColor(guided_mog, cv2.COLOR_BGR2GRAY)
        guided_img_gray = cv2.cvtColor(guided_img, cv2.COLOR_BGR2GRAY)
        if frame_index < 1:
            smoothed_mog = guided_mog_gray
            smoothed_img = guided_img_gray
        else:
            smoothed_mog = self._temporal_blend(guided_mog_gray, self.history_mog)
            smoothed_img = self._temporal_blend(guided_img_gray, self.history_img)

        self.history_mog = smoothed_mog
        self.history_img = smoothed_img

        raw_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return np.dstack([raw_gray, smoothed_img, smoothed_mog]).astype(np.uint8)


def _convert_gray_sequence(gray_frames, params: SFF3CParameters):
    converter = SFF3CConverter(params)
    return [converter.process_frame(frame, index) for index, frame in enumerate(gray_frames)]


def build_sff3c_channels(frames: Sequence[Frame], params: SFF3CParameters) -> list[tuple[Frame, Frame, Frame]]:
    gray_frames = [_as_gray_array(frame) for frame in frames]
    converted = _convert_gray_sequence(gray_frames, params)
    outputs: list[tuple[Frame, Frame, Frame]] = []
    for image in converted:
        outputs.append(
            (
                _rows_from_gray_array(image[:, :, 0]),
                _rows_from_gray_array(image[:, :, 1]),
                _rows_from_gray_array(image[:, :, 2]),
            )
        )
    return outputs


def process_frame_sequence(
    frames: Sequence[Frame],
    domain: str,
    overrides: dict[str, Any] | None = None,
) -> list[tuple[Frame, Frame, Frame]]:
    return build_sff3c_channels(frames, params_for_domain(domain, overrides=overrides))


def _load_gray_frame_path(path: Path):
    Image = _load_pillow_image_module()
    _cv2, np = _load_cv2_modules()
    with Image.open(path) as image:
        return np.asarray(image.convert("L"), dtype=np.uint8)


def _save_rgb_image(path: Path, image) -> None:
    Image = _load_pillow_image_module()
    Image.fromarray(image, mode="RGB").save(path)


def process_frame_paths(
    frame_paths: Sequence[str | Path],
    output_dir: str | Path,
    domain: str,
    overrides: dict[str, Any] | None = None,
) -> tuple[Path, list[Path]]:
    ordered_paths = [Path(path) for path in frame_paths]
    if not ordered_paths:
        raise ValueError("No image frames provided for SFF3C processing")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    output_paths = [output_path / frame_path.name for frame_path in ordered_paths]
    if output_paths and all(path.exists() for path in output_paths):
        return output_path, output_paths

    params = params_for_domain(domain, overrides=overrides)
    gray_frames = [_load_gray_frame_path(frame_path) for frame_path in ordered_paths]
    converted = _convert_gray_sequence(gray_frames, params)
    for target_path, image in zip(output_paths, converted, strict=True):
        _save_rgb_image(target_path, image)
    return output_path, output_paths


def process_frame_paths_with_warmup(
    frame_paths: Sequence[str | Path],
    selected_indices: Sequence[int],
    output_dir: str | Path,
    domain: str,
    warmup_frames: int,
    overrides: dict[str, Any] | None = None,
) -> tuple[Path, list[Path]]:
    ordered_paths = [Path(path) for path in frame_paths]
    if not ordered_paths:
        raise ValueError("No image frames provided for SFF3C processing")
    if not selected_indices:
        raise ValueError("No selected indices provided for SFF3C warmup processing")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    normalized_indices = sorted({int(index) for index in selected_indices})
    output_paths = [output_path / ordered_paths[index].name for index in normalized_indices]
    if output_paths and all(path.exists() for path in output_paths):
        return output_path, output_paths

    params = params_for_domain(domain, overrides=overrides)
    for index, target_path in zip(normalized_indices, output_paths, strict=True):
        if index < 0 or index >= len(ordered_paths):
            raise IndexError(f"Selected frame index {index} is out of range for {len(ordered_paths)} frames")
        start = max(0, index - max(0, warmup_frames))
        segment_paths = ordered_paths[start : index + 1]
        gray_frames = [_load_gray_frame_path(frame_path) for frame_path in segment_paths]
        converted = _convert_gray_sequence(gray_frames, params)[-1]
        _save_rgb_image(target_path, converted)
    return output_path, output_paths


def process_frame_directory(
    input_dir: str | Path,
    output_dir: str | Path,
    domain: str,
    overrides: dict[str, Any] | None = None,
) -> Path:
    input_path = Path(input_dir)
    frame_paths = sorted(path for path in input_path.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if not frame_paths:
        raise ValueError(f"No image frames found in {input_path}")
    output_path, _ = process_frame_paths(frame_paths, output_dir, domain, overrides=overrides)
    return output_path
