from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import combinations
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


FEATURE_NAMES = (
    "pred_count",
    "peak_simultaneous_count",
    "episode_duration_sec",
    "duration_per_peak",
    "confidence",
)


def _safe_float(value: object, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def _prediction_count(prediction: dict[str, Any], direction: str) -> int:
    return _safe_int(prediction.get(f"{direction}_count"), 0)


def _candidate_count(candidate: dict[str, Any]) -> float:
    for key in ("throughput_best_count", "estimated_count", "peak_simultaneous_count"):
        if candidate.get(key) is not None:
            return _safe_float(candidate.get(key), 0.0)
    return 0.0


def dominant_candidate_passage(prediction: dict[str, Any]) -> dict[str, Any]:
    candidates = prediction.get("candidate_passages") or []
    if not isinstance(candidates, list) or not candidates:
        return {}
    ranked = sorted(
        (candidate for candidate in candidates if isinstance(candidate, dict)),
        key=lambda candidate: (
            _candidate_count(candidate),
            _safe_float(candidate.get("episode_duration_sec"), 0.0),
            -_safe_float(candidate.get("timestamp_start_sec"), 0.0),
        ),
        reverse=True,
    )
    return ranked[0] if ranked else {}


@dataclass(frozen=True)
class CalibrationSample:
    clip_key: str
    bucket: str
    gt_count: int
    pred_count: int
    peak_simultaneous_count: float
    episode_duration_sec: float
    duration_per_peak: float
    confidence: float

    def feature_lookup(self) -> dict[str, float]:
        return {
            "pred_count": float(self.pred_count),
            "peak_simultaneous_count": self.peak_simultaneous_count,
            "episode_duration_sec": self.episode_duration_sec,
            "duration_per_peak": self.duration_per_peak,
            "confidence": self.confidence,
        }

    def feature_vector(self, feature_names: Iterable[str] | None = None) -> np.ndarray:
        lookup = self.feature_lookup()
        names = tuple(feature_names or FEATURE_NAMES)
        return np.asarray([lookup[name] for name in names], dtype=float)


@dataclass(frozen=True)
class RidgeCalibrator:
    direction: str
    alpha: float
    feature_names: tuple[str, ...]
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    weights: tuple[float, ...]

    def _transform(self, sample: CalibrationSample) -> np.ndarray:
        features = sample.feature_vector(self.feature_names)
        mean = np.asarray(self.feature_mean, dtype=float)
        scale = np.asarray(self.feature_scale, dtype=float)
        standardized = (features - mean) / scale
        return np.concatenate([np.asarray([1.0]), standardized])

    def predict_value(self, sample: CalibrationSample) -> float:
        vector = self._transform(sample)
        weights = np.asarray(self.weights, dtype=float)
        return float(vector @ weights)

    def predict_count(self, sample: CalibrationSample) -> int:
        return max(0, int(round(self.predict_value(sample))))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def sample_from_batch_result(result: dict[str, Any], direction: str = "right") -> CalibrationSample:
    prediction = result.get("prediction") or {}
    ground_truth = result.get("ground_truth") or {}
    dominant = dominant_candidate_passage(prediction)
    peak = _safe_float(dominant.get("peak_simultaneous_count"), 0.0)
    duration = _safe_float(dominant.get("episode_duration_sec"), 0.0)
    duration_per_peak = duration / max(peak, 1.0)
    return CalibrationSample(
        clip_key=str(result["clip_key"]),
        bucket=str(result.get("bucket", "")),
        gt_count=_safe_int(ground_truth.get(f"{direction}_count"), 0),
        pred_count=_prediction_count(prediction, direction),
        peak_simultaneous_count=peak,
        episode_duration_sec=duration,
        duration_per_peak=duration_per_peak,
        confidence=_safe_float(prediction.get("confidence"), 0.0),
    )


def load_calibration_samples(summary_json_path: str | Path, direction: str = "right") -> list[CalibrationSample]:
    payload = json.loads(Path(summary_json_path).read_text(encoding="utf-8"))
    samples: list[CalibrationSample] = []
    for result in payload.get("results", []):
        if result.get("status") != "completed":
            continue
        if not result.get("ground_truth") or not result.get("prediction"):
            continue
        samples.append(sample_from_batch_result(result, direction=direction))
    return samples


def fit_ridge_calibrator(samples: Iterable[CalibrationSample], *, direction: str = "right", alpha: float = 1.0) -> RidgeCalibrator:
    return fit_ridge_calibrator_with_features(samples, direction=direction, alpha=alpha, feature_names=FEATURE_NAMES)


def fit_ridge_calibrator_with_features(
    samples: Iterable[CalibrationSample],
    *,
    direction: str = "right",
    alpha: float = 1.0,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> RidgeCalibrator:
    rows = list(samples)
    if not rows:
        raise ValueError("at least one calibration sample is required")
    selected_features = tuple(feature_names)
    if not selected_features:
        raise ValueError("at least one feature is required")
    x = np.vstack([sample.feature_vector(selected_features) for sample in rows]).astype(float)
    y = np.asarray([sample.gt_count for sample in rows], dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0.0] = 1.0
    z = (x - mean) / scale
    design = np.concatenate([np.ones((len(rows), 1), dtype=float), z], axis=1)
    reg = np.eye(design.shape[1], dtype=float)
    reg[0, 0] = 0.0
    gram = design.T @ design
    rhs = design.T @ y
    weights = np.linalg.pinv(gram + (alpha * reg)) @ rhs
    return RidgeCalibrator(
        direction=direction,
        alpha=float(alpha),
        feature_names=selected_features,
        feature_mean=tuple(float(value) for value in mean),
        feature_scale=tuple(float(value) for value in scale),
        weights=tuple(float(value) for value in weights),
    )


def absolute_error(y_true: Iterable[int], y_pred: Iterable[int]) -> int:
    return int(sum(abs(int(a) - int(b)) for a, b in zip(y_true, y_pred)))


def evaluate_calibrator(samples: Iterable[CalibrationSample], calibrator: RidgeCalibrator) -> dict[str, Any]:
    rows = list(samples)
    baseline = [sample.pred_count for sample in rows]
    corrected = [calibrator.predict_count(sample) for sample in rows]
    truth = [sample.gt_count for sample in rows]
    return {
        "samples": len(rows),
        "baseline_abs_error": absolute_error(truth, baseline),
        "calibrated_abs_error": absolute_error(truth, corrected),
        "rows": [
            {
                "clip_key": sample.clip_key,
                "bucket": sample.bucket,
                "gt_count": sample.gt_count,
                "baseline_pred_count": sample.pred_count,
                "calibrated_pred_count": calibrated,
                "peak_simultaneous_count": sample.peak_simultaneous_count,
                "episode_duration_sec": sample.episode_duration_sec,
                "duration_per_peak": sample.duration_per_peak,
                "confidence": sample.confidence,
            }
            for sample, calibrated in zip(rows, corrected)
        ],
    }


def leave_one_out_predictions(
    samples: Iterable[CalibrationSample],
    *,
    direction: str = "right",
    alpha: float = 1.0,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> list[dict[str, Any]]:
    rows = list(samples)
    if len(rows) < 2:
        raise ValueError("leave-one-out evaluation requires at least two samples")
    outputs: list[dict[str, Any]] = []
    for index, held_out in enumerate(rows):
        train = rows[:index] + rows[index + 1 :]
        calibrator = fit_ridge_calibrator_with_features(
            train,
            direction=direction,
            alpha=alpha,
            feature_names=feature_names,
        )
        corrected = calibrator.predict_count(held_out)
        outputs.append(
            {
                "clip_key": held_out.clip_key,
                "bucket": held_out.bucket,
                "gt_count": held_out.gt_count,
                "baseline_pred_count": held_out.pred_count,
                "calibrated_pred_count": corrected,
                "abs_error_baseline": abs(held_out.gt_count - held_out.pred_count),
                "abs_error_calibrated": abs(held_out.gt_count - corrected),
            }
        )
    return outputs


def select_alpha(
    samples: Iterable[CalibrationSample],
    *,
    direction: str = "right",
    alpha_grid: Iterable[float] | None = None,
    feature_names: Iterable[str] = FEATURE_NAMES,
) -> dict[str, Any]:
    rows = list(samples)
    if len(rows) < 2:
        alpha = 1.0 if alpha_grid is None else float(next(iter(alpha_grid)))
        return {"alpha": alpha, "loo_abs_error": None, "candidates": []}
    candidates = [float(value) for value in (alpha_grid or (0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0))]
    ranked: list[dict[str, Any]] = []
    for alpha in candidates:
        predictions = leave_one_out_predictions(rows, direction=direction, alpha=alpha, feature_names=feature_names)
        ranked.append(
            {
                "alpha": alpha,
                "loo_abs_error": int(sum(item["abs_error_calibrated"] for item in predictions)),
            }
        )
    ranked.sort(key=lambda item: (item["loo_abs_error"], item["alpha"]))
    return {
        "alpha": ranked[0]["alpha"],
        "loo_abs_error": ranked[0]["loo_abs_error"],
        "candidates": ranked,
    }


def select_calibration_config(
    samples: Iterable[CalibrationSample],
    *,
    direction: str = "right",
    alpha_grid: Iterable[float] | None = None,
    feature_subset_grid: Iterable[Iterable[str]] | None = None,
) -> dict[str, Any]:
    rows = list(samples)
    subsets = (
        [tuple(subset) for subset in feature_subset_grid]
        if feature_subset_grid is not None
        else [
            tuple(subset)
            for length in range(1, len(FEATURE_NAMES) + 1)
            for subset in combinations(FEATURE_NAMES, length)
        ]
    )
    ranked: list[dict[str, Any]] = []
    for subset in subsets:
        alpha_result = select_alpha(rows, direction=direction, alpha_grid=alpha_grid, feature_names=subset)
        ranked.append(
            {
                "feature_names": list(subset),
                "alpha": alpha_result["alpha"],
                "loo_abs_error": alpha_result["loo_abs_error"],
            }
        )
    ranked.sort(key=lambda item: (item["loo_abs_error"], len(item["feature_names"]), item["alpha"]))
    best = ranked[0]
    return {
        "feature_names": best["feature_names"],
        "alpha": best["alpha"],
        "loo_abs_error": best["loo_abs_error"],
        "candidates": ranked,
    }


def calibrator_from_dict(payload: dict[str, Any]) -> RidgeCalibrator:
    return RidgeCalibrator(
        direction=str(payload["direction"]),
        alpha=float(payload["alpha"]),
        feature_names=tuple(str(name) for name in payload["feature_names"]),
        feature_mean=tuple(float(value) for value in payload["feature_mean"]),
        feature_scale=tuple(float(value) for value in payload["feature_scale"]),
        weights=tuple(float(value) for value in payload["weights"]),
    )


def load_calibrator(path: str | Path) -> RidgeCalibrator:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return calibrator_from_dict(payload)
