from __future__ import annotations

from collections import defaultdict
from math import sqrt
from pathlib import Path
from statistics import mean

from ..types import ClipEvalResult, ClipKey, ClipRecord, DirectionalCounts, DomainSummary, EvalReport, PassagePrediction, WeakCountRecord


def normalize_bbox(bbox: list[float], width: int, height: int) -> list[float]:
    return [(bbox[0] - 1) / width, (bbox[1] - 1) / height, bbox[2] / width, bbox[3] / height]


def read_mot_tracks(path: str | Path) -> dict[int, list[list[float]]]:
    tracks: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            frame, track_id, x, y, w, h, *_ = [float(part.strip()) for part in stripped.split(",")]
            tracks[int(track_id)].append((int(frame), [x, y, w, h]))
    return {
        track_id: [bbox for _, bbox in sorted(entries, key=lambda item: item[0])]
        for track_id, entries in tracks.items()
    }


def read_mot_track_sequences(path: str | Path) -> dict[int, list[tuple[int, list[float]]]]:
    tracks: dict[int, list[tuple[int, list[float]]]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            frame, track_id, x, y, w, h, *_ = [float(part.strip()) for part in stripped.split(",")]
            tracks[int(track_id)].append((int(frame), [x, y, w, h]))
    return {
        track_id: sorted(entries, key=lambda item: item[0])
        for track_id, entries in tracks.items()
    }


def count_tracks_like_cfc(
    tracks: dict[int, list[list[float]]],
    width: int,
    height: int,
    filter_dist: float = 0.05,
) -> DirectionalCounts:
    left = 0
    right = 0
    for track in tracks.values():
        if not track:
            continue
        start = normalize_bbox(track[0], width, height)
        end = normalize_bbox(track[-1], width, height)
        x0 = start[0] + (start[2] / 2.0)
        x1 = end[0] + (end[2] / 2.0)
        if filter_dist > 0:
            y0 = start[1] + (start[3] / 2.0)
            y1 = end[1] + (end[3] / 2.0)
            distance = sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            if distance < filter_dist:
                continue
        if x0 < 0.5 <= x1:
            right += 1
        elif x0 >= 0.5 > x1:
            left += 1
    return DirectionalCounts(left=left, right=right)


def evaluate_predictions(
    predictions: dict[ClipKey, PassagePrediction],
    labels: dict[ClipKey, WeakCountRecord],
    clip_index: dict[ClipKey, ClipRecord],
) -> EvalReport:
    clip_results: list[ClipEvalResult] = []
    domain_buckets: dict[str, list[ClipEvalResult]] = defaultdict(list)
    for key, label in labels.items():
        clip = clip_index.get(key)
        fallback_direction = clip.upstream_direction if clip else label.upstream_direction
        prediction = predictions.get(key)
        if prediction is None:
            predicted_counts = DirectionalCounts()
            parse_success = False
            latency_sec = 0.0
        else:
            predicted_counts = prediction.normalized_counts(fallback_direction)
            parse_success = prediction.parse_success
            latency_sec = float(prediction.latency_sec or 0.0)
        result = ClipEvalResult(
            domain=key.domain,
            clip_id=key.clip_id,
            truth=label.counts,
            predicted=predicted_counts,
            abs_error_left=abs(predicted_counts.left - label.counts.left),
            abs_error_right=abs(predicted_counts.right - label.counts.right),
            squared_error_left=(predicted_counts.left - label.counts.left) ** 2,
            squared_error_right=(predicted_counts.right - label.counts.right) ** 2,
            parse_success=parse_success,
            latency_sec=latency_sec,
        )
        clip_results.append(result)
        domain_buckets[result.domain].append(result)
    directional_abs_errors = [value for result in clip_results for value in (result.abs_error_left, result.abs_error_right)]
    directional_sq_errors = [value for result in clip_results for value in (result.squared_error_left, result.squared_error_right)]
    gt_total = sum(result.truth.total for result in clip_results)
    return EvalReport(
        overall_mae=mean(directional_abs_errors) if directional_abs_errors else 0.0,
        overall_rmse=(mean(directional_sq_errors) ** 0.5) if directional_sq_errors else 0.0,
        overall_nmae=(sum(directional_abs_errors) / gt_total) if gt_total else None,
        parse_rate=mean([1.0 if result.parse_success else 0.0 for result in clip_results]) if clip_results else 0.0,
        mean_latency_sec=mean([result.latency_sec for result in clip_results]) if clip_results else 0.0,
        per_domain={domain: _summarize_domain(domain, results) for domain, results in sorted(domain_buckets.items())},
        clip_results=tuple(clip_results),
    )


def _summarize_domain(domain: str, results: list[ClipEvalResult]) -> DomainSummary:
    directional_abs_errors = [value for result in results for value in (result.abs_error_left, result.abs_error_right)]
    directional_sq_errors = [value for result in results for value in (result.squared_error_left, result.squared_error_right)]
    gt_total = sum(result.truth.total for result in results)
    return DomainSummary(
        domain=domain,
        clips=len(results),
        mae=mean(directional_abs_errors) if directional_abs_errors else 0.0,
        rmse=(mean(directional_sq_errors) ** 0.5) if directional_sq_errors else 0.0,
        nmae=(sum(directional_abs_errors) / gt_total) if gt_total else None,
        parse_rate=mean([1.0 if result.parse_success else 0.0 for result in results]) if results else 0.0,
        mean_latency_sec=mean([result.latency_sec for result in results]) if results else 0.0,
    )
