from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .site_profiles import SiteProfile, infer_clip_hints, resolve_site_profile


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


@dataclass(frozen=True)
class DomainShiftAssessment:
    site_profile_id: str
    direction: str
    predicted_total: int
    candidate_count: int
    total_wave_count: int
    total_peak: float
    max_peak: float
    max_duration_sec: float
    confidence: float
    parse_success: bool
    risk_level: str
    clip_hints: tuple[str, ...]
    flags: tuple[str, ...]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _directional_candidates(prediction: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    candidates = prediction.get("candidate_passages") or []
    if not isinstance(candidates, list):
        return []
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("direction") or "").lower() == direction
    ]


def _candidate_wave_count(candidate: dict[str, Any]) -> int:
    value = candidate.get("wave_count")
    if value is None:
        return 1
    return max(0, _safe_int(value, 1))


def _candidate_throughput(candidate: dict[str, Any]) -> int:
    return max(
        0,
        _safe_int(
            candidate.get("throughput_best_count")
            or candidate.get("estimated_count")
            or candidate.get("count")
            or 0,
            0,
        ),
    )


def assess_domain_shift_risk(
    result: dict[str, Any],
    *,
    direction: str = "right",
    site_profile: SiteProfile | None = None,
) -> DomainShiftAssessment:
    prediction = result.get("prediction") or {}
    clip_key = str(result.get("clip_key") or "")
    profile = site_profile or resolve_site_profile(str(result.get("domain") or clip_key.split("/", 1)[0]), clip_key=clip_key)
    candidates = _directional_candidates(prediction, direction)
    predicted_total = _safe_int(prediction.get(f"{direction}_count"), 0)
    confidence = _safe_float(prediction.get("confidence"), 0.0)
    parse_success = bool(prediction.get("parse_success", True))
    candidate_count = len(candidates)
    total_wave_count = sum(_candidate_wave_count(candidate) for candidate in candidates)
    total_peak = sum(_safe_float(candidate.get("peak_simultaneous_count"), 0.0) for candidate in candidates)
    max_peak = max((_safe_float(candidate.get("peak_simultaneous_count"), 0.0) for candidate in candidates), default=0.0)
    max_duration_sec = max((_safe_float(candidate.get("episode_duration_sec"), 0.0) for candidate in candidates), default=0.0)
    candidate_throughputs = sorted((_candidate_throughput(candidate) for candidate in candidates), reverse=True)
    dominant_throughput = candidate_throughputs[0] if candidate_throughputs else 0
    secondary_throughput = candidate_throughputs[1] if len(candidate_throughputs) > 1 else 0
    clip_hints = infer_clip_hints(clip_key)
    window_activity = result.get("window_activity") if isinstance(result.get("window_activity"), dict) else {}
    active_windows = _safe_int(window_activity.get("active_windows"), 0)
    zero_windows = _safe_int(window_activity.get("zero_windows"), 0)
    max_window_total = _safe_int(window_activity.get("max_window_total"), 0)
    sum_positive_window_totals = _safe_int(window_activity.get("sum_positive_window_totals"), 0)
    max_target_window_total = _safe_int(window_activity.get("max_target_window_total"), max_window_total)
    sum_target_window_totals = _safe_int(window_activity.get("sum_target_window_totals"), sum_positive_window_totals)
    opposite_direction = "left" if direction == "right" else "right"
    opposite_direction_total = _safe_int(prediction.get(f"{opposite_direction}_count"), 0)

    flags: list[str] = []

    if not parse_success:
        flags.append("parse_failure_or_no_signal")

    severe_trickle_overcount = candidate_count >= 3 and max_peak <= 2.0 and predicted_total >= 10
    moderate_trickle_overcount = (
        candidate_count >= 2
        and max_peak <= 2.0
        and predicted_total >= 6
        and max_duration_sec >= 30.0
        and confidence <= 0.6
    )
    sparse_singleton_trickle_overcount = (
        candidate_count >= 3
        and total_wave_count >= 4
        and max_peak <= 1.0
        and predicted_total >= 5
        and (
            predicted_total >= candidate_count + 3
            or (predicted_total >= candidate_count + 2 and confidence <= 0.65)
        )
    )
    long_sparse_multiwave_trickle_overcount = (
        candidate_count >= 1
        and total_wave_count >= 4
        and max_peak <= 2.0
        and predicted_total >= 5
        and max_duration_sec >= 60.0
        and confidence <= 0.65
    )
    moderate_peak_sparse_trickle_overcount = (
        candidate_count >= 3
        and total_wave_count >= 6
        and total_peak <= 6.0
        and max_peak <= 3.0
        and predicted_total >= 12
        and predicted_total >= int(total_peak * 3.0)
        and max_duration_sec >= 30.0
        and confidence <= 0.7
    )
    if (
        severe_trickle_overcount
        or moderate_trickle_overcount
        or sparse_singleton_trickle_overcount
        or long_sparse_multiwave_trickle_overcount
        or moderate_peak_sparse_trickle_overcount
    ):
        flags.append("multi_episode_trickle_overcount_risk")

    if candidate_count >= 4 and max_peak <= 1.0 and predicted_total >= 4 and confidence >= 0.8:
        flags.append("sparse_singleton_cluster_undercount_risk")

    if candidate_count == 1 and max_peak <= 2.0 and max_duration_sec >= 20.0 and 3 <= predicted_total <= 4:
        flags.append("long_low_peak_stream_undercount_risk")

    if total_wave_count >= 2 and max_peak >= 3.0 and predicted_total >= 4:
        flags.append("multiwave_high_peak_undercount_risk")

    hidden_crossing_undercount = (
        parse_success
        and candidate_count <= 2
        and total_wave_count <= 2
        and max_peak >= 4.0
        and max_duration_sec >= 18.0
        and predicted_total <= int(round(total_peak + 3.0))
        and confidence <= 0.85
    )
    if hidden_crossing_undercount:
        flags.append("hidden_crossing_undercount_risk")

    single_school_high_throughput_undercount = (
        parse_success
        and profile.site_id.startswith("kenai")
        and "near_view" in clip_hints
        and candidate_count <= 2
        and total_wave_count <= 2
        and (
            (
                max_peak >= 4.0
                and max_duration_sec >= 18.0
                and predicted_total >= 6
                and predicted_total <= int(round(total_peak + 5.0))
                and dominant_throughput >= max(6, predicted_total - 1)
            )
            or (
                max_peak >= 3.0
                and max_duration_sec >= 24.0
                and predicted_total >= 4
                and predicted_total <= int(round(total_peak + 2.0))
                and dominant_throughput >= max(5, predicted_total)
            )
        )
        and secondary_throughput <= 1
        and confidence <= 0.9
    )
    if single_school_high_throughput_undercount:
        flags.append("single_school_high_throughput_undercount_risk")

    far_view_relaxed_dropout = "far_view" in clip_hints and max_peak >= 4.0 and max_duration_sec >= 10.0 and confidence <= 0.9
    near_view_relaxed_dropout = "near_view" in clip_hints and max_peak >= 4.0 and max_duration_sec >= 15.0 and confidence <= 0.65
    single_school_visibility_dropout = (
        parse_success
        and profile.site_id.startswith("kenai")
        and candidate_count == 1
        and total_wave_count <= 1
        and max_peak >= 3.0
        and max_duration_sec >= 13.0
        and predicted_total <= int(round(total_peak + 1.0))
        and confidence <= 0.75
    ) or (
        parse_success
        and profile.site_id.startswith("kenai")
        and candidate_count == 1
        and total_wave_count <= 1
        and predicted_total <= int(round(total_peak + 1.0))
        and (far_view_relaxed_dropout or near_view_relaxed_dropout)
    )
    if single_school_visibility_dropout:
        flags.append("single_school_visibility_dropout_risk")

    dense_stream_throughput = (
        parse_success
        and "stream_throughput" in profile.expected_regime
        and opposite_direction_total <= 1
        and (
            predicted_total >= 4
            or max_target_window_total >= 4
            or sum_target_window_totals >= 8
        )
        and (
            max_peak >= 3.0
            or max_target_window_total >= 4
            or active_windows >= 2
            or max_duration_sec >= 10.0
            or sum_target_window_totals >= 8
        )
        and (
            predicted_total < max(max_target_window_total, max(4, int(round(sum_target_window_totals * 0.6))))
            or (
                active_windows >= 2
                and predicted_total < max(6, int(round(sum_target_window_totals * 0.7)))
            )
            or max_duration_sec >= 12.0
            or max_target_window_total >= 10
        )
    )
    if dense_stream_throughput:
        flags.append("dense_stream_throughput_risk")

    low_count_total_max = _safe_int(profile.risk_thresholds.get("low_count_total_max"), 1)
    low_count_candidate_max = _safe_int(profile.risk_thresholds.get("low_count_candidate_max"), 1)
    low_count_peak_max = _safe_float(profile.risk_thresholds.get("low_count_peak_max"), 1.0)
    clip_level_low_count_regime = any(item in clip_hints for item in {"low_count_sparse", "far_view"})
    site_level_low_count_regime = any(item in profile.expected_regime for item in {"low_count_sparse", "far_view"})
    low_count_regime = clip_level_low_count_regime or site_level_low_count_regime
    if (
        parse_success
        and low_count_regime
        and (
            (
                predicted_total <= low_count_total_max
                and candidate_count <= low_count_candidate_max
                and max_peak <= low_count_peak_max
                and total_wave_count <= 1
            )
            or (
                predicted_total <= max(4, low_count_total_max)
                and candidate_count <= max(1, low_count_candidate_max)
                and max_peak <= max(2.0, low_count_peak_max + 1.0)
                and total_wave_count <= 1
                and max_duration_sec >= 20.0
                and confidence <= 0.7
            )
        )
    ):
        flags.append("low_count_far_view_risk")

    if "parse_failure_or_no_signal" in flags or len(flags) >= 2:
        risk_level = "high"
    elif flags:
        risk_level = "medium"
    else:
        risk_level = "low"

    reason = (
        f"site_profile={profile.site_id} "
        f"predicted_total={predicted_total} candidate_count={candidate_count} "
        f"total_wave_count={total_wave_count} total_peak={total_peak:.1f} "
        f"max_peak={max_peak:.1f} max_duration_sec={max_duration_sec:.1f} "
        f"confidence={confidence:.2f} parse_success={parse_success}"
    )

    return DomainShiftAssessment(
        site_profile_id=profile.site_id,
        direction=direction,
        predicted_total=predicted_total,
        candidate_count=candidate_count,
        total_wave_count=total_wave_count,
        total_peak=total_peak,
        max_peak=max_peak,
        max_duration_sec=max_duration_sec,
        confidence=confidence,
        parse_success=parse_success,
        risk_level=risk_level,
        clip_hints=clip_hints,
        flags=tuple(flags),
        reason=reason,
    )
