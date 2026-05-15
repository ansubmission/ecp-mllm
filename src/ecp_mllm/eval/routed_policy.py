from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .domain_shift_critic import DomainShiftAssessment, assess_domain_shift_risk
from .site_profiles import SiteProfile, resolve_site_profile


ALLOWED_CUSTOM_FLAGS = {
    "parse_failure_or_no_signal",
    "dense_stream_throughput_risk",
    "multi_episode_trickle_overcount_risk",
    "multiwave_high_peak_undercount_risk",
    "hidden_crossing_undercount_risk",
    "single_school_high_throughput_undercount_risk",
    "single_school_visibility_dropout_risk",
    "low_count_far_view_risk",
}

BLOCKED_CUSTOM_FLAGS = {
    "sparse_singleton_cluster_undercount_risk",
    "long_low_peak_stream_undercount_risk",
}


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


def _directional_candidates(prediction: dict[str, Any], direction: str) -> list[dict[str, Any]]:
    candidates = prediction.get("candidate_passages") or []
    if not isinstance(candidates, list):
        return []
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict) and str(candidate.get("direction") or "").lower() == direction
    ]


def _max_peak(prediction: dict[str, Any], direction: str) -> float:
    return max(
        (_safe_float(candidate.get("peak_simultaneous_count"), 0.0) for candidate in _directional_candidates(prediction, direction)),
        default=0.0,
    )


def _total_peak(prediction: dict[str, Any], direction: str) -> float:
    return sum(
        _safe_float(candidate.get("peak_simultaneous_count"), 0.0)
        for candidate in _directional_candidates(prediction, direction)
    )


def _total_wave_count(prediction: dict[str, Any], direction: str) -> int:
    total = 0
    for candidate in _directional_candidates(prediction, direction):
        wave_value = candidate.get("wave_count")
        total += max(0, _safe_int(wave_value, 1 if candidate else 0))
    return total


@dataclass(frozen=True)
class RoutedPolicyDecision:
    direction: str
    site_profile_id: str
    baseline_flags: tuple[str, ...]
    route_candidate: bool
    selected_source: str
    selected_branch: str
    selected_count: int
    rule_name: str
    reason: str
    baseline_assessment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _build_decision(
    *,
    direction: str,
    profile: SiteProfile,
    assessment: DomainShiftAssessment,
    route_candidate: bool,
    selected_source: str,
    selected_branch: str,
    selected_count: int,
    rule_name: str,
    reason: str,
) -> RoutedPolicyDecision:
    return RoutedPolicyDecision(
        direction=direction,
        site_profile_id=profile.site_id,
        baseline_flags=assessment.flags,
        route_candidate=route_candidate,
        selected_source=selected_source,
        selected_branch=selected_branch,
        selected_count=selected_count,
        rule_name=rule_name,
        reason=reason,
        baseline_assessment=assessment.to_dict(),
    )


def apply_routed_policy(
    baseline_result: dict[str, Any],
    custom_result: dict[str, Any] | None,
    *,
    direction: str = "right",
    site_profile: SiteProfile | None = None,
    assessment_override: DomainShiftAssessment | None = None,
) -> RoutedPolicyDecision:
    clip_key = str(baseline_result.get("clip_key") or "")
    profile = site_profile or resolve_site_profile(str(baseline_result.get("domain") or clip_key.split("/", 1)[0]), clip_key=clip_key)
    assessment: DomainShiftAssessment = assessment_override or assess_domain_shift_risk(
        baseline_result,
        direction=direction,
        site_profile=profile,
    )
    baseline_prediction = baseline_result.get("prediction") or {}
    baseline_count = _safe_int(baseline_prediction.get(f"{direction}_count"), 0)
    baseline_left = _safe_int(baseline_prediction.get("left_count"), 0)
    baseline_right = _safe_int(baseline_prediction.get("right_count"), 0)
    baseline_total = baseline_left + baseline_right

    flags = set(assessment.flags)
    blocked_flags = sorted(flags & BLOCKED_CUSTOM_FLAGS)
    allowed_flags = sorted(flags & ALLOWED_CUSTOM_FLAGS)
    stream_profile_enabled = (
        "stream_throughput" in profile.expected_regime
        and "custom_stream_throughput" in profile.enabled_branches
    )
    if "custom_high_throughput" not in profile.enabled_branches:
        allowed_flags = [
            flag
            for flag in allowed_flags
            if flag not in {
                "multiwave_high_peak_undercount_risk",
                "hidden_crossing_undercount_risk",
                "single_school_high_throughput_undercount_risk",
                "single_school_visibility_dropout_risk",
            }
        ]
    if "custom_stream_throughput" not in profile.enabled_branches:
        allowed_flags = [flag for flag in allowed_flags if flag != "dense_stream_throughput_risk"]
    if "custom_trickle_reduction" not in profile.enabled_branches:
        allowed_flags = [flag for flag in allowed_flags if flag != "multi_episode_trickle_overcount_risk"]
    if "custom_low_count" not in profile.enabled_branches:
        allowed_flags = [flag for flag in allowed_flags if flag != "low_count_far_view_risk"]
    route_candidate = (bool(allowed_flags) or stream_profile_enabled) and not (
        blocked_flags and "low_count_far_view_risk" not in allowed_flags
    )

    if not route_candidate:
        reason = "no_allowed_route_flag"
        if blocked_flags:
            reason = f"blocked_flags={','.join(blocked_flags)}"
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=False,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=reason,
        )

    if custom_result is None or custom_result.get("status") != "completed":
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason="custom_missing",
        )

    custom_prediction = custom_result.get("prediction") or {}
    custom_count = _safe_int(custom_prediction.get(f"{direction}_count"), 0)
    custom_left = _safe_int(custom_prediction.get("left_count"), 0)
    custom_right = _safe_int(custom_prediction.get("right_count"), 0)
    custom_total = custom_left + custom_right
    if direction == "left":
        baseline_opposite = baseline_right
        custom_opposite = custom_right
    else:
        baseline_opposite = baseline_left
        custom_opposite = custom_left
    baseline_peak = _max_peak(baseline_prediction, direction)
    baseline_total_peak = _total_peak(baseline_prediction, direction)
    baseline_total_waves = _total_wave_count(baseline_prediction, direction)
    custom_peak = _max_peak(custom_prediction, direction)
    custom_total_peak = _total_peak(custom_prediction, direction)
    custom_total_waves = _total_wave_count(custom_prediction, direction)
    custom_parse_success = bool(custom_prediction.get("parse_success", True))
    baseline_window_activity = baseline_result.get("window_activity") if isinstance(baseline_result.get("window_activity"), dict) else {}
    stream_window_floor = max(
        _safe_int(baseline_window_activity.get("max_window_total"), 0),
        _safe_int(baseline_window_activity.get("max_target_window_total"), 0),
    )

    if stream_profile_enabled:
        if (
            custom_parse_success
            and custom_count > baseline_count
            and custom_count >= max(stream_window_floor, baseline_count + 1)
            and custom_opposite <= min(1, baseline_opposite)
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_stream_throughput",
                selected_count=custom_count,
                rule_name="accept_stream_throughput_gain",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_count={baseline_count} custom_count={custom_count} "
                    f"stream_window_floor={stream_window_floor} "
                    f"baseline_opposite={baseline_opposite} custom_opposite={custom_opposite}"
                ),
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"baseline_count={baseline_count} custom_count={custom_count} "
                f"stream_window_floor={stream_window_floor} "
                f"baseline_opposite={baseline_opposite} custom_opposite={custom_opposite}"
            ),
        )

    if baseline_right > baseline_left and custom_left >= max(3, baseline_total):
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="reject_left_spike",
            reason=(
                f"baseline_right={baseline_right} baseline_left={baseline_left} "
                f"custom_left={custom_left} custom_right={custom_right}"
            ),
        )
    if baseline_left > baseline_right and custom_right >= max(3, baseline_total):
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="reject_right_spike",
            reason=(
                f"baseline_left={baseline_left} baseline_right={baseline_right} "
                f"custom_left={custom_left} custom_right={custom_right}"
            ),
        )

    if custom_total > max(baseline_total * 2, baseline_total + 8) and custom_peak <= 2.0:
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="reject_low_peak_explosion",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"custom_peak={custom_peak:.1f}"
            ),
        )

    if "parse_failure_or_no_signal" in flags:
        if custom_parse_success and custom_total > 0:
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_parse_repair",
                selected_count=custom_count,
                rule_name="accept_parse_repair",
                reason=f"custom_parse_success={custom_parse_success} custom_total={custom_total}",
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=f"custom_parse_success={custom_parse_success} custom_total={custom_total}",
        )

    if "multi_episode_trickle_overcount_risk" in flags:
        conservative_cap = max(3, int(round(custom_peak + custom_total_waves + 1.0)))
        if 0 < custom_total < baseline_total:
            if custom_total <= conservative_cap:
                return _build_decision(
                    direction=direction,
                    profile=profile,
                    assessment=assessment,
                    route_candidate=True,
                    selected_source="custom",
                    selected_branch="custom_trickle_reduction",
                    selected_count=custom_count,
                    rule_name="accept_trickle_reduction",
                    reason=(
                        f"baseline_total={baseline_total} custom_total={custom_total} "
                        f"custom_peak={custom_peak:.1f} custom_total_waves={custom_total_waves} "
                        f"conservative_cap={conservative_cap}"
                    ),
                )
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="baseline",
                selected_branch="baseline",
                selected_count=baseline_count,
                rule_name="reject_trickle_reduction_above_cap",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"custom_peak={custom_peak:.1f} custom_total_waves={custom_total_waves} "
                    f"conservative_cap={conservative_cap}"
                ),
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"custom_peak={custom_peak:.1f} custom_total_waves={custom_total_waves} "
                f"conservative_cap={conservative_cap}"
            ),
        )

    if "hidden_crossing_undercount_risk" in flags:
        hidden_cap = max(
            int(round(max(baseline_peak, baseline_total_peak) * 3.0 + max(0.0, assessment.max_duration_sec - 16.0) / 4.0)),
            baseline_total + 2,
        )
        if (
            custom_parse_success
            and custom_total > baseline_total
            and custom_total <= hidden_cap
            and custom_peak >= max(1.0, baseline_peak)
            and custom_total_waves <= max(2, baseline_total_waves + 1)
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_high_throughput",
                selected_count=custom_count,
                rule_name="accept_hidden_crossing_gain",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_peak={baseline_peak:.1f} baseline_total_peak={baseline_total_peak:.1f} "
                    f"custom_peak={custom_peak:.1f} custom_total_waves={custom_total_waves} "
                    f"hidden_cap={hidden_cap}"
                ),
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"custom_peak={custom_peak:.1f} hidden_cap={hidden_cap}"
            ),
        )

    if "single_school_high_throughput_undercount_risk" in flags:
        dense_school_cap = max(
            int(round(max(baseline_total_peak, custom_total_peak) * 4.0)),
            baseline_total + 6,
            int(round(assessment.max_duration_sec / 2.0)),
        )
        if (
            custom_parse_success
            and custom_total > baseline_total
            and custom_total <= dense_school_cap
            and custom_peak >= max(1.0, baseline_peak - 1.0)
            and custom_total_waves <= max(1, baseline_total_waves + 1)
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_high_throughput",
                selected_count=custom_count,
                rule_name="accept_single_school_high_throughput_gain",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_total_peak={baseline_total_peak:.1f} custom_total_peak={custom_total_peak:.1f} "
                    f"baseline_peak={baseline_peak:.1f} custom_peak={custom_peak:.1f} "
                    f"baseline_total_waves={baseline_total_waves} custom_total_waves={custom_total_waves} "
                    f"duration_sec={assessment.max_duration_sec:.1f} dense_school_cap={dense_school_cap}"
                ),
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"custom_peak={custom_peak:.1f} custom_total_waves={custom_total_waves} "
                f"duration_sec={assessment.max_duration_sec:.1f} dense_school_cap={dense_school_cap}"
            ),
        )

    if "single_school_visibility_dropout_risk" in flags:
        school_cap = max(
            int(round(max(baseline_peak, baseline_total_peak) + 2.0)),
            baseline_total + 2,
        )
        if (
            custom_parse_success
            and custom_total > baseline_total
            and custom_total <= school_cap
            and custom_peak >= max(1.0, baseline_peak)
            and custom_total_waves <= max(1, baseline_total_waves + 1)
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_high_throughput",
                selected_count=custom_count,
                rule_name="accept_single_school_hidden_gain",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_peak={baseline_peak:.1f} baseline_total_peak={baseline_total_peak:.1f} "
                    f"custom_peak={custom_peak:.1f} custom_total_waves={custom_total_waves} "
                    f"school_cap={school_cap}"
                ),
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"custom_peak={custom_peak:.1f} school_cap={school_cap}"
            ),
        )

    if "multiwave_high_peak_undercount_risk" in flags:
        conservative_cap = int(round(custom_total_peak + max(1, custom_total_waves)))
        structural_gain = (
            custom_total_peak > baseline_total_peak
            or custom_total_waves > baseline_total_waves
            or (custom_peak > baseline_peak and custom_total_peak > baseline_total_peak)
        )
        marginal_gain_without_structure = (
            custom_total == baseline_total + 1
            and custom_total_peak <= baseline_total_peak
            and custom_total_waves <= baseline_total_waves
        )
        if (
            custom_parse_success
            and custom_total > baseline_total
            and custom_total <= conservative_cap
            and structural_gain
            and not marginal_gain_without_structure
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_high_throughput",
                selected_count=custom_count,
                rule_name="accept_multiwave_high_peak_gain",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_total_peak={baseline_total_peak:.1f} custom_total_peak={custom_total_peak:.1f} "
                    f"baseline_total_waves={baseline_total_waves} custom_total_waves={custom_total_waves} "
                    f"conservative_cap={conservative_cap}"
                ),
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"custom_total_peak={custom_total_peak:.1f} custom_total_waves={custom_total_waves}"
            ),
        )

    if "low_count_far_view_risk" in flags:
        conservative_cap = int(round(custom_peak + 1.0))
        baseline_minority = baseline_left if direction == "right" else baseline_right
        custom_minority = custom_left if direction == "right" else custom_right
        if (
            custom_parse_success
            and custom_count >= baseline_count
            and custom_total < baseline_total
            and custom_minority < baseline_minority
            and custom_total <= max(2, baseline_total)
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_low_count",
                selected_count=custom_count,
                rule_name="accept_low_count_direction_cleanup",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_count={baseline_count} custom_count={custom_count} "
                    f"baseline_minority={baseline_minority} custom_minority={custom_minority}"
                ),
            )
        if (
            custom_parse_success
            and 0 < custom_total < baseline_total
            and baseline_total <= 4
            and custom_total <= max(2, conservative_cap)
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_low_count",
                selected_count=custom_count,
                rule_name="accept_low_count_sparse_reduction",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_count={baseline_count} custom_count={custom_count} "
                    f"conservative_cap={conservative_cap}"
                ),
            )
        if (
            custom_parse_success
            and custom_total > baseline_total
            and custom_total <= max(1, conservative_cap)
            and custom_total <= max(2, baseline_total + 2)
            and custom_total_peak >= baseline_total_peak
        ):
            return _build_decision(
                direction=direction,
                profile=profile,
                assessment=assessment,
                route_candidate=True,
                selected_source="custom",
                selected_branch="custom_low_count",
                selected_count=custom_count,
                rule_name="accept_low_count_far_view_gain",
                reason=(
                    f"baseline_total={baseline_total} custom_total={custom_total} "
                    f"baseline_total_peak={baseline_total_peak:.1f} custom_total_peak={custom_total_peak:.1f} "
                    f"conservative_cap={conservative_cap}"
                ),
            )
        return _build_decision(
            direction=direction,
            profile=profile,
            assessment=assessment,
            route_candidate=True,
            selected_source="baseline",
            selected_branch="baseline",
            selected_count=baseline_count,
            rule_name="keep_baseline",
            reason=(
                f"baseline_total={baseline_total} custom_total={custom_total} "
                f"custom_total_peak={custom_total_peak:.1f} conservative_cap={conservative_cap}"
            ),
        )

    return _build_decision(
        direction=direction,
        profile=profile,
        assessment=assessment,
        route_candidate=True,
        selected_source="baseline",
        selected_branch="baseline",
        selected_count=baseline_count,
        rule_name="keep_baseline",
        reason="no_matching_route_rule",
    )
