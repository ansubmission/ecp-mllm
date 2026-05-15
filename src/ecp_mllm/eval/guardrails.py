from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


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
class GuardrailDecision:
    direction: str
    baseline_count: int
    corrected_count: int
    applied: bool
    rule_name: str
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


def _max_peak(candidates: list[dict[str, Any]]) -> float:
    return max((_safe_float(candidate.get("peak_simultaneous_count"), 0.0) for candidate in candidates), default=0.0)


def _max_duration(candidates: list[dict[str, Any]]) -> float:
    return max((_safe_float(candidate.get("episode_duration_sec"), 0.0) for candidate in candidates), default=0.0)


def apply_guarded_count(result: dict[str, Any], *, direction: str = "right") -> GuardrailDecision:
    prediction = result.get("prediction") or {}
    candidates = _directional_candidates(prediction, direction)
    baseline_count = _safe_int(prediction.get(f"{direction}_count"), 0)
    confidence = _safe_float(prediction.get("confidence"), 0.0)
    candidate_count = len(candidates)
    max_peak = _max_peak(candidates)
    max_duration = _max_duration(candidates)

    if candidate_count >= 5 and max_peak <= 1.0 and baseline_count >= 5 and confidence <= 0.55:
        corrected = min(baseline_count, 2)
        return GuardrailDecision(
            direction=direction,
            baseline_count=baseline_count,
            corrected_count=corrected,
            applied=corrected != baseline_count,
            rule_name="fragmented_singletons_cap",
            reason=(
                f"candidate_count={candidate_count} max_peak={max_peak:.1f} "
                f"baseline_count={baseline_count} confidence={confidence:.2f}"
            ),
        )

    if (
        candidate_count == 1
        and max_peak >= 4.0
        and max_duration <= 18.0
        and confidence <= 0.65
        and baseline_count > (2.0 * max_peak)
    ):
        moderate_cap = max(int(round(max_peak * 1.5)), int(max_peak + 2))
        corrected = min(baseline_count, moderate_cap)
        return GuardrailDecision(
            direction=direction,
            baseline_count=baseline_count,
            corrected_count=corrected,
            applied=corrected != baseline_count,
            rule_name="short_single_episode_cap",
            reason=(
                f"candidate_count={candidate_count} max_peak={max_peak:.1f} "
                f"max_duration={max_duration:.1f}s baseline_count={baseline_count} "
                f"confidence={confidence:.2f}"
            ),
        )

    return GuardrailDecision(
        direction=direction,
        baseline_count=baseline_count,
        corrected_count=baseline_count,
        applied=False,
        rule_name="none",
        reason=(
            f"candidate_count={candidate_count} max_peak={max_peak:.1f} "
            f"max_duration={max_duration:.1f}s baseline_count={baseline_count} "
            f"confidence={confidence:.2f}"
        ),
    )
