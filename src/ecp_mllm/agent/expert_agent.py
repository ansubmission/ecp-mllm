from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

from ..eval.constraints import audit_prediction_constraints
from ..eval.domain_shift_critic import DomainShiftAssessment
from ..types import AgentMemoryRecord, ClipRecord, PassagePrediction, PerceptionAction
from .event_centric import build_observation_stream, enumerate_perception_actions


@dataclass(frozen=True)
class ExpertSelection:
    action_id: str
    score: float
    reason: str
    prediction: PassagePrediction


def _counts_direction(left: int, right: int) -> str | None:
    if left > right:
        return "left"
    if right > left:
        return "right"
    return None


def _memory_penalty(action: PerceptionAction, memory: Iterable[AgentMemoryRecord], domain: str) -> float:
    penalty = 0.0
    for item in memory:
        if item.domain != domain:
            continue
        if item.preferred_representation == action.representation:
            penalty += 0.25 * max(1, item.count)
        if item.preferred_temporal_mode == action.temporal_mode:
            penalty += 0.15 * max(1, item.count)
        if item.preferred_reasoning_mode == str(action.reasoning_mode):
            penalty += 0.2 * max(1, item.count)
    return penalty


def score_prediction(
    clip: ClipRecord,
    action: PerceptionAction,
    prediction: PassagePrediction,
    *,
    memory: Iterable[AgentMemoryRecord] = (),
) -> float:
    audit = audit_prediction_constraints(
        prediction,
        representation=action.representation,
        fallback_upstream_direction=clip.upstream_direction,
    )
    counts = prediction.normalized_counts(clip.upstream_direction)
    score = 0.0
    if prediction.parse_success:
        score += 5.0
    else:
        score -= 3.0
    if prediction.confidence is not None:
        score += float(prediction.confidence) * 2.0
    if action.repair_enabled:
        score += 0.25
    if str(action.reasoning_mode) == "proposal_guided":
        score += 0.5
    if action.representation == "sff3c":
        score += 0.5
    if not prediction.candidate_passages and not prediction.events:
        score -= 1.0
    if counts.total == 0 and not prediction.candidate_passages:
        score -= 0.5
    if str(action.reasoning_mode) == "proposal_guided" and not prediction.candidate_passages and not prediction.events:
        score -= 1.5
    for finding in audit.findings:
        if finding.severity == "high":
            score -= 3.0
        elif finding.severity == "medium":
            score -= 1.5
        else:
            score -= 0.5
    score -= _memory_penalty(action, memory, clip.domain)
    return score


def _prediction_max_peak(prediction: PassagePrediction, dominant_direction: str | None = None) -> float:
    peaks: list[float] = []
    for candidate in prediction.candidate_passages:
        if not isinstance(candidate, dict):
            continue
        if dominant_direction is not None:
            candidate_direction = str(candidate.get("direction") or "").lower()
            if candidate_direction and candidate_direction != dominant_direction:
                continue
        try:
            peaks.append(float(candidate.get("peak_simultaneous_count") or 0.0))
        except (TypeError, ValueError):
            continue
    return max(peaks, default=0.0)


def _prediction_candidate_count(prediction: PassagePrediction, dominant_direction: str | None = None) -> int:
    count = 0
    for candidate in prediction.candidate_passages:
        if not isinstance(candidate, dict):
            continue
        if dominant_direction is not None:
            candidate_direction = str(candidate.get("direction") or "").lower()
            if candidate_direction and candidate_direction != dominant_direction:
                continue
        count += 1
    return count


def select_expert_prediction(
    clip: ClipRecord,
    action_predictions: Mapping[str, PassagePrediction],
    *,
    memory: Iterable[AgentMemoryRecord] = (),
    risk_assessment: DomainShiftAssessment | None = None,
) -> ExpertSelection:
    actions = {action.action_id: action for action in enumerate_perception_actions(build_observation_stream(clip))}
    best_selection: ExpertSelection | None = None
    scored_candidates: list[tuple[float, str, PassagePrediction, str]] = []
    risk_flags = {str(flag) for flag in (risk_assessment.flags if risk_assessment is not None else ())}
    hidden_crossing_mode = "hidden_crossing_undercount_risk" in risk_flags
    site_profile_id = str(getattr(risk_assessment, "site_profile_id", "") or "")
    multiwave_mode = (
        "multiwave_high_peak_undercount_risk" in risk_flags
        and site_profile_id.startswith("kenai")
    )
    global_candidates = [
        (action_id, prediction)
        for action_id, prediction in action_predictions.items()
        if action_id.endswith(":global")
    ]
    global_prediction = global_candidates[0][1] if global_candidates else None
    global_counts = global_prediction.normalized_counts(clip.upstream_direction) if global_prediction is not None else None
    global_confidence = float(global_prediction.confidence or 0.0) if global_prediction is not None else 0.0
    global_direction = None
    if global_counts is not None:
        global_direction = "left" if global_counts.left > global_counts.right else "right" if global_counts.right > global_counts.left else None
    global_max_peak = _prediction_max_peak(global_prediction, global_direction) if global_prediction is not None else 0.0
    global_candidate_count = _prediction_candidate_count(global_prediction, global_direction) if global_prediction is not None else 0

    for action_id, prediction in action_predictions.items():
        action = actions.get(action_id)
        if action is None:
            continue
        score = score_prediction(clip, action, prediction, memory=memory)
        audit = audit_prediction_constraints(
            prediction,
            representation=action.representation,
            fallback_upstream_direction=clip.upstream_direction,
        )
        counts = prediction.normalized_counts(clip.upstream_direction)
        local_direction = _counts_direction(counts.left, counts.right)
        reason_parts = [
            f"parse_success={prediction.parse_success}",
            f"confidence={prediction.confidence}",
            f"constraint_findings={len(audit.findings)}",
        ]
        if action_id.endswith(":proposal_guided") or action_id.endswith(":local_repair"):
            if not prediction.candidate_passages and not prediction.events:
                score -= 2.0
                reason_parts.append("penalty=no_local_evidence")
            if global_counts is not None and global_counts.total > 0 and counts.total == 0:
                score -= 1.5
                reason_parts.append("penalty=drops_global_signal")
            if global_counts is not None:
                strong_global_reference = global_confidence >= 0.85 and global_counts.total >= max(1, counts.total)
                same_direction = global_direction is None or local_direction is None or local_direction == global_direction
                opposite_count = 0
                if global_direction == "left":
                    opposite_count = counts.right
                elif global_direction == "right":
                    opposite_count = counts.left
                else:
                    opposite_count = min(counts.left, counts.right)
                prediction_max_peak = _prediction_max_peak(prediction, local_direction or global_direction)
                hidden_cap = 0
                hidden_crossing_bonus = False
                multiwave_small_gain_bonus = False
                if hidden_crossing_mode and risk_assessment is not None:
                    hidden_peak_floor = max(2.0, min(global_max_peak, float(risk_assessment.max_peak)) - 1.0)
                    hidden_cap = max(
                        int(
                            round(
                                max(global_max_peak, float(risk_assessment.max_peak)) * 3.0
                                + max(0.0, float(risk_assessment.max_duration_sec) - 16.0) / 4.0
                            )
                        ),
                        global_counts.total + 2,
                    )
                    hidden_crossing_bonus = (
                        same_direction
                        and opposite_count == 0
                        and counts.total > global_counts.total
                        and counts.total <= hidden_cap
                        and prediction_max_peak >= hidden_peak_floor
                    )
                    if hidden_crossing_bonus:
                        score += 1.25 + 0.15 * (counts.total - global_counts.total)
                        reason_parts.append("bonus=hidden_crossing_same_direction_gain")
                if multiwave_mode and risk_assessment is not None and not hidden_crossing_bonus:
                    multiwave_gain = counts.total - global_counts.total
                    multiwave_peak_floor = max(
                        2.0,
                        min(global_max_peak, float(risk_assessment.max_peak)) - 1.0,
                    )
                    multiwave_small_gain_bonus = (
                        same_direction
                        and opposite_count == 0
                        and multiwave_gain in {1, 2}
                        and prediction_max_peak >= multiwave_peak_floor
                        and _prediction_candidate_count(prediction, local_direction or global_direction)
                        >= max(1, global_candidate_count)
                    )
                    if multiwave_small_gain_bonus:
                        score += 0.8 + 0.15 * multiwave_gain
                        reason_parts.append("bonus=multiwave_same_direction_small_gain")
                if strong_global_reference and global_direction and local_direction and global_direction != local_direction:
                    score -= 2.5
                    reason_parts.append("penalty=direction_disagreement_with_global")
                if strong_global_reference and global_counts.left == 0 and global_counts.right > 0 and counts.left > 0:
                    score -= 1.0 + 0.6 * counts.left
                    reason_parts.append("penalty=invented_left_against_right_global")
                if strong_global_reference and global_counts.right == 0 and global_counts.left > 0 and counts.right > 0:
                    score -= 1.0 + 0.6 * counts.right
                    reason_parts.append("penalty=invented_right_against_left_global")
                if global_counts.total >= 4:
                    minority = min(counts.left, counts.right)
                    if minority >= 2:
                        score -= 0.5 * minority
                        reason_parts.append("penalty=large_bidirectional_local_mix")
                if (
                    not hidden_crossing_bonus
                    and not multiwave_small_gain_bonus
                    and
                    global_confidence >= 0.55
                    and global_counts.total > 0
                    and same_direction
                    and counts.total >= global_counts.total + 3
                    and prediction_max_peak <= max(2.0, global_max_peak)
                ):
                    score -= 1.5 + 0.25 * (counts.total - global_counts.total)
                    reason_parts.append("penalty=sparse_low_peak_inflation_against_global")
                prediction_candidate_count = _prediction_candidate_count(prediction, local_direction or global_direction)
                if (
                    not hidden_crossing_bonus
                    and not multiwave_small_gain_bonus
                    and
                    global_confidence >= 0.55
                    and global_counts.total >= 2
                    and global_counts.total <= 4
                    and global_candidate_count >= 2
                    and global_max_peak <= 1.5
                    and same_direction
                    and counts.total > global_counts.total
                    and prediction_candidate_count <= global_candidate_count
                    and prediction_max_peak <= max(2.0, global_max_peak + 1.0)
                ):
                    score -= 1.2 + 0.35 * (counts.total - global_counts.total)
                    reason_parts.append("penalty=sparse_low_total_overshoot_against_global")
        reason = " ".join(reason_parts)
        scored_candidates.append((score, action_id, prediction, reason))

    for score, action_id, prediction, reason in scored_candidates:
        selection = ExpertSelection(action_id=action_id, score=score, reason=reason, prediction=prediction)
        if best_selection is None or selection.score > best_selection.score:
            best_selection = selection

    if best_selection is None:
        raise ValueError("no matching action_predictions could be scored")
    return best_selection
