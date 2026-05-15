from __future__ import annotations

import re
from typing import Iterable

from ..agent.event_centric import prediction_to_event_hypotheses
from ..types import ConstraintAudit, ConstraintFinding, DirectionalCounts, EventHypothesis, PassagePrediction


def _overlap_ratio(first: EventHypothesis, second: EventHypothesis) -> float:
    start = max(first.timestamp_start_sec, second.timestamp_start_sec)
    end = min(first.timestamp_end_sec, second.timestamp_end_sec)
    if end <= start:
        return 0.0
    overlap = end - start
    shorter = min(first.duration_sec, second.duration_sec)
    if shorter <= 0:
        return 0.0
    return overlap / shorter


def _supported_counts(hypotheses: Iterable[EventHypothesis]) -> DirectionalCounts:
    left = 0
    right = 0
    for hypothesis in hypotheses:
        direction = hypothesis.direction.lower()
        if direction == "left":
            left += hypothesis.count
        elif direction == "right":
            right += hypothesis.count
    return DirectionalCounts(left=left, right=right)


def _dominant_direction(counts: DirectionalCounts) -> str | None:
    if counts.left > counts.right:
        return "left"
    if counts.right > counts.left:
        return "right"
    return None


def _text_direction_vote(text: object) -> str | None:
    if not isinstance(text, str):
        return None
    lowered = re.sub(r"\s+", " ", text.strip().lower())
    right_phrases = ("left to right", "left-to-right", "moving rightward", "moves rightward", "towards the right")
    left_phrases = ("right to left", "right-to-left", "moving leftward", "moves leftward", "towards the left")
    has_right = any(phrase in lowered for phrase in right_phrases)
    has_left = any(phrase in lowered for phrase in left_phrases)
    if has_right and not has_left:
        return "right"
    if has_left and not has_right:
        return "left"
    return None


def _peak_count(hypothesis: EventHypothesis) -> float:
    try:
        return float(hypothesis.peak_simultaneous_count or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _should_preserve_short_precursor(first: EventHypothesis, second: EventHypothesis) -> bool:
    if first.direction != second.direction:
        return False
    short_event, long_event = (first, second) if first.duration_sec <= second.duration_sec else (second, first)
    if short_event.duration_sec > 3.0 or long_event.duration_sec < 8.0:
        return False
    if short_event.count != 1 or _peak_count(short_event) > 1.0:
        return False
    if long_event.count < 3 or _peak_count(long_event) < 3.0:
        return False
    if short_event.timestamp_end_sec < long_event.timestamp_start_sec:
        return False
    lead_offset = max(0.0, short_event.timestamp_start_sec - long_event.timestamp_start_sec)
    if lead_offset > 1.0:
        return False
    overlap = min(short_event.timestamp_end_sec, long_event.timestamp_end_sec) - max(
        short_event.timestamp_start_sec,
        long_event.timestamp_start_sec,
    )
    if overlap <= 0:
        return False
    # Preserve a short precursor when it is fully embedded at the leading edge of a much
    # longer, denser event; this often corresponds to a briefly separated entrant that
    # should not be collapsed into the later main wave.
    return short_event.timestamp_end_sec <= long_event.timestamp_start_sec + max(3.0, 0.3 * long_event.duration_sec)


def _text_direction_consistency_findings(
    prediction: PassagePrediction,
    counts: DirectionalCounts,
) -> list[ConstraintFinding]:
    dominant = _dominant_direction(counts)
    if dominant is None:
        return []
    votes: list[str] = []
    for text in (prediction.commentary, prediction.evidence_summary):
        vote = _text_direction_vote(text)
        if vote is not None:
            votes.append(vote)
    for candidate in prediction.candidate_passages:
        if not isinstance(candidate, dict):
            continue
        vote = _text_direction_vote(candidate.get("evidence_note"))
        if vote is not None:
            votes.append(vote)
    if not votes:
        return []
    right_votes = votes.count("right")
    left_votes = votes.count("left")
    text_direction = "right" if right_votes > left_votes else "left" if left_votes > right_votes else None
    if text_direction is None or text_direction == dominant:
        return []
    severity = "high" if max(right_votes, left_votes) >= 2 else "medium"
    return [
        ConstraintFinding(
            code="text_direction_conflicts_with_structured_counts",
            severity=severity,
            message=(
                f"structured_direction={dominant} conflicts with text_direction={text_direction} "
                f"(votes left={left_votes} right={right_votes})"
            ),
        )
    ]


def _merge_overlaps(
    hypotheses: list[EventHypothesis],
    *,
    overlap_tolerance: float,
) -> tuple[list[EventHypothesis], list[ConstraintFinding]]:
    if not hypotheses:
        return [], []
    ordered = sorted(hypotheses, key=lambda item: (item.direction, item.timestamp_start_sec, item.timestamp_end_sec))
    merged: list[EventHypothesis] = []
    findings: list[ConstraintFinding] = []

    for item in ordered:
        if not merged:
            merged.append(item)
            continue
        previous = merged[-1]
        if (
            previous.direction == item.direction
            and _overlap_ratio(previous, item) >= overlap_tolerance
            and not _should_preserve_short_precursor(previous, item)
        ):
            merged[-1] = EventHypothesis(
                hypothesis_id=previous.hypothesis_id,
                timestamp_start_sec=min(previous.timestamp_start_sec, item.timestamp_start_sec),
                timestamp_end_sec=max(previous.timestamp_end_sec, item.timestamp_end_sec),
                direction=previous.direction,
                count=max(previous.count, item.count),
                representation=previous.representation,
                source=previous.source,
                support_score=max(previous.support_score or 0.0, item.support_score or 0.0) or None,
                peak_simultaneous_count=max(previous.peak_simultaneous_count or 0.0, item.peak_simultaneous_count or 0.0) or None,
                wave_count=max(previous.wave_count or 0, item.wave_count or 0) or None,
                evidence_note=previous.evidence_note or item.evidence_note,
                source_event_ids=previous.source_event_ids + item.source_event_ids,
                metadata=dict(previous.metadata),
            )
            findings.append(
                ConstraintFinding(
                    code="overlapping_same_direction_events",
                    severity="medium",
                    message=(
                        f"merged {previous.hypothesis_id} and {item.hypothesis_id} due to "
                        f"same-direction overlap"
                    ),
                    hypothesis_ids=(previous.hypothesis_id, item.hypothesis_id),
                )
            )
        else:
            merged.append(item)
    return merged, findings


def audit_event_hypotheses(
    hypotheses: Iterable[EventHypothesis],
    *,
    final_counts: DirectionalCounts | None = None,
    overlap_tolerance: float = 0.5,
) -> ConstraintAudit:
    source = list(hypotheses)
    merged, findings = _merge_overlaps(source, overlap_tolerance=overlap_tolerance)
    supported = _supported_counts(merged)
    corrected = final_counts
    repaired = bool(findings)

    if final_counts is not None:
        if final_counts.left > supported.left or final_counts.right > supported.right:
            findings.append(
                ConstraintFinding(
                    code="global_count_exceeds_supported_events",
                    severity="high",
                    message=(
                        f"final_counts=({final_counts.left},{final_counts.right}) exceed "
                        f"supported=({supported.left},{supported.right})"
                    ),
                )
            )
            corrected = DirectionalCounts(
                left=min(final_counts.left, supported.left),
                right=min(final_counts.right, supported.right),
            )
            repaired = True

    return ConstraintAudit(
        applied=True,
        repaired=repaired,
        findings=tuple(findings),
        corrected_counts=corrected,
        corrected_event_count=len(merged),
        note="event-centric physical consistency audit",
    )


def audit_prediction_constraints(
    prediction: PassagePrediction,
    *,
    representation: str = "unknown",
    fallback_upstream_direction=None,
    overlap_tolerance: float = 0.5,
) -> ConstraintAudit:
    hypotheses = prediction_to_event_hypotheses(prediction, representation=representation)
    counts = prediction.normalized_counts(fallback_upstream_direction)
    audit = audit_event_hypotheses(hypotheses, final_counts=counts, overlap_tolerance=overlap_tolerance)
    extra_findings = _text_direction_consistency_findings(prediction, counts)
    if not extra_findings:
        return audit
    return ConstraintAudit(
        applied=audit.applied,
        repaired=audit.repaired,
        findings=tuple([*audit.findings, *extra_findings]),
        corrected_counts=audit.corrected_counts,
        corrected_event_count=audit.corrected_event_count,
        note=audit.note,
    )
