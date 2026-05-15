from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from ..types import PassagePrediction, PromptRevision


@dataclass(frozen=True)
class UndercountDecision:
    should_recount: bool
    reason: str
    first_total: int
    dominant_direction: str
    dominant_episode_count: int
    dominant_episode_duration_sec: float


@dataclass(frozen=True)
class PredictionSupport:
    direction: str
    passage_count: int
    total_wave_count: int
    max_peak_simultaneous: int
    total_peak_simultaneous: int
    max_duration_sec: float
    event_count: int


def _prediction_total(prediction: PassagePrediction) -> int:
    return int(prediction.left_count or 0) + int(prediction.right_count or 0)


def _dominant_direction(prediction: PassagePrediction) -> str:
    left = int(prediction.left_count or 0)
    right = int(prediction.right_count or 0)
    if right > left:
        return "right"
    if left > right:
        return "left"
    return "uncertain"


def _episode_count(candidate: dict[str, object]) -> int:
    value = candidate.get("throughput_best_count")
    if value is None:
        value = candidate.get("estimated_count")
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _episode_duration_sec(candidate: dict[str, object]) -> float:
    start = candidate.get("timestamp_start_sec")
    end = candidate.get("timestamp_end_sec")
    try:
        if start is None or end is None:
            return 0.0
        return max(0.0, float(end) - float(start))
    except (TypeError, ValueError):
        return 0.0


def _episode_peak_simultaneous(candidate: dict[str, object]) -> int:
    value = candidate.get("peak_simultaneous_count")
    if value is None:
        return 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _episode_wave_count(candidate: dict[str, object]) -> int:
    value = candidate.get("wave_count")
    if value is None:
        return 1 if _episode_count(candidate) > 0 else 0
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 1 if _episode_count(candidate) > 0 else 0


def _candidate_matches_direction(candidate: dict[str, object], direction: str) -> bool:
    candidate_direction = str(candidate.get("direction") or "").strip().lower()
    return candidate_direction in {direction, "uncertain", ""}


def _episode_support_cap(candidate: dict[str, object]) -> int:
    peak = _episode_peak_simultaneous(candidate)
    if peak <= 0:
        return 0
    return max(peak + 2, int(ceil(peak * 1.75)))


def _event_matches_direction(event: object, direction: str) -> bool:
    event_direction = str(getattr(event, "direction", "") or "").strip().lower()
    return event_direction in {direction, "uncertain", ""}


def _prediction_support(prediction: PassagePrediction, direction: str) -> PredictionSupport:
    passages = [
        candidate
        for candidate in prediction.candidate_passages
        if isinstance(candidate, dict) and _candidate_matches_direction(candidate, direction)
    ]
    peaks = [_episode_peak_simultaneous(candidate) for candidate in passages]
    durations = [_episode_duration_sec(candidate) for candidate in passages]
    total_wave_count = sum(_episode_wave_count(candidate) for candidate in passages)
    event_count = sum(1 for event in prediction.events if _event_matches_direction(event, direction))
    return PredictionSupport(
        direction=direction,
        passage_count=len(passages),
        total_wave_count=total_wave_count,
        max_peak_simultaneous=max(peaks, default=0),
        total_peak_simultaneous=sum(peaks),
        max_duration_sec=max(durations, default=0.0),
        event_count=event_count,
    )


def _matching_passages(prediction: PassagePrediction, direction: str) -> list[dict[str, object]]:
    return [
        candidate
        for candidate in prediction.candidate_passages
        if isinstance(candidate, dict) and _candidate_matches_direction(candidate, direction)
    ]


def detect_undercount_risk(prediction: PassagePrediction) -> UndercountDecision:
    first_total = _prediction_total(prediction)
    dominant_direction = _dominant_direction(prediction)
    if not prediction.parse_success or first_total <= 0 or dominant_direction == "uncertain":
        return UndercountDecision(False, "no_parse_or_no_directional_signal", first_total, dominant_direction, 0, 0.0)

    candidates = [item for item in prediction.candidate_passages if isinstance(item, dict)]
    if not candidates:
        return UndercountDecision(False, "no_candidate_passages", first_total, dominant_direction, 0, 0.0)

    dominant = max(candidates, key=lambda item: _episode_count(item))
    dominant_count = _episode_count(dominant)
    dominant_duration = _episode_duration_sec(dominant)
    dominant_peak = _episode_peak_simultaneous(dominant)
    if dominant_count <= 0:
        return UndercountDecision(False, "candidate_passages_missing_counts", first_total, dominant_direction, 0, dominant_duration)

    if str(dominant.get("direction") or "").strip().lower() not in {dominant_direction, "uncertain"}:
        return UndercountDecision(False, "dominant_episode_direction_mismatch", first_total, dominant_direction, dominant_count, dominant_duration)

    dominant_share = dominant_count / max(1, first_total)
    support = _prediction_support(prediction, dominant_direction)
    suspicious_duration = dominant_duration >= 12.0
    suspicious_share = dominant_share >= 0.7
    suspicious_group = dominant_count >= 3
    suspicious_density = first_total <= max(7, ceil(dominant_duration / 3.0))
    single_episode_evidence = dominant_peak >= 2 or support.event_count >= 2 or support.total_wave_count >= 2
    single_episode_risk = suspicious_duration and suspicious_share and suspicious_group and suspicious_density and single_episode_evidence
    multiwave_risk = (
        support.passage_count >= 2
        and support.total_wave_count >= 2
        and support.total_peak_simultaneous >= 3
        and first_total <= support.total_peak_simultaneous + max(1, support.passage_count - 1)
    )
    should_recount = single_episode_risk or multiwave_risk
    reason = (
        f"dominant_episode_duration={dominant_duration:.1f}s "
        f"dominant_episode_count={dominant_count} first_total={first_total} share={dominant_share:.2f} "
        f"passages={support.passage_count} waves={support.total_wave_count} total_peak={support.total_peak_simultaneous}"
    )
    return UndercountDecision(
        should_recount=should_recount,
        reason=reason,
        first_total=first_total,
        dominant_direction=dominant_direction,
        dominant_episode_count=dominant_count,
        dominant_episode_duration_sec=dominant_duration,
    )


def build_recount_prompt(base_prompt: PromptRevision, first_prediction: PassagePrediction) -> PromptRevision:
    episode_lines = []
    for index, candidate in enumerate(first_prediction.candidate_passages, start=1):
        if not isinstance(candidate, dict):
            continue
        episode_lines.append(
            "- episode "
            f"{index}: start={candidate.get('timestamp_start_sec')} "
            f"end={candidate.get('timestamp_end_sec')} "
            f"direction={candidate.get('direction')} "
            f"estimated_count={candidate.get('estimated_count')} "
            f"peak_simultaneous_count={candidate.get('peak_simultaneous_count')} "
            f"episode_duration_sec={candidate.get('episode_duration_sec')} "
            f"wave_count={candidate.get('wave_count')} "
            f"throughput_best_count={candidate.get('throughput_best_count')} "
            f"note={candidate.get('evidence_note')}"
        )
    prompt_text = "\n".join(
        [
            base_prompt.prompt_text.strip(),
            "",
            "Recount guidance:",
            "High-passage situations often undercount when peak simultaneous occupancy is mistaken for total passage throughput.",
            "Treat the previous pass as a lower-bound seed for episode localization, not as a ceiling on the final count.",
            "Re-evaluate the dominant passage episode for total throughput across time.",
            "Check whether the long stream is better explained as multiple successive waves or mini-bursts rather than one single school summary.",
            "If the long episode contains visible spacing gaps, renewed entries, or density peaks, represent them as multiple `candidate_passages` entries and sum them.",
            "If targets continue entering and exiting over many seconds, count sequential replacements and later waves of entrants over time.",
            "For each candidate passage, explicitly report `peak_simultaneous_count`, `episode_duration_sec`, `wave_count`, and `throughput_best_count`.",
            "Set each passage `estimated_count` equal to its `throughput_best_count`, not to peak occupancy.",
            "Best-effort recount: increase the count when the clip shows sustained replacement, even if only 2 to 3 fish are visible at once.",
            "Do not raise counts without visual evidence of sustained replacement, but do not cap the count at the maximum simultaneously visible fish.",
            "If you only see one sparse or single-wave episode, keep the total close to the distinct visible fish and peak occupancy instead of scaling mainly by duration.",
            "A large increase above peak occupancy needs explicit renewal evidence such as multiple waves, clear spacing gaps, repeated density peaks, or new entrants replacing earlier fish.",
            "Keep the direction conservative and preserve any direction you are confident about.",
            "",
            "Previous pass summary:",
            f"- left_count={first_prediction.left_count}",
            f"- right_count={first_prediction.right_count}",
            f"- commentary={first_prediction.commentary}",
            f"- evidence_summary={first_prediction.evidence_summary}",
            *episode_lines,
        ]
    ).strip()
    return PromptRevision(
        version=base_prompt.version + 1,
        prompt_id=f"{base_prompt.prompt_id}_recount",
        prompt_text=prompt_text,
        critique=base_prompt.critique,
        metrics=base_prompt.metrics,
    )


def should_accept_recount(first_prediction: PassagePrediction, recount_prediction: PassagePrediction) -> tuple[bool, str]:
    if not recount_prediction.parse_success:
        return False, "recount_parse_failed"

    first_total = _prediction_total(first_prediction)
    recount_total = _prediction_total(recount_prediction)
    first_direction = _dominant_direction(first_prediction)
    recount_direction = _dominant_direction(recount_prediction)
    if first_direction != "uncertain" and recount_direction not in {first_direction, "uncertain"}:
        return False, "recount_flipped_direction"

    if recount_total < first_total:
        return False, "recount_lower_total"

    max_allowed = first_total * 2 + 2
    if recount_total > max_allowed:
        return False, f"recount_jump_too_large>{max_allowed}"

    if recount_total == first_total:
        return False, "recount_no_gain"

    support_direction = first_direction if first_direction != "uncertain" else recount_direction
    if support_direction == "uncertain":
        return False, "recount_direction_still_uncertain"

    first_support = _prediction_support(first_prediction, support_direction)
    recount_support = _prediction_support(recount_prediction, support_direction)

    if (
        recount_support.passage_count <= 1
        and recount_support.total_wave_count <= 1
        and recount_support.max_peak_simultaneous > 0
    ):
        single_wave_cap = max(
            recount_support.max_peak_simultaneous + 2,
            int(ceil(recount_support.max_peak_simultaneous * 1.75)),
        )
        if recount_total > single_wave_cap:
            return False, f"recount_single_wave_exceeds_peak_support>{single_wave_cap}"

    if (
        recount_support.passage_count >= 2
        and recount_support.max_peak_simultaneous <= 2
        and recount_support.total_peak_simultaneous > 0
        and recount_support.total_wave_count <= recount_support.passage_count
    ):
        sparse_split_cap = max(
            recount_support.total_peak_simultaneous + 2,
            int(ceil(recount_support.total_peak_simultaneous * 1.5)),
        )
        if recount_total > sparse_split_cap:
            return False, f"recount_sparse_split_exceeds_peak_support>{sparse_split_cap}"

    recount_passages = _matching_passages(recount_prediction, support_direction)
    same_structure_multi_passage_gain = (
        recount_support.passage_count >= 2
        and recount_support.passage_count == first_support.passage_count
        and recount_support.total_wave_count == first_support.total_wave_count
        and recount_support.total_peak_simultaneous >= first_support.total_peak_simultaneous
        and all(_episode_count(candidate) <= _episode_support_cap(candidate) for candidate in recount_passages)
        and recount_total <= recount_support.total_peak_simultaneous + (2 * recount_support.passage_count)
    )

    support_improved = (
        recount_support.passage_count > first_support.passage_count
        or recount_support.total_wave_count > first_support.total_wave_count
        or recount_support.max_peak_simultaneous > first_support.max_peak_simultaneous
        or recount_support.event_count > first_support.event_count
        or recount_support.max_duration_sec > first_support.max_duration_sec + 2.0
        or same_structure_multi_passage_gain
    )
    if not support_improved and recount_total > first_total:
        return False, "recount_increase_without_new_support"

    return True, "recount_increased_total_with_structural_support"
