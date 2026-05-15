from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from ..types import (
    AgentMemoryRecord,
    ClipRecord,
    EventHypothesis,
    EventProposal,
    ObservationStream,
    PassagePrediction,
    PerceptionAction,
    ReasoningMode,
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


def build_observation_stream(clip: ClipRecord, available_variants: Iterable[str] | None = None) -> ObservationStream:
    variants = tuple(
        sorted(
            {
                str(item)
                for item in (
                    available_variants
                    if available_variants is not None
                    else [variant.value for variant in clip.asset_paths.keys()]
                )
                if item
            }
        )
    )
    return ObservationStream(
        dataset=clip.dataset,
        domain=clip.domain,
        stream_id=clip.clip_id,
        duration_seconds=clip.duration_seconds,
        framerate=clip.framerate,
        available_variants=variants,
        metadata=dict(clip.metadata),
    )


def enumerate_perception_actions(stream: ObservationStream) -> list[PerceptionAction]:
    actions: list[PerceptionAction] = []
    for representation in stream.available_variants or ("sff3c",):
        actions.extend(
            [
                PerceptionAction(
                    action_id=f"{representation}:global",
                    representation=representation,
                    temporal_mode="full_stream",
                    reasoning_mode=ReasoningMode.GLOBAL_DIRECT,
                    proposal_mode="none",
                    constraint_mode="audit_only",
                    branch_id="baseline",
                    repair_enabled=False,
                    note="direct global reasoning over the whole observation stream",
                ),
                PerceptionAction(
                    action_id=f"{representation}:proposal_guided",
                    representation=representation,
                    temporal_mode="event_windows",
                    reasoning_mode=ReasoningMode.PROPOSAL_GUIDED,
                    proposal_mode="motion_energy",
                    constraint_mode="repair_if_needed",
                    branch_id="baseline",
                    repair_enabled=True,
                    note="proposal-guided event verification over candidate windows",
                ),
                PerceptionAction(
                    action_id=f"{representation}:local_repair",
                    representation=representation,
                    temporal_mode="event_windows",
                    reasoning_mode=ReasoningMode.REPAIR,
                    proposal_mode="backbone_candidates",
                    constraint_mode="repair_if_needed",
                    branch_id="repair",
                    repair_enabled=True,
                    note="repair-oriented pass over candidate events returned by the backbone",
                ),
                PerceptionAction(
                    action_id=f"{representation}:custom_trickle_reduction",
                    representation=representation,
                    temporal_mode="full_stream",
                    reasoning_mode=ReasoningMode.CUSTOM_TRICKLE_REDUCTION,
                    proposal_mode="routed_prompt",
                    constraint_mode="guardrailed_accept",
                    branch_id="custom_trickle_reduction",
                    repair_enabled=False,
                    note="routed prompt branch for sparse trickle overcount reduction",
                ),
                PerceptionAction(
                    action_id=f"{representation}:custom_high_throughput",
                    representation=representation,
                    temporal_mode="full_stream",
                    reasoning_mode=ReasoningMode.CUSTOM_HIGH_THROUGHPUT,
                    proposal_mode="routed_prompt",
                    constraint_mode="guardrailed_accept",
                    branch_id="custom_high_throughput",
                    repair_enabled=False,
                    note="routed prompt branch for multi-wave high-peak undercount risk",
                ),
                PerceptionAction(
                    action_id=f"{representation}:custom_low_count",
                    representation=representation,
                    temporal_mode="full_stream",
                    reasoning_mode=ReasoningMode.CUSTOM_LOW_COUNT,
                    proposal_mode="routed_prompt",
                    constraint_mode="guardrailed_accept",
                    branch_id="custom_low_count",
                    repair_enabled=False,
                    note="routed prompt branch for low-count far-view detectability risk",
                ),
            ]
        )
    return actions


def prediction_to_event_proposals(
    prediction: PassagePrediction,
    *,
    representation: str = "unknown",
    source: str = "backbone_candidate_passages",
) -> list[EventProposal]:
    proposals: list[EventProposal] = []
    for index, candidate in enumerate(prediction.candidate_passages, start=1):
        if not isinstance(candidate, dict):
            continue
        start_sec = _safe_float(candidate.get("timestamp_start_sec"))
        end_sec = _safe_float(candidate.get("timestamp_end_sec"))
        duration_sec = max(0.0, end_sec - start_sec)
        score = _safe_float(candidate.get("throughput_best_count"))
        if score <= 0:
            score = _safe_float(candidate.get("estimated_count"))
        if score <= 0:
            score = _safe_float(candidate.get("peak_simultaneous_count"))
        if score <= 0:
            score = 1.0 if duration_sec > 0 else 0.0
        proposals.append(
            EventProposal(
                proposal_id=f"{prediction.clip_id}:proposal:{index:02d}",
                timestamp_start_sec=start_sec,
                timestamp_end_sec=end_sec,
                score=score,
                source=source,
                representation=representation,
                reasoning="candidate_passage_projection",
                direction_hint=str(candidate.get("direction")).lower() if candidate.get("direction") else None,
                metadata=dict(candidate),
            )
        )
    return proposals


def prediction_to_event_hypotheses(
    prediction: PassagePrediction,
    *,
    representation: str = "unknown",
    source: str = "backbone_candidate_passages",
) -> list[EventHypothesis]:
    hypotheses: list[EventHypothesis] = []
    proposals = prediction_to_event_proposals(prediction, representation=representation, source=source)
    for index, proposal in enumerate(proposals, start=1):
        metadata = dict(proposal.metadata)
        count = _safe_int(metadata.get("throughput_best_count"))
        if count <= 0:
            count = _safe_int(metadata.get("estimated_count"))
        if count <= 0:
            count = max(1, _safe_int(metadata.get("peak_simultaneous_count"), 1))
        hypotheses.append(
            EventHypothesis(
                hypothesis_id=f"{prediction.clip_id}:event:{index:02d}",
                timestamp_start_sec=proposal.timestamp_start_sec,
                timestamp_end_sec=proposal.timestamp_end_sec,
                direction=str(metadata.get("direction") or proposal.direction_hint or "unknown").lower(),
                count=count,
                representation=representation,
                source=source,
                support_score=float(prediction.confidence) if prediction.confidence is not None else None,
                peak_simultaneous_count=_safe_float(metadata.get("peak_simultaneous_count")) or None,
                wave_count=_safe_int(metadata.get("wave_count")) or None,
                evidence_note=str(metadata.get("evidence_note")) if metadata.get("evidence_note") is not None else None,
                source_event_ids=(proposal.proposal_id,),
                metadata=metadata,
            )
        )
    return hypotheses


def remember_failure(
    memory: Sequence[AgentMemoryRecord],
    *,
    domain: str,
    failure_type: str,
    action: PerceptionAction,
    clip_id: str | None = None,
    prompt_id: str | None = None,
    note: str | None = None,
) -> list[AgentMemoryRecord]:
    updated = list(memory)
    updated.append(
        AgentMemoryRecord(
            domain=domain,
            failure_type=failure_type,
            preferred_representation=action.representation,
            preferred_temporal_mode=action.temporal_mode,
            preferred_reasoning_mode=str(action.reasoning_mode),
            clip_id=clip_id,
            prompt_id=prompt_id,
            count=1,
            last_seen_note=note,
        )
    )
    return updated


def suggest_perception_actions(
    stream: ObservationStream,
    memory: Sequence[AgentMemoryRecord],
    *,
    limit: int = 3,
) -> list[PerceptionAction]:
    actions = enumerate_perception_actions(stream)
    memory_hits: dict[tuple[str, str, str], int] = {}
    for item in memory:
        if item.domain != stream.domain:
            continue
        key = (
            item.preferred_representation or "",
            item.preferred_temporal_mode or "",
            item.preferred_reasoning_mode or "",
        )
        memory_hits[key] = memory_hits.get(key, 0) + max(1, int(item.count))

    scored: list[tuple[int, PerceptionAction]] = []
    for action in actions:
        key = (action.representation, action.temporal_mode, str(action.reasoning_mode))
        penalty = memory_hits.get(key, 0)
        bonus = 0
        if action.representation == "sff3c":
            bonus += 2
        if str(action.reasoning_mode) == ReasoningMode.PROPOSAL_GUIDED.value:
            bonus += 2
        if action.repair_enabled:
            bonus += 1
        scored.append((bonus - penalty, action))
    scored.sort(key=lambda item: (item[0], item[1].action_id), reverse=True)
    return [action for _, action in scored[:limit]]


@dataclass(frozen=True)
class ExpertActionScore:
    action: PerceptionAction
    score: float
    reason: str
