from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class InputVariant(str, Enum):
    RAW = "raw"
    SFF3C = "sff3c"
    CFC_3CHANNEL = "cfc_3channel"
    THERMAL_FILTERED = "thermal_filtered"
    THERMAL_NORMALIZED = "thermal_normalized"
    THERMAL_DUAL = "thermal_dual"

    @classmethod
    def from_value(cls, value: str) -> "InputVariant":
        return cls(value.strip().lower())


class UpstreamDirection(str, Enum):
    LEFT = "left"
    RIGHT = "right"

    @classmethod
    def from_optional(cls, value: str | None) -> "UpstreamDirection | None":
        if value is None or value == "":
            return None
        return cls(value.strip().lower())


@dataclass(frozen=True)
class DirectionalCounts:
    left: int = 0
    right: int = 0

    @property
    def total(self) -> int:
        return self.left + self.right

    def to_dict(self) -> dict[str, int]:
        return {"left_count": self.left, "right_count": self.right, "total_count": self.total}

    @classmethod
    def from_left_right(cls, left: int | None, right: int | None) -> "DirectionalCounts":
        return cls(left=int(left or 0), right=int(right or 0))

    @classmethod
    def from_upstream_downstream(
        cls,
        upstream: int | None,
        downstream: int | None,
        upstream_direction: UpstreamDirection | str | None,
    ) -> "DirectionalCounts":
        direction = UpstreamDirection.from_optional(
            upstream_direction.value if isinstance(upstream_direction, UpstreamDirection) else upstream_direction
        )
        if direction == UpstreamDirection.LEFT:
            return cls(left=int(upstream or 0), right=int(downstream or 0))
        if direction == UpstreamDirection.RIGHT:
            return cls(left=int(downstream or 0), right=int(upstream or 0))
        raise ValueError("upstream_direction is required when upstream/downstream counts are provided")


@dataclass(frozen=True)
class ClipKey:
    domain: str
    clip_id: str

    @property
    def value(self) -> str:
        return f"{self.domain}/{self.clip_id}"


@dataclass
class ClipRecord:
    dataset: str
    domain: str
    clip_id: str
    asset_paths: dict[InputVariant, Path]
    upstream_direction: UpstreamDirection | None = None
    width: int | None = None
    height: int | None = None
    framerate: float | None = None
    duration_seconds: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> ClipKey:
        return ClipKey(self.domain, self.clip_id)

    def get_asset_path(self, variant: InputVariant) -> Path | None:
        return self.asset_paths.get(variant)


@dataclass(frozen=True)
class WeakCountRecord:
    domain: str
    clip_id: str
    counts: DirectionalCounts
    source_type: str
    upstream_direction: UpstreamDirection | None = None
    source_path: Path | None = None

    @property
    def key(self) -> ClipKey:
        return ClipKey(self.domain, self.clip_id)


@dataclass(frozen=True)
class PassageEvent:
    timestamp_sec: float
    direction: str
    confidence: float | None = None
    evidence_note: str | None = None


@dataclass
class PassagePrediction:
    domain: str
    clip_id: str
    scene_assessment: str | None = None
    candidate_passages: list[dict[str, Any]] = field(default_factory=list)
    rejected_targets: list[str] = field(default_factory=list)
    left_count: int | None = None
    right_count: int | None = None
    upstream_count: int | None = None
    downstream_count: int | None = None
    upstream_direction: UpstreamDirection | None = None
    events: list[PassageEvent] = field(default_factory=list)
    confidence: float | None = None
    commentary: str | None = None
    evidence_summary: str | None = None
    raw_response: str | None = None
    latency_sec: float | None = None
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    model_name: str | None = None
    parse_success: bool = True
    prompt_id: str = "base"

    @property
    def key(self) -> ClipKey:
        return ClipKey(self.domain, self.clip_id)

    def normalized_counts(self, fallback_upstream_direction: UpstreamDirection | None = None) -> DirectionalCounts:
        if self.left_count is not None or self.right_count is not None:
            return DirectionalCounts.from_left_right(self.left_count, self.right_count)
        return DirectionalCounts.from_upstream_downstream(
            self.upstream_count,
            self.downstream_count,
            self.upstream_direction or fallback_upstream_direction,
        )


@dataclass(frozen=True)
class ClipEvalResult:
    domain: str
    clip_id: str
    truth: DirectionalCounts
    predicted: DirectionalCounts
    abs_error_left: int
    abs_error_right: int
    squared_error_left: int
    squared_error_right: int
    parse_success: bool
    latency_sec: float

    @property
    def clip_mae(self) -> float:
        return (self.abs_error_left + self.abs_error_right) / 2.0


@dataclass(frozen=True)
class DomainSummary:
    domain: str
    clips: int
    mae: float
    rmse: float
    nmae: float | None
    parse_rate: float
    mean_latency_sec: float


@dataclass(frozen=True)
class EvalReport:
    overall_mae: float
    overall_rmse: float
    overall_nmae: float | None
    parse_rate: float
    mean_latency_sec: float
    per_domain: Mapping[str, DomainSummary]
    clip_results: tuple[ClipEvalResult, ...]


@dataclass(frozen=True)
class PromptRevision:
    version: int
    prompt_id: str
    prompt_text: str
    critique: str = ""
    metrics: Mapping[str, float | None] = field(default_factory=dict)
    assistant_prefill: str | None = None


class ReasoningMode(str, Enum):
    GLOBAL_DIRECT = "global_direct"
    LOCAL_DIRECT = "local_direct"
    PROPOSAL_GUIDED = "proposal_guided"
    REPAIR = "repair"
    CUSTOM_TRICKLE_REDUCTION = "custom_trickle_reduction"
    CUSTOM_HIGH_THROUGHPUT = "custom_high_throughput"
    CUSTOM_LOW_COUNT = "custom_low_count"


@dataclass(frozen=True)
class ObservationStream:
    dataset: str
    domain: str
    stream_id: str
    duration_seconds: float | None
    framerate: float | None
    available_variants: tuple[str, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> ClipKey:
        return ClipKey(self.domain, self.stream_id)


@dataclass(frozen=True)
class PerceptionAction:
    action_id: str
    representation: str
    temporal_mode: str
    reasoning_mode: ReasoningMode | str
    proposal_mode: str
    constraint_mode: str
    branch_id: str = "baseline"
    repair_enabled: bool = False
    note: str | None = None


@dataclass(frozen=True)
class EventProposal:
    proposal_id: str
    timestamp_start_sec: float
    timestamp_end_sec: float
    score: float
    source: str
    representation: str
    reasoning: str
    direction_hint: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.timestamp_end_sec - self.timestamp_start_sec)


@dataclass(frozen=True)
class EventHypothesis:
    hypothesis_id: str
    timestamp_start_sec: float
    timestamp_end_sec: float
    direction: str
    count: int
    representation: str
    source: str
    support_score: float | None = None
    peak_simultaneous_count: float | None = None
    wave_count: int | None = None
    evidence_note: str | None = None
    source_event_ids: tuple[str, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.timestamp_end_sec - self.timestamp_start_sec)


@dataclass(frozen=True)
class ConstraintFinding:
    code: str
    severity: str
    message: str
    hypothesis_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConstraintAudit:
    applied: bool
    repaired: bool
    findings: tuple[ConstraintFinding, ...]
    corrected_counts: DirectionalCounts | None = None
    corrected_event_count: int | None = None
    note: str | None = None


@dataclass(frozen=True)
class AgentMemoryRecord:
    domain: str
    failure_type: str
    preferred_representation: str | None = None
    preferred_temporal_mode: str | None = None
    preferred_reasoning_mode: str | None = None
    clip_id: str | None = None
    prompt_id: str | None = None
    count: int = 1
    last_seen_note: str | None = None


@dataclass(frozen=True)
class ExperimentProtocol:
    name: str
    repeats: int = 3
    paired: bool = True
    split_names: tuple[str, ...] = ()
    report_fields: tuple[str, ...] = ("nmae", "mae", "parse_rate", "latency_sec", "cost")
    notes: str | None = None


@dataclass(frozen=True)
class ThermalEventWindow:
    timestamp_start_sec: float | None = None
    timestamp_end_sec: float | None = None
    label: str | None = None
    confidence: float | None = None
    false_positive_score: float | None = None
    evidence_note: str | None = None
    source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float | None:
        if self.timestamp_start_sec is None or self.timestamp_end_sec is None:
            return None
        return max(0.0, self.timestamp_end_sec - self.timestamp_start_sec)


@dataclass
class ThermalPrediction:
    domain: str
    clip_id: str
    clip_labels: list[str] = field(default_factory=list)
    coarse_label: str | None = None
    animal_present: bool | None = None
    false_positive_score: float | None = None
    center_zone_entered: bool | None = None
    center_zone_first_entry_sec: float | None = None
    center_zone_dwell_sec: float | None = None
    event_windows: list[ThermalEventWindow] = field(default_factory=list)
    event_labels: list[str] = field(default_factory=list)
    confidence: float | None = None
    abstain: bool = False
    commentary: str | None = None
    evidence_summary: str | None = None
    raw_response: str | None = None
    latency_sec: float | None = None
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    model_name: str | None = None
    parse_success: bool = True
    prompt_id: str = "thermal_base"

    @property
    def key(self) -> ClipKey:
        return ClipKey(self.domain, self.clip_id)


@dataclass(frozen=True)
class ThermalClipEvalResult:
    domain: str
    clip_id: str
    truth_false_positive: bool
    predicted_false_positive: bool
    truth_coarse_label: str
    predicted_coarse_label: str
    binary_correct: bool
    coarse_correct: bool
    truth_animal_event_count: int
    predicted_animal_event_count: int
    animal_event_count_error: int
    truth_multi_entity: bool
    predicted_multi_entity: bool
    multi_entity_correct: bool
    animal_event_recall: float | None
    animal_event_precision: float | None
    animal_event_mean_tiou: float | None
    animal_event_label_accuracy: float | None
    truth_center_zone_entered: bool
    predicted_center_zone_entered: bool
    center_zone_entry_correct: bool
    truth_center_zone_first_entry_sec: float | None
    predicted_center_zone_first_entry_sec: float | None
    center_zone_first_entry_error_sec: float | None
    truth_center_zone_dwell_sec: float | None
    predicted_center_zone_dwell_sec: float | None
    center_zone_dwell_error_sec: float | None
    abstain: bool
    latency_sec: float


@dataclass(frozen=True)
class ThermalEvalReport:
    binary_accuracy: float
    binary_f1: float
    binary_balanced_accuracy: float
    coarse_macro_f1: float
    per_class_recall: Mapping[str, float]
    animal_event_count_mae: float | None
    animal_event_window_recall: float | None
    animal_event_window_precision: float | None
    animal_event_window_mean_tiou: float | None
    animal_event_label_accuracy: float | None
    center_zone_entry_accuracy: float | None
    center_zone_entry_f1: float | None
    center_zone_first_entry_mae_sec: float | None
    center_zone_dwell_mae_sec: float | None
    multi_entity_accuracy: float | None
    abstention_rate: float
    mean_latency_sec: float
    clip_results: tuple[ThermalClipEvalResult, ...]


@dataclass(frozen=True)
class TrackletPoint:
    timestamp_sec: float | None = None
    x: float | None = None
    y: float | None = None
    w: float | None = None
    h: float | None = None
    frame_index: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackletObservation:
    dataset: str
    domain: str
    clip_id: str
    tracklet_id: str
    source_track_id: str | None = None
    source: str = "synthetic_frag_v1"
    start_sec: float | None = None
    end_sec: float | None = None
    label_hint: str | None = None
    coarse_label_hint: str | None = None
    direction_hint: str | None = None
    confidence: float | None = None
    is_false_positive_hint: bool | None = None
    points: tuple[TrackletPoint, ...] = ()
    features: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> ClipKey:
        return ClipKey(self.domain, self.clip_id)

    @property
    def duration_sec(self) -> float | None:
        if self.start_sec is None or self.end_sec is None:
            return None
        return max(0.0, self.end_sec - self.start_sec)


@dataclass(frozen=True)
class TrackletGroupPrediction:
    group_id: str
    tracklet_ids: tuple[str, ...]
    label: str | None = None
    direction: str | None = None
    count: int | None = None
    confidence: float | None = None
    false_positive_score: float | None = None
    evidence_note: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackletEventPrediction:
    event_id: str
    tracklet_ids: tuple[str, ...]
    timestamp_start_sec: float | None = None
    timestamp_end_sec: float | None = None
    label: str | None = None
    direction: str | None = None
    count: int | None = None
    confidence: float | None = None
    false_positive_score: float | None = None
    evidence_note: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def duration_sec(self) -> float | None:
        if self.timestamp_start_sec is None or self.timestamp_end_sec is None:
            return None
        return max(0.0, self.timestamp_end_sec - self.timestamp_start_sec)


@dataclass
class TrackletRepairPrediction:
    domain: str
    clip_id: str
    tracklet_groups: list[TrackletGroupPrediction] = field(default_factory=list)
    rejected_tracklets: list[str] = field(default_factory=list)
    event_predictions: list[TrackletEventPrediction] = field(default_factory=list)
    event_count: int | None = None
    clip_labels: list[str] = field(default_factory=list)
    coarse_label: str | None = None
    animal_present: bool | None = None
    false_positive_score: float | None = None
    confidence: float | None = None
    commentary: str | None = None
    evidence_summary: str | None = None
    raw_response: str | None = None
    latency_sec: float | None = None
    usage_metadata: dict[str, Any] = field(default_factory=dict)
    estimated_cost_usd: float | None = None
    model_name: str | None = None
    parse_success: bool = True
    prompt_id: str = "tracklet_repair"

    @property
    def key(self) -> ClipKey:
        return ClipKey(self.domain, self.clip_id)


@dataclass(frozen=True)
class TrackletRepairClipEvalResult:
    dataset: str
    domain: str
    clip_id: str
    truth_event_count: int
    predicted_event_count: int
    event_count_error: int
    merge_pair_accuracy: float | None
    event_recall: float | None
    event_precision: float | None
    event_mean_tiou: float | None
    truth_false_positive: bool | None = None
    predicted_false_positive: bool | None = None
    false_positive_correct: bool | None = None
    latency_sec: float = 0.0


@dataclass(frozen=True)
class TrackletRepairEvalReport:
    event_count_mae: float | None
    merge_pair_accuracy: float | None
    event_recall: float | None
    event_precision: float | None
    event_mean_tiou: float | None
    false_positive_accuracy: float | None
    mean_latency_sec: float
    clip_results: tuple[TrackletRepairClipEvalResult, ...]
