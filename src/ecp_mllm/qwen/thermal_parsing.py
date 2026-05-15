from __future__ import annotations

import json
from typing import Any

from ..thermal import collapse_thermal_labels, map_thermal_label_to_coarse
from ..types import ThermalEventWindow, ThermalPrediction


def _optional_bool(value: object | None) -> bool | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    return None


def _optional_float(value: object | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_json_object(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if not text:
        raise ValueError("empty response")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or start >= end:
            raise
        payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response must decode to a JSON object")
    return payload


def _heuristic_false_positive(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized:
        return False
    strong_terms = (
        "false positive",
        "no animal",
        "no fish",
        "no wildlife",
        "no biological",
        "stationary bright",
        "stationary object",
        "stationary artifact",
        "hot pixel",
        "sensor artifact",
        "fixed clutter",
        "static thermal landscape",
    )
    return any(term in normalized for term in strong_terms)


def _heuristic_positive_animal(text: str) -> bool:
    normalized = text.strip().lower()
    if not normalized or _heuristic_false_positive(normalized):
        return False
    strong_terms = (
        "animal event",
        "real animal",
        "single animal",
        "wildlife",
        "rodent",
        "possum",
        "hedgehog",
        "mustelid",
        "mammal",
        "bird or bat",
        "bird/bat",
        "bird event",
        "bat-like",
    )
    return any(term in normalized for term in strong_terms)


def _candidate_passages_support_animal(payload: dict[str, Any]) -> bool:
    candidates = payload.get("candidate_passages")
    if not isinstance(candidates, list) or not candidates:
        return False
    fallback_text = " ".join(
        str(payload.get(key) or "")
        for key in ("commentary", "evidence_summary", "scene_assessment")
    ).strip()
    normalized = fallback_text.lower()
    if _heuristic_false_positive(normalized):
        return False
    motion_terms = (
        "coherent motion",
        "moves consistently",
        "moving from",
        "moving left",
        "moving right",
        "trajectory",
        "track observed",
        "target moving",
        "drifts from",
        "continuous linear motion",
    )
    motion_hint = any(term in normalized for term in motion_terms)
    best_confidence = float(payload.get("confidence") or 0.0)
    longest_duration = 0.0
    for item in candidates:
        if not isinstance(item, dict):
            continue
        if item.get("confidence") is not None:
            best_confidence = max(best_confidence, float(item["confidence"]))
        start = item.get("timestamp_start_sec")
        end = item.get("timestamp_end_sec")
        if start is not None and end is not None:
            longest_duration = max(longest_duration, max(0.0, float(end) - float(start)))
    return best_confidence >= 0.60 and (motion_hint or longest_duration >= 5.0)


def _salvage_center_zone_fields(
    payload: dict[str, Any],
    *,
    coarse_label: str,
    animal_present: bool | None,
) -> tuple[bool | None, float | None, float | None]:
    entered = _optional_bool(payload.get("center_zone_entered"))
    first_entry_sec = _optional_float(payload.get("center_zone_first_entry_sec"))
    dwell_sec = _optional_float(payload.get("center_zone_dwell_sec"))
    if entered is not None:
        if entered:
            return True, first_entry_sec, dwell_sec
        return False, None, None

    if animal_present is not True and coarse_label == "false_positive":
        return False, None, None

    fallback_text = " ".join(
        str(payload.get(key) or "")
        for key in ("commentary", "evidence_summary", "scene_assessment")
    ).strip().lower()
    if not fallback_text:
        return None, None, None

    negative_terms = (
        "never reaches the center",
        "does not enter the center",
        "stays near the edge",
        "remains at the edge",
        "upper edge",
        "lower edge",
        "left edge",
        "right edge",
    )
    if any(term in fallback_text for term in negative_terms):
        return False, None, None

    positive_terms = (
        "enters the center",
        "into the center",
        "through the center",
        "crosses the center",
        "crosses the middle",
        "through the middle",
        "into mid-frame",
        "mid-frame",
        "middle of the frame",
        "center of the frame",
        "center-right",
        "center-left",
    )
    if any(term in fallback_text for term in positive_terms):
        return True, first_entry_sec, dwell_sec

    return None, first_entry_sec, dwell_sec


def _salvage_payload(payload: dict[str, Any]) -> tuple[list[str], str, bool | None, float | None, list[str]]:
    clip_labels = collapse_thermal_labels(payload.get("clip_labels"))
    coarse_label = map_thermal_label_to_coarse(payload.get("coarse_label"))
    animal_present = _optional_bool(payload.get("animal_present"))
    false_positive_score = float(payload["false_positive_score"]) if payload.get("false_positive_score") is not None else None
    event_labels = collapse_thermal_labels(payload.get("event_labels"))

    if coarse_label != "other" or clip_labels:
        return clip_labels, coarse_label, animal_present, false_positive_score, event_labels

    fallback_text = " ".join(
        str(payload.get(key) or "")
        for key in ("commentary", "evidence_summary", "scene_assessment")
    ).strip()
    if _heuristic_false_positive(fallback_text):
        return ["false_positive"], "false_positive", False if animal_present is None else animal_present, 0.95 if false_positive_score is None else false_positive_score, ["false_positive"]
    if _heuristic_positive_animal(fallback_text):
        return ["other"], "other", True if animal_present is None else animal_present, 0.05 if false_positive_score is None else false_positive_score, ["other"]
    if _candidate_passages_support_animal(payload):
        return ["other"], "other", True if animal_present is None else animal_present, 0.05 if false_positive_score is None else false_positive_score, ["other"]

    return clip_labels, coarse_label, animal_present, false_positive_score, event_labels


def parse_thermal_prediction(
    raw_text: str,
    *,
    domain: str,
    clip_id: str,
    prompt_id: str,
    latency_sec: float | None = None,
    usage_metadata: dict[str, Any] | None = None,
    estimated_cost_usd: float | None = None,
    model_name: str | None = None,
) -> ThermalPrediction:
    try:
        payload = _extract_json_object(raw_text)
        clip_labels, coarse_label, animal_present, false_positive_score, event_labels = _salvage_payload(payload)
        center_zone_entered, center_zone_first_entry_sec, center_zone_dwell_sec = _salvage_center_zone_fields(
            payload,
            coarse_label=coarse_label,
            animal_present=animal_present,
        )
        event_windows = []
        for item in payload.get("event_windows", []):
            if not isinstance(item, dict):
                continue
            event_windows.append(
                ThermalEventWindow(
                    timestamp_start_sec=float(item["timestamp_start_sec"]) if item.get("timestamp_start_sec") is not None else None,
                    timestamp_end_sec=float(item["timestamp_end_sec"]) if item.get("timestamp_end_sec") is not None else None,
                    label=map_thermal_label_to_coarse(item.get("label")),
                    confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                    false_positive_score=float(item["false_positive_score"]) if item.get("false_positive_score") is not None else None,
                    evidence_note=str(item["evidence_note"]) if item.get("evidence_note") is not None else None,
                    source="model",
                )
            )
        if not event_windows and animal_present is True:
            for item in payload.get("candidate_passages", []):
                if not isinstance(item, dict):
                    continue
                event_windows.append(
                    ThermalEventWindow(
                        timestamp_start_sec=float(item["timestamp_start_sec"]) if item.get("timestamp_start_sec") is not None else None,
                        timestamp_end_sec=float(item["timestamp_end_sec"]) if item.get("timestamp_end_sec") is not None else None,
                        label=coarse_label if coarse_label != "false_positive" else "other",
                        confidence=float(item["confidence"]) if item.get("confidence") is not None else float(payload["confidence"]) if payload.get("confidence") is not None else None,
                        false_positive_score=false_positive_score,
                        evidence_note=str(item["evidence_note"]) if item.get("evidence_note") is not None else None,
                        source="model",
                    )
                )
        return ThermalPrediction(
            domain=domain,
            clip_id=clip_id,
            clip_labels=clip_labels,
            coarse_label=coarse_label,
            animal_present=animal_present,
            false_positive_score=false_positive_score,
            center_zone_entered=center_zone_entered,
            center_zone_first_entry_sec=center_zone_first_entry_sec,
            center_zone_dwell_sec=center_zone_dwell_sec,
            event_windows=event_windows,
            event_labels=event_labels,
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
            abstain=bool(payload.get("abstain", False)),
            commentary=str(payload["commentary"]) if payload.get("commentary") is not None else None,
            evidence_summary=str(payload["evidence_summary"]) if payload.get("evidence_summary") is not None else None,
            raw_response=raw_text,
            latency_sec=latency_sec,
            usage_metadata=dict(usage_metadata or {}),
            estimated_cost_usd=estimated_cost_usd,
            model_name=model_name,
            parse_success=True,
            prompt_id=prompt_id,
        )
    except Exception:
        return ThermalPrediction(
            domain=domain,
            clip_id=clip_id,
            clip_labels=[],
            coarse_label="other",
            animal_present=None,
            false_positive_score=0.5,
            center_zone_entered=None,
            center_zone_first_entry_sec=None,
            center_zone_dwell_sec=None,
            event_windows=[],
            event_labels=[],
            confidence=None,
            abstain=True,
            commentary=None,
            evidence_summary=None,
            raw_response=raw_text,
            latency_sec=latency_sec,
            usage_metadata=dict(usage_metadata or {}),
            estimated_cost_usd=estimated_cost_usd,
            model_name=model_name,
            parse_success=False,
            prompt_id=prompt_id,
        )
