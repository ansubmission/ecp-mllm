from __future__ import annotations

import json
from typing import Any

from ..types import PassageEvent, PassagePrediction, UpstreamDirection


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


def parse_passage_prediction(
    raw_text: str,
    domain: str,
    clip_id: str,
    prompt_id: str,
    latency_sec: float | None = None,
    usage_metadata: dict[str, Any] | None = None,
    estimated_cost_usd: float | None = None,
    model_name: str | None = None,
) -> PassagePrediction:
    try:
        payload = _extract_json_object(raw_text)
        events = [
            PassageEvent(
                timestamp_sec=float(item["timestamp_sec"]),
                direction=str(item["direction"]),
                confidence=float(item["confidence"]) if item.get("confidence") is not None else None,
                evidence_note=str(item["evidence_note"]) if item.get("evidence_note") is not None else None,
            )
            for item in payload.get("events", [])
        ]
        return PassagePrediction(
            domain=domain,
            clip_id=clip_id,
            scene_assessment=str(payload["scene_assessment"]) if payload.get("scene_assessment") is not None else None,
            candidate_passages=[
                dict(item) for item in payload.get("candidate_passages", []) if isinstance(item, dict)
            ],
            rejected_targets=[
                str(item) for item in payload.get("rejected_targets", []) if item is not None
            ],
            left_count=int(payload["left_count"]) if payload.get("left_count") is not None else None,
            right_count=int(payload["right_count"]) if payload.get("right_count") is not None else None,
            upstream_count=int(payload["upstream_count"]) if payload.get("upstream_count") is not None else None,
            downstream_count=int(payload["downstream_count"]) if payload.get("downstream_count") is not None else None,
            upstream_direction=UpstreamDirection.from_optional(payload.get("upstream_direction")),
            events=events,
            confidence=float(payload["confidence"]) if payload.get("confidence") is not None else None,
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
        return PassagePrediction(
            domain=domain,
            clip_id=clip_id,
            left_count=0,
            right_count=0,
            raw_response=raw_text,
            latency_sec=latency_sec,
            usage_metadata=dict(usage_metadata or {}),
            estimated_cost_usd=estimated_cost_usd,
            model_name=model_name,
            parse_success=False,
            prompt_id=prompt_id,
        )
