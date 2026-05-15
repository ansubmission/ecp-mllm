from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from ..eval.thermal_eval import evaluate_thermal_predictions
from ..qwen.prompt_archive import write_prompt_artifact
from ..thermal import collapse_thermal_labels
from ..types import ClipRecord, InputVariant, PromptRevision, ThermalEventWindow, ThermalPrediction
from ..video import (
    compose_side_by_side_mp4,
    dual_probe_mp4_path,
    overlay_center_zone_mp4,
    overlay_probe_mp4_path,
    trim_mp4,
    trimmed_probe_mp4_path,
)


@dataclass(frozen=True)
class ThermalBatchClipSpec:
    bucket: str
    clip_id: str


def load_thermal_batch_specs(path: str | Path) -> list[ThermalBatchClipSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("thermal clip file must decode to a list")
    specs: list[ThermalBatchClipSpec] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError("each thermal clip entry must be an object")
        specs.append(
            ThermalBatchClipSpec(
                bucket=str(item.get("bucket") or "thermal").strip(),
                clip_id=str(item["clip_id"]).strip(),
            )
        )
    return specs


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "-" for ch in value).strip("-") or "thermal"


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def format_stage_details(details: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in details.items():
        if value in (None, ""):
            continue
        if isinstance(value, float):
            parts.append(f"{key}={value:.2f}")
        else:
            parts.append(f"{key}={value}")
    return " ".join(parts)


def thermal_prediction_payload(prediction: ThermalPrediction) -> dict[str, Any]:
    return {
        "clip_labels": list(prediction.clip_labels),
        "coarse_label": prediction.coarse_label,
        "animal_present": prediction.animal_present,
        "false_positive_score": prediction.false_positive_score,
        "center_zone_entered": prediction.center_zone_entered,
        "center_zone_first_entry_sec": prediction.center_zone_first_entry_sec,
        "center_zone_dwell_sec": prediction.center_zone_dwell_sec,
        "event_windows": [asdict(window) for window in prediction.event_windows],
        "event_labels": list(prediction.event_labels),
        "confidence": prediction.confidence,
        "abstain": prediction.abstain,
        "commentary": prediction.commentary,
        "evidence_summary": prediction.evidence_summary,
        "latency_sec": prediction.latency_sec,
        "estimated_cost_usd": prediction.estimated_cost_usd,
        "usage_metadata": dict(prediction.usage_metadata),
        "model_name": prediction.model_name,
        "parse_success": prediction.parse_success,
        "prompt_id": prediction.prompt_id,
    }


def thermal_prediction_from_payload(
    payload: dict[str, Any],
    *,
    domain: str,
    clip_id: str,
) -> ThermalPrediction:
    return ThermalPrediction(
        domain=domain,
        clip_id=clip_id,
        clip_labels=list(payload.get("clip_labels") or []),
        coarse_label=payload.get("coarse_label"),
        animal_present=payload.get("animal_present"),
        false_positive_score=payload.get("false_positive_score"),
        center_zone_entered=payload.get("center_zone_entered"),
        center_zone_first_entry_sec=payload.get("center_zone_first_entry_sec"),
        center_zone_dwell_sec=payload.get("center_zone_dwell_sec"),
        event_windows=[
            ThermalEventWindow(
                timestamp_start_sec=item.get("timestamp_start_sec"),
                timestamp_end_sec=item.get("timestamp_end_sec"),
                label=item.get("label"),
                confidence=item.get("confidence"),
                false_positive_score=item.get("false_positive_score"),
                evidence_note=item.get("evidence_note"),
                source=item.get("source"),
                metadata=dict(item.get("metadata") or {}),
            )
            for item in payload.get("event_windows", [])
            if isinstance(item, dict)
        ],
        event_labels=list(payload.get("event_labels") or []),
        confidence=payload.get("confidence"),
        abstain=bool(payload.get("abstain", False)),
        commentary=payload.get("commentary"),
        evidence_summary=payload.get("evidence_summary"),
        latency_sec=payload.get("latency_sec"),
        usage_metadata=dict(payload.get("usage_metadata") or {}),
        estimated_cost_usd=payload.get("estimated_cost_usd"),
        model_name=payload.get("model_name"),
        parse_success=bool(payload.get("parse_success", True)),
        prompt_id=str(payload.get("prompt_id") or "thermal"),
    )


def materialize_thermal_variant(
    settings,
    clip: ClipRecord,
    variant: InputVariant,
    *,
    stitch_height: int = 360,
    stitch_crf: int = 30,
    zone_overlay: bool = False,
) -> tuple[ClipRecord, Path | None]:
    metadata = dict(clip.metadata)
    if variant == InputVariant.THERMAL_DUAL:
        filtered_path = clip.get_asset_path(InputVariant.THERMAL_FILTERED)
        normalized_path = clip.get_asset_path(InputVariant.THERMAL_NORMALIZED)
        if filtered_path is None or normalized_path is None:
            raise RuntimeError(f"thermal_dual requires filtered and normalized assets for {clip.key.value}")
        output_path = dual_probe_mp4_path(
            settings.paths.output_root,
            clip.key.value,
            InputVariant.THERMAL_FILTERED.value,
            InputVariant.THERMAL_NORMALIZED.value,
            stitch_height,
            stitch_crf,
        )
        if not output_path.exists():
            compose_side_by_side_mp4(
                filtered_path,
                normalized_path,
                output_path,
                target_height=stitch_height if stitch_height > 0 else None,
                crf=stitch_crf,
            )
        metadata["thermal_dual"] = {
            "filtered_path": str(filtered_path),
            "normalized_path": str(normalized_path),
            "video_path": str(output_path),
        }
    else:
        output_path = clip.get_asset_path(variant)

    if output_path is None:
        raise RuntimeError(f"No asset path available for {clip.key.value} / {variant.value}")

    if zone_overlay:
        overlay_path = overlay_probe_mp4_path(
            settings.paths.output_root,
            clip.key.value,
            variant.value,
            "centerzone",
            crf=stitch_crf,
        )
        if not overlay_path.exists():
            overlay_center_zone_mp4(
                output_path,
                overlay_path,
                dual_layout=(variant == InputVariant.THERMAL_DUAL),
                crf=stitch_crf,
            )
        output_path = overlay_path
        metadata["zone_overlay"] = {
            "type": "center",
            "video_path": str(output_path),
            "dual_layout": variant == InputVariant.THERMAL_DUAL,
        }

    asset_paths = dict(clip.asset_paths)
    asset_paths[variant] = output_path
    if variant == InputVariant.THERMAL_DUAL:
        asset_paths[InputVariant.THERMAL_DUAL] = output_path
    return replace(clip, asset_paths=asset_paths, metadata=metadata), output_path


def materialize_thermal_window(
    settings,
    clip: ClipRecord,
    variant: InputVariant,
    *,
    start_sec: float,
    end_sec: float,
    label: str,
    crf: int = 30,
    zone_overlay: bool = False,
) -> Path:
    asset_path = clip.get_asset_path(variant)
    if asset_path is None:
        raise RuntimeError(f"No asset path available for {clip.key.value} / {variant.value}")
    output_path = trimmed_probe_mp4_path(
        settings.paths.output_root,
        clip.key.value,
        variant.value,
        start_sec,
        end_sec,
        label=label,
        crf=crf,
    )
    if not output_path.exists():
        trim_mp4(asset_path, output_path, start_sec=start_sec, end_sec=end_sec, crf=crf)
    if not zone_overlay:
        return output_path
    overlay_path = overlay_probe_mp4_path(
        settings.paths.output_root,
        clip.key.value,
        variant.value,
        f"{label}.centerzone",
        start_sec=start_sec,
        end_sec=end_sec,
        crf=crf,
    )
    if not overlay_path.exists():
        overlay_center_zone_mp4(
            output_path,
            overlay_path,
            dual_layout=(variant == InputVariant.THERMAL_DUAL),
            crf=crf,
        )
    return overlay_path


def resolve_specs(adapter, *, clips_path: str | None, pack: str | None, split: str, include_calibration: bool) -> list[ThermalBatchClipSpec]:
    if clips_path:
        return load_thermal_batch_specs(clips_path)
    if pack:
        return [ThermalBatchClipSpec(bucket=item["bucket"], clip_id=item["clip_id"]) for item in adapter.build_batch_specs(pack, split=split, include_calibration=include_calibration)]
    raise ValueError("Either clips_path or pack is required")


def _proposal_label(index: int, total: int) -> str:
    return f"window-{index + 1:02d}-of-{total:02d}"


def derive_thermal_proposals(
    clip: ClipRecord,
    *,
    proposal_source: str = "raw_video",
    max_proposals: int = 4,
    min_duration_sec: float = 2.0,
    overlap_ratio: float = 0.25,
) -> list[dict[str, Any]]:
    if proposal_source == "track_oracle":
        proposals: list[dict[str, Any]] = []
        target_duration = max(float(min_duration_sec), 4.0)
        clip_end = float(clip.duration_seconds or 0.0)
        for index, item in enumerate(clip.metadata.get("truth_event_windows") or []):
            start_sec = float(item.get("start_sec") or 0.0)
            end_sec = float(item.get("end_sec") or start_sec + min_duration_sec)
            duration = max(0.05, end_sec - start_sec)
            if duration < target_duration:
                pad_total = target_duration - duration
                left_pad = pad_total / 2.0
                right_pad = pad_total - left_pad
                start_sec = max(0.0, start_sec - left_pad)
                end_sec = min(clip_end if clip_end > 0.0 else end_sec + right_pad, end_sec + right_pad)
                if end_sec - start_sec < target_duration and clip_end > 0.0:
                    missing = target_duration - (end_sec - start_sec)
                    start_sec = max(0.0, start_sec - missing)
            proposals.append(
                {
                    "proposal_id": _proposal_label(index, max_proposals),
                    "start_sec": start_sec,
                    "end_sec": max(end_sec, start_sec + 0.05),
                    "source": "track_oracle",
                    "label_hint": item.get("label"),
                }
            )
        return proposals[:max_proposals]

    active_intervals = clip.metadata.get("active_intervals") or [(0.0, float(clip.duration_seconds or 0.0))]
    proposals = []
    for start_sec, end_sec in active_intervals:
        duration = max(0.0, float(end_sec) - float(start_sec))
        if duration < max(0.05, min_duration_sec):
            continue
        if len(proposals) >= max_proposals:
            break
        proposals.append((float(start_sec), float(end_sec)))
        if duration <= min_duration_sec * 2.0:
            continue
        remaining_slots = max_proposals - len(proposals)
        if remaining_slots <= 0:
            break
        tiles = min(remaining_slots, max(1, int(round(duration / max(min_duration_sec * 2.0, 4.0)))))
        tile_span = duration / max(1, tiles)
        step = max(min_duration_sec, tile_span * max(0.25, 1.0 - overlap_ratio))
        cursor = float(start_sec)
        while len(proposals) < max_proposals and cursor + min_duration_sec <= float(end_sec):
            tile_end = min(float(end_sec), cursor + max(min_duration_sec, tile_span))
            proposals.append((cursor, tile_end))
            cursor += step
    deduped: list[tuple[float, float]] = []
    for start_sec, end_sec in proposals:
        if any(abs(start_sec - existing_start) < 0.1 and abs(end_sec - existing_end) < 0.1 for existing_start, existing_end in deduped):
            continue
        deduped.append((start_sec, end_sec))
    return [
        {
            "proposal_id": _proposal_label(index, len(deduped)),
            "start_sec": start_sec,
            "end_sec": end_sec,
            "source": "raw_video",
            "label_hint": None,
        }
        for index, (start_sec, end_sec) in enumerate(deduped[:max_proposals])
    ]


def _window_duration(window: ThermalEventWindow) -> float:
    return max(0.05, float(window.duration_sec or 0.05))


def _window_overlap(a: ThermalEventWindow, b: ThermalEventWindow) -> float:
    if None in {a.timestamp_start_sec, a.timestamp_end_sec, b.timestamp_start_sec, b.timestamp_end_sec}:
        return 0.0
    start = max(float(a.timestamp_start_sec), float(b.timestamp_start_sec))
    end = min(float(a.timestamp_end_sec), float(b.timestamp_end_sec))
    return max(0.0, end - start)


def merge_thermal_event_windows(
    windows: Iterable[ThermalEventWindow],
    *,
    merge_gap_sec: float = 0.75,
) -> list[ThermalEventWindow]:
    merged: list[ThermalEventWindow] = []
    for window in sorted(
        [item for item in windows if item.timestamp_start_sec is not None and item.timestamp_end_sec is not None],
        key=lambda item: (float(item.timestamp_start_sec or 0.0), float(item.timestamp_end_sec or 0.0), str(item.label or "")),
    ):
        if merged:
            previous = merged[-1]
            same_label = str(previous.label or "") == str(window.label or "")
            close_enough = float(window.timestamp_start_sec or 0.0) <= float(previous.timestamp_end_sec or 0.0) + merge_gap_sec
            if same_label and close_enough:
                merged[-1] = ThermalEventWindow(
                    timestamp_start_sec=min(float(previous.timestamp_start_sec or 0.0), float(window.timestamp_start_sec or 0.0)),
                    timestamp_end_sec=max(float(previous.timestamp_end_sec or 0.0), float(window.timestamp_end_sec or 0.0)),
                    label=previous.label,
                    confidence=max(float(previous.confidence or 0.0), float(window.confidence or 0.0)) or None,
                    false_positive_score=max(
                        float(previous.false_positive_score or 0.0),
                        float(window.false_positive_score or 0.0),
                    )
                    or None,
                    evidence_note=previous.evidence_note or window.evidence_note,
                    source=previous.source or window.source,
                    metadata={**dict(previous.metadata), **dict(window.metadata)},
                )
                continue
        merged.append(window)
    return merged


def _positive_non_fp_labels(prediction: ThermalPrediction) -> list[str]:
    labels: list[str] = []
    if prediction.coarse_label:
        labels.append(prediction.coarse_label)
    labels.extend(prediction.clip_labels)
    labels.extend(prediction.event_labels)
    return [label for label in collapse_thermal_labels(labels) if label and label != "false_positive"]


def _text_supports_false_positive(prediction: ThermalPrediction) -> bool:
    text = " ".join(filter(None, [prediction.commentary, prediction.evidence_summary])).strip().lower()
    if not text:
        return False
    false_positive_terms = (
        "false positive",
        "no animal",
        "no animal activity",
        "no animal events",
        "no wildlife",
        "no biological",
        "no biological targets",
        "no moving targets",
        "no moving objects",
        "no coherent motion",
        "lack of coherent motion",
        "no translational movement",
        "no translational motion",
        "no directional movement",
        "no fish passage",
        "no fish passages",
        "no defensible passages",
        "stationary bright",
        "stationary object",
        "stationary artifact",
        "stationary bright spot",
        "single stationary bright spot",
        "sensor artifact",
        "static bright",
        "static bright spot",
        "static target",
        "static scene",
        "static thermal landscape",
        "remains stationary",
        "stays fixed",
        "fixed bright spot",
        "random clutter",
        "effectively empty",
        "no evidence supports",
    )
    return any(term in text for term in false_positive_terms)


def _text_supports_positive_animal(prediction: ThermalPrediction) -> bool:
    text = " ".join(filter(None, [prediction.commentary, prediction.evidence_summary])).strip().lower()
    if not text or _text_supports_false_positive(prediction):
        return False
    positive_terms = (
        "animal event",
        "real animal",
        "single animal",
        "wildlife",
        "biological target",
        "biological event",
        "coherent motion",
        "coherent track",
        "coherent bright target",
        "persistent coherent motion",
        "persistence and coherence",
        "persistence distinguishes it from random noise",
        "distinguishes it from random noise",
        "showing net",
        "displacement observed",
        "moving rightward",
        "moving leftward",
        "rightward drift",
        "leftward drift",
        "drifting slowly",
        "trajectory",
    )
    return any(term in text for term in positive_terms)


def _has_positive_animal_evidence(prediction: ThermalPrediction) -> bool:
    if prediction.animal_present is True:
        return True
    if any((window.label or "") != "false_positive" for window in prediction.event_windows if window.label):
        return True
    positive_labels = _positive_non_fp_labels(prediction)
    if not positive_labels:
        return False
    informative_labels = [label for label in positive_labels if label != "other"]
    if not informative_labels:
        if _text_supports_positive_animal(prediction):
            if prediction.false_positive_score is not None and float(prediction.false_positive_score) >= 0.8:
                return False
            return float(prediction.confidence or 0.0) >= 0.4
        return False
    if prediction.false_positive_score is not None and float(prediction.false_positive_score) < 0.5:
        return True
    return float(prediction.confidence or 0.0) >= 0.75


def _fallback_global_window(clip: ClipRecord, prediction: ThermalPrediction) -> ThermalEventWindow | None:
    if prediction.coarse_label in {None, "false_positive"}:
        return None
    if not _has_positive_animal_evidence(prediction):
        return None
    active_intervals = clip.metadata.get("active_intervals") or []
    if active_intervals:
        start_sec, end_sec = max(
            ((float(start), float(end)) for start, end in active_intervals),
            key=lambda item: max(0.05, item[1] - item[0]),
        )
    else:
        start_sec, end_sec = 0.0, float(clip.duration_seconds or 0.0)
    if end_sec <= start_sec:
        end_sec = start_sec + max(0.05, float(clip.duration_seconds or 0.05))
    return ThermalEventWindow(
        timestamp_start_sec=start_sec,
        timestamp_end_sec=end_sec,
        label=prediction.coarse_label or "other",
        confidence=prediction.confidence,
        false_positive_score=prediction.false_positive_score,
        evidence_note=prediction.evidence_summary or prediction.commentary,
        source="global_fallback",
    )


def aggregate_thermal_predictions(
    clip: ClipRecord,
    global_prediction: ThermalPrediction,
    local_predictions: list[dict[str, Any]],
) -> ThermalPrediction:
    local_label_scores: dict[str, float] = {}
    event_candidates: list[ThermalEventWindow] = []
    fp_scores: list[float] = []
    confidences = [float(global_prediction.confidence or 0.0)]
    clip_labels = set(global_prediction.clip_labels)
    event_labels = set(global_prediction.event_labels)
    global_positive_evidence = _has_positive_animal_evidence(global_prediction)
    positive_local_count = 0
    strong_fp_local_count = 0
    zone_entry_candidates: list[float] = []
    zone_dwell_candidates: list[float] = []

    if global_positive_evidence:
        clip_labels.update(label for label in global_prediction.clip_labels if label != "false_positive")
        event_labels.update(label for label in global_prediction.event_labels if label != "false_positive")
        for window in global_prediction.event_windows:
            if (window.label or "") == "false_positive":
                continue
            event_candidates.append(window)
        if not event_candidates:
            fallback_window = _fallback_global_window(clip, global_prediction)
            if fallback_window is not None:
                event_candidates.append(fallback_window)
        if global_prediction.center_zone_entered:
            if global_prediction.center_zone_first_entry_sec is not None:
                zone_entry_candidates.append(float(global_prediction.center_zone_first_entry_sec))
            if global_prediction.center_zone_dwell_sec is not None:
                zone_dwell_candidates.append(max(0.0, float(global_prediction.center_zone_dwell_sec)))

    for item in local_predictions:
        prediction: ThermalPrediction = item["prediction"]
        proposal = item["proposal"]
        label = prediction.coarse_label or "other"
        confidence = float(prediction.confidence or 0.5)
        confidences.append(confidence)
        if prediction.false_positive_score is not None:
            fp_scores.append(float(prediction.false_positive_score))
        positive_local_evidence = _has_positive_animal_evidence(prediction)
        if positive_local_evidence:
            positive_local_count += 1
            if prediction.center_zone_entered:
                local_zone_start = float(proposal["start_sec"])
                if prediction.center_zone_first_entry_sec is not None:
                    local_zone_start += max(0.0, float(prediction.center_zone_first_entry_sec))
                zone_entry_candidates.append(local_zone_start)
                if prediction.center_zone_dwell_sec is not None:
                    zone_dwell_candidates.append(max(0.0, float(prediction.center_zone_dwell_sec)))
                elif prediction.event_windows:
                    zone_dwell_candidates.append(
                        sum(max(0.05, float(window.duration_sec or 0.05)) for window in prediction.event_windows if (window.label or "") != "false_positive")
                    )
                else:
                    zone_dwell_candidates.append(max(0.05, float(proposal["end_sec"]) - float(proposal["start_sec"])))
        if (prediction.coarse_label == "false_positive" or float(prediction.false_positive_score or 0.0) >= 0.8) and float(prediction.confidence or 0.0) >= 0.8:
            strong_fp_local_count += 1
        clip_labels.update(label for label in prediction.clip_labels if label == "false_positive")
        event_labels.update(label for label in prediction.event_labels if label == "false_positive")
        if positive_local_evidence:
            clip_labels.update(label for label in prediction.clip_labels if label != "false_positive")
            event_labels.update(label for label in prediction.event_labels if label != "false_positive")
        duration = max(0.05, float(proposal["end_sec"]) - float(proposal["start_sec"]))
        score = confidence * duration
        if label != "false_positive" and positive_local_evidence:
            local_label_scores[label] = local_label_scores.get(label, 0.0) + score
        if prediction.event_windows and positive_local_evidence:
            for window in prediction.event_windows:
                local_start = float(window.timestamp_start_sec) if window.timestamp_start_sec is not None else 0.0
                local_end = float(window.timestamp_end_sec) if window.timestamp_end_sec is not None else duration
                global_start = float(proposal["start_sec"]) + max(0.0, local_start)
                global_end = min(float(proposal["end_sec"]), float(proposal["start_sec"]) + max(local_start + 0.05, local_end))
                event_candidates.append(
                    ThermalEventWindow(
                        timestamp_start_sec=global_start,
                        timestamp_end_sec=global_end,
                        label=window.label or label,
                        confidence=window.confidence or prediction.confidence,
                        false_positive_score=window.false_positive_score if window.false_positive_score is not None else prediction.false_positive_score,
                        evidence_note=window.evidence_note or prediction.evidence_summary or prediction.commentary,
                        source=str(proposal["source"]),
                        metadata={"proposal_id": proposal["proposal_id"]},
                    )
                )
        elif label != "false_positive" and positive_local_evidence:
            event_candidates.append(
                ThermalEventWindow(
                    timestamp_start_sec=float(proposal["start_sec"]),
                    timestamp_end_sec=float(proposal["end_sec"]),
                    label=label,
                    confidence=prediction.confidence,
                    false_positive_score=prediction.false_positive_score,
                    evidence_note=prediction.evidence_summary or prediction.commentary,
                    source=str(proposal["source"]),
                    metadata={"proposal_id": proposal["proposal_id"]},
                )
            )

    dominant_local_label = max(local_label_scores.items(), key=lambda item: (item[1], item[0]))[0] if local_label_scores else None
    dominant_global_label = global_prediction.coarse_label or "other"
    final_label = dominant_global_label
    global_fp = float(global_prediction.false_positive_score or 0.0)
    strong_fp_support = max([global_fp, *fp_scores], default=0.0) >= 0.8 or "false_positive" in clip_labels
    if dominant_local_label is not None:
        local_strength = float(local_label_scores.get(dominant_local_label, 0.0))
        global_strength = float(global_prediction.confidence or 0.0) * max(1.0, float(clip.duration_seconds or 1.0))
        if dominant_global_label == "false_positive" or local_strength >= global_strength or global_fp >= 0.65:
            final_label = dominant_local_label
    elif dominant_global_label == "false_positive" or (strong_fp_support and not global_positive_evidence):
        final_label = "false_positive"
    merged_windows = merge_thermal_event_windows(event_candidates)
    if (
        final_label != "false_positive"
        and positive_local_count == 0
        and strong_fp_local_count >= 2
        and strong_fp_support
        and float(global_prediction.confidence or 0.0) < 0.65
    ):
        final_label = "false_positive"
        merged_windows = []
    if merged_windows and final_label == "false_positive":
        final_label = merged_windows[0].label or "other"
    if final_label == "false_positive":
        final_fp_score = max([float(global_prediction.false_positive_score or 0.0), *fp_scores], default=1.0)
    else:
        final_fp_score = min([float(global_prediction.false_positive_score or 0.5), *fp_scores], default=0.5)
        final_fp_score = min(final_fp_score, 0.49)
    final_zone_entered = (final_label != "false_positive") and bool(zone_entry_candidates or (global_prediction.center_zone_entered and global_positive_evidence))
    final_zone_first_entry_sec = min(zone_entry_candidates) if final_zone_entered and zone_entry_candidates else (
        float(global_prediction.center_zone_first_entry_sec)
        if final_zone_entered and global_prediction.center_zone_entered and global_prediction.center_zone_first_entry_sec is not None
        else None
    )
    final_zone_dwell_sec = (
        max(zone_dwell_candidates)
        if final_zone_entered and zone_dwell_candidates
        else (
            float(global_prediction.center_zone_dwell_sec)
            if final_zone_entered and global_prediction.center_zone_entered and global_prediction.center_zone_dwell_sec is not None
            else None
        )
    )
    clip_labels.add(final_label)
    event_labels.update(window.label for window in merged_windows if window.label)
    return ThermalPrediction(
        domain=clip.domain,
        clip_id=clip.clip_id,
        clip_labels=collapse_thermal_labels(clip_labels),
        coarse_label=final_label,
        animal_present=(final_label != "false_positive"),
        false_positive_score=final_fp_score,
        center_zone_entered=final_zone_entered,
        center_zone_first_entry_sec=final_zone_first_entry_sec,
        center_zone_dwell_sec=final_zone_dwell_sec,
        event_windows=merged_windows,
        event_labels=collapse_thermal_labels(event_labels),
        confidence=max(confidences) if confidences else global_prediction.confidence,
        abstain=global_prediction.abstain and not merged_windows and dominant_local_label is None,
        commentary=global_prediction.commentary,
        evidence_summary=global_prediction.evidence_summary,
        raw_response=global_prediction.raw_response,
        latency_sec=(float(global_prediction.latency_sec or 0.0) + sum(float(item["prediction"].latency_sec or 0.0) for item in local_predictions)),
        usage_metadata=dict(global_prediction.usage_metadata),
        estimated_cost_usd=(float(global_prediction.estimated_cost_usd or 0.0) + sum(float(item["prediction"].estimated_cost_usd or 0.0) for item in local_predictions)),
        model_name=global_prediction.model_name,
        parse_success=global_prediction.parse_success and all(item["prediction"].parse_success for item in local_predictions),
        prompt_id=global_prediction.prompt_id,
    )


def build_thermal_run_metrics(state: dict[str, Any], clip_index: dict[str, ClipRecord]) -> dict[str, float | None]:
    completed = [item for item in state.get("results", []) if item.get("status") == "completed"]
    if not completed:
        return {
            "binary_accuracy": 0.0,
            "binary_f1": 0.0,
            "binary_balanced_accuracy": 0.0,
            "coarse_macro_f1": 0.0,
            "event_window_recall": None,
            "event_window_mean_tiou": None,
            "center_zone_entry_accuracy": None,
            "center_zone_entry_f1": None,
            "center_zone_first_entry_mae_sec": None,
            "center_zone_dwell_mae_sec": None,
            "abstention_rate": 0.0,
            "mean_latency_sec": 0.0,
        }
    predictions = [
        thermal_prediction_from_payload(item["prediction"], domain=clip_index[item["clip_key"]].domain, clip_id=clip_index[item["clip_key"]].clip_id)
        for item in completed
        if item.get("clip_key") in clip_index
    ]
    clips = [clip_index[item["clip_key"]] for item in completed if item.get("clip_key") in clip_index]
    report = evaluate_thermal_predictions(clips, predictions)
    return {
        "binary_accuracy": report.binary_accuracy,
        "binary_f1": report.binary_f1,
        "binary_balanced_accuracy": report.binary_balanced_accuracy,
        "coarse_macro_f1": report.coarse_macro_f1,
        "animal_event_count_mae": report.animal_event_count_mae,
        "animal_event_window_recall": report.animal_event_window_recall,
        "animal_event_window_precision": report.animal_event_window_precision,
        "animal_event_window_mean_tiou": report.animal_event_window_mean_tiou,
        "animal_event_label_accuracy": report.animal_event_label_accuracy,
        "center_zone_entry_accuracy": report.center_zone_entry_accuracy,
        "center_zone_entry_f1": report.center_zone_entry_f1,
        "center_zone_first_entry_mae_sec": report.center_zone_first_entry_mae_sec,
        "center_zone_dwell_mae_sec": report.center_zone_dwell_mae_sec,
        "multi_entity_accuracy": report.multi_entity_accuracy,
        "abstention_rate": report.abstention_rate,
        "mean_latency_sec": report.mean_latency_sec,
    }


def write_thermal_summary(path: str | Path, payload: dict[str, Any], metrics: dict[str, float | None]) -> None:
    def _fmt(value: object, digits: int = 3) -> str:
        if value in (None, ""):
            return ""
        if isinstance(value, (float, int)):
            return f"{float(value):.{digits}f}"
        return str(value)

    lines = [
        f"# {payload['name']}",
        "",
        f"- split: `{payload.get('split')}`",
        f"- variant: `{payload['variant']}`",
        f"- model: `{payload['model']}`",
        f"- transport: `{payload['transport']}`",
        f"- proposal_source: `{payload.get('proposal_source', 'clip_only')}`",
        f"- completed: `{payload['completed_count']}/{payload['total_clips']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in metrics.items():
        lines.append(f"| `{key}` | {_fmt(value)} |")
    lines.extend(
        [
            "",
            "| Status | Bucket | Clip | Truth | Pred | FP Truth | FP Score | Latency | Pass | Note |",
            "|---|---|---|---|---|---|---:|---:|---|---|",
        ]
    )
    for item in payload.get("results", []):
        truth = item.get("ground_truth") or {}
        pred = item.get("prediction") or {}
        note = item.get("error") or pred.get("commentary") or ""
        lines.append(
            f"| {item.get('status')} | {item.get('bucket')} | `{item.get('clip_key')}` | "
            f"{truth.get('coarse_label', '')} | {pred.get('coarse_label', '')} | "
            f"{truth.get('is_false_positive', '')} | {_fmt(pred.get('false_positive_score'))} | "
            f"{_fmt(pred.get('latency_sec'), 2)} | {item.get('selected_pass', '')} | {str(note).replace('|', '/')} |"
        )
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def archive_thermal_prompt(
    output_root: Path,
    *,
    variant: InputVariant,
    prompt: PromptRevision,
    bucket: str,
    clip: ClipRecord,
    model: str,
    transport: str,
    rendered_prompt: str,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    payload = {
        "bucket": bucket,
        "domain": clip.domain,
        "clip_id": clip.clip_id,
        "model": model,
        "transport": transport,
        **(metadata or {}),
    }
    return write_prompt_artifact(
        output_root=output_root,
        scope="thermal_batch",
        variant=variant.value,
        prompt_id=prompt.prompt_id,
        version=prompt.version,
        prompt_text=prompt.prompt_text,
        critique=prompt.critique,
        metrics=dict(prompt.metrics),
        rendered_prompt=rendered_prompt,
        metadata=payload,
    )
