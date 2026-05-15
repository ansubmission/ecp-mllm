from __future__ import annotations

from ..thermal import THERMAL_COARSE_LABELS
from ..types import ClipRecord, InputVariant, PromptRevision


THERMAL_JSON_SCHEMA = """Return a single JSON object with this schema:
{
  "clip_labels": [string],
  "coarse_label": "false_positive|bird|rodent|possum|cat|hedgehog|mustelid|other",
  "animal_present": bool | null,
  "false_positive_score": float | null,
  "center_zone_entered": bool | null,
  "center_zone_first_entry_sec": float | null,
  "center_zone_dwell_sec": float | null,
  "event_windows": [
    {
      "timestamp_start_sec": float | null,
      "timestamp_end_sec": float | null,
      "label": "false_positive|bird|rodent|possum|cat|hedgehog|mustelid|other",
      "confidence": float | null,
      "false_positive_score": float | null,
      "evidence_note": string | null
    }
  ],
  "event_labels": [string],
  "confidence": float | null,
  "abstain": bool,
  "commentary": string | null,
  "evidence_summary": string | null
}
Use `clip_labels` for all plausible coarse labels visible anywhere in the clip.
Use `coarse_label` for the single best clip-level label.
Use `false_positive_score` as the probability that this clip is a false positive rather than an animal event.
Use `center_zone_*` for the central 40% of the frame (roughly x,y in [0.3, 0.7]).
Set `center_zone_entered=true` only when a real animal event clearly enters that central zone.
Use `center_zone_first_entry_sec` and `center_zone_dwell_sec` only when `center_zone_entered=true`; otherwise return null.
Always include all three `center_zone_*` keys in the JSON, even when the answer is null/false.
Use `event_windows` for temporally localized animal or false-positive episodes you relied on.
Use `event_labels` for the distinct coarse labels supported by those windows.
Use `abstain=true` only when the clip is too ambiguous to support a confident class decision.
Do not add markdown fences or explanation outside the JSON object."""


def _variant_guidance(variant: InputVariant) -> list[str]:
    if variant == InputVariant.THERMAL_FILTERED:
        return [
            "This is filtered thermal footage emphasizing moving heat signatures.",
            "False positives can still arise from vegetation, thermal shimmer, moving background clutter, and calibration artifacts.",
        ]
    if variant == InputVariant.THERMAL_NORMALIZED:
        return [
            "This is normalized thermal footage emphasizing scene structure and relative heat contrast.",
            "Normalized intensity helps with body shape and motion context, but it can flatten absolute heat differences.",
        ]
    if variant == InputVariant.THERMAL_DUAL:
        return [
            "This clip is a synchronized side-by-side thermal view: filtered on the left, normalized on the right.",
            "Use the filtered side for candidate event timing and the normalized side for shape, body plan, and scene context.",
        ]
    return []


def build_thermal_prompt(
    clip: ClipRecord,
    variant: InputVariant,
    revision: PromptRevision,
    *,
    task_mode: str = "clip",
    proposal_source: str = "raw_video",
    window_hint: tuple[float, float] | None = None,
) -> str:
    asset_path = clip.get_asset_path(variant)
    metadata_lines = [
        f"dataset={clip.dataset}",
        f"location={clip.domain}",
        f"clip_id={clip.clip_id}",
        f"asset_path={asset_path}",
        f"variant={variant.value}",
        f"width={clip.width}",
        f"height={clip.height}",
        f"framerate={clip.framerate}",
        f"duration_seconds={clip.duration_seconds}",
        f"location_split={clip.metadata.get('split')}",
        f"calibration_frames_present={bool(clip.metadata.get('calibration_frames'))}",
        f"proposal_source={proposal_source}",
    ]
    if window_hint is not None:
        metadata_lines.append(f"window_start_sec={window_hint[0]:.3f}")
        metadata_lines.append(f"window_end_sec={window_hint[1]:.3f}")
    zone_overlay = clip.metadata.get("zone_overlay")
    task_lines = [
        "This is a thermal wildlife clip captured at night.",
        "This is NOT a fish-counting or line-crossing task.",
        "Do not return count fields such as left_count, right_count, upstream_count, downstream_count, passages, or candidate_passages.",
        "Reason about whether the clip contains a real animal event or a false positive trigger.",
        "A real animal event can be stationary, perched, resting, or only weakly moving; do NOT require translational motion to call something an animal.",
        "Do not classify a clip as false_positive merely because the target is stationary. Use false_positive only for artifacts, clutter, shimmer, calibration issues, or non-biological triggers.",
        "Use the coarse taxonomy only: " + ", ".join(THERMAL_COARSE_LABELS) + ".",
        "First decide whether animal evidence exists at all. Then choose the best coarse label.",
        "If you see a biologically plausible warm body but cannot assign a supported species group, return `coarse_label=\"other\"` with `animal_present=true` instead of `false_positive`.",
        "Also judge whether any real animal event enters the center zone, defined as the central 40% of the frame.",
        "If an animal enters the center zone, estimate its first entry time and approximate total dwell time within that zone.",
        "Treat the center zone question as mandatory: explicitly answer yes/no and fill the timing fields accordingly.",
        "If the target never reaches the center zone, return `center_zone_entered=false`, `center_zone_first_entry_sec=null`, and `center_zone_dwell_sec=null`.",
        "When there are multiple short episodes, summarize them in `event_windows` before deciding the clip-level label.",
        "If the clip contains repeated appearance, disappearance, or reappearance, treat that as one continuing event unless there is clear evidence for a separate episode.",
        "Suppress contradictory species guesses unless there is clear temporal evidence for different animals in different windows.",
        "Use `false_positive_score` to quantify uncertainty about non-animal triggers.",
        "Base the answer only on the provided thermal clip. Do not use dataset metadata or external knowledge.",
    ]
    if isinstance(zone_overlay, dict) and zone_overlay.get("type") == "center":
        task_lines.extend(
            [
                "A bright green box is drawn directly onto the video to mark the center zone.",
                "Use that drawn box as the authoritative center-zone boundary instead of estimating it mentally.",
            ]
        )
    if task_mode == "window":
        task_lines.append("This clip is a local temporal window cut from a larger thermal clip; focus on what is visible in this window only.")
    if proposal_source == "track_oracle":
        task_lines.append("This window was proposed by an oracle track interval for ablation; still classify only what is visually supported by the video.")
    task_lines.extend(_variant_guidance(variant))
    return "\n".join(
        [
            revision.prompt_text.strip(),
            "",
            "Task guidance:",
            *task_lines,
            "",
            "Clip metadata:",
            *metadata_lines,
            "",
            THERMAL_JSON_SCHEMA,
        ]
    )
