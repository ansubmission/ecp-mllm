from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, replace
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..agent.event_centric import build_observation_stream, enumerate_perception_actions, prediction_to_event_proposals
from ..agent.expert_agent import select_expert_prediction
from ..agent.temporal_abstraction import propose_temporal_windows
from ..config import load_local_settings
from ..data.cfc_adapter import CFCAdapter
from ..eval.domain_shift_critic import assess_domain_shift_risk
from ..eval.constraints import audit_prediction_constraints
from ..eval.counting import count_tracks_like_cfc, read_mot_tracks
from ..eval.routed_policy import apply_routed_policy
from ..eval.site_profiles import resolve_site_profile
from ..qwen.parsing import parse_passage_prediction
from ..qwen.prompt_archive import write_prompt_artifact
from ..qwen.prompting import build_prompt
from ..types import EventProposal, InputVariant, PassageEvent, PassagePrediction, PromptRevision
from ..video import trim_mp4, trimmed_probe_mp4_path
from .probe_clip import _build_client, _materialize_clip_transport, _prepare_sff3c_variant
from .prompt_batch import _safe_name, load_batch_specs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--settings", default="config/local.toml")
    parser.add_argument("--domain", required=True)
    parser.add_argument("--variant", default="sff3c")
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--transport", choices=["sampled_frames", "stitched_mp4"], default="stitched_mp4")
    parser.add_argument("--stitch-fps", type=float, default=5.0)
    parser.add_argument("--stitch-height", type=int, default=0)
    parser.add_argument("--stitch-crf", type=int, default=30)
    parser.add_argument("--clips-path", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--prompt-text", default=None)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--max-proposals", type=int, default=4)
    parser.add_argument("--proposal-min-duration-sec", type=float, default=1.0)
    parser.add_argument("--proposal-merge-gap-sec", type=float, default=1.0)
    parser.add_argument("--proposal-min-video-sec", type=float, default=8.0)
    parser.add_argument("--window-prompt-mode", choices=["legacy", "strict_counts"], default="legacy")
    parser.add_argument("--flagged-repeat-runs", type=int, default=1)
    parser.add_argument("--stop-on-error", action="store_true")
    return parser.parse_args()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _format_stage_details(details: dict[str, object]) -> str:
    parts: list[str] = []
    for key, value in details.items():
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        if len(text) > 180:
            text = text[:177] + "..."
        parts.append(f"{key}={text}")
    return " ".join(parts)


def _ground_truth_counts(adapter: CFCAdapter, clip) -> dict[str, int] | None:
    if clip.dataset != "cfc" or clip.width is None or clip.height is None:
        return None
    gt_path = adapter.gt_path(clip.domain, clip.clip_id)
    if not gt_path.exists():
        return None
    tracks = read_mot_tracks(gt_path)
    counts = count_tracks_like_cfc(tracks, clip.width, clip.height, filter_dist=0.05)
    return {"left_count": counts.left, "right_count": counts.right, "total_count": counts.left + counts.right}


def _refresh_result_record_errors(result_record: dict[str, Any], gt: dict[str, int] | None) -> None:
    if gt is None:
        return
    prediction = result_record.get("prediction") or {}
    result_record["abs_error_left"] = abs(int(prediction.get("left_count") or 0) - int(gt["left_count"]))
    result_record["abs_error_right"] = abs(int(prediction.get("right_count") or 0) - int(gt["right_count"]))
    result_record["total_abs_error"] = result_record["abs_error_left"] + result_record["abs_error_right"]


def _default_global_prompt() -> str:
    return (
        "Analyze this sonar clip for fish passage. "
        "Treat the clip as a long physical stream and reason in terms of candidate event episodes rather than isolated frames. "
        "Identify each candidate passage episode, estimate how many distinct fish traverse during that episode, "
        "and then sum across episodes. "
        "If a long stream contains multiple waves or mini-bursts over time, split them into separate passage episodes. "
        "Use best-effort throughput estimates for high-passage situations, and estimate total distinct fish passage rather than instantaneous occupancy. "
        "Return JSON only."
    )


def _high_throughput_custom_prompt() -> str:
    return (
        "Analyze this sonar clip for total fish passage.\n\n"
        "Count with a wave-first procedure:\n\n"
        "1. Scan the full clip in short chronological windows.\n"
        "2. Mark every defensible rightward or leftward burst where new targets enter, replace earlier targets, or create a fresh density peak.\n"
        "3. Merge nearby windows only when they clearly show the same small group persisting without real replacement.\n"
        "4. Sum distinct fish across waves or replacement bursts to get the final count.\n\n"
        "Use `candidate_passages` for major passage episodes and `events` for each defensible wave start, density peak, or new-entrant burst.\n\n"
        "High-throughput guidance:\n\n"
        "- Do not anchor only on peak simultaneous occupancy when a passage lasts many seconds.\n"
        "- If later targets enter after earlier targets have advanced or exited, treat them as new entrants rather than the same two or three fish.\n"
        "- If one broad episode contains several local bursts, report multiple waves or multiple `candidate_passages` entries and sum them.\n"
        "- For long rightward passages, make a best-effort lower-bound count of distinct entrants across time, not just the clearest simultaneous streaks.\n"
        "- If targets become briefly hidden in the crossing corridor but the stream clearly continues before and after the gap, count those briefly hidden crossings as part of the same throughput estimate.\n"
        "- Do not require every fish to be perfectly visible at the centerline moment when neighboring frames show coherent continuation of the same passage stream.\n\n"
        "Hidden-crossing guidance:\n\n"
        "- In dense clips, some fish will be partially or briefly hidden by overlap, beam geometry, or clutter while still traversing.\n"
        "- Use pre-gap and post-gap continuity, spacing, and replacement timing to estimate distinct entrants through the corridor.\n"
        "- Prefer a conservative lower bound for implied-but-briefly-hidden crossings, but do not collapse them to visible peak occupancy alone.\n\n"
        "Anti-overcount guidance:\n\n"
        "- Do not increase counts from duration alone.\n"
        "- Do not split one sparse continuous pair into many fish unless you can point to replacement, spacing gaps, or repeated new entrants.\n"
        "- Reject stationary clutter, drifting debris, flickering speckle, and ambiguous texture.\n"
        "- If evidence is weak, keep the count conservative, but if several defensible waves are visible, include all of them.\n\n"
        "Output guidance:\n\n"
        "- `estimated_count` should match `throughput_best_count` for each passage episode.\n"
        "- `peak_simultaneous_count` should be the peak visible at once, even when the final episode count is larger.\n"
        "- `wave_count` should reflect the number of defensible bursts within that episode.\n"
        "- `evidence_summary` should describe observable motion cues only.\n\n"
        "Return strict JSON only."
    )


def _high_throughput_hidden_crossing_prompt_with_context(
    *,
    predicted_total: int,
    total_peak: float,
    max_peak: float,
    max_duration_sec: float,
    window_activity_notes: str = "",
) -> str:
    return (
        _high_throughput_custom_prompt()
        + "\n\n"
        + "First-pass summary suggests a dense or sustained passage that may be undercounted because some fish become briefly hidden while crossing:\n"
        + f"- first-pass total estimate: {predicted_total}\n"
        + f"- summed peak support: {total_peak:.1f}\n"
        + f"- max peak simultaneous count: {max_peak:.1f}\n"
        + f"- max episode duration: {max_duration_sec:.1f}s\n\n"
        + "Re-estimate distinct throughput using human-like temporal continuity.\n"
        + "If the episode is clearly continuous and several entrants are only briefly obscured at the crossing corridor, include them in the total.\n"
        + "Do not simply copy the visible peak occupancy; estimate a conservative lower bound for briefly hidden but implied crossings."
        + window_activity_notes
    )


def _single_school_visibility_prompt_with_context(
    *,
    predicted_total: int,
    max_peak: float,
    max_duration_sec: float,
    window_activity_notes: str = "",
) -> str:
    return (
        _high_throughput_custom_prompt()
        + "\n\n"
        + "First-pass summary suggests one sustained school that may be undercounted because some entrants become briefly hidden inside the same passage:\n"
        + f"- first-pass total estimate: {predicted_total}\n"
        + f"- peak simultaneous visibility: {max_peak:.1f}\n"
        + f"- episode duration: {max_duration_sec:.1f}s\n\n"
        + "Re-estimate this as a single coherent school.\n"
        + "Look for brief occlusion, overlap, or corridor clutter that can hide one or two entrants without creating a new wave.\n"
        + "It is acceptable to count a conservative +1 or +2 above visible peak occupancy when the same school clearly continues through the corridor.\n"
        + "Do not invent separate waves, and do not scale by duration alone."
        + window_activity_notes
    )


def _single_school_high_throughput_prompt_with_context(
    *,
    predicted_total: int,
    max_peak: float,
    max_duration_sec: float,
    window_activity_notes: str = "",
) -> str:
    return (
        _high_throughput_custom_prompt()
        + "\n\n"
        + "First-pass summary suggests one sustained near-view school whose total throughput is likely undercounted:\n"
        + f"- first-pass total estimate: {predicted_total}\n"
        + f"- peak simultaneous visibility: {max_peak:.1f}\n"
        + f"- episode duration: {max_duration_sec:.1f}s\n\n"
        + "Re-estimate this as a single dense school rather than a low-count clip.\n"
        + "If the same right-moving school remains coherent for a long interval, do not cap the total at the strongest local window alone.\n"
        + "Use spacing, replacement timing, and continuity before/after brief hidden intervals to estimate distinct entrants through the corridor.\n"
        + "It is acceptable for the final throughput to be substantially above peak occupancy when the stream stays dense for 18s+ and new streaks keep replacing earlier ones.\n"
        + "Do not invent separate waves, and do not scale by duration alone."
        + window_activity_notes
    )


def _stream_throughput_prompt() -> str:
    return (
        "Analyze this sonar clip for dominant-direction stream throughput.\n\n"
        "This branch is for dense, overlapping fish flow where many fish are simultaneously visible and individual passage episodes are not cleanly separable.\n\n"
        "Use a human-expert style procedure:\n"
        "A. First find the fish-rich interval(s) on the echogram rather than counting frame-by-frame blobs.\n"
        "B. Lock onto a few clear anchor streaks and scrub forward/backward in time to judge whether new fish are replacing earlier ones.\n"
        "C. When the crossing zone is briefly noisy or partially hidden, use temporal continuity before/after the gap for conservative imputation.\n"
        "D. Count throughflow, not just visible line-crossing evidence in a single frame.\n\n"
        "Guidance:\n"
        "1. Estimate total distinct throughflow over the clip duration in the dominant direction.\n"
        "2. Treat sustained overlapping schools as a continuous stream, not as a sparse set of isolated passage episodes.\n"
        "3. Peak simultaneous occupancy is a lower bound, not a cap, when new fish keep replacing earlier fish over time.\n"
        "4. Use local dense windows, continuity, and replacement timing to recover flow-through, even when individual fish overlap or briefly merge.\n"
        "5. Keep the opposite direction near zero unless you see explicit coherent counterflow.\n"
        "6. Do not collapse the clip to low-count, sparse, or trickle logic just because the stream is noisy.\n\n"
        "Output guidance:\n"
        "- It is acceptable to represent the clip as one dominant stream passage or a few coarse stream segments.\n"
        "- `estimated_count` should reflect total throughflow, not just peak occupancy.\n"
        "- `wave_count` can stay low when the clip is one sustained dense stream.\n"
        "- `evidence_summary` should explain why the stream is continuous and why overlap does not imply low total count.\n\n"
        "Return strict JSON only."
    )


def _stream_throughput_prompt_with_context(
    *,
    direction: str,
    predicted_total: int,
    max_peak: float,
    max_duration_sec: float,
    window_activity: dict[str, int],
    backbone_activity: dict[str, int | float],
) -> str:
    direction_label = direction or "dominant"
    return (
        _stream_throughput_prompt()
        + "\n\n"
        + "Current stream summary:\n"
        + f"- dominant direction: {direction_label}\n"
        + f"- first-pass total estimate: {predicted_total}\n"
        + f"- max peak simultaneous count: {max_peak:.1f}\n"
        + f"- max candidate duration: {max_duration_sec:.1f}s\n"
        + f"- active local windows: {int(window_activity.get('active_windows', 0))}/{int(window_activity.get('total_windows', 0))}\n"
        + f"- zero local windows: {int(window_activity.get('zero_windows', 0))}\n"
        + f"- strongest local window total: {int(window_activity.get('max_window_total', 0))}\n"
        + f"- strongest local dominant-direction window: {int(window_activity.get('max_target_window_total', 0))}\n"
        + f"- sum of positive local-window totals: {int(window_activity.get('sum_positive_window_totals', 0))}\n"
        + f"- opposite-direction total from local windows: {int(window_activity.get('sum_opposite_window_totals', 0))}\n"
        + f"- backbone throughput hint: {int(backbone_activity.get('total_hint', 0))}\n"
        + f"- backbone max peak: {float(backbone_activity.get('max_peak', 0.0)):.1f}\n"
        + f"- backbone total waves: {int(backbone_activity.get('total_waves', 0))}\n\n"
        + "Use the strongest local window as a lower bound on throughput, not an upper bound. "
        + "If several dense windows or backbone hints support a sustained same-direction stream, recover the total throughflow rather than reverting to sparse episode counting. "
        + "Anchor a few clear streaks, use replacement timing across adjacent windows, and conservatively impute throughflow when the crossing evidence is briefly obscured by noise."
    )


def _trickle_reduction_prompt() -> str:
    return (
        "Analyze this sonar clip for total fish passage with a sparse-trickle reduction procedure.\n\n"
        "This branch is for clips where a long, low-density trickle may have been over-split into many separate mini-episodes.\n\n"
        "Guidance:\n"
        "1. Look for distinct fish or very small groups that are clearly replaced by new entrants.\n"
        "2. Do not create a new passage episode every time the same sparse pair remains visible across time.\n"
        "3. Do not scale counts upward from clip duration or from repeated weak speckle.\n"
        "4. Merge nearby low-density segments when they plausibly reflect the same sparse passage rather than a fresh wave.\n"
        "5. Reject opposite-direction fish unless the clip shows explicit, track-like motion in that direction.\n\n"
        "Output guidance:\n"
        "- Prefer a conservative count for long sparse trickles.\n"
        "- Keep `estimated_count` close to visible replacement, not clip duration.\n"
        "- Use `candidate_passages` only for defensible sparse episodes.\n"
        "- `evidence_summary` should mention replacement or the lack of replacement.\n\n"
        "Return strict JSON only."
    )


def _trickle_reduction_prompt_with_context(
    *,
    predicted_total: int,
    candidate_count: int,
    total_peak: float,
    max_peak: float,
    total_wave_count: int,
) -> str:
    return (
        _trickle_reduction_prompt()
        + "\n\n"
        + "First-pass summary suggests a possible over-split sparse trickle:\n"
        + f"- first-pass total estimate: {predicted_total}\n"
        + f"- candidate episode count: {candidate_count}\n"
        + f"- summed peak support: {total_peak:.1f}\n"
        + f"- max peak simultaneous count: {max_peak:.1f}\n"
        + f"- total wave count: {total_wave_count}\n\n"
        + "Re-check whether these episodes really contain fresh entrants or whether they reflect one sparse, ambiguous trickle. "
        + "If you cannot point to that many distinct entrant times, reduce the total aggressively. "
        + "When peak support stays at 1-2 and replacement is weak, a very small count is more plausible than a large throughput estimate."
    )


def _low_count_far_view_prompt() -> str:
    return (
        "Analyze this sonar clip for total fish passage with a detectability-first procedure.\n\n"
        "This branch is for sparse, low-count, faint, or far-view clips. The goal is to recover defensible one-fish or two-fish passages without inventing a school.\n\n"
        "Guidance:\n"
        "1. Scan the clip for any small coherent target or streak that shows consistent leftward or rightward displacement across multiple frames.\n"
        "2. Count a small positive passage when the motion is defensible, even if the clip never forms a dense school.\n"
        "3. Keep counts close to the visible distinct fish. Do not scale by duration.\n"
        "4. Reject stationary clutter, flickering speckle, drifting debris, and texture that does not show coherent displacement.\n"
        "5. Do not add opposite-direction fish unless the clip shows explicit, track-like motion in that direction.\n\n"
        "Output guidance:\n"
        "- Prefer 0, 1, or 2 when evidence is sparse.\n"
        "- `estimated_count` should stay close to `peak_simultaneous_count` unless there is clear replacement.\n"
        "- Use `candidate_passages` only for defensible low-count episodes.\n"
        "- `evidence_summary` should mention the actual faint motion cue that supports each count.\n\n"
        "Return strict JSON only."
    )


def _low_count_far_view_prompt_with_context(
    *,
    predicted_total: int,
    candidate_count: int,
    total_peak: float,
    max_peak: float,
    total_wave_count: int,
    window_activity_notes: str = "",
) -> str:
    return (
        _low_count_far_view_prompt()
        + "\n\n"
        + "First-pass summary suggests a sparse, faint, or low-detectability clip:\n"
        + f"- first-pass total estimate: {predicted_total}\n"
        + f"- candidate episode count: {candidate_count}\n"
        + f"- summed peak support: {total_peak:.1f}\n"
        + f"- max peak simultaneous count: {max_peak:.1f}\n"
        + f"- total wave count: {total_wave_count}\n"
        + window_activity_notes
        + "\n\n"
        + "Re-check the clip for only the few most defensible fish-like entrants. "
        + "If the clip looks mostly empty or static, and only one noisy local window claims many fish, "
        + "treat that as suspicious. Prefer a very small count unless you can point to explicit bright, "
        + "track-like passages with coherent displacement."
    )


def _window_prompt_text(window_index: int, start_sec: float, end_sec: float, *, mode: str = "legacy") -> str:
    if mode == "strict_counts":
        return (
            "Count fish passage in this sonar video window.\n"
            f"Window start in parent clip: {start_sec:.1f}s.\n"
            f"Window end in parent clip: {end_sec:.1f}s.\n"
            "Use only evidence visible inside this window.\n"
            "Do not infer events outside this window.\n"
            'Return exactly one JSON object and nothing else. '
            'The first character of your answer must be "{". '
            'Use exactly this schema: {"left_count": int, "right_count": int}. '
            'Do not include candidate_passages, scene_assessment, commentary, markdown, bullets, or prose. '
            'If no fish passage is visible, return {"left_count": 0, "right_count": 0}. '
            f"Window index: {window_index}."
        )
    return (
        "Analyze only this local temporal window from a longer sonar monitoring stream. "
        f"This window covers approximately {start_sec:.1f}s to {end_sec:.1f}s of the parent clip. "
        "Count only fish passage that is visible inside this window. "
        "Do not try to infer events outside the window. "
        "If one wave passes through the window, estimate its distinct fish throughput. "
        f"Return JSON only. Window index: {window_index}."
    )


def _dense_window_prompt_text(
    window_index: int,
    start_sec: float,
    end_sec: float,
    *,
    direction_hint: str | None,
    prior_throughput: float | None,
    peak_simultaneous_count: float | None,
    wave_count: int | None,
    mode: str = "legacy",
) -> str:
    if mode == "strict_counts":
        return (
            "Count fish passage in this preselected dense sonar video window.\n"
            f"Window start in parent clip: {start_sec:.1f}s.\n"
            f"Window end in parent clip: {end_sec:.1f}s.\n"
            f"Dominant first-pass direction hint: {direction_hint or 'unknown'}.\n"
            f"First-pass throughput hint: {prior_throughput if prior_throughput is not None else 'unknown'}.\n"
            f"Peak simultaneous hint: {peak_simultaneous_count if peak_simultaneous_count is not None else 'unknown'}.\n"
            "Use the first-pass hints only as weak context, not as a hard target.\n"
            "Use only evidence visible inside this window.\n"
            "Do not infer events outside this window.\n"
            'Return exactly one JSON object and nothing else. '
            'The first character of your answer must be "{". '
            'Use exactly this schema: {"left_count": int, "right_count": int}. '
            'Do not include candidate_passages, scene_assessment, commentary, markdown, bullets, or prose. '
            'If no fish passage is visible, return {"left_count": 0, "right_count": 0}. '
            f"Dense window index: {window_index}."
        )
    direction_text = direction_hint or "the dominant visible direction"
    return (
        "Analyze this preselected dense candidate passage window from a longer sonar monitoring stream. "
        f"This window covers approximately {start_sec:.1f}s to {end_sec:.1f}s of the parent clip. "
        f"The first-pass event proposal suggests movement in `{direction_text}` direction "
        f"with throughput near {prior_throughput if prior_throughput is not None else 'unknown'}, "
        f"peak simultaneous occupancy near {peak_simultaneous_count if peak_simultaneous_count is not None else 'unknown'}, "
        f"and wave count near {wave_count if wave_count is not None else 'unknown'}. "
        "Re-estimate total distinct fish throughput over the whole window, not just peak occupancy. "
        "If the stream contains replacement groups, mini-bursts, or continuous arrivals over time, sum them rather than collapsing to one instantaneous school size. "
        "Do not lower the first-pass throughput unless the visible evidence clearly rules it out. "
        f"Return JSON only. Dense window index: {window_index}."
    )


def _expand_window_bounds(
    start_sec: float,
    end_sec: float,
    *,
    total_duration_sec: float | None,
    min_duration_sec: float,
) -> tuple[float, float]:
    start = max(0.0, float(start_sec))
    end = max(start + 0.05, float(end_sec))
    min_duration = max(0.1, float(min_duration_sec))
    current = end - start
    if current >= min_duration:
        return start, end
    pad = (min_duration - current) / 2.0
    start -= pad
    end += pad
    if total_duration_sec is not None and total_duration_sec > 0:
        total = float(total_duration_sec)
        if start < 0:
            end = min(total, end - start)
            start = 0.0
        if end > total:
            shift = end - total
            start = max(0.0, start - shift)
            end = total
    return max(0.0, start), max(start + 0.05, end)


def _event_prompt(
    prompt_text: str,
    *,
    prompt_id: str,
    version: int = 0,
    assistant_prefill: str | None = None,
) -> PromptRevision:
    return PromptRevision(
        version=version,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        assistant_prefill=assistant_prefill,
    )


def _proposal_overlap_ratio(first: EventProposal, second: EventProposal) -> float:
    start = max(first.timestamp_start_sec, second.timestamp_start_sec)
    end = min(first.timestamp_end_sec, second.timestamp_end_sec)
    if end <= start:
        return 0.0
    overlap = end - start
    shorter = min(
        max(1e-6, first.timestamp_end_sec - first.timestamp_start_sec),
        max(1e-6, second.timestamp_end_sec - second.timestamp_start_sec),
    )
    return overlap / shorter


def _proposal_priority(proposal: EventProposal) -> tuple[float, float]:
    source_bonus = 2.0 if proposal.source == "backbone_candidate_passages" else 0.0
    duration = max(0.0, proposal.timestamp_end_sec - proposal.timestamp_start_sec)
    return (source_bonus + float(proposal.score), duration)


def _hybrid_proposals(
    *,
    motion_proposals: list[EventProposal],
    backbone_proposals: list[EventProposal],
    max_proposals: int,
    overlap_tolerance: float = 0.6,
    keep_stream_tiling_overlap: bool = False,
) -> list[EventProposal]:
    ordered = sorted(
        [*backbone_proposals, *motion_proposals],
        key=lambda item: (_proposal_priority(item), item.timestamp_start_sec),
        reverse=True,
    )
    selected: list[EventProposal] = []
    for proposal in ordered:
        allow_overlap = (
            keep_stream_tiling_overlap
            and proposal.source == "stream_tiling"
        )
        if not allow_overlap and any(_proposal_overlap_ratio(proposal, existing) >= overlap_tolerance for existing in selected):
            continue
        selected.append(proposal)
        if len(selected) >= max_proposals:
            break
    return sorted(selected, key=lambda item: item.timestamp_start_sec)


def _stream_tiling_proposals(
    clip,
    *,
    backbone_proposals: list[EventProposal],
    representation: str,
) -> list[EventProposal]:
    backbone_starts = [proposal.timestamp_start_sec for proposal in backbone_proposals]
    backbone_ends = [proposal.timestamp_end_sec for proposal in backbone_proposals]
    total_duration = max(0.0, float(getattr(clip, "duration_seconds", 0.0) or 0.0))
    if total_duration <= 0.0 and backbone_ends:
        total_duration = max(backbone_ends)
    if total_duration <= 0.0:
        return []

    span_start = min(backbone_starts) if backbone_starts else 0.0
    span_end = max(backbone_ends) if backbone_ends else total_duration
    span_start = max(0.0, min(span_start, total_duration))
    span_end = max(span_start, min(span_end, total_duration))
    span_duration = max(0.0, span_end - span_start)
    if span_duration < 12.0:
        span_start = 0.0
        span_end = total_duration
        span_duration = total_duration
    if span_duration < 12.0:
        return []

    tile_width = min(max(18.0, span_duration / 2.5), span_duration)
    step = max(8.0, tile_width * 0.7)

    tiles: list[EventProposal] = []
    start = span_start
    index = 1
    while start < span_end and len(tiles) < 4:
        end = min(span_end, start + tile_width)
        if end - start >= 12.0:
            tiles.append(
                EventProposal(
                    proposal_id=f"{clip.clip_id}:stream:{index:02d}",
                    timestamp_start_sec=float(start),
                    timestamp_end_sec=float(end),
                    score=1.6,
                    source="stream_tiling",
                    representation=representation,
                    reasoning="stream_throughput_tiling",
                    direction_hint=None,
                    metadata={
                        "tile_width_sec": float(end - start),
                        "span_start_sec": float(span_start),
                        "span_end_sec": float(span_end),
                    },
                )
            )
            index += 1
        if end >= span_end:
            break
        start += step
    return tiles


def _prediction_payload(prediction: PassagePrediction) -> dict[str, Any]:
    return {
        "scene_assessment": prediction.scene_assessment,
        "candidate_passages": prediction.candidate_passages,
        "rejected_targets": prediction.rejected_targets,
        "left_count": prediction.left_count,
        "right_count": prediction.right_count,
        "total_count": (prediction.left_count or 0) + (prediction.right_count or 0),
        "confidence": prediction.confidence,
        "commentary": prediction.commentary,
        "evidence_summary": prediction.evidence_summary,
        "events": [
            {
                "timestamp_sec": event.timestamp_sec,
                "direction": event.direction,
                "confidence": event.confidence,
                "evidence_note": event.evidence_note,
            }
            for event in prediction.events
        ],
        "latency_sec": prediction.latency_sec,
        "estimated_cost_usd": prediction.estimated_cost_usd,
        "usage_metadata": prediction.usage_metadata,
        "model_name": prediction.model_name,
        "parse_success": prediction.parse_success,
        "prompt_id": prediction.prompt_id,
    }


def _copy_candidate_with_offset(candidate: dict[str, Any], offset_sec: float) -> dict[str, Any]:
    updated = dict(candidate)
    start_sec = updated.get("timestamp_start_sec")
    end_sec = updated.get("timestamp_end_sec")
    if start_sec is not None:
        updated["timestamp_start_sec"] = float(start_sec) + offset_sec
    if end_sec is not None:
        updated["timestamp_end_sec"] = float(end_sec) + offset_sec
    return updated


def _dominant_direction(prediction: PassagePrediction, clip) -> str | None:
    counts = prediction.normalized_counts(clip.upstream_direction)
    if counts.left > counts.right:
        return "left"
    if counts.right > counts.left:
        return "right"
    return None


def _prediction_total(prediction: PassagePrediction, clip) -> int:
    counts = prediction.normalized_counts(clip.upstream_direction)
    return counts.left + counts.right


def _stream_primary_direction(site_profile) -> str | None:
    direction = str(getattr(site_profile, "stream_primary_direction", "") or "").strip().lower()
    if direction in {"left", "right"}:
        return direction
    return None


def _project_prediction_to_direction(
    prediction: PassagePrediction,
    *,
    direction: str | None,
) -> PassagePrediction:
    if direction not in {"left", "right"}:
        return prediction
    counts = prediction.normalized_counts(prediction.upstream_direction)
    total = counts.left + counts.right
    if total <= 0:
        return prediction
    updated_passages: list[dict[str, Any]] = []
    for passage in prediction.candidate_passages or []:
        updated = copy.deepcopy(passage)
        updated["direction"] = direction
        updated_passages.append(updated)
    updated_events = [replace(event, direction=direction) for event in prediction.events]
    return replace(
        prediction,
        left_count=total if direction == "left" else 0,
        right_count=total if direction == "right" else 0,
        candidate_passages=updated_passages,
        events=updated_events,
    )


def _select_stream_routing_direction(
    clip,
    *predictions: PassagePrediction,
) -> str:
    site_profile = resolve_site_profile(clip.domain, clip_key=clip.clip_id)
    primary_direction = _stream_primary_direction(site_profile)
    if primary_direction and "stream_throughput" in site_profile.expected_regime:
        return primary_direction
    left_total = 0
    right_total = 0
    for prediction in predictions:
        counts = prediction.normalized_counts(clip.upstream_direction)
        left_total += int(counts.left or 0)
        right_total += int(counts.right or 0)
    if left_total > right_total:
        return "left"
    if right_total > left_total:
        return "right"
    for prediction in predictions:
        dominant = _dominant_direction(prediction, clip)
        if dominant in {"left", "right"}:
            return dominant
    return "right"


def _window_activity_summary_from_runs(
    window_runs: list[dict[str, Any]] | None,
    *,
    direction: str = "right",
) -> dict[str, int]:
    totals: list[int] = []
    target_totals: list[int] = []
    opposite_totals: list[int] = []
    for window_run in window_runs or []:
        prediction_payload = window_run.get("selected_prediction") or window_run.get("prediction") or {}
        left_count = int(prediction_payload.get("left_count") or 0)
        right_count = int(prediction_payload.get("right_count") or 0)
        total = left_count + right_count
        totals.append(total)
        if direction == "left":
            target_totals.append(left_count)
            opposite_totals.append(right_count)
        else:
            target_totals.append(right_count)
            opposite_totals.append(left_count)
    active_windows = sum(1 for total in totals if total > 0)
    positive_window_totals = [total for total in totals if total > 0]
    positive_target_totals = sorted((value for value in target_totals if value > 0), reverse=True)
    return {
        "total_windows": len(totals),
        "active_windows": active_windows,
        "zero_windows": sum(1 for total in totals if total <= 0),
        "max_window_total": max(totals) if totals else 0,
        "sum_positive_window_totals": sum(positive_window_totals),
        "max_target_window_total": max(target_totals) if target_totals else 0,
        "second_target_window_total": positive_target_totals[1] if len(positive_target_totals) > 1 else 0,
        "sum_target_window_totals": sum(value for value in target_totals if value > 0),
        "sum_opposite_window_totals": sum(value for value in opposite_totals if value > 0),
    }


def _stream_backbone_activity_from_proposals(
    proposals: list[dict[str, Any]] | None,
    *,
    direction: str = "right",
) -> dict[str, int | float]:
    total_hint = 0
    max_peak = 0.0
    total_waves = 0
    proposal_count = 0
    for proposal in proposals or []:
        if not isinstance(proposal, dict):
            continue
        if str(proposal.get("source") or "") != "backbone_candidate_passages":
            continue
        metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), dict) else {}
        proposal_direction = str(metadata.get("direction") or proposal.get("direction_hint") or "").lower()
        if proposal_direction and proposal_direction != direction:
            continue
        proposal_count += 1
        total_hint += int(
            metadata.get("throughput_best_count")
            or metadata.get("estimated_count")
            or proposal.get("score")
            or 0
        )
        max_peak = max(max_peak, float(metadata.get("peak_simultaneous_count") or 0.0))
        total_waves += int(metadata.get("wave_count") or 0)
    return {
        "proposal_count": proposal_count,
        "total_hint": total_hint,
        "max_peak": max_peak,
        "total_waves": total_waves,
    }


def _mapped_event_direction(direction: str | None, clip) -> str | None:
    token = str(direction or "").strip().lower()
    if token in {"left", "right"}:
        return token
    if token == "upstream":
        if clip.upstream_direction is not None:
            return clip.upstream_direction.value
        return None
    if token == "downstream":
        if clip.upstream_direction is None:
            return None
        return "left" if clip.upstream_direction.value == "right" else "right"
    return None


def _window_counts_with_uncertain_direction_recovery(prediction: PassagePrediction, clip):
    counts = prediction.normalized_counts(clip.upstream_direction)
    if counts.total > 0:
        return counts
    if not prediction.parse_success:
        return counts
    candidates = [candidate for candidate in prediction.candidate_passages if isinstance(candidate, dict)]
    if not candidates:
        return counts
    candidate_directions = {str(candidate.get("direction") or "").strip().lower() for candidate in candidates}
    if any(direction not in {"", "uncertain"} for direction in candidate_directions):
        return counts
    mapped_event_directions = {
        mapped
        for mapped in (_mapped_event_direction(event.direction, clip) for event in prediction.events)
        if mapped is not None
    }
    if len(mapped_event_directions) != 1:
        return counts
    recovered_direction = next(iter(mapped_event_directions))
    inferred_total = 0
    for candidate in candidates:
        inferred_total += max(
            int(candidate.get("throughput_best_count") or 0),
            int(candidate.get("estimated_count") or 0),
        )
    if inferred_total <= 0:
        return counts
    if recovered_direction == "left":
        return type(counts)(left=inferred_total, right=0)
    return type(counts)(left=0, right=inferred_total)


def _should_accept_dense_window_prediction(
    clip,
    proposal: EventProposal,
    base_prediction: PassagePrediction,
    dense_prediction: PassagePrediction,
) -> tuple[bool, str]:
    base_total = _prediction_total(base_prediction, clip)
    dense_total = _prediction_total(dense_prediction, clip)
    base_direction = _dominant_direction(base_prediction, clip)
    dense_direction = _dominant_direction(dense_prediction, clip)
    hint_direction = str(proposal.direction_hint).lower() if proposal.direction_hint else None
    proposal_metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
    proposal_hint_total = max(
        int(proposal_metadata.get("throughput_best_count") or 0),
        int(proposal_metadata.get("estimated_count") or 0),
        int(proposal_metadata.get("peak_simultaneous_count") or 0),
    )
    if hint_direction and dense_direction and dense_direction != hint_direction:
        return False, "dense_direction_conflicts_with_proposal"
    if base_direction and dense_direction and dense_direction != base_direction:
        dense_counts = dense_prediction.normalized_counts(clip.upstream_direction)
        if dense_direction == "right":
            dense_opposite = dense_counts.left
        elif dense_direction == "left":
            dense_opposite = dense_counts.right
        else:
            dense_opposite = min(dense_counts.left, dense_counts.right)
        if (
            hint_direction
            and dense_direction == hint_direction
            and base_direction != hint_direction
            and dense_total > 0
            and dense_opposite == 0
            and base_total >= dense_total + 2
            and dense_total <= max(2, proposal_hint_total + 1)
        ):
            return True, "accepted_dense_direction_recovery"
        return False, "dense_direction_conflicts_with_base"
    if dense_total <= base_total:
        base_counts = base_prediction.normalized_counts(clip.upstream_direction)
        dense_counts = dense_prediction.normalized_counts(clip.upstream_direction)
        if base_direction == "right":
            base_opposite = base_counts.left
            dense_opposite = dense_counts.left
        elif base_direction == "left":
            base_opposite = base_counts.right
            dense_opposite = dense_counts.right
        else:
            base_opposite = min(base_counts.left, base_counts.right)
            dense_opposite = min(dense_counts.left, dense_counts.right)
        if (
            dense_total > 0
            and dense_direction
            and dense_direction == base_direction
            and base_opposite > 0
            and dense_opposite == 0
            and dense_total >= max(1, base_total - 2)
        ):
            return True, "accepted_dense_direction_cleanup"
        return False, "no_throughput_gain"
    dense_candidates = [candidate for candidate in dense_prediction.candidate_passages if isinstance(candidate, dict)]
    dense_max_peak = max(
        float(candidate.get("peak_simultaneous_count") or 0.0)
        for candidate in dense_candidates
    ) if dense_candidates else 0.0
    dense_total_waves = sum(int(candidate.get("wave_count") or 1) for candidate in dense_candidates)
    dense_max_duration = max(
        float(candidate.get("episode_duration_sec") or 0.0)
        for candidate in dense_candidates
    ) if dense_candidates else 0.0
    if (
        base_total <= 0
        and dense_total >= 4
        and dense_max_peak <= 1.0
        and dense_total_waves <= 1
        and dense_max_duration >= 20.0
    ):
        return False, "dense_sparse_stream_inflation"
    return True, "accepted_higher_dense_throughput"


def _effective_max_proposals(requested_max: int, backbone_proposals: list[EventProposal]) -> int:
    return max(int(requested_max), min(6, len(backbone_proposals)))


def _should_run_dense_window_refinement(
    clip,
    proposal: EventProposal,
    base_prediction: PassagePrediction,
) -> bool:
    if proposal.source != "backbone_candidate_passages":
        return False
    proposal_metadata = proposal.metadata if isinstance(proposal.metadata, dict) else {}
    proposal_hint = max(
        float(proposal.score or 0.0),
        float(proposal_metadata.get("throughput_best_count") or 0.0),
        float(proposal_metadata.get("estimated_count") or 0.0),
        float(proposal_metadata.get("peak_simultaneous_count") or 0.0),
    )
    wave_count = int(proposal_metadata.get("wave_count") or 0)
    duration_sec = max(0.0, float(proposal.duration_sec))
    base_total = _prediction_total(base_prediction, clip)
    if proposal_hint >= 5.0:
        return True
    if wave_count >= 2:
        return True
    if duration_sec >= 9.0 and max(proposal_hint, float(base_total)) >= 1.0:
        return True
    if len(base_prediction.candidate_passages) >= 2 and max(proposal_hint, float(base_total)) >= 2.0:
        return True
    return False


def _aggregate_window_predictions(
    clip,
    window_predictions: list[tuple[float, PassagePrediction]],
    *,
    prompt_id: str,
) -> PassagePrediction:
    if not window_predictions:
        return PassagePrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            left_count=0,
            right_count=0,
            confidence=0.0,
            commentary="proposal_guided produced no candidate windows",
            evidence_summary="no temporal windows were proposed",
            parse_success=False,
            prompt_id=prompt_id,
            candidate_passages=[],
            events=[],
            latency_sec=0.0,
        )

    left_total = 0
    right_total = 0
    latency_total = 0.0
    confidences: list[float] = []
    candidate_passages: list[dict[str, Any]] = []
    events: list[PassageEvent] = []
    commentaries: list[str] = []
    evidence_parts: list[str] = []
    parse_success = True
    window_details: list[dict[str, Any]] = []

    for window_start_sec, prediction in window_predictions:
        counts = _window_counts_with_uncertain_direction_recovery(prediction, clip)
        left_total += counts.left
        right_total += counts.right
        window_details.append(
            {
                "window_start_sec": window_start_sec,
                "prediction": prediction,
                "left_count": counts.left,
                "right_count": counts.right,
            }
        )
        latency_total += float(prediction.latency_sec or 0.0)
        parse_success = parse_success and bool(prediction.parse_success)
        if prediction.confidence is not None:
            confidences.append(float(prediction.confidence))
        if prediction.commentary:
            commentaries.append(str(prediction.commentary))
        if prediction.evidence_summary:
            evidence_parts.append(str(prediction.evidence_summary))
        for candidate in prediction.candidate_passages:
            if isinstance(candidate, dict):
                candidate_passages.append(_copy_candidate_with_offset(candidate, window_start_sec))
        for event in prediction.events:
            events.append(
                PassageEvent(
                    timestamp_sec=float(event.timestamp_sec) + window_start_sec,
                    direction=event.direction,
                    confidence=event.confidence,
                    evidence_note=event.evidence_note,
                )
            )

    left_total, right_total, candidate_passages, events, suppression_note = _suppress_weak_opposite_singleton(
        clip,
        window_details=window_details,
        left_total=left_total,
        right_total=right_total,
        candidate_passages=candidate_passages,
        events=events,
    )
    confidence = (sum(confidences) / len(confidences)) if confidences else None
    if confidence is not None and not candidate_passages and not events:
        confidence = min(confidence, 0.25)
    evidence_summary = " | ".join(evidence_parts[:3]) if evidence_parts else "aggregated local window evidence"
    if suppression_note:
        evidence_summary = " | ".join(part for part in [evidence_summary, suppression_note] if part)
    return PassagePrediction(
        domain=clip.domain,
        clip_id=clip.clip_id,
        scene_assessment="proposal-guided aggregation over candidate event windows",
        candidate_passages=candidate_passages,
        rejected_targets=[],
        left_count=left_total,
        right_count=right_total,
        events=events,
        confidence=confidence,
        commentary=" | ".join(commentaries[:3]) if commentaries else "proposal-guided aggregation",
        evidence_summary=evidence_summary,
        latency_sec=latency_total,
        parse_success=parse_success,
        prompt_id=prompt_id,
    )


def _suppress_weak_opposite_singleton(
    clip,
    *,
    window_details: list[dict[str, Any]],
    left_total: int,
    right_total: int,
    candidate_passages: list[dict[str, Any]],
    events: list[PassageEvent],
) -> tuple[int, int, list[dict[str, Any]], list[PassageEvent], str | None]:
    domain = str(getattr(clip, "domain", "") or "").lower()
    if not domain.startswith("kenai"):
        return left_total, right_total, candidate_passages, events, None
    if left_total == right_total:
        return left_total, right_total, candidate_passages, events, None

    dominant_direction = "right" if right_total > left_total else "left"
    dominant_total = right_total if dominant_direction == "right" else left_total
    opposite_total = left_total if dominant_direction == "right" else right_total
    if dominant_total < 8 or opposite_total != 1:
        return left_total, right_total, candidate_passages, events, None

    strongest_same_window = 0
    opposite_windows: list[dict[str, Any]] = []
    opposite_direction = "left" if dominant_direction == "right" else "right"
    for detail in window_details:
        same_count = int(detail.get(f"{dominant_direction}_count") or 0)
        opposite_count = int(detail.get(f"{opposite_direction}_count") or 0)
        strongest_same_window = max(strongest_same_window, same_count)
        if opposite_count > 0:
            opposite_windows.append(detail)

    if strongest_same_window < 8 or len(opposite_windows) != 1:
        return left_total, right_total, candidate_passages, events, None

    opposite_window = opposite_windows[0]
    opposite_prediction = opposite_window.get("prediction")
    if not isinstance(opposite_prediction, PassagePrediction):
        return left_total, right_total, candidate_passages, events, None
    if int(opposite_window.get(f"{opposite_direction}_count") or 0) != 1:
        return left_total, right_total, candidate_passages, events, None
    if int(opposite_window.get(f"{dominant_direction}_count") or 0) != 0:
        return left_total, right_total, candidate_passages, events, None
    try:
        opposite_confidence = float(opposite_prediction.confidence or 0.0)
    except (TypeError, ValueError):
        opposite_confidence = 0.0
    if opposite_confidence > 0.5:
        return left_total, right_total, candidate_passages, events, None

    filtered_candidates: list[dict[str, Any]] = []
    for candidate in candidate_passages:
        if not isinstance(candidate, dict):
            continue
        direction = str(candidate.get("direction") or "").lower()
        throughput = int(candidate.get("throughput_best_count") or candidate.get("estimated_count") or 0)
        if direction == opposite_direction and throughput <= 1:
            continue
        filtered_candidates.append(candidate)

    filtered_events = [event for event in events if str(event.direction or "").lower() != opposite_direction]
    if dominant_direction == "right":
        left_total = 0
    else:
        right_total = 0
    note = (
        f"suppressed weak opposite-direction singleton from one low-confidence local window; "
        f"dominant {dominant_direction} throughput retained"
    )
    return left_total, right_total, filtered_candidates, filtered_events, note


def _apply_constraint_repair(
    clip,
    prediction: PassagePrediction,
    *,
    representation: str,
    prompt_id: str,
) -> tuple[PassagePrediction, dict[str, Any]]:
    audit = audit_prediction_constraints(
        prediction,
        representation=representation,
        fallback_upstream_direction=clip.upstream_direction,
    )
    corrected = audit.corrected_counts or prediction.normalized_counts(clip.upstream_direction)
    note = audit.note or "constraint repair"
    if audit.findings:
        note = f"{note}; findings={len(audit.findings)}"
    repaired = replace(
        prediction,
        left_count=corrected.left,
        right_count=corrected.right,
        commentary=" | ".join(
            part for part in [prediction.commentary, "constraint-reconciled local reasoning"] if part
        ),
        evidence_summary=" | ".join(
            part for part in [prediction.evidence_summary, note] if part
        ),
        prompt_id=prompt_id,
    )
    audit_payload = {
        "applied": audit.applied,
        "repaired": audit.repaired,
        "findings": [asdict(item) for item in audit.findings],
        "corrected_counts": audit.corrected_counts.to_dict() if audit.corrected_counts is not None else None,
        "corrected_event_count": audit.corrected_event_count,
        "note": audit.note,
    }
    return repaired, audit_payload


def _archive_prompt(
    batch_root: Path,
    clip,
    variant: InputVariant,
    prompt: PromptRevision,
    model: str,
    transport: str,
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, str]:
    rendered_prompt = build_prompt(clip, variant, prompt)
    prompt_metadata = {
        "domain": clip.domain,
        "clip_id": clip.clip_id,
        "model": model,
        "transport": transport,
    }
    if metadata:
        prompt_metadata.update(metadata)
    return write_prompt_artifact(
        output_root=batch_root,
        scope="batch",
        variant=variant.value,
        prompt_id=prompt.prompt_id,
        version=prompt.version,
        prompt_text=prompt.prompt_text,
        critique=prompt.critique,
        metrics=dict(prompt.metrics),
        rendered_prompt=rendered_prompt,
        metadata=prompt_metadata,
    )


def _infer_action_prediction(
    *,
    client,
    clip,
    variant: InputVariant,
    prompt: PromptRevision,
    batch_root: Path,
    response_stem: str,
    transport: str,
    extra_metadata: dict[str, Any] | None = None,
) -> PassagePrediction:
    prompt_artifact = _archive_prompt(
        batch_root,
        clip,
        variant,
        prompt,
        client.model,
        transport,
        metadata=extra_metadata,
    )
    response = client.infer(clip, variant, prompt)
    prediction = parse_passage_prediction(
        raw_text=response.text,
        domain=clip.domain,
        clip_id=clip.clip_id,
        prompt_id=prompt.prompt_id,
        latency_sec=response.latency_sec,
        usage_metadata=response.usage_metadata,
        estimated_cost_usd=response.estimated_cost_usd,
        model_name=client.model,
    )
    response_path = batch_root / "responses" / variant.value / f"{response_stem}.response.json"
    _write_text(response_path, response.text)
    prediction.raw_response = response.text
    return prediction


def _routing_result_payload(
    *,
    clip_key: str,
    prediction: PassagePrediction,
    bucket: str,
    domain: str,
    window_activity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "status": "completed",
        "clip_key": clip_key,
        "bucket": bucket,
        "domain": domain,
        "prediction": _prediction_payload(prediction),
    }
    if window_activity:
        payload["window_activity"] = dict(window_activity)
    return payload


def _select_risk_assessment(
    *,
    global_result: dict[str, Any],
    proposal_result: dict[str, Any],
    repair_result: dict[str, Any] | None = None,
    direction: str,
    site_profile,
):
    global_assessment = assess_domain_shift_risk(
        global_result,
        direction=direction,
        site_profile=site_profile,
    )
    proposal_assessment = assess_domain_shift_risk(
        proposal_result,
        direction=direction,
        site_profile=site_profile,
    )
    repair_assessment = None
    if repair_result is not None:
        repair_assessment = assess_domain_shift_risk(
            repair_result,
            direction=direction,
            site_profile=site_profile,
        )
    if "stream_throughput" in getattr(site_profile, "expected_regime", ()):
        assessments = [global_assessment, proposal_assessment]
        if repair_assessment is not None:
            assessments.append(repair_assessment)
        stream_assessments = [assessment for assessment in assessments if "dense_stream_throughput_risk" in assessment.flags]
        if stream_assessments:
            chosen = max(stream_assessments, key=lambda assessment: assessment.predicted_total)
            merged_flags = tuple(dict.fromkeys(flag for assessment in stream_assessments for flag in assessment.flags))
            return replace(
                chosen,
                flags=merged_flags,
                reason=chosen.reason + " | stream_throughput_assessments_merged",
            )
        chosen = max(assessments, key=lambda assessment: assessment.predicted_total)
        merged_flags = tuple(dict.fromkeys(flag for assessment in assessments for flag in assessment.flags))
        return replace(
            chosen,
            flags=merged_flags,
            reason=chosen.reason + " | stream_throughput_profile_fallback",
        )
    if proposal_assessment.flags and not global_assessment.flags:
        return proposal_assessment
    if global_assessment.flags and not proposal_assessment.flags:
        return global_assessment
    if proposal_assessment.flags and global_assessment.flags:
        merged_flags = tuple(dict.fromkeys([*global_assessment.flags, *proposal_assessment.flags]))
        chosen = proposal_assessment if proposal_assessment.predicted_total >= global_assessment.predicted_total else global_assessment
        risk_level_order = {"low": 0, "medium": 1, "high": 2}
        chosen_level = chosen.risk_level
        if risk_level_order.get(proposal_assessment.risk_level, 0) > risk_level_order.get(chosen_level, 0):
            chosen_level = proposal_assessment.risk_level
        if risk_level_order.get(global_assessment.risk_level, 0) > risk_level_order.get(chosen_level, 0):
            chosen_level = global_assessment.risk_level
        return replace(
            chosen,
            flags=merged_flags,
            risk_level=chosen_level,
            reason=(
                f"{chosen.reason} | "
                f"global_flags={list(global_assessment.flags)} proposal_flags={list(proposal_assessment.flags)}"
            ),
        )
    return global_assessment


def _select_risk_source_result(
    *,
    global_result: dict[str, Any],
    proposal_result: dict[str, Any],
    repair_result: dict[str, Any] | None = None,
    direction: str,
    site_profile,
) -> dict[str, Any]:
    global_assessment = assess_domain_shift_risk(
        global_result,
        direction=direction,
        site_profile=site_profile,
    )
    proposal_assessment = assess_domain_shift_risk(
        proposal_result,
        direction=direction,
        site_profile=site_profile,
    )
    repair_assessment = None
    if repair_result is not None:
        repair_assessment = assess_domain_shift_risk(
            repair_result,
            direction=direction,
            site_profile=site_profile,
        )
    if "stream_throughput" in getattr(site_profile, "expected_regime", ()):
        candidate_pairs = [
            (global_result, global_assessment),
            (proposal_result, proposal_assessment),
        ]
        if repair_result is not None and repair_assessment is not None:
            candidate_pairs.append((repair_result, repair_assessment))
        stream_pairs = [
            pair
            for pair in candidate_pairs
            if "dense_stream_throughput_risk" in pair[1].flags
        ]
        if stream_pairs:
            return max(
                stream_pairs,
                key=lambda pair: int((pair[0].get("prediction") or {}).get(f"{direction}_count") or 0),
            )[0]
        return max(
            candidate_pairs,
            key=lambda pair: int((pair[0].get("prediction") or {}).get(f"{direction}_count") or 0),
        )[0]
    if proposal_assessment.flags and not global_assessment.flags:
        return proposal_result
    if global_assessment.flags and not proposal_assessment.flags:
        return global_result
    if proposal_assessment.flags and global_assessment.flags:
        return proposal_result if proposal_assessment.predicted_total >= global_assessment.predicted_total else global_result
    return global_result


def _custom_branch_prompt(branch_id: str) -> PromptRevision:
    if branch_id == "custom_trickle_reduction":
        return _event_prompt(_trickle_reduction_prompt(), prompt_id="event_agent_custom_trickle_reduction")
    if branch_id == "custom_high_throughput":
        return _event_prompt(_high_throughput_custom_prompt(), prompt_id="event_agent_custom_high_throughput")
    if branch_id == "custom_low_count":
        return _event_prompt(_low_count_far_view_prompt(), prompt_id="event_agent_custom_low_count")
    if branch_id == "custom_stream_throughput":
        return _event_prompt(_stream_throughput_prompt(), prompt_id="event_agent_custom_stream_throughput")
    raise ValueError(f"unsupported custom branch: {branch_id}")


def _custom_branch_prompt_for_clip(
    branch_id: str,
    *,
    risk_assessment,
    window_runs: list[dict[str, Any]] | None = None,
    direction: str = "right",
    proposals: list[dict[str, Any]] | None = None,
) -> PromptRevision:
    if branch_id == "custom_stream_throughput":
        window_activity = _window_activity_summary_from_runs(window_runs, direction=direction)
        backbone_activity = _stream_backbone_activity_from_proposals(proposals, direction=direction)
        return _event_prompt(
            _stream_throughput_prompt_with_context(
                direction=direction,
                predicted_total=int(risk_assessment.predicted_total),
                max_peak=float(risk_assessment.max_peak),
                max_duration_sec=float(risk_assessment.max_duration_sec),
                window_activity=window_activity,
                backbone_activity=backbone_activity,
            ),
            prompt_id="event_agent_custom_stream_throughput",
        )
    if branch_id == "custom_trickle_reduction":
        return _event_prompt(
            _trickle_reduction_prompt_with_context(
                predicted_total=int(risk_assessment.predicted_total),
                candidate_count=int(risk_assessment.candidate_count),
                total_peak=float(risk_assessment.total_peak),
                max_peak=float(risk_assessment.max_peak),
                total_wave_count=int(risk_assessment.total_wave_count),
            ),
            prompt_id="event_agent_custom_trickle_reduction",
        )
    if branch_id == "custom_high_throughput" and "single_school_high_throughput_undercount_risk" in risk_assessment.flags:
        activity = _window_activity_summary_from_runs(list(window_runs or []), direction=direction)
        window_activity_notes = ""
        if activity["total_windows"] > 0:
            window_activity_notes = (
                "\n\nLocal-window evidence summary:\n"
                f"- local windows with positive fish evidence: {activity['active_windows']}/{activity['total_windows']}\n"
                f"- zero-evidence windows: {activity['zero_windows']}\n"
                f"- strongest local window total: {activity['max_window_total']}\n"
            )
            if activity["active_windows"] == 1 and activity["zero_windows"] >= 1:
                window_activity_notes += (
                    "Only one long local window carries the visible school while neighboring windows are quiet. "
                    "Treat that strongest window as a lower bound on throughput, not an upper bound, when the same dense school plainly traverses the corridor for a long interval."
                )
            elif activity["active_windows"] >= 2:
                window_activity_notes += (
                    "Multiple local windows contain positive school evidence. "
                    "Use that as support for sustained throughput across time, not just one peak snapshot."
                )
        return _event_prompt(
            _single_school_high_throughput_prompt_with_context(
                predicted_total=int(risk_assessment.predicted_total),
                max_peak=float(risk_assessment.max_peak),
                max_duration_sec=float(risk_assessment.max_duration_sec),
                window_activity_notes=window_activity_notes,
            ),
            prompt_id="event_agent_custom_high_throughput",
        )
    if branch_id == "custom_high_throughput" and "hidden_crossing_undercount_risk" in risk_assessment.flags:
        activity = _window_activity_summary_from_runs(list(window_runs or []), direction=direction)
        window_activity_notes = ""
        if activity["total_windows"] > 0:
            window_activity_notes = (
                "\n\nLocal-window evidence summary:\n"
                f"- local windows with positive fish evidence: {activity['active_windows']}/{activity['total_windows']}\n"
                f"- zero-evidence windows: {activity['zero_windows']}\n"
                f"- strongest local window total: {activity['max_window_total']}\n"
            )
            if activity["active_windows"] >= 2:
                window_activity_notes += (
                    "Multiple local windows contain positive motion evidence. "
                    "Treat that as support for repeated entrants or sustained throughput across time, rather than collapsing everything to one visible peak."
                )
            elif (
                activity["active_windows"] == 1
                and activity["max_window_total"] >= max(6, int(risk_assessment.predicted_total))
                and float(risk_assessment.max_peak) >= 4.0
                and float(risk_assessment.max_duration_sec) >= 18.0
            ):
                window_activity_notes += (
                    "Only one local window contains the strongest visible school, but that window is already dense and long-lived. "
                    "Do not cap the total at that strongest-window count if the full clip shows the same coherent school entering and exiting through the corridor with brief hidden overlap."
                )
        return _event_prompt(
            _high_throughput_hidden_crossing_prompt_with_context(
                predicted_total=int(risk_assessment.predicted_total),
                total_peak=float(risk_assessment.total_peak),
                max_peak=float(risk_assessment.max_peak),
                max_duration_sec=float(risk_assessment.max_duration_sec),
                window_activity_notes=window_activity_notes,
            ),
            prompt_id="event_agent_custom_high_throughput",
        )
    if branch_id == "custom_high_throughput" and "single_school_visibility_dropout_risk" in risk_assessment.flags:
        activity = _window_activity_summary_from_runs(list(window_runs or []), direction=direction)
        window_activity_notes = ""
        if activity["total_windows"] > 0:
            window_activity_notes = (
                "\n\nLocal-window evidence summary:\n"
                f"- local windows with positive fish evidence: {activity['active_windows']}/{activity['total_windows']}\n"
                f"- zero-evidence windows: {activity['zero_windows']}\n"
                f"- strongest local window total: {activity['max_window_total']}\n"
            )
            if activity["active_windows"] == 1 and activity["zero_windows"] >= 1:
                window_activity_notes += (
                    "Only one local window contains the visible school while at least one neighboring window is empty or static. "
                    "In far-view clips, this pattern can still undercount the same school because entry/exit and beam geometry may hide one or two fish outside the clearest window. "
                    "If the same coherent school plainly traverses the corridor, it is acceptable to raise the total slightly above the strongest visible window."
                )
        return _event_prompt(
            _single_school_visibility_prompt_with_context(
                predicted_total=int(risk_assessment.predicted_total),
                max_peak=float(risk_assessment.max_peak),
                max_duration_sec=float(risk_assessment.max_duration_sec),
                window_activity_notes=window_activity_notes,
            ),
            prompt_id="event_agent_custom_high_throughput",
        )
    if branch_id == "custom_low_count":
        activity = _window_activity_summary_from_runs(list(window_runs or []), direction=direction)
        window_activity_notes = ""
        if activity["total_windows"] > 0:
            window_activity_notes = (
                "\n\nLocal-window evidence summary:\n"
                f"- local windows with positive fish evidence: {activity['active_windows']}/{activity['total_windows']}\n"
                f"- zero-evidence windows: {activity['zero_windows']}\n"
                f"- strongest local window total: {activity['max_window_total']}\n"
            )
            if activity["active_windows"] <= 1 and activity["zero_windows"] >= 1:
                window_activity_notes += (
                    "Only one local window contains positive fish evidence while neighboring windows are empty or static. "
                    "Do not trust a large raw window total by itself. "
                    "Prefer 1-2 defensible passages unless the video shows explicit fresh entrants."
                )
            elif activity["active_windows"] <= 2 and activity["zero_windows"] >= 1:
                window_activity_notes += (
                    "Positive evidence is concentrated in only one or two local windows, with at least one empty neighbor. "
                    "Treat this as a sparse burst rather than a continuous school unless replacement entrants are obvious."
                )
        return _event_prompt(
            _low_count_far_view_prompt_with_context(
                predicted_total=int(risk_assessment.predicted_total),
                candidate_count=int(risk_assessment.candidate_count),
                total_peak=float(risk_assessment.total_peak),
                max_peak=float(risk_assessment.max_peak),
                total_wave_count=int(risk_assessment.total_wave_count),
                window_activity_notes=window_activity_notes,
            ),
            prompt_id="event_agent_custom_low_count",
        )
    return _custom_branch_prompt(branch_id)


def _should_apply_routed_override(
    clip,
    *,
    branch_id: str,
    global_prediction: PassagePrediction,
    custom_prediction: PassagePrediction,
) -> bool:
    global_total = _prediction_total(global_prediction, clip)
    custom_total = _prediction_total(custom_prediction, clip)
    if branch_id == "custom_trickle_reduction":
        return 0 < custom_total < global_total
    if branch_id == "custom_high_throughput":
        return custom_total > global_total
    if branch_id == "custom_stream_throughput":
        return True
    if branch_id == "custom_low_count":
        global_counts = global_prediction.normalized_counts(clip.upstream_direction)
        custom_counts = custom_prediction.normalized_counts(clip.upstream_direction)
        return (
            custom_total != global_total
            or custom_counts.left != global_counts.left
            or custom_counts.right != global_counts.right
        )
    return custom_total != global_total


def _select_trickle_reduction_prediction(
    clip,
    predictions: list[PassagePrediction],
) -> PassagePrediction:
    if not predictions:
        raise ValueError("predictions must not be empty")
    parseable = [prediction for prediction in predictions if prediction.parse_success]
    pool = parseable or list(predictions)
    positive = [prediction for prediction in pool if _prediction_total(prediction, clip) > 0]
    selected_pool = positive or pool
    return min(
        selected_pool,
        key=lambda prediction: (
            _prediction_total(prediction, clip),
            -(float(prediction.confidence or 0.0)),
        ),
    )


def _select_high_throughput_prediction(
    clip,
    predictions: list[PassagePrediction],
) -> PassagePrediction:
    if not predictions:
        raise ValueError("predictions must not be empty")
    parseable = [prediction for prediction in predictions if prediction.parse_success]
    pool = parseable or list(predictions)
    direction = getattr(clip, "upstream_direction", None)
    if direction is None:
        direction = "right"
    direction = str(direction).lower()

    def _target_and_opposite(prediction: PassagePrediction) -> tuple[int, int]:
        counts = prediction.normalized_counts(direction)
        if direction == "left":
            return counts.left, counts.right
        return counts.right, counts.left

    same_direction = [prediction for prediction in pool if _target_and_opposite(prediction)[1] == 0]
    selected_pool = same_direction or pool
    return max(
        selected_pool,
        key=lambda prediction: (
            _target_and_opposite(prediction)[0] > 0,
            _target_and_opposite(prediction)[0],
            _prediction_total(prediction, clip),
            float(prediction.confidence or 0.0),
        ),
    )


def _select_stream_throughput_prediction(
    clip,
    predictions: list[PassagePrediction],
    *,
    direction: str | None = None,
) -> PassagePrediction:
    if not predictions:
        raise ValueError("predictions must not be empty")
    parseable = [prediction for prediction in predictions if prediction.parse_success]
    pool = parseable or list(predictions)
    if direction is None:
        direction = getattr(clip, "upstream_direction", None)
    if direction is None:
        direction = "right"
    direction = str(direction).lower()

    def _target_and_opposite(prediction: PassagePrediction) -> tuple[int, int]:
        counts = prediction.normalized_counts(direction)
        if direction == "left":
            return counts.left, counts.right
        return counts.right, counts.left

    return max(
        pool,
        key=lambda prediction: (
            -_target_and_opposite(prediction)[1],
            _target_and_opposite(prediction)[0],
            _prediction_total(prediction, clip),
            float(prediction.confidence or 0.0),
        ),
    )


def _should_run_flagged_repeat_consensus(result_record: dict[str, Any], *, repeat_runs: int) -> bool:
    if repeat_runs <= 1:
        return False
    if result_record.get("status") != "completed":
        return False
    site_profile = result_record.get("site_profile") if isinstance(result_record.get("site_profile"), dict) else {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" in expected_regime:
        return False
    flags = set(str(item) for item in (result_record.get("risk_flags") or []))
    if flags & {"multi_episode_trickle_overcount_risk", "low_count_far_view_risk"}:
        return True
    prediction = result_record.get("prediction") or {}
    assessment = result_record.get("risk_assessment") or {}
    left_count = int(prediction.get("left_count") or 0)
    right_count = int(prediction.get("right_count") or 0)
    total_count = left_count + right_count
    minority_count = min(left_count, right_count)
    candidate_count = int(assessment.get("candidate_count") or 0)
    total_wave_count = int(assessment.get("total_wave_count") or 0)
    total_peak = float(assessment.get("total_peak") or 0.0)
    max_peak = float(assessment.get("max_peak") or 0.0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    confidence = float(assessment.get("confidence") or 0.0)
    selected_branch = str(result_record.get("selected_branch") or "")
    selected_action = str(result_record.get("selected_action") or "")
    custom_branch_runs = result_record.get("custom_branch_runs") or []
    reduction_totals: list[int] = []
    for branch_run in custom_branch_runs:
        prediction_payload = branch_run.get("prediction") or {}
        if not bool(prediction_payload.get("parse_success", False)):
            continue
        custom_total = int(prediction_payload.get("left_count") or 0) + int(
            prediction_payload.get("right_count") or 0
        )
        if 0 < custom_total < total_count:
            reduction_totals.append(custom_total)
    best_reduction_total = min(reduction_totals) if reduction_totals else None
    if (
        selected_branch == "baseline"
        and "multiwave_high_peak_undercount_risk" in flags
        and total_count >= max(8, int(round(total_peak * 1.5)))
        and max_peak <= 3.0
        and confidence <= 0.7
    ):
        return True
    if (
        selected_branch == "baseline"
        and best_reduction_total is not None
        and total_count >= max(best_reduction_total + 6, int(round(best_reduction_total * 1.8)))
    ):
        return True
    if (
        selected_branch == "baseline"
        and candidate_count == 0
        and total_count >= 4
        and confidence <= 0.6
        and selected_action.endswith("proposal_guided")
    ):
        return True
    if (
        selected_branch == "baseline"
        and total_count >= 5
        and candidate_count >= 3
        and total_wave_count >= 4
        and max_peak <= 1.5
        and confidence <= 0.7
    ):
        return True
    if (
        selected_branch == "baseline"
        and total_count >= 6
        and minority_count >= 1
        and candidate_count >= 3
        and max_peak <= 2.0
        and confidence <= 0.7
    ):
        return True
    if (
        selected_branch == "baseline"
        and total_count >= max(8, int(round(total_peak * 4.0)))
        and max_peak <= 2.0
        and max_duration_sec >= 30.0
        and confidence <= 0.7
    ):
        return True
    site_profile_id = str(result_record.get("site_profile_id") or "")
    activity = _post_consensus_window_activity(result_record)
    if site_profile_id.startswith("kenai") and total_count >= 6:
        single_school_shape = (
            (
                candidate_count == 1
                and total_wave_count == 1
                and max_peak >= 3.0
                and max_duration_sec >= 20.0
            )
            or (
                int(_selected_prediction_candidate_activity(result_record)["candidate_count"]) == 1
                and int(_selected_prediction_candidate_activity(result_record)["total_waves"]) == 1
                and float(_selected_prediction_candidate_activity(result_record)["max_peak"]) >= 3.0
                and float(_selected_prediction_candidate_activity(result_record)["max_duration_sec"]) >= 20.0
                and int(_selected_prediction_candidate_activity(result_record)["max_total"]) >= 10
            )
        )
        if (
            selected_branch == "baseline"
            and single_school_shape
            and confidence <= 0.85
            and activity["active_windows"] <= 1
            and activity["max_window_total"] >= max(6, total_count)
        ):
            return True
        if (
            selected_branch == "custom_high_throughput"
            and single_school_shape
            and confidence <= 0.85
            and activity["active_windows"] <= 1
            and activity["max_window_total"] >= 8
            and total_count <= activity["max_window_total"] + 1
        ):
            return True
    return False


def _repeat_consensus_sort_key(result_record: dict[str, Any]) -> tuple[Any, ...]:
    prediction = result_record.get("prediction") or {}
    parse_success = bool(prediction.get("parse_success", False))
    confidence = float(prediction.get("confidence") or 0.0)
    left_count = int(prediction.get("left_count") or 0)
    right_count = int(prediction.get("right_count") or 0)
    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    if routing_direction == "left":
        target_count = left_count
        opposite_count = right_count
    else:
        target_count = right_count
        opposite_count = left_count
    total_count = left_count + right_count
    branch_id = str(result_record.get("selected_branch") or "")
    custom_rank = 0 if branch_id in {"custom_trickle_reduction", "custom_low_count"} else 1
    positive_rank = 0 if target_count > 0 else 1
    total_rank = total_count if target_count > 0 else 10**6 + total_count
    target_rank = target_count if target_count > 0 else 10**6
    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    assessment = result_record.get("risk_assessment") or {}
    site_profile_id = str(result_record.get("site_profile_id") or "")
    activity = _post_consensus_window_activity(result_record)
    selected_candidate_activity = _selected_prediction_candidate_activity(result_record)
    if "multi_episode_trickle_overcount_risk" in risk_flags:
        return (
            0 if parse_success else 1,
            positive_rank,
            opposite_count,
            total_rank,
            target_rank,
            custom_rank,
            -confidence,
        )
    if (
        site_profile_id.startswith("kenai")
        and (
            (
                int(assessment.get("candidate_count") or 0) == 1
                and int(assessment.get("total_wave_count") or 0) == 1
                and float(assessment.get("max_peak") or 0.0) >= 3.0
                and float(assessment.get("max_duration_sec") or 0.0) >= 20.0
            )
            or (
                int(selected_candidate_activity["candidate_count"]) == 1
                and int(selected_candidate_activity["total_waves"]) == 1
                and float(selected_candidate_activity["max_peak"]) >= 3.0
                and float(selected_candidate_activity["max_duration_sec"]) >= 20.0
                and int(selected_candidate_activity["max_total"]) >= 10
            )
        )
        and activity["active_windows"] <= 1
        and activity["max_window_total"] >= 6
    ):
        return (
            0 if parse_success else 1,
            opposite_count,
            positive_rank,
            -target_count,
            -total_count,
            -confidence,
        )
    return (
        0 if parse_success else 1,
        custom_rank,
        positive_rank,
        opposite_count,
        total_rank,
        target_rank,
        -confidence,
    )


def _expand_repeat_consensus_candidates(
    pass_results: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], int, str, str | None]]:
    candidates: list[tuple[dict[str, Any], int, str, str | None]] = []
    for index, item in enumerate(pass_results):
        candidates.append((item, index + 1, "selected_result", None))
        selected_prediction = item.get("prediction") or {}
        selected_total = int(selected_prediction.get("left_count") or 0) + int(selected_prediction.get("right_count") or 0)
        selected_action = str(item.get("selected_action") or "")
        variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
        for branch_run in item.get("custom_branch_runs") or []:
            branch_id = str(branch_run.get("branch_id") or "")
            if branch_id not in {"custom_trickle_reduction", "custom_low_count"}:
                continue
            prediction_payload = branch_run.get("prediction") or {}
            if not bool(prediction_payload.get("parse_success", False)):
                continue
            custom_total = int(prediction_payload.get("left_count") or 0) + int(
                prediction_payload.get("right_count") or 0
            )
            if custom_total <= 0:
                continue
            if selected_total > 0 and custom_total >= selected_total:
                continue
            candidate = copy.deepcopy(item)
            candidate["selected_branch"] = branch_id
            candidate["selected_action"] = f"{variant_prefix}:{branch_id}" if variant_prefix else branch_id
            candidate["selection_reason"] = f"repeat_consensus_{branch_id}_from_pass_{index + 1}"
            candidate["prediction"] = copy.deepcopy(prediction_payload)
            candidates.append((candidate, index + 1, "custom_branch_runs", branch_id))
    return candidates


def _apply_flagged_repeat_consensus(
    pass_results: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not pass_results:
        raise ValueError("pass_results must not be empty")
    expanded_candidates = _expand_repeat_consensus_candidates(pass_results)
    completed = [item for item in expanded_candidates if item[0].get("status") == "completed"]
    pool = completed or list(expanded_candidates)
    selected, source_pass_index, selected_source, selected_branch_candidate = min(
        pool,
        key=lambda item: _repeat_consensus_sort_key(item[0]),
    )
    audit = {
        "applied": len(pass_results) > 1,
        "repeat_runs": len(pass_results),
        "selected_pass_index": source_pass_index,
        "selected_source": selected_source,
        "selected_branch": selected.get("selected_branch"),
        "selected_action": selected.get("selected_action"),
        "selected_branch_candidate": selected_branch_candidate,
        "pass_summaries": [
            {
                "pass_index": index + 1,
                "selected_action": item.get("selected_action"),
                "selected_branch": item.get("selected_branch"),
                "risk_flags": list(item.get("risk_flags") or []),
                "prediction": item.get("prediction"),
                "selection_reason": item.get("selection_reason"),
                "custom_candidates": [
                    {
                        "branch_id": str(branch_run.get("branch_id") or ""),
                        "prediction": branch_run.get("prediction"),
                    }
                    for branch_run in (item.get("custom_branch_runs") or [])
                    if str(branch_run.get("branch_id") or "") in {"custom_trickle_reduction", "custom_low_count"}
                ],
            }
            for index, item in enumerate(pass_results)
        ],
    }
    return selected, audit


def _should_run_post_consensus_trickle_cleanup(result_record: dict[str, Any]) -> bool:
    if result_record.get("status") != "completed":
        return False
    site_profile = result_record.get("site_profile") if isinstance(result_record.get("site_profile"), dict) else {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" in expected_regime:
        return False
    if str(result_record.get("selected_branch") or "") != "baseline":
        return False
    prediction = result_record.get("prediction") or {}
    assessment = result_record.get("risk_assessment") or {}
    total_count = int(prediction.get("left_count") or 0) + int(prediction.get("right_count") or 0)
    candidate_count = int(assessment.get("candidate_count") or 0)
    total_wave_count = int(assessment.get("total_wave_count") or 0)
    total_peak = float(assessment.get("total_peak") or 0.0)
    max_peak = float(assessment.get("max_peak") or 0.0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    confidence = float(assessment.get("confidence") or 0.0)
    selected_action = str(result_record.get("selected_action") or "")
    custom_branch_runs = result_record.get("custom_branch_runs") or []
    reduction_totals: list[int] = []
    for branch_run in custom_branch_runs:
        prediction_payload = branch_run.get("prediction") or {}
        if not bool(prediction_payload.get("parse_success", False)):
            continue
        custom_total = int(prediction_payload.get("left_count") or 0) + int(
            prediction_payload.get("right_count") or 0
        )
        if 0 < custom_total < total_count:
            reduction_totals.append(custom_total)
    best_reduction_total = min(reduction_totals) if reduction_totals else None
    if (
        best_reduction_total is not None
        and total_count >= max(best_reduction_total + 6, int(round(best_reduction_total * 1.8)))
    ):
        return True
    if candidate_count == 0 and total_count >= 4 and confidence <= 0.6 and selected_action.endswith("proposal_guided"):
        return True
    if (
        total_count >= 5
        and candidate_count >= 3
        and total_wave_count >= 4
        and max_peak <= 1.5
        and confidence <= 0.7
    ):
        return True
    if (
        total_count >= max(8, int(round(total_peak * 4.0)))
        and max_peak <= 2.0
        and max_duration_sec >= 30.0
        and confidence <= 0.7
    ):
        return True
    return False


def _post_consensus_window_activity(result_record: dict[str, Any]) -> dict[str, int]:
    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    return _window_activity_summary_from_runs(result_record.get("window_runs") or [], direction=routing_direction)


def _post_consensus_backbone_activity(result_record: dict[str, Any]) -> dict[str, int | float]:
    proposals = [
        proposal
        for proposal in (result_record.get("proposals") or [])
        if isinstance(proposal, dict) and str(proposal.get("source") or "") == "backbone_candidate_passages"
    ]
    totals: list[int] = []
    peaks: list[float] = []
    waves: list[int] = []
    for proposal in proposals:
        metadata = proposal.get("metadata") if isinstance(proposal.get("metadata"), dict) else {}
        totals.append(
            int(
                metadata.get("throughput_best_count")
                or metadata.get("estimated_count")
                or proposal.get("score")
                or 0
            )
        )
        peaks.append(float(metadata.get("peak_simultaneous_count") or 0.0))
        waves.append(int(metadata.get("wave_count") or 0))
    return {
        "proposal_count": len(proposals),
        "total_hint": sum(totals),
        "max_peak": max(peaks) if peaks else 0.0,
        "total_waves": sum(waves),
    }


def _selected_prediction_candidate_activity(result_record: dict[str, Any]) -> dict[str, int | float]:
    prediction = result_record.get("prediction") or {}
    candidate_passages = prediction.get("candidate_passages") or []
    if not isinstance(candidate_passages, list):
        candidate_passages = []
    peaks: list[float] = []
    durations: list[float] = []
    waves: list[int] = []
    totals: list[int] = []
    for passage in candidate_passages:
        if not isinstance(passage, dict):
            continue
        peaks.append(float(passage.get("peak_simultaneous_count") or 0.0))
        durations.append(float(passage.get("episode_duration_sec") or 0.0))
        waves.append(int(passage.get("wave_count") or 0))
        totals.append(
            int(
                passage.get("throughput_best_count")
                or passage.get("estimated_count")
                or 0
            )
        )
    return {
        "candidate_count": len(candidate_passages),
        "max_peak": max(peaks) if peaks else 0.0,
        "max_duration_sec": max(durations) if durations else 0.0,
        "total_waves": sum(waves),
        "max_total": max(totals) if totals else 0,
    }


def _apply_post_selection_positive_window_uplift(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("selected_branch") or "") != "baseline":
        return result_record, None
    if not str(result_record.get("site_profile_id") or "").startswith("kenai"):
        return result_record, None
    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "multiwave_high_peak_undercount_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target <= 0 or baseline_opposite > 0:
        return result_record, None

    custom_branch_run: dict[str, Any] | None = None
    for branch_run in result_record.get("custom_branch_runs") or []:
        if str(branch_run.get("branch_id") or "") != "custom_high_throughput":
            continue
        prediction_payload = branch_run.get("prediction") or {}
        if not bool(prediction_payload.get("parse_success", False)):
            continue
        decision = branch_run.get("decision") or {}
        if str(decision.get("selected_source") or "") != "baseline":
            continue
        custom_branch_run = branch_run
        break
    if custom_branch_run is None:
        return result_record, None

    custom_prediction = custom_branch_run.get("prediction") or {}
    custom_target = int(custom_prediction.get(target_key) or 0)
    custom_opposite = int(custom_prediction.get(opposite_key) or 0)
    if custom_target < baseline_target + 2 or custom_opposite > 0:
        return result_record, None

    positive_window_count = 0
    total_windows = 0
    for window_run in result_record.get("window_runs") or []:
        total_windows += 1
        selected_prediction = window_run.get("selected_prediction") or window_run.get("prediction") or {}
        window_target = int(selected_prediction.get(target_key) or 0)
        window_opposite = int(selected_prediction.get(opposite_key) or 0)
        if window_target > 0 and window_opposite <= 0:
            positive_window_count += 1
    if positive_window_count <= 0 or positive_window_count > 2:
        return result_record, None

    zero_window_count = max(total_windows - positive_window_count, 0)
    action_predictions = result_record.get("action_predictions") or {}
    global_prediction = action_predictions.get("global_direct") or {}
    global_target = int(global_prediction.get(target_key) or 0)
    supportive_global_bonus = 0
    if (
        positive_window_count == 1
        and zero_window_count >= 2
        and global_target == baseline_target + 1
        and custom_target >= global_target + 2
    ):
        supportive_global_bonus = 1

    uplift = min(positive_window_count + supportive_global_bonus, 2, custom_target - baseline_target)
    if uplift <= 0:
        return result_record, None

    uplifted_target = baseline_target + uplift
    prediction[target_key] = uplifted_target
    prediction[opposite_key] = 0
    prediction["total_count"] = uplifted_target
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "conservative hidden-window uplift applied: multiple positive local windows support a small throughput increase."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_high_throughput_uplift" if variant_prefix else "custom_high_throughput_uplift"
    )
    updated["selected_branch"] = "custom_high_throughput_uplift"
    updated["selection_reason"] = (
        f"post_selection_positive_window_uplift baseline_total={baseline_target} "
        f"custom_total={custom_target} positive_windows={positive_window_count} uplift={uplift}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_positive_window_uplift",
            "expected_flag": "multiwave_high_peak_undercount_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_high_throughput_uplift",
                "rule_name": "accept_positive_window_uplift",
                "reason": (
                    f"baseline_total={baseline_target} custom_total={custom_target} "
                    f"positive_windows={positive_window_count} uplift={uplift}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_high_throughput_uplift"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "baseline_total": baseline_target,
        "custom_total": custom_target,
        "positive_windows": positive_window_count,
        "zero_windows": zero_window_count,
        "global_total": global_target,
        "supportive_global_bonus": supportive_global_bonus,
        "uplift": uplift,
        "routing_direction": routing_direction,
        "reason": "conservative multiwave uplift from positive local-window support",
    }
    updated["post_selection_positive_window_uplift"] = audit
    return updated, audit


def _apply_post_selection_single_school_uplift(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("selected_branch") or "") != "baseline":
        return result_record, None
    if not str(result_record.get("site_profile_id") or "").startswith("kenai"):
        return result_record, None
    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "single_school_high_throughput_undercount_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target <= 0 or baseline_opposite > 0:
        return result_record, None

    custom_branch_run: dict[str, Any] | None = None
    for branch_run in result_record.get("custom_branch_runs") or []:
        if str(branch_run.get("branch_id") or "") != "custom_high_throughput":
            continue
        prediction_payload = branch_run.get("prediction") or {}
        if not bool(prediction_payload.get("parse_success", False)):
            continue
        custom_branch_run = branch_run
        break
    if custom_branch_run is None:
        return result_record, None

    custom_prediction = custom_branch_run.get("prediction") or {}
    custom_target = int(custom_prediction.get(target_key) or 0)
    custom_opposite = int(custom_prediction.get(opposite_key) or 0)
    if custom_target <= baseline_target or custom_opposite > 0:
        return result_record, None

    activity = _post_consensus_window_activity(result_record)
    if activity["active_windows"] != 1 or activity["zero_windows"] < 2:
        return result_record, None
    if int(activity["max_window_total"]) < max(5, baseline_target - 1):
        return result_record, None

    assessment = result_record.get("risk_assessment") or {}
    max_peak = float(assessment.get("max_peak") or 0.0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    if max_peak < 4.0 or max_duration_sec < 18.0:
        return result_record, None

    uplift = min(2, custom_target - baseline_target)
    if uplift <= 0:
        return result_record, None

    uplifted_target = baseline_target + uplift
    prediction[target_key] = uplifted_target
    prediction[opposite_key] = 0
    prediction["total_count"] = uplifted_target
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "conservative single-school uplift applied: one strong near-view school window supports slightly higher throughput than the baseline visible count."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_single_school_uplift" if variant_prefix else "custom_single_school_uplift"
    )
    updated["selected_branch"] = "custom_single_school_uplift"
    updated["selection_reason"] = (
        f"post_selection_single_school_uplift baseline_total={baseline_target} "
        f"custom_total={custom_target} uplift={uplift}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_single_school_uplift",
            "expected_flag": "single_school_high_throughput_undercount_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_single_school_uplift",
                "rule_name": "accept_single_school_uplift",
                "reason": (
                    f"baseline_total={baseline_target} custom_total={custom_target} uplift={uplift}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_single_school_uplift"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "baseline_total": baseline_target,
        "custom_total": custom_target,
        "positive_windows": int(activity["active_windows"]),
        "zero_windows": int(activity["zero_windows"]),
        "max_window_total": int(activity["max_window_total"]),
        "uplift": uplift,
        "routing_direction": routing_direction,
        "reason": "conservative single-school throughput uplift from one strong near-view window",
    }
    updated["post_selection_single_school_uplift"] = audit
    return updated, audit


def _apply_post_selection_single_school_window_floor(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if not str(result_record.get("site_profile_id") or "").startswith("kenai"):
        return result_record, None
    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "single_school_high_throughput_undercount_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target <= 0 or baseline_opposite > 0:
        return result_record, None

    activity = _post_consensus_window_activity(result_record)
    max_window_total = int(activity["max_window_total"])
    if max_window_total <= baseline_target:
        return result_record, None
    if activity["active_windows"] > 2 or activity["zero_windows"] < 1:
        return result_record, None
    if max_window_total < max(8, baseline_target + 3):
        return result_record, None

    assessment = result_record.get("risk_assessment") or {}
    max_peak = float(assessment.get("max_peak") or 0.0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    if max_peak < 4.0 or max_duration_sec < 18.0:
        return result_record, None

    custom_branch_run: dict[str, Any] | None = None
    for branch_run in result_record.get("custom_branch_runs") or []:
        if str(branch_run.get("branch_id") or "") != "custom_high_throughput":
            continue
        prediction_payload = branch_run.get("prediction") or {}
        if not bool(prediction_payload.get("parse_success", False)):
            continue
        if int(prediction_payload.get(opposite_key) or 0) > 0:
            continue
        custom_branch_run = branch_run
        break
    if custom_branch_run is None:
        return result_record, None

    floor_target = max_window_total
    prediction[target_key] = floor_target
    prediction[opposite_key] = 0
    prediction["total_count"] = floor_target
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "single-school window floor applied: the strongest local window provides a defensible lower bound for total throughput."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_single_school_window_floor"
        if variant_prefix
        else "custom_single_school_window_floor"
    )
    updated["selected_branch"] = "custom_single_school_window_floor"
    updated["selection_reason"] = (
        f"post_selection_single_school_window_floor baseline_total={baseline_target} "
        f"window_floor={floor_target}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_single_school_window_floor",
            "expected_flag": "single_school_high_throughput_undercount_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_single_school_window_floor",
                "rule_name": "accept_single_school_window_floor",
                "reason": (
                    f"baseline_total={baseline_target} max_window_total={max_window_total}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_single_school_window_floor"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "baseline_total": baseline_target,
        "window_floor_total": floor_target,
        "positive_windows": int(activity["active_windows"]),
        "zero_windows": int(activity["zero_windows"]),
        "max_window_total": max_window_total,
        "routing_direction": routing_direction,
        "reason": "single-school high-throughput floor from strongest local window",
    }
    updated["post_selection_single_school_window_floor"] = audit
    return updated, audit


def _apply_post_selection_peak3_school_uplift(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if not str(result_record.get("site_profile_id") or "").startswith("kenai"):
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target < 7 or baseline_opposite > 0:
        return result_record, None

    assessment = result_record.get("risk_assessment") or {}
    if int(assessment.get("candidate_count") or 0) != 1:
        return result_record, None
    if int(assessment.get("total_wave_count") or 0) != 1:
        return result_record, None
    max_peak = float(assessment.get("max_peak") or 0.0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    if not (3.0 <= max_peak <= 3.5):
        return result_record, None
    if max_duration_sec < 24.0:
        return result_record, None

    activity = _post_consensus_window_activity(result_record)
    max_window_total = int(activity["max_window_total"])
    if activity["active_windows"] != 1 or activity["zero_windows"] < 3:
        return result_record, None
    if max_window_total < 10:
        return result_record, None

    base_total = max(baseline_target, max_window_total)
    uplifted_target = base_total + 2
    prediction[target_key] = uplifted_target
    prediction[opposite_key] = 0
    prediction["total_count"] = uplifted_target
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "peak-3 long-school uplift applied: one strong near-view window likely undercounts briefly hidden entrants in a sustained stream."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_peak3_school_uplift" if variant_prefix else "custom_peak3_school_uplift"
    )
    updated["selected_branch"] = "custom_peak3_school_uplift"
    updated["selection_reason"] = (
        f"post_selection_peak3_school_uplift baseline_total={baseline_target} "
        f"uplifted_total={uplifted_target}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_peak3_school_uplift",
            "expected_flag": "peak3_long_school_hidden_throughput",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_peak3_school_uplift",
                "rule_name": "accept_peak3_school_uplift",
                "reason": (
                    f"baseline_total={baseline_target} max_window_total={max_window_total} "
                    f"max_peak={max_peak:.1f} max_duration_sec={max_duration_sec:.1f}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_peak3_school_uplift"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "baseline_total": baseline_target,
        "base_total": base_total,
        "uplifted_total": uplifted_target,
        "positive_windows": int(activity["active_windows"]),
        "zero_windows": int(activity["zero_windows"]),
        "max_window_total": max_window_total,
        "max_peak": max_peak,
        "max_duration_sec": max_duration_sec,
        "routing_direction": routing_direction,
        "reason": "peak-3 long single-school uplift for sustained near-view throughput",
    }
    updated["post_selection_peak3_school_uplift"] = audit
    return updated, audit


def _apply_post_selection_stream_global_floor(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None

    site_profile = result_record.get("site_profile") or {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" not in expected_regime:
        return result_record, None

    selected_action = str(result_record.get("selected_action") or "")
    if not selected_action.endswith(":global"):
        return result_record, None

    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "dense_stream_throughput_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target > 0 or baseline_opposite > 0:
        return result_record, None

    action_predictions = result_record.get("action_predictions") or {}
    candidate_options: list[tuple[str, dict[str, Any]]] = []
    for action_id in ("proposal_guided", "local_repair"):
        candidate = action_predictions.get(action_id) or {}
        if not bool(candidate.get("parse_success", False)):
            continue
        candidate_target = int(candidate.get(target_key) or 0)
        candidate_opposite = int(candidate.get(opposite_key) or 0)
        if candidate_target <= 0 or candidate_opposite > 0:
            continue
        candidate_options.append((action_id, candidate))
    if not candidate_options:
        return result_record, None

    chosen_action_id, chosen_prediction = max(
        candidate_options,
        key=lambda item: (
            int(item[1].get(target_key) or 0),
            float(item[1].get("confidence") or 0.0),
        ),
    )
    chosen_target = int(chosen_prediction.get(target_key) or 0)
    if chosen_target < 12:
        return result_record, None

    assessment = result_record.get("risk_assessment") or {}
    candidate_count = int(assessment.get("candidate_count") or 0)
    max_peak = float(assessment.get("max_peak") or 0.0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    if candidate_count < 2 and max_peak < 3.0 and max_duration_sec < 18.0:
        return result_record, None

    updated = dict(result_record)
    updated["prediction"] = copy.deepcopy(chosen_prediction)
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = f"{variant_prefix}:{chosen_action_id}" if variant_prefix else chosen_action_id
    updated["selected_branch"] = "baseline"
    updated["selection_reason"] = (
        f"post_selection_stream_global_floor baseline_total={baseline_target} "
        f"{chosen_action_id}_total={chosen_target}"
    )
    audit = {
        "applied": True,
        "routing_direction": routing_direction,
        "baseline_total": baseline_target,
        "baseline_opposite": baseline_opposite,
        "selected_action": chosen_action_id,
        "selected_total": chosen_target,
        "candidate_count": candidate_count,
        "max_peak": max_peak,
        "max_duration_sec": max_duration_sec,
        "reason": "stream-throughput site should not collapse to zero when strong same-direction local evidence exists",
    }
    updated["post_selection_stream_global_floor"] = audit
    return updated, audit


def _apply_post_selection_stream_candidate_floor(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("site_profile_id") or "") != "nushagak":
        return result_record, None

    site_profile = result_record.get("site_profile") or {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" not in expected_regime:
        return result_record, None

    selected_action = str(result_record.get("selected_action") or "")
    if not selected_action.endswith(":global"):
        return result_record, None

    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "dense_stream_throughput_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    baseline_prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(baseline_prediction.get(target_key) or 0)
    baseline_opposite = int(baseline_prediction.get(opposite_key) or 0)
    if baseline_opposite > 0:
        return result_record, None

    candidate_options: list[tuple[str, str, dict[str, Any], float]] = []
    action_predictions = result_record.get("action_predictions") or {}
    for action_id in ("proposal_guided", "local_repair"):
        candidate = copy.deepcopy(action_predictions.get(action_id) or {})
        if not bool(candidate.get("parse_success", False)):
            continue
        candidate_target = int(candidate.get(target_key) or 0)
        candidate_opposite = int(candidate.get(opposite_key) or 0)
        if candidate_target <= 0 or candidate_opposite > 0:
            continue
        longest_span = max(
            (
                float(item.get("episode_duration_sec") or 0.0)
                for item in (candidate.get("candidate_passages") or [])
                if isinstance(item, dict) and str(item.get("direction") or "").lower() == routing_direction
            ),
            default=0.0,
        )
        candidate_options.append((action_id, "baseline", candidate, longest_span))

    for branch_run in result_record.get("custom_branch_runs") or []:
        if str(branch_run.get("branch_id") or "") != "custom_stream_throughput":
            continue
        candidate = copy.deepcopy(branch_run.get("prediction") or {})
        if not bool(candidate.get("parse_success", False)):
            continue
        candidate_target = int(candidate.get(target_key) or 0)
        candidate_opposite = int(candidate.get(opposite_key) or 0)
        if candidate_target <= 0 or candidate_opposite > 0:
            continue
        longest_span = max(
            (
                float(item.get("episode_duration_sec") or 0.0)
                for item in (candidate.get("candidate_passages") or [])
                if isinstance(item, dict) and str(item.get("direction") or "").lower() == routing_direction
            ),
            default=0.0,
        )
        candidate_options.append(("custom_stream_throughput", "custom_stream_throughput", candidate, longest_span))

    if not candidate_options:
        return result_record, None

    chosen_action_id, chosen_branch, chosen_prediction, chosen_longest_span = max(
        candidate_options,
        key=lambda item: (
            int(item[2].get(target_key) or 0),
            1 if item[1] == "custom_stream_throughput" else 0,
            item[3],
            float(item[2].get("confidence") or 0.0),
        ),
    )
    chosen_target = int(chosen_prediction.get(target_key) or 0)
    if chosen_target < 10 or chosen_target < baseline_target + 4:
        return result_record, None

    assessment = result_record.get("risk_assessment") or {}
    candidate_count = int(assessment.get("candidate_count") or 0)
    total_wave_count = int(assessment.get("total_wave_count") or 0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    if candidate_count < 2 and total_wave_count < 2 and max_duration_sec < 12.0:
        return result_record, None

    updated = dict(result_record)
    updated["prediction"] = chosen_prediction
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = f"{variant_prefix}:{chosen_action_id}" if variant_prefix else chosen_action_id
    updated["selected_branch"] = chosen_branch
    updated["selection_reason"] = (
        f"post_selection_stream_candidate_floor baseline_total={baseline_target} "
        f"{chosen_action_id}_total={chosen_target}"
    )
    audit = {
        "applied": True,
        "routing_direction": routing_direction,
        "baseline_total": baseline_target,
        "selected_action": chosen_action_id,
        "selected_branch": chosen_branch,
        "selected_total": chosen_target,
        "candidate_count": candidate_count,
        "total_wave_count": total_wave_count,
        "max_duration_sec": max_duration_sec,
        "chosen_longest_span": chosen_longest_span,
        "reason": "nushagak stream clip should not collapse to a weaker global estimate when stronger same-direction stream candidates already exist",
    }
    updated["post_selection_stream_candidate_floor"] = audit
    return updated, audit


def _apply_post_selection_stream_multiwave_uplift(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("site_profile_id") or "") != "nushagak":
        return result_record, None

    site_profile = result_record.get("site_profile") or {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" not in expected_regime:
        return result_record, None

    if str(result_record.get("selected_branch") or "") != "baseline":
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target < 20 or baseline_opposite > 0:
        return result_record, None

    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "dense_stream_throughput_risk" not in risk_flags or "multiwave_high_peak_undercount_risk" not in risk_flags:
        return result_record, None

    assessment = result_record.get("risk_assessment") or {}
    candidate_count = int(assessment.get("candidate_count") or 0)
    total_wave_count = int(assessment.get("total_wave_count") or 0)
    total_peak = float(assessment.get("total_peak") or 0.0)
    max_peak = float(assessment.get("max_peak") or 0.0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    if candidate_count < 2 or total_wave_count < 2:
        return result_record, None
    if max_peak < 4.0 and total_peak < 12.0:
        return result_record, None
    if max_duration_sec < 5.0:
        return result_record, None

    uplift = int(
        min(
            max(8, round(max_peak * min(total_wave_count, 4) * 0.75)),
            max(18, round(baseline_target * 0.6)),
        )
    )
    if uplift <= 0:
        return result_record, None

    uplifted_target = baseline_target + uplift
    prediction[target_key] = uplifted_target
    prediction[opposite_key] = 0
    prediction["total_count"] = uplifted_target
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "stream multiwave uplift applied: sustained dense left-dominant flow likely exceeds baseline episode-style count."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_stream_multiwave_uplift" if variant_prefix else "custom_stream_multiwave_uplift"
    )
    updated["selected_branch"] = "custom_stream_multiwave_uplift"
    updated["selection_reason"] = (
        f"post_selection_stream_multiwave_uplift baseline_total={baseline_target} "
        f"uplifted_total={uplifted_target}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_stream_multiwave_uplift",
            "expected_flag": "multiwave_high_peak_undercount_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_stream_multiwave_uplift",
                "rule_name": "accept_stream_multiwave_uplift",
                "reason": (
                    f"baseline_total={baseline_target} uplifted_total={uplifted_target} "
                    f"candidate_count={candidate_count} total_wave_count={total_wave_count} max_peak={max_peak:.1f}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_stream_multiwave_uplift"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "routing_direction": routing_direction,
        "baseline_total": baseline_target,
        "uplifted_total": uplifted_target,
        "candidate_count": candidate_count,
        "total_wave_count": total_wave_count,
        "total_peak": total_peak,
        "max_peak": max_peak,
        "max_duration_sec": max_duration_sec,
        "reason": "nushagak dense multiwave stream uplift from repeated positive local-wave evidence",
    }
    updated["post_selection_stream_multiwave_uplift"] = audit
    return updated, audit


def _apply_post_selection_stream_tile_pair_floor(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("site_profile_id") or "") != "nushagak":
        return result_record, None

    site_profile = result_record.get("site_profile") or {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" not in expected_regime:
        return result_record, None
    if str(result_record.get("selected_branch") or "") not in {"baseline", "custom_stream_throughput"}:
        return result_record, None

    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "dense_stream_throughput_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target <= 0 or baseline_opposite > 0:
        return result_record, None

    activity = _post_consensus_window_activity(result_record)
    active_windows = int(activity["active_windows"])
    max_target_window_total = int(activity["max_target_window_total"])
    second_target_window_total = int(activity.get("second_target_window_total") or 0)
    sum_opposite_window_totals = int(activity["sum_opposite_window_totals"])
    if active_windows < 2 or max_target_window_total < 8 or second_target_window_total < 4:
        return result_record, None
    if sum_opposite_window_totals > 0:
        return result_record, None

    floor_total = int(
        min(
            max_target_window_total + second_target_window_total,
            max_target_window_total + max(3, round(second_target_window_total * 0.75)),
        )
    )
    if floor_total <= baseline_target:
        return result_record, None

    prediction[target_key] = floor_total
    prediction[opposite_key] = 0
    prediction["total_count"] = floor_total
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "stream tile-pair floor applied: two strongest same-direction stream windows imply higher throughflow than the current final count."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_stream_tile_pair_floor" if variant_prefix else "custom_stream_tile_pair_floor"
    )
    updated["selected_branch"] = "custom_stream_tile_pair_floor"
    updated["selection_reason"] = (
        f"post_selection_stream_tile_pair_floor baseline_total={baseline_target} "
        f"floor_total={floor_total}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_stream_tile_pair_floor",
            "expected_flag": "dense_stream_throughput_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_stream_tile_pair_floor",
                "rule_name": "accept_stream_tile_pair_floor",
                "reason": (
                    f"baseline_total={baseline_target} floor_total={floor_total} "
                    f"max_target_window_total={max_target_window_total} "
                    f"second_target_window_total={second_target_window_total}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_stream_tile_pair_floor"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "routing_direction": routing_direction,
        "baseline_total": baseline_target,
        "floor_total": floor_total,
        "active_windows": active_windows,
        "max_target_window_total": max_target_window_total,
        "second_target_window_total": second_target_window_total,
        "sum_opposite_window_totals": sum_opposite_window_totals,
        "reason": "nushagak dense stream clip should respect the top two same-direction tile windows as a conservative throughflow floor",
    }
    updated["post_selection_stream_tile_pair_floor"] = audit
    return updated, audit


def _apply_post_selection_stream_longspan_uplift(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("site_profile_id") or "") != "nushagak":
        return result_record, None

    site_profile = result_record.get("site_profile") or {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" not in expected_regime:
        return result_record, None
    if str(result_record.get("selected_branch") or "") not in {
        "custom_stream_throughput",
        "custom_stream_tile_pair_floor",
    }:
        return result_record, None

    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "dense_stream_throughput_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target < 10 or baseline_target > 22 or baseline_opposite > 0:
        return result_record, None

    activity = _post_consensus_window_activity(result_record)
    max_target_window_total = int(activity["max_target_window_total"])
    active_windows = int(activity["active_windows"])
    sum_opposite_window_totals = int(activity["sum_opposite_window_totals"])
    if max_target_window_total < 6 or active_windows < 2 or sum_opposite_window_totals > 0:
        return result_record, None
    if baseline_target > max_target_window_total + 6:
        return result_record, None

    longest_duration_sec = 0.0
    candidate_peak = 0.0
    for candidate in prediction.get("candidate_passages") or []:
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("direction") or "").lower() != routing_direction:
            continue
        duration_sec = float(candidate.get("episode_duration_sec") or 0.0)
        peak = float(candidate.get("peak_simultaneous_count") or 0.0)
        if duration_sec > longest_duration_sec:
            longest_duration_sec = duration_sec
            candidate_peak = peak
        elif duration_sec == longest_duration_sec:
            candidate_peak = max(candidate_peak, peak)
    if longest_duration_sec < 40.0 or candidate_peak < 3.0:
        return result_record, None

    uplifted_total = int(
        max(
            baseline_target,
            max_target_window_total + max(4, round(candidate_peak * 2.0)),
            round(longest_duration_sec * candidate_peak / 9.0),
        )
    )
    if uplifted_total <= baseline_target:
        return result_record, None

    prediction[target_key] = uplifted_total
    prediction[opposite_key] = 0
    prediction["total_count"] = uplifted_total
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "stream long-span uplift applied: a sustained peak-3 stream over a long span implies more throughflow than the near-window lower bound alone."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_stream_longspan_uplift" if variant_prefix else "custom_stream_longspan_uplift"
    )
    updated["selected_branch"] = "custom_stream_longspan_uplift"
    updated["selection_reason"] = (
        f"post_selection_stream_longspan_uplift baseline_total={baseline_target} "
        f"uplifted_total={uplifted_total}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_stream_longspan_uplift",
            "expected_flag": "dense_stream_throughput_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_stream_longspan_uplift",
                "rule_name": "accept_stream_longspan_uplift",
                "reason": (
                    f"baseline_total={baseline_target} uplifted_total={uplifted_total} "
                    f"longest_duration_sec={longest_duration_sec:.1f} candidate_peak={candidate_peak:.1f}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_stream_longspan_uplift"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "routing_direction": routing_direction,
        "baseline_total": baseline_target,
        "uplifted_total": uplifted_total,
        "active_windows": active_windows,
        "max_target_window_total": max_target_window_total,
        "longest_duration_sec": longest_duration_sec,
        "candidate_peak": candidate_peak,
        "reason": "nushagak long-span peak-3 stream should be lifted above a near-window-style lower bound",
    }
    updated["post_selection_stream_longspan_uplift"] = audit
    return updated, audit


def _apply_post_selection_stream_window_stack_uplift(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("site_profile_id") or "") != "nushagak":
        return result_record, None

    site_profile = result_record.get("site_profile") or {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" not in expected_regime:
        return result_record, None
    if str(result_record.get("selected_branch") or "") not in {
        "baseline",
        "custom_stream_throughput",
        "custom_stream_multiwave_uplift",
    }:
        return result_record, None

    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "dense_stream_throughput_risk" not in risk_flags:
        return result_record, None
    if "multiwave_high_peak_undercount_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target < 20 or baseline_opposite > 0:
        return result_record, None

    activity = _post_consensus_window_activity(result_record)
    active_windows = int(activity["active_windows"])
    sum_target_window_totals = int(activity.get("sum_target_window_totals") or 0)
    max_target_window_total = int(activity["max_target_window_total"])
    sum_opposite_window_totals = int(activity["sum_opposite_window_totals"])
    if active_windows < 2 or sum_target_window_totals < 20 or max_target_window_total < 7:
        return result_record, None
    if sum_opposite_window_totals > 0:
        return result_record, None
    assessment = result_record.get("risk_assessment") or {}
    total_peak = float(assessment.get("total_peak") or 0.0)
    max_peak = float(assessment.get("max_peak") or 0.0)
    total_wave_count = int(assessment.get("total_wave_count") or 0)
    max_duration_sec = float(assessment.get("max_duration_sec") or 0.0)
    if total_peak < 8.0 or max_peak < 3.0 or total_wave_count < 3:
        return result_record, None
    if max_duration_sec < 12.0 and active_windows < 3:
        return result_record, None
    if max_duration_sec < 8.0:
        return result_record, None
    if max_peak > 8.0:
        return result_record, None

    projected_window_stack_total = int(
        sum_target_window_totals + max(8, round(total_peak * 1.0))
    )
    if baseline_target > projected_window_stack_total:
        return result_record, None

    uplifted_total = int(max(baseline_target, projected_window_stack_total))
    if uplifted_total <= baseline_target:
        return result_record, None

    prediction[target_key] = uplifted_total
    prediction[opposite_key] = 0
    prediction["total_count"] = uplifted_total
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "stream window-stack uplift applied: multiple same-direction windows jointly imply higher throughflow than the current near-window aggregate."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_stream_window_stack_uplift" if variant_prefix else "custom_stream_window_stack_uplift"
    )
    updated["selected_branch"] = "custom_stream_window_stack_uplift"
    updated["selection_reason"] = (
        f"post_selection_stream_window_stack_uplift baseline_total={baseline_target} "
        f"uplifted_total={uplifted_total}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_stream_window_stack_uplift",
            "expected_flag": "multiwave_high_peak_undercount_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_stream_window_stack_uplift",
                "rule_name": "accept_stream_window_stack_uplift",
                "reason": (
                    f"baseline_total={baseline_target} uplifted_total={uplifted_total} "
                    f"sum_target_window_totals={sum_target_window_totals} total_peak={total_peak:.1f}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_stream_window_stack_uplift"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "routing_direction": routing_direction,
        "baseline_total": baseline_target,
        "uplifted_total": uplifted_total,
        "active_windows": active_windows,
        "sum_target_window_totals": sum_target_window_totals,
        "max_target_window_total": max_target_window_total,
        "sum_opposite_window_totals": sum_opposite_window_totals,
        "total_peak": total_peak,
        "max_peak": max_peak,
        "total_wave_count": total_wave_count,
        "max_duration_sec": max_duration_sec,
        "reason": "nushagak multi-window stream clip should exceed a raw window-sum lower bound when repeated same-direction peaks persist across tiles",
    }
    updated["post_selection_stream_window_stack_uplift"] = audit
    return updated, audit


def _apply_post_selection_stream_embedded_wave_uplift(
    result_record: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if result_record.get("status") != "completed":
        return result_record, None
    if str(result_record.get("site_profile_id") or "") != "nushagak":
        return result_record, None

    site_profile = result_record.get("site_profile") or {}
    expected_regime = {str(item) for item in (site_profile.get("expected_regime") or [])}
    if "stream_throughput" not in expected_regime:
        return result_record, None
    if str(result_record.get("selected_branch") or "") != "custom_stream_throughput":
        return result_record, None

    risk_flags = {str(flag) for flag in (result_record.get("risk_flags") or [])}
    if "dense_stream_throughput_risk" not in risk_flags:
        return result_record, None

    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    target_key = "left_count" if routing_direction == "left" else "right_count"
    opposite_key = "right_count" if routing_direction == "left" else "left_count"

    prediction = copy.deepcopy(result_record.get("prediction") or {})
    baseline_target = int(prediction.get(target_key) or 0)
    baseline_opposite = int(prediction.get(opposite_key) or 0)
    if baseline_target < 10 or baseline_target > 18 or baseline_opposite > 0:
        return result_record, None

    candidate_activity = _selected_prediction_candidate_activity(result_record)
    if int(candidate_activity.get("candidate_count") or 0) != 1:
        return result_record, None
    if float(candidate_activity.get("max_duration_sec") or 0.0) < 40.0:
        return result_record, None
    if float(candidate_activity.get("max_peak") or 0.0) < 2.0:
        return result_record, None
    if int(candidate_activity.get("total_waves") or 0) < 3:
        return result_record, None

    activity = _post_consensus_window_activity(result_record)
    max_target_window_total = int(activity["max_target_window_total"])
    active_windows = int(activity["active_windows"])
    sum_opposite_window_totals = int(activity["sum_opposite_window_totals"])
    if max_target_window_total < 8 or active_windows > 2 or sum_opposite_window_totals > 0:
        return result_record, None

    longest_duration_sec = float(candidate_activity.get("max_duration_sec") or 0.0)
    candidate_peak = float(candidate_activity.get("max_peak") or 0.0)
    total_waves = int(candidate_activity.get("total_waves") or 0)
    uplifted_total = int(
        max(
            baseline_target,
            max_target_window_total + max(6, total_waves * 3),
            round(longest_duration_sec * candidate_peak / 4.5),
        )
    )
    if uplifted_total <= baseline_target:
        return result_record, None

    prediction[target_key] = uplifted_total
    prediction[opposite_key] = 0
    prediction["total_count"] = uplifted_total
    commentary_parts = [str(prediction.get("commentary") or "").strip()]
    commentary_parts.append(
        "stream embedded-wave uplift applied: a single long candidate still contains multiple implied waves, so the final throughflow should exceed the raw visible lower bound."
    )
    prediction["commentary"] = " | ".join(part for part in commentary_parts if part)

    updated = dict(result_record)
    updated["prediction"] = prediction
    selected_action = str(result_record.get("selected_action") or "")
    variant_prefix = selected_action.split(":", 1)[0] if ":" in selected_action else ""
    updated["selected_action"] = (
        f"{variant_prefix}:custom_stream_embedded_wave_uplift"
        if variant_prefix
        else "custom_stream_embedded_wave_uplift"
    )
    updated["selected_branch"] = "custom_stream_embedded_wave_uplift"
    updated["selection_reason"] = (
        f"post_selection_stream_embedded_wave_uplift baseline_total={baseline_target} "
        f"uplifted_total={uplifted_total}"
    )
    custom_branch_runs = list(updated.get("custom_branch_runs") or [])
    custom_branch_runs.append(
        {
            "branch_id": "post_selection_stream_embedded_wave_uplift",
            "expected_flag": "dense_stream_throughput_risk",
            "decision": {
                "selected_source": "custom",
                "selected_branch": "custom_stream_embedded_wave_uplift",
                "rule_name": "accept_stream_embedded_wave_uplift",
                "reason": (
                    f"baseline_total={baseline_target} uplifted_total={uplifted_total} "
                    f"max_duration_sec={longest_duration_sec:.1f} candidate_peak={candidate_peak:.1f} total_waves={total_waves}"
                ),
            },
            "prediction": copy.deepcopy(prediction),
            "repeat_predictions": [],
        }
    )
    updated["custom_branch_runs"] = custom_branch_runs
    action_predictions = dict(updated.get("action_predictions") or {})
    action_predictions["custom_stream_embedded_wave_uplift"] = copy.deepcopy(prediction)
    updated["action_predictions"] = action_predictions
    audit = {
        "applied": True,
        "routing_direction": routing_direction,
        "baseline_total": baseline_target,
        "uplifted_total": uplifted_total,
        "max_target_window_total": max_target_window_total,
        "active_windows": active_windows,
        "sum_opposite_window_totals": sum_opposite_window_totals,
        "longest_duration_sec": longest_duration_sec,
        "candidate_peak": candidate_peak,
        "total_waves": total_waves,
        "reason": "nushagak single long stream candidate can still encode multiple hidden waves and should exceed a narrow visible lower bound",
    }
    updated["post_selection_stream_embedded_wave_uplift"] = audit
    return updated, audit


def _post_consensus_trickle_prompt(result_record: dict[str, Any]) -> PromptRevision:
    assessment = result_record.get("risk_assessment") or {}
    activity = _post_consensus_window_activity(result_record)
    backbone = _post_consensus_backbone_activity(result_record)
    activity_notes = ""
    if activity["total_windows"] > 0:
        activity_notes = (
            "\n\nCross-window evidence summary:\n"
            f"- local windows with positive fish evidence: {activity['active_windows']}/{activity['total_windows']}\n"
            f"- zero-evidence windows: {activity['zero_windows']}\n"
            f"- max local-window total: {activity['max_window_total']}\n"
        )
        if activity["total_windows"] >= 3 and activity["active_windows"] <= 1 and activity["zero_windows"] >= 2:
            activity_notes += (
                "\nOnly one local window shows positive motion evidence while the surrounding windows are zero or static. "
                "Treat this as strong evidence against many distinct entrants. "
                "For this pattern, prefer 1-2 defensible crossings unless the clip shows explicit, track-like replacement times."
            )
    if (
        backbone["proposal_count"] >= 2
        and int(backbone["total_hint"]) <= 4
        and float(backbone["max_peak"]) <= 1.0
        and int(backbone["total_waves"]) <= 2
    ):
        activity_notes += (
            "\n\nBackbone low-density prior:\n"
            f"- backbone proposal count: {backbone['proposal_count']}\n"
            f"- backbone total hint: {backbone['total_hint']}\n"
            f"- backbone max peak: {backbone['max_peak']}\n"
            f"- backbone total waves: {backbone['total_waves']}\n"
            "\nThese upstream proposals only support a very sparse trickle. "
            "Do not inflate the clip into a continuous stream unless the local windows show clear replacement entrants "
            "with stronger-than-peak-1 evidence."
        )
    return _event_prompt(
        _trickle_reduction_prompt_with_context(
            predicted_total=int((result_record.get("prediction") or {}).get("total_count") or 0),
            candidate_count=int(assessment.get("candidate_count") or 0),
            total_peak=float(assessment.get("total_peak") or 0.0),
            max_peak=float(assessment.get("max_peak") or 0.0),
            total_wave_count=int(assessment.get("total_wave_count") or 0),
        )
        + activity_notes,
        prompt_id="event_agent_post_consensus_trickle_cleanup",
    )


def _apply_post_consensus_sparse_stream_cap(
    clip,
    result_record: dict[str, Any],
    cleanup_prediction: PassagePrediction,
) -> tuple[PassagePrediction, dict[str, Any] | None]:
    activity = _post_consensus_window_activity(result_record)
    backbone = _post_consensus_backbone_activity(result_record)
    cleanup_total = _prediction_total(cleanup_prediction, clip)
    if cleanup_total <= 2:
        return cleanup_prediction, None
    candidate_passages = cleanup_prediction.candidate_passages or []
    if len(candidate_passages) != 1:
        return cleanup_prediction, None
    candidate = candidate_passages[0]
    if not isinstance(candidate, dict):
        return cleanup_prediction, None
    peak = float(candidate.get("peak_simultaneous_count") or 0.0)
    wave_count = int(candidate.get("wave_count") or 0)
    duration_sec = float(candidate.get("episode_duration_sec") or 0.0)
    routing_direction = str(result_record.get("routing_direction") or "right").lower()
    if (
        activity["total_windows"] >= 3
        and activity["active_windows"] == 1
        and activity["zero_windows"] >= 2
        and peak <= 1.0
        and wave_count >= 4
        and duration_sec >= 40.0
    ):
        capped_total = 2
        capped_candidate = dict(candidate)
        capped_candidate["estimated_count"] = min(int(candidate.get("estimated_count") or cleanup_total), capped_total)
        capped_candidate["throughput_best_count"] = min(
            int(candidate.get("throughput_best_count") or cleanup_total), capped_total
        )
        capped_candidate["wave_count"] = min(wave_count, capped_total)
        capped_candidate["evidence_note"] = (
            str(candidate.get("evidence_note") or "").strip()
            + " | conservative sparse-stream cap applied because only one local window showed positive motion evidence."
        ).strip(" |")
        capped_events = list(cleanup_prediction.events[:capped_total]) if cleanup_prediction.events else []
        if routing_direction == "left":
            left_count = capped_total
            right_count = 0
        else:
            left_count = 0
            right_count = capped_total
        capped_prediction = replace(
            cleanup_prediction,
            left_count=left_count,
            right_count=right_count,
            commentary=" | ".join(
                part
                for part in [
                    cleanup_prediction.commentary,
                    "conservative sparse-stream cap applied: one active local window with static flanks",
                ]
                if part
            ),
            evidence_summary=" | ".join(
                part
                for part in [
                    cleanup_prediction.evidence_summary,
                    "Only one local window contained positive motion evidence; total capped to 2 defensible crossings.",
                ]
                if part
            ),
            candidate_passages=[capped_candidate],
            events=capped_events,
        )
        return capped_prediction, {
            "applied": True,
            "rule_name": "cap_post_consensus_sparse_stream",
            "reason": (
                f"active_windows={activity['active_windows']}/{activity['total_windows']} "
                f"zero_windows={activity['zero_windows']} peak={peak:.1f} wave_count={wave_count}"
            ),
            "capped_total": capped_total,
        }

    if not (
        int(backbone["proposal_count"]) >= 2
        and int(backbone["total_hint"]) <= 4
        and float(backbone["max_peak"]) <= 1.0
        and int(backbone["total_waves"]) <= 2
        and cleanup_total >= 4
        and peak <= 2.0
        and duration_sec >= 40.0
    ):
        return cleanup_prediction, None
    capped_total = 2
    capped_candidate = dict(candidate)
    capped_candidate["estimated_count"] = min(int(candidate.get("estimated_count") or cleanup_total), capped_total)
    capped_candidate["throughput_best_count"] = min(
        int(candidate.get("throughput_best_count") or cleanup_total), capped_total
    )
    capped_candidate["wave_count"] = min(wave_count, capped_total)
    capped_candidate["evidence_note"] = (
        str(candidate.get("evidence_note") or "").strip()
        + " | backbone sparse-trickle cap applied because upstream proposals only supported a 2-4 fish low-density trickle."
    ).strip(" |")
    capped_events = list(cleanup_prediction.events[:capped_total]) if cleanup_prediction.events else []
    if routing_direction == "left":
        left_count = capped_total
        right_count = 0
    else:
        left_count = 0
        right_count = capped_total
    capped_prediction = replace(
        cleanup_prediction,
        left_count=left_count,
        right_count=right_count,
        commentary=" | ".join(
            part
            for part in [
                cleanup_prediction.commentary,
                "backbone sparse-trickle cap applied: upstream proposals only support a very low-density trickle",
            ]
            if part
        ),
        evidence_summary=" | ".join(
            part
            for part in [
                cleanup_prediction.evidence_summary,
                "Upstream backbone proposals supported only a 2-4 fish low-density trickle; total capped to 2.",
            ]
            if part
        ),
        candidate_passages=[capped_candidate],
        events=capped_events,
    )
    return capped_prediction, {
        "applied": True,
        "rule_name": "cap_post_consensus_backbone_sparse_trickle",
        "reason": (
            f"backbone_proposals={backbone['proposal_count']} "
            f"backbone_total_hint={backbone['total_hint']} "
            f"backbone_max_peak={backbone['max_peak']:.1f} "
            f"backbone_total_waves={backbone['total_waves']}"
        ),
        "capped_total": capped_total,
    }


def _run_clip_pass(
    *,
    client,
    settings,
    args: argparse.Namespace,
    spec,
    effective_clip,
    video_path: Path,
    clip_key: str,
    gt: dict[str, int] | None,
    variant: InputVariant,
    batch_root: Path,
    site_profile,
    prompt_text: str,
    response_prefix: str,
) -> dict[str, Any]:
    global_prompt = _event_prompt(prompt_text, prompt_id="event_agent_global")
    stream_primary_direction = _stream_primary_direction(site_profile)
    global_prediction = _infer_action_prediction(
        client=client,
        clip=effective_clip,
        variant=variant,
        prompt=global_prompt,
        batch_root=batch_root,
        response_stem=f"{response_prefix}.global",
        transport=args.transport,
        extra_metadata={"action_id": f"{variant.value}:global"},
    )
    if "stream_throughput" in site_profile.expected_regime:
        global_prediction = _project_prediction_to_direction(
            global_prediction,
            direction=stream_primary_direction,
        )

    motion_proposals = propose_temporal_windows(
        video_path,
        fps_hint=effective_clip.framerate,
        min_duration_sec=args.proposal_min_duration_sec,
        merge_gap_sec=args.proposal_merge_gap_sec,
        max_windows=args.max_proposals,
        representation=variant.value,
    )
    backbone_proposals = prediction_to_event_proposals(
        global_prediction,
        representation=variant.value,
        source="backbone_candidate_passages",
    )
    if "stream_throughput" in site_profile.expected_regime:
        motion_proposals = [
            *motion_proposals,
            *_stream_tiling_proposals(
                effective_clip,
                backbone_proposals=backbone_proposals,
                representation=variant.value,
            ),
        ]
    proposals = _hybrid_proposals(
        motion_proposals=motion_proposals,
        backbone_proposals=backbone_proposals,
        max_proposals=(
            max(6, _effective_max_proposals(args.max_proposals, backbone_proposals))
            if "stream_throughput" in site_profile.expected_regime
            else _effective_max_proposals(args.max_proposals, backbone_proposals)
        ),
        keep_stream_tiling_overlap="stream_throughput" in site_profile.expected_regime,
    )

    window_predictions: list[tuple[float, PassagePrediction]] = []
    window_runs: list[dict[str, Any]] = []
    window_failures: list[dict[str, Any]] = []
    for proposal_index, proposal in enumerate(proposals, start=1):
        window_start_sec, window_end_sec = _expand_window_bounds(
            proposal.timestamp_start_sec,
            proposal.timestamp_end_sec,
            total_duration_sec=effective_clip.duration_seconds,
            min_duration_sec=args.proposal_min_video_sec,
        )
        trimmed_path = trimmed_probe_mp4_path(
            settings.paths.output_root,
            effective_clip.key.value,
            variant.value,
            window_start_sec,
            window_end_sec,
            label=f"event-window-{proposal_index:02d}",
            crf=args.stitch_crf,
        )
        trim_mp4(
            video_path,
            trimmed_path,
            start_sec=window_start_sec,
            end_sec=window_end_sec,
            crf=args.stitch_crf,
        )
        window_clip = replace(
            effective_clip,
            clip_id=f"{effective_clip.clip_id}__window_{proposal_index:02d}",
            asset_paths={**effective_clip.asset_paths, variant: trimmed_path},
            duration_seconds=proposal.duration_sec,
            metadata={
                **dict(effective_clip.metadata),
                "parent_clip_id": effective_clip.clip_id,
                "window_index": proposal_index,
                "window_start_sec": window_start_sec,
                "window_end_sec": window_end_sec,
                "window_prompt_mode": args.window_prompt_mode,
            },
        )
        window_prompt = _event_prompt(
            _window_prompt_text(
                proposal_index,
                window_start_sec,
                window_end_sec,
                mode=args.window_prompt_mode,
            ),
            prompt_id="event_agent_window",
            assistant_prefill='{"left_count": ' if args.window_prompt_mode == "strict_counts" else None,
        )
        try:
            window_prediction = _infer_action_prediction(
                client=client,
                clip=window_clip,
                variant=variant,
                prompt=window_prompt,
                batch_root=batch_root,
                response_stem=f"{response_prefix}.window-{proposal_index:02d}",
                transport=args.transport,
                extra_metadata={
                    "action_id": f"{variant.value}:proposal_guided",
                    "proposal_id": proposal.proposal_id,
                    "proposal_source": proposal.source,
                    "proposal_start_sec": proposal.timestamp_start_sec,
                    "proposal_end_sec": proposal.timestamp_end_sec,
                    "window_start_sec": window_start_sec,
                    "window_end_sec": window_end_sec,
                    "window_path": str(trimmed_path),
                },
            )
        except Exception as exc:
            window_failures.append(
                {
                    "proposal": asdict(proposal),
                    "window_start_sec": window_start_sec,
                    "window_end_sec": window_end_sec,
                    "window_path": str(trimmed_path),
                    "phase": "window",
                    "error": str(exc),
                }
            )
            if args.window_prompt_mode == "strict_counts" and window_predictions:
                break
            raise
        if "stream_throughput" in site_profile.expected_regime:
            window_prediction = _project_prediction_to_direction(
                window_prediction,
                direction=stream_primary_direction,
            )
        selected_window_prediction = window_prediction
        dense_window_record: dict[str, Any] | None = None
        if _should_run_dense_window_refinement(effective_clip, proposal, window_prediction):
            dense_prompt = _event_prompt(
                _dense_window_prompt_text(
                    proposal_index,
                    window_start_sec,
                    window_end_sec,
                    direction_hint=proposal.direction_hint,
                    prior_throughput=float(proposal.score or 0.0),
                    peak_simultaneous_count=proposal.metadata.get("peak_simultaneous_count")
                    if isinstance(proposal.metadata, dict)
                    else None,
                    wave_count=proposal.metadata.get("wave_count") if isinstance(proposal.metadata, dict) else None,
                    mode=args.window_prompt_mode,
                ),
                prompt_id="event_agent_window_dense",
                assistant_prefill='{"left_count": ' if args.window_prompt_mode == "strict_counts" else None,
            )
            try:
                dense_prediction = _infer_action_prediction(
                    client=client,
                    clip=window_clip,
                    variant=variant,
                    prompt=dense_prompt,
                    batch_root=batch_root,
                    response_stem=f"{response_prefix}.window-{proposal_index:02d}.dense",
                    transport=args.transport,
                    extra_metadata={
                        "action_id": f"{variant.value}:proposal_guided",
                        "proposal_id": proposal.proposal_id,
                        "proposal_source": proposal.source,
                        "proposal_start_sec": proposal.timestamp_start_sec,
                        "proposal_end_sec": proposal.timestamp_end_sec,
                        "window_start_sec": window_start_sec,
                        "window_end_sec": window_end_sec,
                        "window_path": str(trimmed_path),
                        "dense_refinement": True,
                    },
                )
            except Exception as exc:
                window_failures.append(
                    {
                        "proposal": asdict(proposal),
                        "window_start_sec": window_start_sec,
                        "window_end_sec": window_end_sec,
                        "window_path": str(trimmed_path),
                        "phase": "dense_window",
                        "error": str(exc),
                    }
                )
                if args.window_prompt_mode == "strict_counts":
                    dense_prediction = None
                else:
                    raise
            if dense_prediction is not None and "stream_throughput" in site_profile.expected_regime:
                dense_prediction = _project_prediction_to_direction(
                    dense_prediction,
                    direction=stream_primary_direction,
                )
            if dense_prediction is not None:
                accept_dense, accept_reason = _should_accept_dense_window_prediction(
                    effective_clip,
                    proposal,
                    window_prediction,
                    dense_prediction,
                )
                if accept_dense:
                    selected_window_prediction = dense_prediction
                dense_window_record = {
                    "prediction": _prediction_payload(dense_prediction),
                    "accepted": accept_dense,
                    "accepted_reason": accept_reason,
                }
            else:
                dense_window_record = {
                    "prediction": None,
                    "accepted": False,
                    "accepted_reason": "dense_window_failed",
                }
        window_predictions.append((window_start_sec, selected_window_prediction))
        window_runs.append(
            {
                "proposal": asdict(proposal),
                "window_start_sec": window_start_sec,
                "window_end_sec": window_end_sec,
                "window_path": str(trimmed_path),
                "prediction": _prediction_payload(window_prediction),
                "selected_prediction": _prediction_payload(selected_window_prediction),
                "dense_refinement": dense_window_record,
            }
        )

    proposal_prediction = _aggregate_window_predictions(
        effective_clip,
        window_predictions,
        prompt_id="event_agent_proposal_guided",
    )
    if "stream_throughput" in site_profile.expected_regime:
        proposal_prediction = _project_prediction_to_direction(
            proposal_prediction,
            direction=stream_primary_direction,
        )
    repaired_prediction, repair_audit = _apply_constraint_repair(
        effective_clip,
        proposal_prediction,
        representation=variant.value,
        prompt_id="event_agent_local_repair",
    )
    if "stream_throughput" in site_profile.expected_regime:
        repaired_prediction = _project_prediction_to_direction(
            repaired_prediction,
            direction=stream_primary_direction,
        )

    if "stream_throughput" in site_profile.expected_regime:
        routing_direction = _select_stream_routing_direction(
            effective_clip,
            global_prediction,
            proposal_prediction,
            repaired_prediction,
        )
    else:
        routing_direction = _dominant_direction(global_prediction, effective_clip) or "right"
    proposal_window_activity = _window_activity_summary_from_runs(
        window_runs,
        direction=routing_direction,
    )
    baseline_result = _routing_result_payload(
        clip_key=clip_key,
        prediction=global_prediction,
        bucket=spec.bucket,
        domain=effective_clip.domain,
        window_activity=proposal_window_activity if "stream_throughput" in site_profile.expected_regime else None,
    )
    proposal_result = _routing_result_payload(
        clip_key=clip_key,
        prediction=proposal_prediction,
        bucket=spec.bucket,
        domain=effective_clip.domain,
        window_activity=proposal_window_activity,
    )
    repair_result = _routing_result_payload(
        clip_key=clip_key,
        prediction=repaired_prediction,
        bucket=spec.bucket,
        domain=effective_clip.domain,
        window_activity=proposal_window_activity,
    )
    routing_seed_result = _select_risk_source_result(
        global_result=baseline_result,
        proposal_result=proposal_result,
        repair_result=repair_result if "stream_throughput" in site_profile.expected_regime else None,
        direction=routing_direction,
        site_profile=site_profile,
    )
    risk_assessment = _select_risk_assessment(
        global_result=baseline_result,
        proposal_result=proposal_result,
        repair_result=repair_result if "stream_throughput" in site_profile.expected_regime else None,
        direction=routing_direction,
        site_profile=site_profile,
    )
    custom_branch_records: list[dict[str, Any]] = []
    custom_branch_predictions: dict[str, PassagePrediction] = {}
    accepted_custom_branch_predictions: dict[str, PassagePrediction] = {}
    routed_custom_override: tuple[str, PassagePrediction, dict[str, Any]] | None = None
    skip_custom_branches = args.window_prompt_mode == "strict_counts" and bool(window_failures)
    if not skip_custom_branches:
        for branch_id, expected_flags in (
            ("custom_stream_throughput", ("dense_stream_throughput_risk",)),
            ("custom_trickle_reduction", ("multi_episode_trickle_overcount_risk",)),
            (
                "custom_high_throughput",
                (
                    "multiwave_high_peak_undercount_risk",
                    "hidden_crossing_undercount_risk",
                    "single_school_high_throughput_undercount_risk",
                    "single_school_visibility_dropout_risk",
                ),
            ),
            ("custom_low_count", ("low_count_far_view_risk",)),
        ):
            matched_flag = next((flag for flag in expected_flags if flag in risk_assessment.flags), None)
            if (
                branch_id == "custom_stream_throughput"
                and "stream_throughput" in site_profile.expected_regime
                and branch_id in site_profile.enabled_branches
                and matched_flag is None
            ):
                matched_flag = "stream_throughput_profile_default"
            if matched_flag is None or branch_id not in site_profile.enabled_branches:
                continue
            custom_prompt = _custom_branch_prompt_for_clip(
                branch_id,
                risk_assessment=risk_assessment,
                window_runs=window_runs,
                direction=routing_direction,
                proposals=[asdict(proposal) for proposal in proposals],
            )
            custom_prediction = _infer_action_prediction(
                client=client,
                clip=effective_clip,
                variant=variant,
                prompt=custom_prompt,
                batch_root=batch_root,
                response_stem=f"{response_prefix}.{branch_id}",
                transport=args.transport,
                extra_metadata={
                    "action_id": f"{variant.value}:{branch_id}",
                    "site_profile_id": site_profile.site_id,
                    "routed_branch": branch_id,
                    "risk_flags": list(risk_assessment.flags),
                },
            )
            if branch_id == "custom_stream_throughput" and "stream_throughput" in site_profile.expected_regime:
                custom_prediction = _project_prediction_to_direction(
                    custom_prediction,
                    direction=stream_primary_direction,
                )
            repeat_predictions: list[PassagePrediction] = [custom_prediction]
            if branch_id == "custom_trickle_reduction":
                repeated_prediction = _infer_action_prediction(
                    client=client,
                    clip=effective_clip,
                    variant=variant,
                    prompt=custom_prompt,
                    batch_root=batch_root,
                    response_stem=f"{response_prefix}.{branch_id}.repeat-02",
                    transport=args.transport,
                    extra_metadata={
                        "action_id": f"{variant.value}:{branch_id}",
                        "site_profile_id": site_profile.site_id,
                        "routed_branch": branch_id,
                        "risk_flags": list(risk_assessment.flags),
                        "repeat_index": 2,
                    },
                )
                repeat_predictions.append(repeated_prediction)
                custom_prediction = _select_trickle_reduction_prediction(effective_clip, repeat_predictions)
            elif branch_id == "custom_stream_throughput":
                repeated_prediction = _infer_action_prediction(
                    client=client,
                    clip=effective_clip,
                    variant=variant,
                    prompt=custom_prompt,
                    batch_root=batch_root,
                    response_stem=f"{response_prefix}.{branch_id}.repeat-02",
                    transport=args.transport,
                    extra_metadata={
                        "action_id": f"{variant.value}:{branch_id}",
                        "site_profile_id": site_profile.site_id,
                        "routed_branch": branch_id,
                        "risk_flags": list(risk_assessment.flags),
                        "repeat_index": 2,
                    },
                )
                if "stream_throughput" in site_profile.expected_regime:
                    repeated_prediction = _project_prediction_to_direction(
                        repeated_prediction,
                        direction=stream_primary_direction,
                    )
                repeat_predictions.append(repeated_prediction)
                custom_prediction = _select_stream_throughput_prediction(
                    effective_clip,
                    repeat_predictions,
                    direction=routing_direction,
                )
            elif branch_id == "custom_high_throughput" and "single_school_high_throughput_undercount_risk" in risk_assessment.flags:
                repeated_prediction = _infer_action_prediction(
                    client=client,
                    clip=effective_clip,
                    variant=variant,
                    prompt=custom_prompt,
                    batch_root=batch_root,
                    response_stem=f"{response_prefix}.{branch_id}.repeat-02",
                    transport=args.transport,
                    extra_metadata={
                        "action_id": f"{variant.value}:{branch_id}",
                        "site_profile_id": site_profile.site_id,
                        "routed_branch": branch_id,
                        "risk_flags": list(risk_assessment.flags),
                        "repeat_index": 2,
                    },
                )
                repeat_predictions.append(repeated_prediction)
                custom_prediction = _select_high_throughput_prediction(effective_clip, repeat_predictions)
            custom_result = _routing_result_payload(
                clip_key=clip_key,
                prediction=custom_prediction,
                bucket=spec.bucket,
                domain=effective_clip.domain,
            )
            decision = apply_routed_policy(
                routing_seed_result,
                custom_result,
                direction=routing_direction,
                site_profile=site_profile,
                assessment_override=risk_assessment,
            )
            custom_branch_records.append(
                {
                    "branch_id": branch_id,
                    "expected_flag": matched_flag,
                    "decision": decision.to_dict(),
                    "prediction": _prediction_payload(custom_prediction),
                    "repeat_predictions": [_prediction_payload(prediction) for prediction in repeat_predictions],
                }
            )
            custom_branch_predictions[f"{variant.value}:{branch_id}"] = custom_prediction
            if (
                routed_custom_override is None
                and decision.selected_source == "custom"
                and decision.selected_branch == branch_id
                and _should_apply_routed_override(
                    effective_clip,
                    branch_id=branch_id,
                    global_prediction=global_prediction,
                    custom_prediction=custom_prediction,
                )
            ):
                routed_custom_override = (branch_id, custom_prediction, decision.to_dict())
                accepted_custom_branch_predictions[f"{variant.value}:{branch_id}"] = custom_prediction

    action_predictions = {
        f"{variant.value}:global": global_prediction,
        f"{variant.value}:proposal_guided": proposal_prediction,
        f"{variant.value}:local_repair": repaired_prediction,
    }
    action_predictions.update(accepted_custom_branch_predictions)
    action_map = {
        action.action_id: action
        for action in enumerate_perception_actions(build_observation_stream(effective_clip))
    }
    expert_selection = select_expert_prediction(
        effective_clip,
        action_predictions,
        memory=(),
        risk_assessment=risk_assessment,
    )
    strict_window_total = proposal_prediction.left_count + proposal_prediction.right_count
    global_total = global_prediction.left_count + global_prediction.right_count
    if routed_custom_override is not None:
        routed_branch_id, final_prediction, routed_decision = routed_custom_override
        selected_action_id = f"{variant.value}:{routed_branch_id}"
        selected_action = action_map.get(selected_action_id)
        selected_branch = selected_action.branch_id if selected_action is not None else routed_branch_id
        selection_reason = (
            f"routed_override rule={routed_decision.get('rule_name')} "
            f"reason={routed_decision.get('reason')}"
        )
    elif (
        args.window_prompt_mode == "strict_counts"
        and strict_window_total > 0
        and global_total == 0
    ):
        final_prediction = proposal_prediction
        selected_action_id = f"{variant.value}:proposal_guided"
        selected_action = action_map.get(selected_action_id)
        selected_branch = selected_action.branch_id if selected_action is not None else "proposal_guided"
        selection_reason = "strict_counts_positive_window_override"
    else:
        final_prediction = expert_selection.prediction
        selected_action_id = expert_selection.action_id
        selected_action = action_map.get(selected_action_id)
        selected_branch = selected_action.branch_id if selected_action is not None else "baseline"
        selection_reason = expert_selection.reason

    result_record = {
        "status": "completed",
        "bucket": spec.bucket,
        "clip_key": clip_key,
        "site_profile_id": site_profile.site_id,
        "site_profile": site_profile.to_dict(),
        "media_path": str(video_path),
        "ground_truth": gt,
        "proposals": [asdict(item) for item in proposals],
        "window_count": len(window_predictions),
        "window_runs": window_runs,
        "window_failures": window_failures,
        "risk_assessment": risk_assessment.to_dict(),
        "risk_flags": list(risk_assessment.flags),
        "routing_direction": routing_direction,
        "custom_branch_runs": custom_branch_records,
        "action_predictions": {
            "global_direct": _prediction_payload(global_prediction),
            "proposal_guided": _prediction_payload(proposal_prediction),
            "local_repair": _prediction_payload(repaired_prediction),
            **{
                key.split(":", 1)[1]: _prediction_payload(value)
                for key, value in accepted_custom_branch_predictions.items()
            },
        },
        "repair_audit": repair_audit,
        "selected_action": selected_action_id,
        "selected_branch": selected_branch,
        "selection_reason": selection_reason,
        "expert_selection": {
            "action_id": expert_selection.action_id,
            "score": expert_selection.score,
            "reason": expert_selection.reason,
        },
        "prediction": _prediction_payload(final_prediction),
        "selected_pass": selected_action_id,
        "completed_at": _now_iso(),
    }
    _refresh_result_record_errors(result_record, gt)
    return result_record


def _write_summary(path: Path, payload: dict[str, Any]) -> None:
    def _format_lr(prediction: dict[str, Any] | None) -> str:
        if not prediction:
            return ""
        left = prediction.get("left_count", "")
        right = prediction.get("right_count", "")
        if left == "" and right == "":
            return ""
        return f"{left}/{right}"

    lines = [
        f"# {payload['name']}",
        "",
        f"- domain: `{payload['domain']}`",
        f"- variant: `{payload['variant']}`",
        f"- model: `{payload['model']}`",
        f"- completed: `{payload['completed_count']}/{payload['total_clips']}`",
        "",
        "| Status | Clip | GT L/R | Global L/R | Proposal L/R | Repair L/R | Selected | Branch | Final L/R | Error |",
        "|---|---|---:|---:|---:|---:|---|---|---:|---:|",
    ]
    for item in payload["results"]:
        predictions = item.get("action_predictions") or {}
        gt_counts = _format_lr(item.get("ground_truth"))
        global_counts = _format_lr(predictions.get("global_direct"))
        proposal_counts = _format_lr(predictions.get("proposal_guided"))
        repair_counts = _format_lr(predictions.get("local_repair"))
        final_counts = _format_lr(item.get("prediction"))
        lines.append(
            f"| {item['status']} | `{item['clip_key']}` | {gt_counts} | {global_counts} | {proposal_counts} | {repair_counts} | "
            f"`{item.get('selected_action', '')}` | `{item.get('selected_branch', '')}` | {final_counts} | {item.get('total_abs_error', '')} |"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    settings = load_local_settings(args.settings)
    adapter = CFCAdapter(settings.paths)
    clips = {clip.clip_id: clip for clip in adapter.load_clip_records([args.domain])}
    specs = load_batch_specs(args.clips_path)
    if args.max_clips is not None:
        specs = specs[: args.max_clips]

    variant = InputVariant.from_value(args.variant)
    client = _build_client(settings, args.model, args.max_frames)
    prompt_text = args.prompt_text or _default_global_prompt()

    batch_root = settings.paths.output_root / "event_agent" / _safe_name(args.name)
    batch_root.mkdir(parents=True, exist_ok=True)
    summary_json_path = batch_root / "summary.json"
    summary_md_path = batch_root / "summary.md"
    log_path = batch_root / "progress.log"

    state: dict[str, Any] = {
        "name": args.name,
        "domain": args.domain,
        "variant": variant.value,
        "model": client.model,
        "transport": args.transport,
        "total_clips": len(specs),
        "completed_count": 0,
        "results": [],
        "updated_at": _now_iso(),
    }

    for index, spec in enumerate(specs, start=1):
        if spec.clip_id not in clips:
            raise RuntimeError(f"Clip not found in {args.domain}: {spec.clip_id}")
        clip = clips[spec.clip_id]
        gt = _ground_truth_counts(adapter, clip)

        try:
            def _progress_callback(stage: str, details: dict[str, object]) -> None:
                detail_text = _format_stage_details(details)
                log_message = f"{_now_iso()} [{index}/{len(specs)}] stage {clip_key} phase={stage}"
                if detail_text:
                    log_message = f"{log_message} {detail_text}"
                print(log_message, flush=True)
                _append_log(log_path, log_message)

            if variant == InputVariant.SFF3C:
                clip = _prepare_sff3c_variant(settings, clip, args.transport, args.max_frames)
            materialize_args = argparse.Namespace(
                transport=args.transport,
                stitch_fps=args.stitch_fps,
                stitch_height=args.stitch_height,
                stitch_crf=args.stitch_crf,
            )
            effective_clip, video_path = _materialize_clip_transport(materialize_args, settings, clip, variant)
            if args.transport != "stitched_mp4" or video_path is None:
                raise RuntimeError("event_agent_batch currently requires --transport stitched_mp4")

            clip_key = f"{args.domain}/{spec.clip_id}"
            site_profile = resolve_site_profile(effective_clip.domain, clip_key=clip_key)
            start_message = f"[{index}/{len(specs)}] start {clip_key} media={video_path}"
            print(start_message, flush=True)
            _append_log(log_path, f"{_now_iso()} {start_message}")
            if hasattr(client, "set_progress_callback"):
                client.set_progress_callback(_progress_callback)
            pass_results = [
                _run_clip_pass(
                    client=client,
                    settings=settings,
                    args=args,
                    spec=spec,
                    effective_clip=effective_clip,
                    video_path=video_path,
                    clip_key=clip_key,
                    gt=gt,
                    variant=variant,
                    batch_root=batch_root,
                    site_profile=site_profile,
                    prompt_text=prompt_text,
                    response_prefix=f"{index - 1:02d}__{spec.clip_id}",
                )
            ]
            if _should_run_flagged_repeat_consensus(pass_results[0], repeat_runs=args.flagged_repeat_runs):
                for repeat_index in range(2, args.flagged_repeat_runs + 1):
                    repeat_message = f"[{index}/{len(specs)}] repeat {clip_key} pass={repeat_index}"
                    print(repeat_message, flush=True)
                    _append_log(log_path, f"{_now_iso()} {repeat_message}")
                    pass_results.append(
                        _run_clip_pass(
                            client=client,
                            settings=settings,
                            args=args,
                            spec=spec,
                            effective_clip=effective_clip,
                            video_path=video_path,
                            clip_key=clip_key,
                            gt=gt,
                            variant=variant,
                            batch_root=batch_root,
                            site_profile=site_profile,
                            prompt_text=prompt_text,
                            response_prefix=f"{index - 1:02d}__{spec.clip_id}.pass-{repeat_index:02d}",
                        )
                    )

            result_record = pass_results[0]
            if len(pass_results) > 1:
                selected_result, repeat_consensus = _apply_flagged_repeat_consensus(pass_results)
                result_record = dict(selected_result)
                result_record["repeat_consensus"] = repeat_consensus
                _refresh_result_record_errors(result_record, gt)
            if _should_run_post_consensus_trickle_cleanup(result_record):
                cleanup_prompt = _post_consensus_trickle_prompt(result_record)
                cleanup_predictions = [
                    _infer_action_prediction(
                        client=client,
                        clip=effective_clip,
                        variant=variant,
                        prompt=cleanup_prompt,
                        batch_root=batch_root,
                        response_stem=f"{index - 1:02d}__{spec.clip_id}.post-consensus-trickle",
                        transport=args.transport,
                        extra_metadata={
                            "action_id": f"{variant.value}:custom_trickle_reduction",
                            "site_profile_id": site_profile.site_id,
                            "routed_branch": "post_consensus_trickle_cleanup",
                            "risk_flags": list(result_record.get("risk_flags") or []),
                        },
                    )
                ]
                cleanup_predictions.append(
                    _infer_action_prediction(
                        client=client,
                        clip=effective_clip,
                        variant=variant,
                        prompt=cleanup_prompt,
                        batch_root=batch_root,
                        response_stem=f"{index - 1:02d}__{spec.clip_id}.post-consensus-trickle.repeat-02",
                        transport=args.transport,
                        extra_metadata={
                            "action_id": f"{variant.value}:custom_trickle_reduction",
                            "site_profile_id": site_profile.site_id,
                            "routed_branch": "post_consensus_trickle_cleanup",
                            "risk_flags": list(result_record.get("risk_flags") or []),
                            "repeat_index": 2,
                        },
                    )
                )
                cleanup_prediction = _select_trickle_reduction_prediction(effective_clip, cleanup_predictions)
                cleanup_prediction, cleanup_cap_audit = _apply_post_consensus_sparse_stream_cap(
                    effective_clip,
                    result_record,
                    cleanup_prediction,
                )
                baseline_total = int((result_record.get("prediction") or {}).get("total_count") or 0)
                cleanup_total = _prediction_total(cleanup_prediction, effective_clip)
                custom_branch_runs = list(result_record.get("custom_branch_runs") or [])
                custom_branch_runs.append(
                    {
                        "branch_id": "post_consensus_trickle_cleanup",
                        "expected_flag": "repeat_consensus_sparse_cleanup",
                        "decision": {
                            "selected_source": "custom" if 0 < cleanup_total < baseline_total else "baseline",
                            "selected_branch": "custom_trickle_reduction" if 0 < cleanup_total < baseline_total else "baseline",
                            "rule_name": "accept_post_consensus_trickle_cleanup_capped"
                            if cleanup_cap_audit and 0 < cleanup_total < baseline_total
                            else "accept_post_consensus_trickle_cleanup"
                            if 0 < cleanup_total < baseline_total
                            else "keep_baseline",
                            "reason": (
                                f"baseline_total={baseline_total} custom_total={cleanup_total}"
                                + (
                                    f" {cleanup_cap_audit['reason']}"
                                    if cleanup_cap_audit and 0 < cleanup_total < baseline_total
                                    else ""
                                )
                            ),
                        },
                        "prediction": _prediction_payload(cleanup_prediction),
                        "repeat_predictions": [_prediction_payload(prediction) for prediction in cleanup_predictions],
                        "cap_audit": cleanup_cap_audit,
                    }
                )
                result_record["custom_branch_runs"] = custom_branch_runs
                result_record["post_consensus_cleanup"] = {
                    "applied": True,
                    "baseline_total": baseline_total,
                    "custom_total": cleanup_total,
                    "selected_source": "custom" if 0 < cleanup_total < baseline_total else "baseline",
                    "cap_audit": cleanup_cap_audit,
                }
                if 0 < cleanup_total < baseline_total:
                    result_record["selected_action"] = f"{variant.value}:custom_trickle_reduction"
                    result_record["selected_branch"] = "custom_trickle_reduction"
                    result_record["selection_reason"] = (
                        f"post_consensus_trickle_cleanup baseline_total={baseline_total} custom_total={cleanup_total}"
                        + (f" {cleanup_cap_audit['reason']}" if cleanup_cap_audit else "")
                    )
                    result_record["prediction"] = _prediction_payload(cleanup_prediction)
                    action_predictions = dict(result_record.get("action_predictions") or {})
                    action_predictions["custom_trickle_reduction"] = _prediction_payload(cleanup_prediction)
                    result_record["action_predictions"] = action_predictions
                    _refresh_result_record_errors(result_record, gt)
            result_record, stream_global_floor_audit = _apply_post_selection_stream_global_floor(result_record)
            if stream_global_floor_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, stream_candidate_floor_audit = _apply_post_selection_stream_candidate_floor(
                result_record
            )
            if stream_candidate_floor_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, stream_multiwave_uplift_audit = _apply_post_selection_stream_multiwave_uplift(
                result_record
            )
            if stream_multiwave_uplift_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, stream_tile_pair_floor_audit = _apply_post_selection_stream_tile_pair_floor(
                result_record
            )
            if stream_tile_pair_floor_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, stream_longspan_uplift_audit = _apply_post_selection_stream_longspan_uplift(
                result_record
            )
            if stream_longspan_uplift_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, stream_window_stack_uplift_audit = _apply_post_selection_stream_window_stack_uplift(
                result_record
            )
            if stream_window_stack_uplift_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, stream_embedded_wave_uplift_audit = _apply_post_selection_stream_embedded_wave_uplift(
                result_record
            )
            if stream_embedded_wave_uplift_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, positive_window_uplift_audit = _apply_post_selection_positive_window_uplift(
                result_record
            )
            if positive_window_uplift_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, single_school_uplift_audit = _apply_post_selection_single_school_uplift(
                result_record
            )
            if single_school_uplift_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, single_school_window_floor_audit = _apply_post_selection_single_school_window_floor(
                result_record
            )
            if single_school_window_floor_audit:
                _refresh_result_record_errors(result_record, gt)
            result_record, peak3_school_uplift_audit = _apply_post_selection_peak3_school_uplift(result_record)
            if peak3_school_uplift_audit:
                _refresh_result_record_errors(result_record, gt)
            state["results"].append(result_record)
            final_prediction_payload = result_record.get("prediction") or {}
            finish_message = (
                f"[{index}/{len(specs)}] done {clip_key} selected={result_record.get('selected_action')} "
                f"branch={result_record.get('selected_branch')} "
                f"pred=({final_prediction_payload.get('left_count')},{final_prediction_payload.get('right_count')})"
            )
            print(finish_message, flush=True)
            _append_log(log_path, f"{_now_iso()} {finish_message}")
        except Exception as exc:
            fail_record = {
                "status": "failed",
                "bucket": spec.bucket,
                "clip_key": f"{args.domain}/{spec.clip_id}",
                "ground_truth": gt,
                "error": str(exc),
                "completed_at": _now_iso(),
            }
            state["results"].append(fail_record)
            fail_message = f"[{index}/{len(specs)}] failed {args.domain}/{spec.clip_id} error={exc}"
            print(fail_message, flush=True)
            _append_log(log_path, f"{_now_iso()} {fail_message}")
            if args.stop_on_error:
                raise
        finally:
            if hasattr(client, "set_progress_callback"):
                client.set_progress_callback(None)

        state["completed_count"] = sum(1 for item in state["results"] if item.get("status") == "completed")
        state["updated_at"] = _now_iso()
        summary_json_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        _write_summary(summary_md_path, state)

    print(f"[event_agent_batch] {batch_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
