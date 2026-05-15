from __future__ import annotations

from ..types import ClipRecord, InputVariant, PromptRevision


STRICT_JSON_SCHEMA = """Return a single JSON object with this schema:
{
  "scene_assessment": string | null,
  "candidate_passages": [
    {
      "timestamp_start_sec": float | null,
      "timestamp_end_sec": float | null,
      "direction": "left|right|uncertain",
      "estimated_count": int | null,
      "peak_simultaneous_count": int | null,
      "episode_duration_sec": float | null,
      "wave_count": int | null,
      "throughput_best_count": int | null,
      "evidence_note": string | null
    }
  ],
  "rejected_targets": [string],
  "left_count": int,
  "right_count": int,
  "upstream_count": int | null,
  "downstream_count": int | null,
  "confidence": float | null,
  "commentary": string | null,
  "evidence_summary": string | null,
  "events": [
    {
      "timestamp_sec": float,
      "direction": "left|right|upstream|downstream",
      "confidence": float | null,
      "evidence_note": string | null
    }
  ]
}
Use `scene_assessment` for a short summary of visibility, clutter, and whether fish-like motion is present.
Use `candidate_passages` to summarize the main candidate fish-passage episodes you considered before final counting.
For each `candidate_passages` entry, use `peak_simultaneous_count` for the maximum simultaneously visible fish in that episode, `episode_duration_sec` for the episode duration, `wave_count` for the number of defensible waves or mini-bursts, and `throughput_best_count` for your best total-throughput estimate for that episode.
Set `estimated_count` equal to the same best throughput estimate you want counted for that episode, not just the peak simultaneous occupancy.
Use `rejected_targets` for clutter, debris, stationary blobs, or ambiguous targets that you deliberately did not count.
Always provide integer `left_count` and `right_count`. Use 0 if no fish are observed moving in that direction.
Do not return count ranges such as `5-7`; choose one best integer per direction.
Leave `upstream_count` and `downstream_count` as null unless they are explicitly easier to infer than left/right.
Use `commentary` for a short plain-language interpretation of the scene and your conclusion.
Use `evidence_summary` for brief observable evidence only. Do not provide hidden reasoning or long step-by-step analysis.
Base the answer only on this clip. Do not cite papers, websites, or generic sonar references.
Do not add markdown fences or explanation."""


def _variant_guidance(variant: InputVariant) -> list[str]:
    if variant == InputVariant.RAW:
        return [
            "The input is raw sonar imagery. Bright or textured regions may be fish, debris, bottom structure, or noise.",
            "In raw sonar, isolated bright speckles or static blobs are not enough to count as fish.",
        ]
    if variant == InputVariant.SFF3C:
        return [
            "This is an algorithmic 3-channel representation of the same sonar scene, not natural RGB color video.",
            "The channels were generated to emphasize moving targets and suppress background structure, but they can still highlight debris or artifacts.",
            "Yellow, orange, or bright compact blobs are candidate targets only; count them as fish only if they move coherently over time.",
        ]
    if variant == InputVariant.CFC_3CHANNEL:
        return [
            "This is a false-color derived 3-channel representation of the same sonar scene, not natural RGB color video.",
            "Warm or bright blobs are candidate targets, not guaranteed fish.",
            "Use coherent motion, consistent shape, and temporal persistence to separate fish from clutter or noise.",
        ]
    return []


def build_prompt(clip: ClipRecord, variant: InputVariant, revision: PromptRevision) -> str:
    asset_path = clip.get_asset_path(variant)
    frame_paths_by_variant = clip.metadata.get("frame_paths_by_variant", {})
    frame_count = None
    if isinstance(frame_paths_by_variant, dict):
        variant_frames = frame_paths_by_variant.get(variant.value)
        if isinstance(variant_frames, list):
            frame_count = len(variant_frames)
    metadata_lines = [
        f"dataset={clip.dataset}",
        f"domain={clip.domain}",
        f"clip_id={clip.clip_id}",
        f"asset_path={asset_path}",
        f"variant={variant.value}",
        f"frame_count={frame_count}",
        f"width={clip.width}",
        f"height={clip.height}",
        f"framerate={clip.framerate}",
        f"duration_seconds={clip.duration_seconds}",
        f"upstream_direction={clip.upstream_direction.value if clip.upstream_direction else None}",
    ]
    task_lines = [
        "This is an underwater sonar clip.",
        "The frames are chronological samples from one clip.",
        "Work in this order: scene assessment, candidate passages, rejected targets, then final counts.",
        "Estimate unique fish passage counts over the whole clip, not per frame.",
        "First distinguish fish-like targets from debris, stationary clutter, bottom returns, and random speckle noise.",
        "Count a fish only when it shows coherent motion across time and behaves like one individual fish or one member of a small school.",
        "Do not count stationary clutter, flickering speckle, or the same fish twice across adjacent frames.",
        "Use `left_count` and `right_count` for net movement direction across the clip. Do not require a literal image-center crossing unless it is clearly visible and relevant.",
        "If one compact school passes together, estimate the number of distinct fish in that episode and record it in `candidate_passages` and `events`.",
        "If a long stream appears to contain multiple waves, mini-bursts, or replacement groups over time, split them into separate `candidate_passages` entries instead of collapsing them into one broad school event.",
        "For each `candidate_passages` entry, fill in `peak_simultaneous_count`, `episode_duration_sec`, `wave_count`, and `throughput_best_count` when they can be inferred from the clip.",
        "Make `estimated_count` match `throughput_best_count` for each passage episode.",
        "Use one `events` entry for each defensible wave start or density peak, not only one event for the whole long episode.",
        "If visibility is poor, report lower confidence and explain the limiting evidence in `scene_assessment`, `commentary`, or `evidence_summary`, but do not default to the lowest plausible count.",
    ]
    task_lines.extend(_variant_guidance(variant))
    return "\n".join([revision.prompt_text.strip(), "", "Task guidance:", *task_lines, "", "Clip metadata:", *metadata_lines, "", STRICT_JSON_SCHEMA])


def build_revised_prompt(previous: PromptRevision, critique: str, next_version: int) -> PromptRevision:
    return PromptRevision(
        version=next_version,
        prompt_id=f"r{next_version}",
        prompt_text="\n".join([previous.prompt_text.strip(), "", "Revision guidance:", critique.strip()]).strip(),
        critique=critique,
        metrics=previous.metrics,
    )
