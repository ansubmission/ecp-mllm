from __future__ import annotations

from csv import DictReader
import json
from pathlib import Path
from typing import Any, Iterable

from ..config import PathsConfig
from ..thermal import THERMAL_COARSE_LABELS, collapse_thermal_labels, infer_false_positive_from_labels, map_thermal_label_to_coarse
from ..types import ClipRecord, InputVariant


def _optional_float(value: object | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def _optional_int(value: object | None) -> int | None:
    if value in (None, ""):
        return None
    return int(float(value))


def _parse_jsonish(value: object) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("{") or text.startswith("["):
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return value
    return value


def _load_rows_from_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return [dict(item) for item in payload if isinstance(item, dict)]
        if isinstance(payload, dict):
            if isinstance(payload.get("clips"), list):
                return [dict(item) for item in payload["clips"] if isinstance(item, dict)]
            if payload and all(isinstance(value, list) for value in payload.values()):
                rows: list[dict[str, Any]] = []
                for split_name, items in payload.items():
                    for item in items:
                        if not isinstance(item, dict):
                            continue
                        row = dict(item)
                        row.setdefault("split", str(split_name))
                        rows.append(row)
                return rows
            return [dict(payload)]
        return []
    if path.suffix.lower() == ".jsonl":
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if isinstance(payload, dict):
                rows.append(dict(payload))
        return rows
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in DictReader(handle)]
    return []


def _load_rows(path: Path | None) -> list[dict[str, Any]]:
    if path is None or not path.exists():
        return []
    if path.is_file():
        return _load_rows_from_file(path)
    rows: list[dict[str, Any]] = []
    for child in sorted(path.rglob("*")):
        if not child.is_file() or child.suffix.lower() not in {".json", ".jsonl", ".csv"}:
            continue
        rows.extend(_load_rows_from_file(child))
    return rows


def _coerce_labels(value: object) -> list[str]:
    parsed = _parse_jsonish(value)
    if parsed is None or parsed == "":
        return []
    if isinstance(parsed, str):
        return [part.strip() for part in parsed.replace("|", ",").split(",") if part.strip()]
    if isinstance(parsed, dict):
        for key in ("label", "name", "class", "species"):
            if parsed.get(key):
                return [str(parsed[key]).strip()]
        return []
    if isinstance(parsed, list):
        labels: list[str] = []
        for item in parsed:
            if isinstance(item, dict):
                for key in ("label", "name", "class", "species"):
                    if item.get(key):
                        labels.append(str(item[key]).strip())
                        break
            elif item not in (None, ""):
                labels.append(str(item).strip())
        return [label for label in labels if label]
    return []


def _seconds_from_track_value(track: dict[str, Any], start_key: str, end_key: str, framerate: float | None) -> tuple[float | None, float | None]:
    start = _optional_float(track.get(start_key))
    end = _optional_float(track.get(end_key))
    if start is not None and end is not None:
        return start, end
    frame_start = _optional_float(track.get("start_frame") if start is None else None)
    frame_end = _optional_float(track.get("end_frame") if end is None else None)
    if frame_start is not None and frame_end is not None and framerate not in (None, 0.0):
        return frame_start / float(framerate), frame_end / float(framerate)
    positions = track.get("positions") or track.get("bounds") or track.get("points")
    if isinstance(positions, list):
        seconds: list[float] = []
        for item in positions:
            if not isinstance(item, dict):
                continue
            if item.get("time_s") not in (None, ""):
                seconds.append(float(item["time_s"]))
                continue
            if item.get("frame_number") not in (None, "") and framerate not in (None, 0.0):
                seconds.append(float(item["frame_number"]) / float(framerate))
        if seconds:
            return min(seconds), max(seconds)
    return start, end


def _normalize_track(track: object, framerate: float | None) -> dict[str, Any] | None:
    parsed = _parse_jsonish(track)
    if not isinstance(parsed, dict):
        return None
    start_sec, end_sec = _seconds_from_track_value(parsed, "start_s", "end_s", framerate)
    if start_sec is None or end_sec is None:
        start_sec, end_sec = _seconds_from_track_value(parsed, "start", "end", framerate)
    tags = parsed.get("tags")
    first_tag = tags[0] if isinstance(tags, list) and tags and isinstance(tags[0], dict) else {}
    label = str(
        parsed.get("label")
        or parsed.get("class")
        or parsed.get("species")
        or first_tag.get("label")
        or first_tag.get("name")
        or ""
    ).strip() or None
    confidence = _optional_float(parsed.get("confidence"))
    if confidence is None and isinstance(first_tag, dict):
        confidence = _optional_float(first_tag.get("confidence"))
    points: list[dict[str, float | None]] = []
    raw_points = parsed.get("points") or parsed.get("positions") or []
    if isinstance(raw_points, list):
        for item in raw_points:
            normalized = _point_xy_time(item, framerate)
            if normalized is None:
                continue
            x, y, time_sec = normalized
            frame_index: int | None = None
            if isinstance(item, dict) and item.get("frame_number") not in (None, ""):
                try:
                    frame_index = int(float(item["frame_number"]))
                except (TypeError, ValueError):
                    frame_index = None
            elif isinstance(item, (list, tuple)) and len(item) >= 3 and item[2] not in (None, ""):
                try:
                    frame_index = int(float(item[2]))
                except (TypeError, ValueError):
                    frame_index = None
            points.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "time_sec": float(time_sec) if time_sec is not None else None,
                    "frame_index": frame_index,
                }
            )
    return {
        "track_id": str(parsed.get("id") or parsed.get("track_id") or "").strip() or None,
        "label": label,
        "coarse_label": map_thermal_label_to_coarse(label),
        "start_sec": start_sec,
        "end_sec": end_sec,
        "confidence": confidence,
        "points": points,
    }


def _coerce_tracks(value: object, framerate: float | None) -> list[dict[str, Any]]:
    parsed = _parse_jsonish(value)
    if not isinstance(parsed, list):
        return []
    tracks: list[dict[str, Any]] = []
    for item in parsed:
        normalized = _normalize_track(item, framerate)
        if normalized is not None:
            tracks.append(normalized)
    return tracks


def _coerce_calibration_frames(value: object, framerate: float | None) -> list[dict[str, float]]:
    parsed = _parse_jsonish(value)
    if parsed in (None, "", []):
        return []
    if isinstance(parsed, dict):
        parsed = [parsed]
    if not isinstance(parsed, list):
        return []
    ranges: list[dict[str, float]] = []
    for item in parsed:
        if isinstance(item, dict):
            start = _optional_float(item.get("start_sec"))
            end = _optional_float(item.get("end_sec"))
            if start is None or end is None:
                frame_start = _optional_float(item.get("start_frame"))
                frame_end = _optional_float(item.get("end_frame"))
                if frame_start is not None and frame_end is not None and framerate not in (None, 0.0):
                    start = frame_start / float(framerate)
                    end = frame_end / float(framerate)
            if start is not None and end is not None:
                ranges.append({"start_sec": float(start), "end_sec": float(end)})
        elif isinstance(item, (int, float)) and framerate not in (None, 0.0):
            second = float(item) / float(framerate)
            ranges.append({"start_sec": second, "end_sec": second})
    return ranges


def _point_xy_time(point: object, framerate: float | None) -> tuple[float, float, float | None] | None:
    if isinstance(point, (list, tuple)) and len(point) >= 2:
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        time_sec: float | None = None
        if len(point) >= 3 and point[2] not in (None, "") and framerate not in (None, 0.0):
            try:
                time_sec = float(point[2]) / float(framerate)
            except (TypeError, ValueError):
                time_sec = None
        return x, y, time_sec
    if isinstance(point, dict):
        try:
            x = float(point.get("x"))
            y = float(point.get("y"))
        except (TypeError, ValueError):
            return None
        if point.get("time_s") not in (None, ""):
            try:
                return x, y, float(point["time_s"])
            except (TypeError, ValueError):
                return x, y, None
        if point.get("frame_number") not in (None, "") and framerate not in (None, 0.0):
            try:
                return x, y, float(point["frame_number"]) / float(framerate)
            except (TypeError, ValueError):
                return x, y, None
        return x, y, None
    return None


def _coerce_center_zone_events(
    tracks: object,
    *,
    width: float | None,
    height: float | None,
    framerate: float | None,
    zone_lo: float = 0.3,
    zone_hi: float = 0.7,
) -> list[dict[str, Any]]:
    parsed = _parse_jsonish(tracks)
    if not isinstance(parsed, list):
        return []
    zone_events: list[dict[str, Any]] = []
    width_value = float(width or 0.0)
    height_value = float(height or 0.0)
    for item in parsed:
        if not isinstance(item, dict):
            continue
        coarse_label = map_thermal_label_to_coarse(
            str(
                item.get("label")
                or item.get("class")
                or item.get("species")
                or ((item.get("tags") or [{}])[0].get("label") if isinstance(item.get("tags"), list) and item.get("tags") else "")
            ).strip()
        )
        if coarse_label == "false_positive":
            continue
        points = item.get("points") or item.get("positions") or []
        in_zone_seconds: list[float] = []
        for point in points:
            normalized = _point_xy_time(point, framerate)
            if normalized is None:
                continue
            x, y, time_sec = normalized
            x_norm = (x / width_value) if width_value > 0 else 0.0
            y_norm = (y / height_value) if height_value > 0 else 0.0
            if zone_lo <= x_norm <= zone_hi and zone_lo <= y_norm <= zone_hi:
                if time_sec is not None:
                    in_zone_seconds.append(float(time_sec))
        if not in_zone_seconds:
            continue
        zone_events.append(
            {
                "track_id": str(item.get("id") or item.get("track_id") or "").strip() or None,
                "label": coarse_label,
                "enter_sec": min(in_zone_seconds),
                "exit_sec": max(in_zone_seconds),
                "dwell_sec": max(0.0, max(in_zone_seconds) - min(in_zone_seconds)),
            }
        )
    return zone_events


def _coerce_bool(value: object | None) -> bool | None:
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


def _clip_id_for_row(row: dict[str, Any]) -> str:
    for key in ("clip_id", "id", "uuid", "clip_uuid"):
        if row.get(key) not in (None, ""):
            return str(row[key]).strip()
    for key in ("filtered_video_filename", "normalized_video_filename", "video_filename"):
        if row.get(key):
            return Path(str(row[key])).stem
    raise ValueError(f"Unable to infer clip id from row keys: {sorted(row.keys())}")


def _lookup_path(root: Path | None, filename: str | None) -> Path | None:
    if not filename:
        return None
    candidate = Path(filename)
    if candidate.is_absolute():
        return candidate
    if root is None:
        return candidate
    return root / candidate


def _subtract_mask(duration_seconds: float | None, masks: list[dict[str, float]]) -> list[tuple[float, float]]:
    if duration_seconds is None or duration_seconds <= 0:
        return [(0.0, 0.0)]
    active = [(0.0, float(duration_seconds))]
    for mask in sorted(masks, key=lambda item: (item["start_sec"], item["end_sec"])):
        next_active: list[tuple[float, float]] = []
        for start, end in active:
            if mask["end_sec"] <= start or mask["start_sec"] >= end:
                next_active.append((start, end))
                continue
            if mask["start_sec"] > start:
                next_active.append((start, mask["start_sec"]))
            if mask["end_sec"] < end:
                next_active.append((mask["end_sec"], end))
        active = [(max(0.0, start), max(0.0, end)) for start, end in next_active if end - start >= 0.05]
    return active or [(0.0, float(duration_seconds))]


class NzThermalAdapter:
    def __init__(self, paths: PathsConfig) -> None:
        self.paths = paths

    @staticmethod
    def _clip_duration_sec(clip: ClipRecord) -> float:
        return float(clip.duration_seconds or 0.0)

    @staticmethod
    def _track_count(clip: ClipRecord) -> int:
        return len(clip.metadata.get("tracks") or [])

    @staticmethod
    def _max_track_duration_sec(clip: ClipRecord) -> float:
        durations = [
            float(track["end_sec"]) - float(track["start_sec"])
            for track in (clip.metadata.get("tracks") or [])
            if track.get("start_sec") is not None and track.get("end_sec") is not None
        ]
        return max(durations, default=0.0)

    @staticmethod
    def _zone_dwell_sec(clip: ClipRecord) -> float:
        return float(clip.metadata.get("truth_center_zone_dwell_sec") or 0.0)

    @staticmethod
    def _dedupe_preserve_order(items: list[ClipRecord]) -> list[ClipRecord]:
        seen: set[str] = set()
        ordered: list[ClipRecord] = []
        for clip in items:
            if clip.clip_id in seen:
                continue
            seen.add(clip.clip_id)
            ordered.append(clip)
        return ordered

    @staticmethod
    def _balanced_pick(
        candidates: list[ClipRecord],
        count: int,
        *,
        score_key,
    ) -> list[ClipRecord]:
        if count <= 0 or not candidates:
            return []
        ordered = sorted(candidates, key=lambda clip: (float(score_key(clip)), clip.key.value))
        if len(ordered) <= count:
            return ordered
        low_count = max(1, count // 2)
        high_count = count - low_count
        selected = ordered[:low_count]
        if high_count > 0:
            selected.extend(reversed(ordered[-high_count:]))
        deduped = NzThermalAdapter._dedupe_preserve_order(selected)
        if len(deduped) < count:
            for clip in ordered:
                if clip.clip_id in {item.clip_id for item in deduped}:
                    continue
                deduped.append(clip)
                if len(deduped) >= count:
                    break
        return deduped[:count]

    def _inventory_score(self, clip: ClipRecord) -> float:
        coarse_labels = clip.metadata.get("coarse_labels") or []
        return (
            self._track_count(clip) * 4.0
            + self._max_track_duration_sec(clip)
            + self._clip_duration_sec(clip) / 6.0
            + self._zone_dwell_sec(clip) / 6.0
            + max(0, len(coarse_labels) - 1) * 2.0
        )

    def _false_positive_score(self, clip: ClipRecord) -> float:
        return (
            self._max_track_duration_sec(clip)
            + self._clip_duration_sec(clip) / 5.0
            + self._track_count(clip) * 1.5
        )

    def _inventory32_specs(self, clips: list[ClipRecord]) -> list[dict[str, Any]]:
        manageable = [
            clip
            for clip in clips
            if 5.0 <= self._clip_duration_sec(clip) <= 60.0
        ]
        false_positive = [
            clip
            for clip in manageable
            if bool(clip.metadata.get("is_false_positive"))
        ]
        animals = [
            clip
            for clip in manageable
            if not bool(clip.metadata.get("is_false_positive"))
        ]

        fp_sorted = sorted(false_positive, key=lambda clip: (self._false_positive_score(clip), clip.key.value))
        fp_easy = fp_sorted[:8]
        fp_hard = list(reversed(fp_sorted[-8:]))
        fp_selected = self._dedupe_preserve_order(fp_easy + fp_hard)
        if len(fp_selected) < 16:
            for clip in fp_sorted:
                if clip.clip_id in {item.clip_id for item in fp_selected}:
                    continue
                fp_selected.append(clip)
                if len(fp_selected) >= 16:
                    break

        animal_specs: list[dict[str, Any]] = []
        label_plan: list[tuple[str, int]] = [
            ("rodent", 2),
            ("bird", 2),
            ("possum", 2),
            ("cat", 2),
            ("hedgehog", 2),
            ("mustelid", 2),
            ("other", 4),
        ]
        selected_ids: set[str] = set()
        for coarse_label, count in label_plan:
            candidates = [
                clip
                for clip in animals
                if str(clip.metadata.get("coarse_label")) == coarse_label and clip.clip_id not in selected_ids
            ]
            picked = self._balanced_pick(candidates, count, score_key=self._inventory_score)
            for clip in picked:
                selected_ids.add(clip.clip_id)
                animal_specs.append({"bucket": coarse_label, "clip": clip})
        if len(animal_specs) < 16:
            remaining = [
                clip
                for clip in sorted(animals, key=lambda clip: (self._inventory_score(clip), clip.key.value))
                if clip.clip_id not in selected_ids
            ]
            for clip in remaining:
                selected_ids.add(clip.clip_id)
                animal_specs.append({"bucket": str(clip.metadata.get("coarse_label") or "other"), "clip": clip})
                if len(animal_specs) >= 16:
                    break

        specs = [{"bucket": "false_positive_easy", "clip": clip} for clip in fp_easy]
        specs.extend({"bucket": "false_positive_hard", "clip": clip} for clip in fp_hard if clip.clip_id not in {item["clip"].clip_id for item in specs})
        if len(specs) < 16:
            for clip in fp_selected:
                if clip.clip_id in {item["clip"].clip_id for item in specs}:
                    continue
                specs.append({"bucket": "false_positive_support", "clip": clip})
                if len(specs) >= 16:
                    break
        specs.extend(animal_specs)
        if len(specs) < 32:
            used_ids = {item["clip"].clip_id for item in specs}
            remaining_animals = [
                clip
                for clip in sorted(animals, key=lambda clip: (self._inventory_score(clip), clip.key.value))
                if clip.clip_id not in used_ids
            ]
            for clip in remaining_animals:
                specs.append({"bucket": str(clip.metadata.get("coarse_label") or "other"), "clip": clip})
                if len(specs) >= 32:
                    break
        return specs[:32]

    def _zone16_meaningful_specs(self, clips: list[ClipRecord]) -> list[dict[str, Any]]:
        manageable = [
            clip
            for clip in clips
            if 5.0 <= self._clip_duration_sec(clip) <= 60.0
        ]
        positive_candidates = [
            clip
            for clip in manageable
            if not bool(clip.metadata.get("is_false_positive")) and self._zone_dwell_sec(clip) >= 1.0
        ]
        negative_animal_candidates = [
            clip
            for clip in manageable
            if not bool(clip.metadata.get("is_false_positive")) and self._zone_dwell_sec(clip) < 0.5
        ]
        false_positive_candidates = [
            clip
            for clip in manageable
            if bool(clip.metadata.get("is_false_positive"))
        ]

        label_order = ["rodent", "bird", "possum", "cat", "hedgehog", "mustelid", "other"]
        selected_positive: list[ClipRecord] = []
        selected_positive_ids: set[str] = set()
        for coarse_label in label_order:
            candidates = [
                clip
                for clip in positive_candidates
                if str(clip.metadata.get("coarse_label")) == coarse_label and clip.clip_id not in selected_positive_ids
            ]
            if not candidates:
                continue
            chosen = max(
                candidates,
                key=lambda clip: (
                    self._zone_dwell_sec(clip),
                    self._max_track_duration_sec(clip),
                    -self._clip_duration_sec(clip),
                    clip.key.value,
                ),
            )
            selected_positive.append(chosen)
            selected_positive_ids.add(chosen.clip_id)
            if len(selected_positive) >= 8:
                break
        if len(selected_positive) < 8:
            ranked_positive = sorted(
                positive_candidates,
                key=lambda clip: (
                    -self._zone_dwell_sec(clip),
                    -self._max_track_duration_sec(clip),
                    self._clip_duration_sec(clip),
                    clip.key.value,
                ),
            )
            for clip in ranked_positive:
                if clip.clip_id in selected_positive_ids:
                    continue
                selected_positive.append(clip)
                selected_positive_ids.add(clip.clip_id)
                if len(selected_positive) >= 8:
                    break

        selected_negative_animals: list[ClipRecord] = []
        selected_negative_ids: set[str] = set()
        for coarse_label in label_order:
            candidates = [
                clip
                for clip in negative_animal_candidates
                if str(clip.metadata.get("coarse_label")) == coarse_label and clip.clip_id not in selected_negative_ids
            ]
            if not candidates:
                continue
            chosen = max(
                candidates,
                key=lambda clip: (
                    self._inventory_score(clip),
                    self._clip_duration_sec(clip),
                    clip.key.value,
                ),
            )
            selected_negative_animals.append(chosen)
            selected_negative_ids.add(chosen.clip_id)
            if len(selected_negative_animals) >= 4:
                break
        if len(selected_negative_animals) < 4:
            ranked_negative_animals = sorted(
                negative_animal_candidates,
                key=lambda clip: (
                    -self._inventory_score(clip),
                    -self._clip_duration_sec(clip),
                    clip.key.value,
                ),
            )
            for clip in ranked_negative_animals:
                if clip.clip_id in selected_negative_ids:
                    continue
                selected_negative_animals.append(clip)
                selected_negative_ids.add(clip.clip_id)
                if len(selected_negative_animals) >= 4:
                    break

        fp_sorted = sorted(false_positive_candidates, key=lambda clip: (self._false_positive_score(clip), clip.key.value))
        selected_false_positive = self._dedupe_preserve_order(fp_sorted[:2] + list(reversed(fp_sorted[-2:])))
        if len(selected_false_positive) < 4:
            for clip in fp_sorted:
                if clip.clip_id in {item.clip_id for item in selected_false_positive}:
                    continue
                selected_false_positive.append(clip)
                if len(selected_false_positive) >= 4:
                    break

        specs = [{"bucket": "zone_positive_meaningful", "clip": clip} for clip in selected_positive[:8]]
        specs.extend({"bucket": "zone_negative_animal", "clip": clip} for clip in selected_negative_animals[:4])
        specs.extend({"bucket": "zone_negative_fp", "clip": clip} for clip in selected_false_positive[:4])
        return specs[:16]

    def _build_fixed_pack_specs(
        self,
        pack_name: str,
        *,
        split: str = "test",
        include_calibration: bool = False,
    ) -> list[dict[str, Any]]:
        clips = self.load_clip_records(split=split, include_calibration=include_calibration)
        if pack_name == "thermal_fp16":
            false_positive = sorted(
                [clip for clip in clips if bool(clip.metadata.get("is_false_positive"))],
                key=lambda clip: clip.key.value,
            )[:8]
            animal = sorted(
                [clip for clip in clips if not bool(clip.metadata.get("is_false_positive"))],
                key=lambda clip: clip.key.value,
            )[:8]
            return [{"bucket": "false_positive", "clip": clip} for clip in false_positive] + [
                {"bucket": str(clip.metadata.get("coarse_label") or "other"), "clip": clip} for clip in animal
            ]
        if pack_name == "thermal_coarse32":
            selected: list[dict[str, Any]] = []
            for coarse_label in THERMAL_COARSE_LABELS:
                bucket = sorted(
                    [clip for clip in clips if str(clip.metadata.get("coarse_label")) == coarse_label],
                    key=lambda clip: clip.key.value,
                )[:4]
                selected.extend({"bucket": coarse_label, "clip": clip} for clip in bucket)
            return selected
        if pack_name == "thermal_mixed16":
            def _difficulty(clip: ClipRecord) -> tuple[float, str]:
                tracks = clip.metadata.get("tracks") or []
                coarse_labels = clip.metadata.get("coarse_labels") or []
                difficulty = (
                    float(len(tracks)) * 10.0
                    + float(max(0, len(coarse_labels) - 1)) * 5.0
                    + float(clip.duration_seconds or 0.0) / 10.0
                    + (2.0 if not bool(clip.metadata.get("is_false_positive")) else 0.0)
                )
                return (-difficulty, clip.key.value)

            return [
                {"bucket": str(clip.metadata.get("coarse_label") or "other"), "clip": clip}
                for clip in sorted(clips, key=_difficulty)[:16]
            ]
        if pack_name == "thermal_inventory32":
            return self._inventory32_specs(clips)
        if pack_name == "thermal_zone16_meaningful":
            return self._zone16_meaningful_specs(clips)
        raise ValueError(f"Unknown thermal pack: {pack_name}")

    def load_clip_records(
        self,
        split: str | None = None,
        *,
        include_calibration: bool = False,
        clip_ids: Iterable[str] | None = None,
    ) -> list[ClipRecord]:
        rows = _load_rows(self.paths.resolved_nz_thermal_metadata_path())
        clip_rows = _load_rows(self.paths.resolved_nz_thermal_clip_metadata_path())
        split_rows = _load_rows(self.paths.resolved_nz_thermal_splits_path())
        clip_id_filter = {str(item).strip() for item in clip_ids or () if str(item).strip()}

        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            clip_id = _clip_id_for_row(row)
            if clip_id_filter and clip_id not in clip_id_filter:
                continue
            merged[clip_id] = {k: _parse_jsonish(v) for k, v in row.items()}
        for row in clip_rows:
            clip_id = _clip_id_for_row(row)
            if clip_id_filter and clip_id not in clip_id_filter:
                continue
            merged.setdefault(clip_id, {})
            merged[clip_id].update({k: _parse_jsonish(v) for k, v in row.items()})
        split_map: dict[str, str] = {}
        for row in split_rows:
            clip_id = _clip_id_for_row(row)
            if clip_id_filter and clip_id not in clip_id_filter:
                continue
            split_value = row.get("split") or row.get("recommended_split") or row.get("partition")
            if split_value not in (None, ""):
                split_map[clip_id] = str(split_value).strip().lower()

        filtered_root = self.paths.resolved_variant_root(InputVariant.THERMAL_FILTERED)
        normalized_root = self.paths.resolved_variant_root(InputVariant.THERMAL_NORMALIZED)
        records: list[ClipRecord] = []

        for clip_id in sorted(merged):
            row = merged[clip_id]
            framerate = _optional_float(row.get("framerate") or row.get("frame_rate") or row.get("fps"))
            duration_seconds = _optional_float(row.get("duration_seconds") or row.get("duration"))
            width = _optional_int(row.get("width") or row.get("res_x") or row.get("frame_width"))
            height = _optional_int(row.get("height") or row.get("res_y") or row.get("frame_height"))
            labels = _coerce_labels(row.get("labels") or row.get("label"))
            coarse_labels = collapse_thermal_labels(labels)
            tracks = _coerce_tracks(row.get("tracks"), framerate)
            center_zone_events = _coerce_center_zone_events(
                row.get("tracks"),
                width=width,
                height=height,
                framerate=framerate,
            )
            if duration_seconds is None and framerate not in (None, 0.0):
                track_end_seconds = [float(track["end_sec"]) for track in tracks if track.get("end_sec") is not None]
                if track_end_seconds:
                    duration_seconds = max(track_end_seconds)
            calibration_frames = _coerce_calibration_frames(row.get("calibration_frames"), framerate)
            is_false_positive = (
                _coerce_bool(row.get("is_false_positive"))
                if _coerce_bool(row.get("is_false_positive")) is not None
                else infer_false_positive_from_labels(labels)
            )
            coarse_label = (
                map_thermal_label_to_coarse(str(row.get("coarse_label")))
                if row.get("coarse_label") not in (None, "")
                else (coarse_labels[0] if len(coarse_labels) == 1 else ("false_positive" if is_false_positive else "other"))
            )
            split_value = str(row.get("split") or row.get("recommended_split") or split_map.get(clip_id) or "").strip().lower()
            if split is not None and split_value != split.strip().lower():
                continue
            if calibration_frames and not include_calibration:
                continue
            location = str(row.get("location") or row.get("site") or row.get("station") or split_value or "unknown").strip()
            filtered_name = str(row.get("filtered_video_filename") or row.get("filtered_filename") or row.get("video_filename") or "").strip() or None
            normalized_name = str(
                row.get("normalized_video_filename")
                or row.get("normalized_filename")
                or row.get("video_filename")
                or ""
            ).strip() or None
            asset_paths: dict[InputVariant, Path] = {}
            filtered_path = _lookup_path(filtered_root, filtered_name)
            if filtered_path is not None:
                asset_paths[InputVariant.THERMAL_FILTERED] = filtered_path
            normalized_path = _lookup_path(normalized_root, normalized_name)
            if normalized_path is not None:
                asset_paths[InputVariant.THERMAL_NORMALIZED] = normalized_path
            metadata = dict(row)
            metadata.update(
                {
                    "location": location,
                    "labels": labels,
                    "coarse_labels": coarse_labels,
                    "coarse_label": coarse_label,
                    "split": split_value or None,
                    "tracks": tracks,
                    "truth_event_windows": [
                        {
                            "start_sec": track["start_sec"],
                            "end_sec": track["end_sec"],
                            "label": track["coarse_label"],
                            "track_id": track["track_id"],
                        }
                        for track in tracks
                        if track.get("start_sec") is not None and track.get("end_sec") is not None
                    ],
                    "center_zone_definition": {"x_lo": 0.3, "x_hi": 0.7, "y_lo": 0.3, "y_hi": 0.7},
                    "truth_center_zone_events": center_zone_events,
                    "truth_center_zone_entered": bool(center_zone_events),
                    "truth_center_zone_event_count": len(center_zone_events),
                    "truth_center_zone_first_entry_sec": min((float(item["enter_sec"]) for item in center_zone_events), default=None),
                    "truth_center_zone_dwell_sec": sum(float(item["dwell_sec"]) for item in center_zone_events) if center_zone_events else None,
                    "calibration_frames": calibration_frames,
                    "active_intervals": _subtract_mask(duration_seconds, calibration_frames),
                    "is_false_positive": is_false_positive,
                }
            )
            records.append(
                ClipRecord(
                    dataset="nz_thermal",
                    domain=location,
                    clip_id=clip_id,
                    asset_paths=asset_paths,
                    width=width,
                    height=height,
                    framerate=framerate,
                    duration_seconds=duration_seconds,
                    metadata=metadata,
                )
            )
        return records

    def index_clips(
        self,
        split: str | None = None,
        *,
        include_calibration: bool = False,
        clip_ids: Iterable[str] | None = None,
    ) -> dict[str, ClipRecord]:
        return {
            clip.clip_id: clip
            for clip in self.load_clip_records(
                split=split,
                include_calibration=include_calibration,
                clip_ids=clip_ids,
            )
        }

    def build_fixed_pack(
        self,
        pack_name: str,
        *,
        split: str = "test",
        include_calibration: bool = False,
    ) -> list[ClipRecord]:
        return [item["clip"] for item in self._build_fixed_pack_specs(pack_name, split=split, include_calibration=include_calibration)]

    def build_batch_specs(
        self,
        pack_name: str,
        *,
        split: str = "test",
        include_calibration: bool = False,
    ) -> list[dict[str, str]]:
        specs: list[dict[str, str]] = []
        for item in self._build_fixed_pack_specs(pack_name, split=split, include_calibration=include_calibration):
            clip = item["clip"]
            specs.append(
                {
                    "bucket": str(item.get("bucket") or clip.metadata.get("coarse_label") or "other"),
                    "clip_id": clip.clip_id,
                }
            )
        return specs

    @staticmethod
    def truth_payload(clip: ClipRecord) -> dict[str, Any]:
        return {
            "location": clip.metadata.get("location"),
            "split": clip.metadata.get("split"),
            "labels": list(clip.metadata.get("labels") or []),
            "coarse_labels": list(clip.metadata.get("coarse_labels") or []),
            "coarse_label": str(clip.metadata.get("coarse_label") or "other"),
            "is_false_positive": bool(clip.metadata.get("is_false_positive")),
            "event_windows": [dict(item) for item in clip.metadata.get("truth_event_windows") or []],
            "center_zone_entered": bool(clip.metadata.get("truth_center_zone_entered")),
            "center_zone_event_count": int(clip.metadata.get("truth_center_zone_event_count") or 0),
            "center_zone_first_entry_sec": clip.metadata.get("truth_center_zone_first_entry_sec"),
            "center_zone_dwell_sec": clip.metadata.get("truth_center_zone_dwell_sec"),
        }
