from __future__ import annotations

import collections
import json
from pathlib import Path
import struct
from typing import Iterable

from ..config import PathsConfig
from ..eval.counting import count_tracks_like_cfc, read_mot_tracks
from ..types import ClipRecord, InputVariant, UpstreamDirection, WeakCountRecord


LOCATION_IMAGE_DIR = {
    "kenai-train": "kenai",
    "kenai-val": "kenai",
    "kenai-rightbank": "rightbank",
    "kenai-channel": "channel",
    "nushagak": "nushagak",
    "elwha": "elwha",
}


class CFCAdapter:
    def __init__(self, paths: PathsConfig) -> None:
        self.paths = paths

    def _existing_metadata_dir(self) -> Path | None:
        path = self.paths.resolved_cfc_metadata_path()
        if path is None or not path.exists():
            return None
        return path

    def _existing_file_lists_dir(self) -> Path | None:
        if self.paths.cfc_data_path is None:
            return None
        path = self.paths.cfc_data_path / "file_lists"
        if not path.exists():
            return None
        return path

    def metadata_dir(self) -> Path:
        path = self._existing_metadata_dir()
        if path is None:
            raise RuntimeError("CFC metadata path is not configured")
        return path

    def annotations_dir(self) -> Path:
        path = self.paths.resolved_cfc_annotations_path()
        if path is None:
            raise RuntimeError("CFC annotations path is not configured")
        return path

    def list_locations(self) -> list[str]:
        metadata_dir = self._existing_metadata_dir()
        if metadata_dir is not None:
            return sorted(path.stem for path in metadata_dir.glob("*.json"))
        annotations_dir = self.annotations_dir()
        if annotations_dir.exists():
            return sorted(path.name for path in annotations_dir.iterdir() if path.is_dir())
        file_lists_dir = self._existing_file_lists_dir()
        if file_lists_dir is not None:
            return sorted(
                path.stem
                for path in file_lists_dir.glob("*.txt")
                if path.stem in LOCATION_IMAGE_DIR
            )
        return []

    def _file_list_path(self, location: str) -> Path | None:
        file_lists_dir = self._existing_file_lists_dir()
        if file_lists_dir is None:
            return None
        path = file_lists_dir / f"{location}.txt"
        if not path.exists():
            return None
        return path

    def _normalize_file_list_entry(self, value: str) -> Path:
        parts = Path(value.strip().replace("\\", "/")).parts
        if "images" in parts:
            return Path(*parts[parts.index("images") + 1 :])
        return Path(*parts)

    def _load_location_file_lists(self, location: str) -> dict[str, list[Path]]:
        file_list_path = self._file_list_path(location)
        if file_list_path is None:
            return {}
        grouped: dict[str, list[Path]] = collections.OrderedDict()
        for raw_line in file_list_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            relative_path = self._normalize_file_list_entry(line)
            clip_name = relative_path.stem.rsplit("_", 1)[0]
            grouped.setdefault(clip_name, []).append(relative_path)
        return grouped

    def _resolve_variant_frame_paths(self, variant_root: Path | None, relative_paths: list[Path]) -> list[Path]:
        if variant_root is None:
            return []
        return [variant_root / relative_path for relative_path in relative_paths if (variant_root / relative_path).exists()]

    def _read_image_size(self, path: Path) -> tuple[int, int]:
        with path.open("rb") as handle:
            signature = handle.read(26)
            if signature.startswith(b"\x89PNG\r\n\x1a\n"):
                width, height = struct.unpack(">II", signature[16:24])
                return int(width), int(height)
            if signature[:2] == b"\xff\xd8":
                handle.seek(2)
                while True:
                    marker = handle.read(1)
                    while marker == b"\xff":
                        marker = handle.read(1)
                    if not marker:
                        break
                    marker_code = marker[0]
                    while marker_code == 0xFF:
                        marker = handle.read(1)
                        if not marker:
                            break
                        marker_code = marker[0]
                    size_bytes = handle.read(2)
                    if len(size_bytes) != 2:
                        break
                    segment_size = struct.unpack(">H", size_bytes)[0]
                    if marker_code in {0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7, 0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF}:
                        _precision = handle.read(1)
                        height, width = struct.unpack(">HH", handle.read(4))
                        return int(width), int(height)
                    handle.seek(max(0, segment_size - 2), 1)
        raise ValueError(f"Unable to infer image dimensions from {path}")

    @staticmethod
    def _optional_int(value: object | None) -> int | None:
        if value in (None, ""):
            return None
        return int(value)

    @staticmethod
    def _optional_float(value: object | None) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    def _enrich_entries_with_frame_paths(
        self,
        location: str,
        entries: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        frame_lists_by_clip = self._load_location_file_lists(location)
        if not frame_lists_by_clip:
            return [dict(item) for item in entries]

        raw_root = self.paths.resolved_variant_root(InputVariant.RAW)
        cfc_3channel_root = self.paths.resolved_variant_root(InputVariant.CFC_3CHANNEL)
        entry_order = [str(item["clip_name"]) for item in entries]
        entry_map = {str(item["clip_name"]): dict(item) for item in entries}

        for clip_name, relative_paths in frame_lists_by_clip.items():
            entry = entry_map.get(clip_name, {"clip_name": clip_name})
            raw_frame_paths = self._resolve_variant_frame_paths(raw_root, relative_paths)
            cfc_frame_paths = self._resolve_variant_frame_paths(cfc_3channel_root, relative_paths)
            frame_paths_by_variant = {
                variant.value: [str(path) for path in paths]
                for variant, paths in (
                    (InputVariant.RAW, raw_frame_paths),
                    (InputVariant.CFC_3CHANNEL, cfc_frame_paths),
                )
                if paths
            }
            if frame_paths_by_variant:
                entry["frame_paths_by_variant"] = frame_paths_by_variant
            if entry.get("num_frames") in (None, ""):
                entry["num_frames"] = len(raw_frame_paths) if raw_frame_paths else len(relative_paths)
            if entry.get("width") in (None, "") or entry.get("height") in (None, ""):
                probe_path = raw_frame_paths[0] if raw_frame_paths else (cfc_frame_paths[0] if cfc_frame_paths else None)
                if probe_path is not None:
                    width, height = self._read_image_size(probe_path)
                    entry.setdefault("width", width)
                    entry.setdefault("height", height)
            entry_map[clip_name] = entry
            if clip_name not in entry_order:
                entry_order.append(clip_name)
        return [entry_map[clip_name] for clip_name in entry_order]

    def load_location_metadata(self, location: str) -> list[dict[str, object]]:
        metadata_dir = self._existing_metadata_dir()
        if metadata_dir is not None:
            metadata_path = metadata_dir / f"{location}.json"
            if metadata_path.exists():
                entries = json.loads(metadata_path.read_text(encoding="utf-8"))
                return self._enrich_entries_with_frame_paths(location, entries)
        return self._enrich_entries_with_frame_paths(location, [])

    def load_clip_records(self, locations: Iterable[str] | None = None) -> list[ClipRecord]:
        selected_locations = list(locations) if locations is not None else self.list_locations()
        raw_root = self.paths.resolved_variant_root(InputVariant.RAW)
        cfc_3channel_root = self.paths.resolved_variant_root(InputVariant.CFC_3CHANNEL)
        clips: list[ClipRecord] = []
        for location in selected_locations:
            image_dir_name = LOCATION_IMAGE_DIR.get(location, location)
            for item in self.load_location_metadata(location):
                clip_name = str(item["clip_name"])
                metadata = dict(item)
                frame_paths_by_variant = metadata.get("frame_paths_by_variant", {})
                asset_paths = {}
                if raw_root is not None:
                    raw_dir = raw_root / image_dir_name / clip_name
                    if raw_dir.exists():
                        asset_paths[InputVariant.RAW] = raw_dir
                    elif isinstance(frame_paths_by_variant, dict) and frame_paths_by_variant.get(InputVariant.RAW.value):
                        asset_paths[InputVariant.RAW] = Path(frame_paths_by_variant[InputVariant.RAW.value][0])
                if cfc_3channel_root is not None:
                    cfc_dir = cfc_3channel_root / image_dir_name / clip_name
                    if cfc_dir.exists():
                        asset_paths[InputVariant.CFC_3CHANNEL] = cfc_dir
                    elif isinstance(frame_paths_by_variant, dict) and frame_paths_by_variant.get(InputVariant.CFC_3CHANNEL.value):
                        asset_paths[InputVariant.CFC_3CHANNEL] = Path(frame_paths_by_variant[InputVariant.CFC_3CHANNEL.value][0])
                clips.append(
                    ClipRecord(
                        dataset="cfc",
                        domain=location,
                        clip_id=clip_name,
                        asset_paths=asset_paths,
                        upstream_direction=UpstreamDirection.from_optional(
                            str(item.get("upstream_direction")) if item.get("upstream_direction") is not None else None
                        ),
                        width=self._optional_int(item.get("width")),
                        height=self._optional_int(item.get("height")),
                        framerate=self._optional_float(item.get("framerate")),
                        duration_seconds=(
                            self._optional_float(item.get("num_frames")) / self._optional_float(item.get("framerate"))
                            if self._optional_float(item.get("num_frames")) is not None
                            and self._optional_float(item.get("framerate")) not in (None, 0.0)
                            else None
                        ),
                        metadata=metadata,
                    )
                )
        return clips

    def gt_path(self, location: str, clip_id: str) -> Path:
        return self.annotations_dir() / location / clip_id / "gt.txt"

    def derive_ground_truth_counts(self, locations: Iterable[str] | None = None, filter_dist: float = 0.05) -> list[WeakCountRecord]:
        labels: list[WeakCountRecord] = []
        for clip in self.load_clip_records(locations):
            if clip.width is None or clip.height is None:
                raise ValueError(f"CFC clip {clip.key.value} is missing width/height metadata")
            tracks = read_mot_tracks(self.gt_path(clip.domain, clip.clip_id))
            counts = count_tracks_like_cfc(tracks, clip.width, clip.height, filter_dist=filter_dist)
            labels.append(
                WeakCountRecord(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    counts=counts,
                    source_type="mot",
                    upstream_direction=clip.upstream_direction,
                    source_path=self.gt_path(clip.domain, clip.clip_id),
                )
            )
        return labels
