from __future__ import annotations

from csv import DictReader
from pathlib import Path
from typing import Iterable

from ..types import ClipKey, ClipRecord, InputVariant, UpstreamDirection


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(float(value))


def _asset_paths_from_row(row: dict[str, str]) -> dict[InputVariant, Path]:
    mapping: dict[InputVariant, Path] = {}
    for column, variant in {
        "raw_path": InputVariant.RAW,
        "sff3c_path": InputVariant.SFF3C,
        "cfc_3channel_path": InputVariant.CFC_3CHANNEL,
    }.items():
        value = row.get(column)
        if value:
            mapping[variant] = Path(value).expanduser()
    return mapping


def load_private_clip_manifest(path: str | Path) -> list[ClipRecord]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = DictReader(handle)
        clips: list[ClipRecord] = []
        for row in reader:
            clips.append(
                ClipRecord(
                    dataset="private",
                    domain=row["domain"],
                    clip_id=row["clip_id"],
                    asset_paths=_asset_paths_from_row(row),
                    upstream_direction=UpstreamDirection.from_optional(row.get("upstream_direction")),
                    width=_optional_int(row.get("width")),
                    height=_optional_int(row.get("height")),
                    framerate=_optional_float(row.get("framerate")),
                    duration_seconds=_optional_float(row.get("duration_seconds")),
                    metadata={k: v for k, v in row.items() if k not in {"domain", "clip_id"}},
                )
            )
    return clips


def index_clips(clips: Iterable[ClipRecord]) -> dict[ClipKey, ClipRecord]:
    return {clip.key: clip for clip in clips}

