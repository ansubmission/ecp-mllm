from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib

from .types import InputVariant


def _path_or_none(value: str | None) -> Path | None:
    if value is None or str(value).strip() == "":
        return None
    return Path(value).expanduser()


def _first_existing(base: Path | None, candidates: list[str]) -> Path | None:
    if base is None:
        return None
    for candidate in candidates:
        path = base / candidate
        if path.exists():
            return path
    return base / candidates[0]


def _first_matching_child(base: Path | None, relative_parent: str, suffix: str) -> Path | None:
    if base is None:
        return None
    parent = base / relative_parent
    if not parent.exists():
        return None
    matches = sorted(path for path in parent.iterdir() if path.is_dir() and path.name.endswith(suffix))
    return matches[0] if matches else None


@dataclass(frozen=True)
class PathsConfig:
    cfc_repo_path: Path | None = None
    cfc_data_path: Path | None = None
    cfc_metadata_path: Path | None = None
    cfc_annotations_path: Path | None = None
    cfc_raw_root: Path | None = None
    cfc_3channel_root: Path | None = None
    nz_thermal_data_path: Path | None = None
    nz_thermal_metadata_path: Path | None = None
    nz_thermal_clip_metadata_path: Path | None = None
    nz_thermal_splits_path: Path | None = None
    nz_thermal_filtered_root: Path | None = None
    nz_thermal_normalized_root: Path | None = None
    private_manifest_path: Path | None = None
    private_counts_path: Path | None = None
    output_root: Path = Path("outputs")

    def resolved_cfc_metadata_path(self) -> Path | None:
        return self.cfc_metadata_path or _first_existing(self.cfc_data_path, ["metadata", "cfc_v1.1/metadata"])

    def resolved_cfc_annotations_path(self) -> Path | None:
        return self.cfc_annotations_path or _first_existing(self.cfc_data_path, ["annotations", "cfc_v1.1/annotations"])

    def resolved_nz_thermal_metadata_path(self) -> Path | None:
        return self.nz_thermal_metadata_path or _first_existing(
            self.nz_thermal_data_path,
            [
                "new-zealand-wildlife-thermal-imaging.json",
                "metadata.json",
                "metadata.jsonl",
                "clips-metadata.json",
                "clips-metadata.jsonl",
                "metadata",
            ],
        )

    def resolved_nz_thermal_clip_metadata_path(self) -> Path | None:
        return self.nz_thermal_clip_metadata_path or _first_existing(
            self.nz_thermal_data_path,
            [
                "individual-metadata",
                "individual_metadata",
                "clip-metadata.json",
                "clip-metadata.jsonl",
                "clip_metadata.json",
                "clip_metadata.jsonl",
                "clip_metadata",
            ],
        )

    def resolved_nz_thermal_splits_path(self) -> Path | None:
        return self.nz_thermal_splits_path or _first_existing(
            self.nz_thermal_data_path,
            [
                "recommended_splits.json",
                "recommended_splits.jsonl",
                "splits.json",
                "splits.jsonl",
                "splits.csv",
                "splits",
            ],
        )

    def resolved_variant_root(self, variant: InputVariant) -> Path | None:
        if variant == InputVariant.RAW:
            return self.cfc_raw_root or _first_existing(self.cfc_data_path, ["raw", "images", "cfc_v1.1/images"])
        if variant == InputVariant.CFC_3CHANNEL:
            if self.cfc_3channel_root is not None:
                return self.cfc_3channel_root
            direct_path = _first_existing(
                self.cfc_data_path,
                ["3channel", "images_3channel", "cfc_v1.1/images_3channel"],
            )
            if direct_path is not None and direct_path.exists():
                return direct_path
            extracted_path = _first_matching_child(self.cfc_data_path, "images", "-3channel")
            if extracted_path is not None:
                return extracted_path
            return direct_path
        if variant == InputVariant.THERMAL_FILTERED:
            return self.nz_thermal_filtered_root or _first_existing(
                self.nz_thermal_data_path,
                [
                    "videos",
                    "filtered",
                    "filtered_videos",
                    "videos/filtered",
                    "clips/filtered",
                ],
            )
        if variant == InputVariant.THERMAL_NORMALIZED:
            return self.nz_thermal_normalized_root or _first_existing(
                self.nz_thermal_data_path,
                [
                    "videos",
                    "normalized",
                    "normalized_videos",
                    "videos/normalized",
                    "clips/normalized",
                ],
            )
        return None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.cfc_repo_path and not self.cfc_repo_path.exists():
            errors.append(f"cfc_repo_path does not exist: {self.cfc_repo_path}")
        if self.cfc_data_path and not self.cfc_data_path.exists():
            errors.append(f"cfc_data_path does not exist: {self.cfc_data_path}")
        if self.nz_thermal_data_path and not self.nz_thermal_data_path.exists():
            errors.append(f"nz_thermal_data_path does not exist: {self.nz_thermal_data_path}")
        for path in [self.private_manifest_path, self.private_counts_path]:
            if path and not path.exists():
                errors.append(f"path does not exist: {path}")
        return errors


@dataclass(frozen=True)
class QwenSettings:
    provider: str = "mock"
    model: str = "qwen2.5-vl-7b-instruct"
    scripted_responses_path: Path | None = None
    api_key: str | None = None
    api_key_env: str = "DASHSCOPE_API_KEY"
    base_url: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    timeout_seconds: int = 120
    max_frames: int = 16
    temperature: float = 0.1
    thinking_level: str | None = None
    prefer_temporary_oss_upload: bool | None = None


@dataclass(frozen=True)
class LocalSettings:
    paths: PathsConfig
    qwen: QwenSettings


def load_local_settings(path: str | Path) -> LocalSettings:
    data = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    paths_section = data.get("paths", {})
    qwen_section = data.get("qwen", {})
    paths = PathsConfig(
        cfc_repo_path=_path_or_none(paths_section.get("cfc_repo_path")),
        cfc_data_path=_path_or_none(paths_section.get("cfc_data_path")),
        cfc_metadata_path=_path_or_none(paths_section.get("cfc_metadata_path")),
        cfc_annotations_path=_path_or_none(paths_section.get("cfc_annotations_path")),
        cfc_raw_root=_path_or_none(paths_section.get("cfc_raw_root")),
        cfc_3channel_root=_path_or_none(paths_section.get("cfc_3channel_root")),
        nz_thermal_data_path=_path_or_none(paths_section.get("nz_thermal_data_path")),
        nz_thermal_metadata_path=_path_or_none(paths_section.get("nz_thermal_metadata_path")),
        nz_thermal_clip_metadata_path=_path_or_none(paths_section.get("nz_thermal_clip_metadata_path")),
        nz_thermal_splits_path=_path_or_none(paths_section.get("nz_thermal_splits_path")),
        nz_thermal_filtered_root=_path_or_none(paths_section.get("nz_thermal_filtered_root")),
        nz_thermal_normalized_root=_path_or_none(paths_section.get("nz_thermal_normalized_root")),
        private_manifest_path=_path_or_none(paths_section.get("private_manifest_path")),
        private_counts_path=_path_or_none(paths_section.get("private_counts_path")),
        output_root=_path_or_none(paths_section.get("output_root")) or Path("outputs"),
    )
    qwen = QwenSettings(
        provider=str(qwen_section.get("provider", "mock")).strip().lower(),
        model=str(qwen_section.get("model", "qwen2.5-vl-7b-instruct")).strip(),
        scripted_responses_path=_path_or_none(qwen_section.get("scripted_responses_path")),
        api_key=(str(qwen_section.get("api_key")).strip() if qwen_section.get("api_key") not in (None, "") else None),
        api_key_env=str(qwen_section.get("api_key_env", "DASHSCOPE_API_KEY")).strip(),
        base_url=qwen_section.get("base_url") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        timeout_seconds=int(qwen_section.get("timeout_seconds", 120)),
        max_frames=int(qwen_section.get("max_frames", 16)),
        temperature=float(qwen_section.get("temperature", 0.1)),
        thinking_level=(
            str(qwen_section.get("thinking_level")).strip().lower()
            if qwen_section.get("thinking_level") not in (None, "")
            else None
        ),
        prefer_temporary_oss_upload=(
            bool(qwen_section.get("prefer_temporary_oss_upload"))
            if "prefer_temporary_oss_upload" in qwen_section
            else None
        ),
    )
    return LocalSettings(paths=paths, qwen=qwen)
