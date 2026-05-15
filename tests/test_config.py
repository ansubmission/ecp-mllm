from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.config import load_local_settings
from ecp_mllm.types import InputVariant


class ConfigTests(unittest.TestCase):
    def test_load_local_settings_and_path_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            metadata_dir = root / "metadata"
            annotations_dir = root / "annotations"
            raw_dir = root / "raw"
            data_dir = root / "cfc"
            metadata_dir.mkdir()
            annotations_dir.mkdir()
            raw_dir.mkdir()
            repo_dir = root / "repo"
            repo_dir.mkdir()
            local_toml = root / "local.toml"
            local_toml.write_text(
                "\n".join(
                    [
                        "[paths]",
                        f'cfc_repo_path = "{repo_dir.as_posix()}"',
                        f'cfc_data_path = "{data_dir.as_posix()}"',
                        f'cfc_metadata_path = "{metadata_dir.as_posix()}"',
                        f'cfc_annotations_path = "{annotations_dir.as_posix()}"',
                        f'cfc_raw_root = "{raw_dir.as_posix()}"',
                        "",
                        "[qwen]",
                        'provider = "mock"',
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_local_settings(local_toml)
            self.assertEqual(settings.qwen.provider, "mock")
            self.assertEqual(settings.qwen.max_frames, 16)
            self.assertIsNone(settings.qwen.thinking_level)
            self.assertEqual(settings.paths.resolved_cfc_metadata_path(), metadata_dir)
            self.assertEqual(settings.paths.resolved_cfc_annotations_path(), annotations_dir)
            self.assertEqual(settings.paths.resolved_variant_root(InputVariant.RAW), raw_dir)

    def test_auto_detects_extracted_3channel_root_under_images(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "cfc"
            extracted_dir = data_dir / "images" / "kenai-3channel"
            extracted_dir.mkdir(parents=True)
            local_toml = root / "local.toml"
            local_toml.write_text(
                "\n".join(
                    [
                        "[paths]",
                        f'cfc_data_path = "{data_dir.as_posix()}"',
                        "",
                        "[qwen]",
                        'provider = "mock"',
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_local_settings(local_toml)
            self.assertEqual(settings.paths.resolved_variant_root(InputVariant.CFC_3CHANNEL), extracted_dir)

    def test_loads_qwen_thinking_level(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            local_toml = root / "local.toml"
            local_toml.write_text(
                "\n".join(
                    [
                        "[paths]",
                        "",
                        "[qwen]",
                        'provider = "gemini_api"',
                        'thinking_level = "HIGH"',
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_local_settings(local_toml)
            self.assertEqual(settings.qwen.provider, "gemini_api")
            self.assertEqual(settings.qwen.thinking_level, "high")

    def test_loads_nz_thermal_paths_and_variant_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "thermal"
            filtered_dir = data_dir / "filtered"
            normalized_dir = data_dir / "normalized"
            metadata_path = data_dir / "metadata.json"
            splits_path = data_dir / "splits.json"
            filtered_dir.mkdir(parents=True)
            normalized_dir.mkdir(parents=True)
            metadata_path.write_text("[]", encoding="utf-8")
            splits_path.write_text("[]", encoding="utf-8")
            local_toml = root / "local.toml"
            local_toml.write_text(
                "\n".join(
                    [
                        "[paths]",
                        f'nz_thermal_data_path = "{data_dir.as_posix()}"',
                        f'nz_thermal_metadata_path = "{metadata_path.as_posix()}"',
                        f'nz_thermal_splits_path = "{splits_path.as_posix()}"',
                        "",
                        "[qwen]",
                        'provider = "mock"',
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_local_settings(local_toml)
            self.assertEqual(settings.paths.resolved_nz_thermal_metadata_path(), metadata_path)
            self.assertEqual(settings.paths.resolved_nz_thermal_splits_path(), splits_path)
            self.assertEqual(settings.paths.resolved_variant_root(InputVariant.THERMAL_FILTERED), filtered_dir)
            self.assertEqual(settings.paths.resolved_variant_root(InputVariant.THERMAL_NORMALIZED), normalized_dir)

    def test_auto_detects_official_nz_thermal_zip_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "thermal"
            videos_dir = data_dir / "videos"
            individual_dir = data_dir / "individual-metadata"
            metadata_path = data_dir / "new-zealand-wildlife-thermal-imaging.json"
            splits_path = data_dir / "splits.json"
            videos_dir.mkdir(parents=True)
            individual_dir.mkdir(parents=True)
            metadata_path.write_text("[]", encoding="utf-8")
            splits_path.write_text("[]", encoding="utf-8")
            local_toml = root / "local.toml"
            local_toml.write_text(
                "\n".join(
                    [
                        "[paths]",
                        f'nz_thermal_data_path = "{data_dir.as_posix()}"',
                        f'nz_thermal_splits_path = "{splits_path.as_posix()}"',
                        "",
                        "[qwen]",
                        'provider = "mock"',
                    ]
                ),
                encoding="utf-8",
            )
            settings = load_local_settings(local_toml)
            self.assertEqual(settings.paths.resolved_nz_thermal_metadata_path(), metadata_path)
            self.assertEqual(settings.paths.resolved_nz_thermal_clip_metadata_path(), individual_dir)
            self.assertEqual(settings.paths.resolved_variant_root(InputVariant.THERMAL_FILTERED), videos_dir)
            self.assertEqual(settings.paths.resolved_variant_root(InputVariant.THERMAL_NORMALIZED), videos_dir)


if __name__ == "__main__":
    unittest.main()

