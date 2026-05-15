from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.config import LocalSettings, PathsConfig, QwenSettings
from ecp_mllm.experiments.probe_clip import _prepare_sff3c_variant
from ecp_mllm.preprocess.sff3c import (
    build_sff3c_channels,
    params_cache_token,
    params_for_domain,
    process_frame_paths,
    process_frame_paths_with_warmup,
    process_frame_sequence,
)
from ecp_mllm.types import ClipRecord, InputVariant


FRAMES = [
    [[10.0, 20.0], [30.0, 40.0]],
    [[20.0, 30.0], [40.0, 50.0]],
    [[15.0, 25.0], [35.0, 45.0]],
]


class SFF3CTests(unittest.TestCase):
    def test_process_frame_sequence_is_deterministic(self) -> None:
        first = process_frame_sequence(FRAMES, "haida")
        second = process_frame_sequence(FRAMES, "haida")
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertEqual(len(first[0]), 3)

    def test_domain_params_can_be_overridden(self) -> None:
        baseline = params_for_domain("haida")
        tuned = params_for_domain("haida", {"history_factor": 0.8})
        self.assertNotEqual(baseline.history_factor, tuned.history_factor)
        self.assertNotEqual(params_cache_token("haida"), params_cache_token("haida", {"history_factor": 0.8}))

    def test_caltech_kenai_preset_uses_requested_tuning(self) -> None:
        tuned = params_for_domain("kenai-val")
        self.assertEqual(tuned.gaussian_blur, (3, 3))
        self.assertEqual(tuned.gaussian_sigma, 1.5)
        self.assertEqual(tuned.history_factor, 0.6)
        self.assertEqual(tuned.guided_filter_radius, 10)
        self.assertEqual(tuned.guided_filter_eps, 0.01)

    def test_process_frame_paths_writes_rgb_outputs_in_input_order(self) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            self.fail(f"Pillow should be available for SFF3C disk processing tests: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_paths: list[Path] = []
            for index, value in enumerate((32, 96, 160)):
                image = Image.new("L", (2, 2), color=value)
                path = root / f"frame_{index}.png"
                image.save(path)
                input_paths.append(path)

            output_dir, output_paths = process_frame_paths(input_paths, root / "out", "haida")
            self.assertEqual(output_dir, root / "out")
            self.assertEqual([path.name for path in output_paths], [path.name for path in input_paths])
            self.assertTrue(all(path.exists() for path in output_paths))

            with Image.open(output_paths[0]) as image:
                self.assertEqual(image.mode, "RGB")

    def test_process_frame_paths_with_warmup_writes_only_selected_targets(self) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            self.fail(f"Pillow should be available for SFF3C disk processing tests: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_paths: list[Path] = []
            for index, value in enumerate((16, 32, 64, 96, 128)):
                image = Image.new("L", (2, 2), color=value)
                path = root / f"frame_{index}.png"
                image.save(path)
                input_paths.append(path)

            output_dir, output_paths = process_frame_paths_with_warmup(
                input_paths,
                selected_indices=[2, 4],
                output_dir=root / "out_warmup",
                domain="haida",
                warmup_frames=2,
            )
            self.assertEqual(output_dir, root / "out_warmup")
            self.assertEqual([path.name for path in output_paths], ["frame_2.png", "frame_4.png"])
            self.assertTrue(all(path.exists() for path in output_paths))

            with Image.open(output_paths[0]) as image:
                self.assertEqual(image.mode, "RGB")

    def test_sampled_probe_preprocesses_full_clip_before_sampling(self) -> None:
        try:
            from PIL import Image
        except ImportError as exc:
            self.fail(f"Pillow should be available for SFF3C disk processing tests: {exc}")

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_dir = root / "raw"
            raw_dir.mkdir()
            raw_paths: list[Path] = []
            for index, value in enumerate((16, 32, 64, 96, 128, 160)):
                image = Image.new("L", (2, 2), color=value)
                path = raw_dir / f"frame_{index}.png"
                image.save(path)
                raw_paths.append(path)

            settings = LocalSettings(
                paths=PathsConfig(output_root=root / "outputs"),
                qwen=QwenSettings(max_frames=2),
            )
            clip = ClipRecord(
                dataset="cfc",
                domain="kenai-val",
                clip_id="clip_001",
                asset_paths={InputVariant.RAW: raw_dir},
                metadata={"frame_paths_by_variant": {"raw": [str(path) for path in raw_paths]}},
            )

            prepared_clip = _prepare_sff3c_variant(settings, clip, "sampled_frames", max_frames=2)
            prepared_dir = prepared_clip.get_asset_path(InputVariant.SFF3C)
            self.assertIsNotNone(prepared_dir)
            self.assertTrue(prepared_dir.is_dir())
            written_paths = sorted(prepared_dir.iterdir())
            self.assertEqual(len(written_paths), len(raw_paths))
            sampled_paths = prepared_clip.metadata["frame_paths_by_variant"]["sff3c"]
            self.assertEqual(len(sampled_paths), 2)
            self.assertTrue(all(Path(path).exists() for path in sampled_paths))


if __name__ == "__main__":
    unittest.main()

