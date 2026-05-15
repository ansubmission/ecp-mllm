from __future__ import annotations

from pathlib import Path
import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.qwen.prompting import build_prompt
from ecp_mllm.types import ClipRecord, InputVariant, PromptRevision


class PromptingTests(unittest.TestCase):
    def _make_clip(self, variant: InputVariant) -> ClipRecord:
        return ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_001",
            asset_paths={variant: Path("dummy_asset")},
            width=640,
            height=480,
            framerate=5.0,
            duration_seconds=10.0,
            metadata={"frame_paths_by_variant": {variant.value: ["a.jpg", "b.jpg"]}},
        )

    def test_prompt_includes_clip_only_and_single_integer_guidance(self) -> None:
        prompt = build_prompt(
            self._make_clip(InputVariant.RAW),
            InputVariant.RAW,
            PromptRevision(version=0, prompt_id="base", prompt_text="Analyze the sonar clip."),
        )
        self.assertIn("Base the answer only on this clip.", prompt)
        self.assertIn("Do not return count ranges", prompt)
        self.assertIn('"scene_assessment": string | null', prompt)
        self.assertIn('"candidate_passages": [', prompt)
        self.assertIn('"peak_simultaneous_count": int | null', prompt)
        self.assertIn('"wave_count": int | null', prompt)
        self.assertIn('"throughput_best_count": int | null', prompt)
        self.assertIn('"rejected_targets": [string]', prompt)
        self.assertIn("First distinguish fish-like targets from debris", prompt)
        self.assertIn("Work in this order: scene assessment, candidate passages, rejected targets, then final counts.", prompt)

    def test_prompt_includes_variant_guidance_for_derived_color_inputs(self) -> None:
        prompt = build_prompt(
            self._make_clip(InputVariant.SFF3C),
            InputVariant.SFF3C,
            PromptRevision(version=0, prompt_id="base", prompt_text="Analyze the sonar clip."),
        )
        self.assertIn("not natural RGB color video", prompt)
        self.assertIn("Yellow, orange, or bright compact blobs are candidate targets only", prompt)


if __name__ == "__main__":
    unittest.main()

