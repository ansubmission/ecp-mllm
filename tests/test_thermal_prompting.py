from __future__ import annotations

from pathlib import Path
import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.qwen.thermal_parsing import parse_thermal_prediction
from ecp_mllm.qwen.thermal_prompting import build_thermal_prompt
from ecp_mllm.types import ClipRecord, InputVariant, PromptRevision


class ThermalPromptingTests(unittest.TestCase):
    def _make_clip(self, variant: InputVariant) -> ClipRecord:
        return ClipRecord(
            dataset="nz_thermal",
            domain="loc_a",
            clip_id="clip_001",
            asset_paths={variant: Path("clip.mp4")},
            width=160,
            height=120,
            framerate=9.0,
            duration_seconds=8.0,
            metadata={"split": "test", "calibration_frames": []},
        )

    def test_prompt_includes_thermal_schema_and_dual_guidance(self) -> None:
        prompt = build_thermal_prompt(
            self._make_clip(InputVariant.THERMAL_DUAL),
            InputVariant.THERMAL_DUAL,
            PromptRevision(version=0, prompt_id="thermal", prompt_text="Analyze the clip."),
        )
        self.assertIn('"clip_labels": [string]', prompt)
        self.assertIn('"coarse_label": "false_positive|bird|rodent|possum|cat|hedgehog|mustelid|other"', prompt)
        self.assertIn('"center_zone_entered": bool | null', prompt)
        self.assertIn('"center_zone_first_entry_sec": float | null', prompt)
        self.assertIn("filtered on the left, normalized on the right", prompt)
        self.assertIn("false positive trigger", prompt)
        self.assertIn("NOT a fish-counting or line-crossing task", prompt)
        self.assertIn("Do not return count fields", prompt)
        self.assertIn("do NOT require translational motion", prompt)
        self.assertIn("stationary. Use false_positive only for artifacts", prompt)
        self.assertIn('coarse_label="other"', prompt)
        self.assertIn("center zone", prompt)
        self.assertIn("mandatory", prompt)
        self.assertIn("center_zone_entered=false", prompt)

    def test_prompt_mentions_drawn_center_zone_overlay_when_present(self) -> None:
        clip = self._make_clip(InputVariant.THERMAL_FILTERED)
        clip.metadata["zone_overlay"] = {"type": "center", "video_path": "clip.centerzone.mp4"}
        prompt = build_thermal_prompt(
            clip,
            InputVariant.THERMAL_FILTERED,
            PromptRevision(version=0, prompt_id="thermal", prompt_text="Analyze the clip."),
        )
        self.assertIn("bright green box", prompt)
        self.assertIn("authoritative center-zone boundary", prompt)

    def test_parse_thermal_prediction_reads_event_windows(self) -> None:
        raw = """
        {
          "clip_labels": ["bird"],
          "coarse_label": "bird",
          "animal_present": true,
          "false_positive_score": 0.1,
          "center_zone_entered": true,
          "center_zone_first_entry_sec": 1.5,
          "center_zone_dwell_sec": 1.2,
          "event_windows": [
            {
              "timestamp_start_sec": 1.2,
              "timestamp_end_sec": 3.6,
              "label": "bird",
              "confidence": 0.9,
              "false_positive_score": 0.1,
              "evidence_note": "hot moving body with bird-like gait"
            }
          ],
          "event_labels": ["bird"],
          "confidence": 0.88,
          "abstain": false,
          "commentary": "One bird walks through the scene.",
          "evidence_summary": "Compact warm body crosses from left to right."
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="loc_a",
            clip_id="clip_001",
            prompt_id="thermal_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertEqual(prediction.coarse_label, "bird")
        self.assertEqual(prediction.clip_labels, ["bird"])
        self.assertTrue(prediction.center_zone_entered)
        self.assertAlmostEqual(float(prediction.center_zone_first_entry_sec or 0.0), 1.5)
        self.assertEqual(len(prediction.event_windows), 1)
        self.assertAlmostEqual(prediction.event_windows[0].timestamp_start_sec, 1.2)
        self.assertEqual(prediction.event_labels, ["bird"])

    def test_parse_failure_returns_abstaining_prediction(self) -> None:
        prediction = parse_thermal_prediction(
            "not-json",
            domain="loc_a",
            clip_id="clip_001",
            prompt_id="thermal_global",
        )
        self.assertFalse(prediction.parse_success)
        self.assertTrue(prediction.abstain)
        self.assertEqual(prediction.coarse_label, "other")

    def test_parse_salvages_false_positive_from_counting_style_response(self) -> None:
        raw = """
        {
          "scene_assessment": "The clip shows a static thermal landscape at night.",
          "candidate_passages": [],
          "left_count": 0,
          "right_count": 0,
          "confidence": 0.95,
          "commentary": "No animal motion is detected in this clip. The only target is a stationary bright spot, characteristic of a false positive trigger.",
          "evidence_summary": "A stationary bright artifact remains fixed throughout the clip.",
          "events": []
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="loc_a",
            clip_id="clip_001",
            prompt_id="thermal_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertEqual(prediction.coarse_label, "false_positive")
        self.assertEqual(prediction.clip_labels, ["false_positive"])
        self.assertFalse(prediction.animal_present)
        self.assertGreaterEqual(float(prediction.false_positive_score or 0.0), 0.9)


if __name__ == "__main__":
    unittest.main()
