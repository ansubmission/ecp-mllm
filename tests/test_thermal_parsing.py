from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.qwen.thermal_parsing import parse_thermal_prediction


class ThermalParsingTests(unittest.TestCase):
    def test_salvages_positive_animal_from_legacy_counting_schema(self) -> None:
        raw = """
        {
          "candidate_passages": [
            {
              "timestamp_start_sec": 3.5,
              "timestamp_end_sec": 14.0,
              "confidence": 0.85,
              "evidence_note": "Single bright thermal target traverses the frame."
            }
          ],
          "confidence": 0.85,
          "commentary": "The clip shows a single animal event, likely a bird or bat, moving right to left.",
          "evidence_summary": "One bright thermal target appears at 3.5s and exits by 14.0s."
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="test",
            clip_id="1000577",
            prompt_id="thermal_event_agent_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertTrue(prediction.animal_present)
        self.assertEqual(prediction.coarse_label, "other")
        self.assertEqual(len(prediction.event_windows), 1)
        self.assertEqual(prediction.event_windows[0].label, "other")
        self.assertIsNone(prediction.center_zone_entered)

    def test_does_not_salvage_fish_hallucination_as_positive_animal(self) -> None:
        raw = """
        {
          "candidate_passages": [
            {
              "timestamp_start_sec": 1.0,
              "timestamp_end_sec": 9.0,
              "confidence": 0.4,
              "evidence_note": "Faint elongated target."
            }
          ],
          "confidence": 0.4,
          "commentary": "A faint fish-like target drifts through the frame.",
          "evidence_summary": "Low-visibility moving streak."
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="test",
            clip_id="1007168",
            prompt_id="thermal_event_agent_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertIsNone(prediction.animal_present)
        self.assertEqual(prediction.coarse_label, "other")
        self.assertEqual(prediction.event_windows, [])

    def test_salvages_confident_candidate_passages_as_animal_event(self) -> None:
        raw = """
        {
          "candidate_passages": [
            {
              "timestamp_start_sec": 3.4,
              "timestamp_end_sec": 14.0,
              "evidence_note": "Single bright target visible with clear leftward trajectory."
            }
          ],
          "confidence": 0.6,
          "commentary": "A single coherent target moves from right to left over approximately 10 seconds.",
          "evidence_summary": "One bright target visible from 3.4s to 14.0s with clear leftward trajectory."
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="test",
            clip_id="1000577",
            prompt_id="thermal_event_agent_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertTrue(prediction.animal_present)
        self.assertEqual(prediction.coarse_label, "other")
        self.assertEqual(len(prediction.event_windows), 1)

    def test_parses_center_zone_fields_when_present(self) -> None:
        raw = """
        {
          "clip_labels": ["rodent"],
          "coarse_label": "rodent",
          "animal_present": true,
          "false_positive_score": 0.1,
          "center_zone_entered": true,
          "center_zone_first_entry_sec": 2.4,
          "center_zone_dwell_sec": 1.7,
          "event_windows": [],
          "event_labels": ["rodent"],
          "confidence": 0.8,
          "abstain": false
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="test",
            clip_id="1001878",
            prompt_id="thermal_event_agent_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertTrue(prediction.center_zone_entered)
        self.assertAlmostEqual(float(prediction.center_zone_first_entry_sec or 0.0), 2.4)
        self.assertAlmostEqual(float(prediction.center_zone_dwell_sec or 0.0), 1.7)

    def test_salvages_negative_center_zone_from_commentary(self) -> None:
        raw = """
        {
          "clip_labels": ["rodent"],
          "coarse_label": "rodent",
          "animal_present": true,
          "false_positive_score": 0.2,
          "event_windows": [],
          "event_labels": ["rodent"],
          "confidence": 0.7,
          "commentary": "A single rodent remains near the left edge and does not enter the center of the frame.",
          "evidence_summary": "Warm body stays near the left edge for the full clip."
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="test",
            clip_id="1001878",
            prompt_id="thermal_event_agent_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertFalse(bool(prediction.center_zone_entered))
        self.assertIsNone(prediction.center_zone_first_entry_sec)

    def test_salvages_positive_center_zone_from_commentary(self) -> None:
        raw = """
        {
          "clip_labels": ["other"],
          "coarse_label": "other",
          "animal_present": true,
          "false_positive_score": 0.1,
          "event_windows": [],
          "event_labels": ["other"],
          "confidence": 0.7,
          "commentary": "One animal passes through the middle of the frame before exiting.",
          "evidence_summary": "The target crosses the center of the frame around 6 seconds."
        }
        """
        prediction = parse_thermal_prediction(
            raw,
            domain="test",
            clip_id="1000577",
            prompt_id="thermal_event_agent_global",
        )
        self.assertTrue(prediction.parse_success)
        self.assertTrue(bool(prediction.center_zone_entered))


if __name__ == "__main__":
    unittest.main()
