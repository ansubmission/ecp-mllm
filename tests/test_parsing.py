from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.qwen.parsing import parse_passage_prediction


class ParsingTests(unittest.TestCase):
    def test_parse_prediction_with_commentary_fields(self) -> None:
        raw = """
        {
          "scene_assessment": "Moderate clutter with a short burst of coherent rightward motion.",
          "candidate_passages": [
            {
              "timestamp_start_sec": 50.0,
              "timestamp_end_sec": 65.0,
              "direction": "right",
              "estimated_count": 3,
              "peak_simultaneous_count": 2,
              "episode_duration_sec": 15.0,
              "wave_count": 2,
              "throughput_best_count": 3,
              "evidence_note": "Three elongated bright targets move together."
            }
          ],
          "rejected_targets": ["static bottom return near lower edge"],
          "left_count": 1,
          "right_count": 3,
          "upstream_count": null,
          "downstream_count": null,
          "confidence": 0.8,
          "commentary": "The clip appears to show several fish moving rightward.",
          "evidence_summary": "Multiple elongated bright returns move consistently from left to right.",
          "events": []
        }
        """
        prediction = parse_passage_prediction(raw, domain="kenai-val", clip_id="clip_a", prompt_id="probe")
        self.assertTrue(prediction.parse_success)
        self.assertEqual(
            prediction.scene_assessment,
            "Moderate clutter with a short burst of coherent rightward motion.",
        )
        self.assertEqual(len(prediction.candidate_passages), 1)
        self.assertEqual(prediction.candidate_passages[0]["direction"], "right")
        self.assertEqual(prediction.candidate_passages[0]["peak_simultaneous_count"], 2)
        self.assertEqual(prediction.candidate_passages[0]["wave_count"], 2)
        self.assertEqual(prediction.candidate_passages[0]["throughput_best_count"], 3)
        self.assertEqual(prediction.rejected_targets, ["static bottom return near lower edge"])
        self.assertEqual(prediction.left_count, 1)
        self.assertEqual(prediction.right_count, 3)
        self.assertEqual(prediction.commentary, "The clip appears to show several fish moving rightward.")
        self.assertEqual(
            prediction.evidence_summary,
            "Multiple elongated bright returns move consistently from left to right.",
        )

    def test_parse_prediction_preserves_usage_and_cost_metadata(self) -> None:
        raw = '{"left_count": 0, "right_count": 2, "confidence": 0.8, "events": []}'
        prediction = parse_passage_prediction(
            raw,
            domain="kenai-val",
            clip_id="clip_b",
            prompt_id="probe",
            latency_sec=1.25,
            usage_metadata={"promptTokenCount": 123, "candidatesTokenCount": 45},
            estimated_cost_usd=0.0123,
            model_name="gemini-3.1-pro-preview",
        )
        self.assertTrue(prediction.parse_success)
        self.assertEqual(prediction.usage_metadata["promptTokenCount"], 123)
        self.assertEqual(prediction.estimated_cost_usd, 0.0123)
        self.assertEqual(prediction.model_name, "gemini-3.1-pro-preview")


if __name__ == "__main__":
    unittest.main()

