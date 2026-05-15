from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.agent.undercount_recount import (
    build_recount_prompt,
    detect_undercount_risk,
    should_accept_recount,
)
from ecp_mllm.types import PassagePrediction, PromptRevision


class UndercountRecountTests(unittest.TestCase):
    def test_detects_sustained_single_episode_undercount_risk(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_a",
            left_count=0,
            right_count=7,
            candidate_passages=[
                {
                    "timestamp_start_sec": 50.0,
                    "timestamp_end_sec": 75.0,
                    "direction": "right",
                    "estimated_count": 7,
                    "peak_simultaneous_count": 3,
                    "evidence_note": "continuous stream",
                }
            ],
            commentary="continuous school-like stream",
            evidence_summary="sequential entry of fish over many seconds",
            parse_success=True,
        )
        decision = detect_undercount_risk(prediction)
        self.assertTrue(decision.should_recount)
        self.assertEqual(decision.dominant_direction, "right")
        self.assertEqual(decision.dominant_episode_count, 7)
        self.assertAlmostEqual(decision.dominant_episode_duration_sec, 25.0)

    def test_skips_recount_for_split_small_events(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_b",
            left_count=0,
            right_count=4,
            candidate_passages=[
                {"timestamp_start_sec": 0.0, "timestamp_end_sec": 10.0, "direction": "right", "estimated_count": 1},
                {"timestamp_start_sec": 45.0, "timestamp_end_sec": 70.0, "direction": "right", "estimated_count": 2},
                {"timestamp_start_sec": 75.0, "timestamp_end_sec": 95.0, "direction": "right", "estimated_count": 1},
            ],
            parse_success=True,
        )
        decision = detect_undercount_risk(prediction)
        self.assertFalse(decision.should_recount)

    def test_skips_recount_for_single_sparse_peak_one_stream(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_b1",
            left_count=0,
            right_count=5,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 108.0,
                    "direction": "right",
                    "estimated_count": 5,
                    "peak_simultaneous_count": 1,
                    "wave_count": 1,
                }
            ],
            parse_success=True,
        )
        decision = detect_undercount_risk(prediction)
        self.assertFalse(decision.should_recount)

    def test_detects_multiwave_underflow_risk(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_b2",
            left_count=0,
            right_count=5,
            candidate_passages=[
                {
                    "timestamp_start_sec": 20.0,
                    "timestamp_end_sec": 38.0,
                    "direction": "right",
                    "estimated_count": 3,
                    "peak_simultaneous_count": 3,
                    "wave_count": 1,
                },
                {
                    "timestamp_start_sec": 46.0,
                    "timestamp_end_sec": 58.0,
                    "direction": "right",
                    "estimated_count": 2,
                    "peak_simultaneous_count": 2,
                    "wave_count": 1,
                },
            ],
            parse_success=True,
        )
        decision = detect_undercount_risk(prediction)
        self.assertTrue(decision.should_recount)

    def test_build_recount_prompt_preserves_base_and_adds_guidance(self) -> None:
        base_prompt = PromptRevision(version=0, prompt_id="probe", prompt_text="Analyze the clip.")
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_c",
            left_count=0,
            right_count=5,
            candidate_passages=[{"timestamp_start_sec": 10.0, "timestamp_end_sec": 30.0, "direction": "right", "estimated_count": 5}],
            commentary="school moving rightward",
            evidence_summary="sequential targets",
        )
        recount_prompt = build_recount_prompt(base_prompt, prediction)
        self.assertEqual(recount_prompt.prompt_id, "probe_recount")
        self.assertIn("peak simultaneous occupancy", recount_prompt.prompt_text)
        self.assertIn("lower-bound seed", recount_prompt.prompt_text)
        self.assertIn("multiple successive waves", recount_prompt.prompt_text)
        self.assertIn("one sparse or single-wave episode", recount_prompt.prompt_text)
        self.assertIn("right_count=5", recount_prompt.prompt_text)

    def test_accepts_recount_when_total_increases_with_stronger_support(self) -> None:
        first_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d",
            left_count=0,
            right_count=3,
            candidate_passages=[
                {
                    "timestamp_start_sec": 45.0,
                    "timestamp_end_sec": 60.0,
                    "direction": "right",
                    "estimated_count": 3,
                    "peak_simultaneous_count": 3,
                    "wave_count": 1,
                }
            ],
            parse_success=True,
        )
        recount_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d",
            left_count=0,
            right_count=6,
            candidate_passages=[
                {
                    "timestamp_start_sec": 45.0,
                    "timestamp_end_sec": 65.0,
                    "direction": "right",
                    "estimated_count": 6,
                    "peak_simultaneous_count": 4,
                    "wave_count": 1,
                }
            ],
            parse_success=True,
        )
        accept, reason = should_accept_recount(first_prediction, recount_prediction)
        self.assertTrue(accept)
        self.assertEqual(reason, "recount_increased_total_with_structural_support")

    def test_rejects_single_wave_recount_that_scales_far_above_peak(self) -> None:
        first_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d2",
            left_count=0,
            right_count=12,
            candidate_passages=[
                {
                    "timestamp_start_sec": 40.0,
                    "timestamp_end_sec": 108.0,
                    "direction": "right",
                    "estimated_count": 12,
                    "peak_simultaneous_count": 4,
                    "wave_count": 1,
                }
            ],
            parse_success=True,
        )
        recount_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d2",
            left_count=0,
            right_count=17,
            candidate_passages=[
                {
                    "timestamp_start_sec": 40.0,
                    "timestamp_end_sec": 105.0,
                    "direction": "right",
                    "estimated_count": 17,
                    "peak_simultaneous_count": 5,
                    "wave_count": 1,
                }
            ],
            parse_success=True,
        )
        accept, reason = should_accept_recount(first_prediction, recount_prediction)
        self.assertFalse(accept)
        self.assertEqual(reason, "recount_single_wave_exceeds_peak_support>9")

    def test_rejects_sparse_split_recount_that_scales_far_above_total_peak(self) -> None:
        first_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d3",
            left_count=0,
            right_count=5,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 108.0,
                    "direction": "right",
                    "estimated_count": 5,
                    "peak_simultaneous_count": 1,
                    "wave_count": 1,
                }
            ],
            parse_success=True,
        )
        recount_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d3",
            left_count=0,
            right_count=10,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 54.0,
                    "direction": "right",
                    "estimated_count": 5,
                    "peak_simultaneous_count": 2,
                    "wave_count": 1,
                },
                {
                    "timestamp_start_sec": 54.0,
                    "timestamp_end_sec": 108.0,
                    "direction": "right",
                    "estimated_count": 5,
                    "peak_simultaneous_count": 2,
                    "wave_count": 1,
                },
            ],
            parse_success=True,
        )
        accept, reason = should_accept_recount(first_prediction, recount_prediction)
        self.assertFalse(accept)
        self.assertEqual(reason, "recount_sparse_split_exceeds_peak_support>6")

    def test_accepts_multi_passage_recount_with_conservative_local_caps(self) -> None:
        first_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d4",
            left_count=0,
            right_count=5,
            candidate_passages=[
                {
                    "timestamp_start_sec": 21.0,
                    "timestamp_end_sec": 40.0,
                    "direction": "right",
                    "estimated_count": 3,
                    "peak_simultaneous_count": 3,
                    "wave_count": 1,
                },
                {
                    "timestamp_start_sec": 46.0,
                    "timestamp_end_sec": 60.0,
                    "direction": "right",
                    "estimated_count": 2,
                    "peak_simultaneous_count": 2,
                    "wave_count": 1,
                },
            ],
            parse_success=True,
        )
        recount_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d4",
            left_count=0,
            right_count=9,
            candidate_passages=[
                {
                    "timestamp_start_sec": 21.0,
                    "timestamp_end_sec": 40.0,
                    "direction": "right",
                    "estimated_count": 5,
                    "peak_simultaneous_count": 3,
                    "wave_count": 1,
                },
                {
                    "timestamp_start_sec": 46.0,
                    "timestamp_end_sec": 60.0,
                    "direction": "right",
                    "estimated_count": 4,
                    "peak_simultaneous_count": 2,
                    "wave_count": 1,
                },
            ],
            parse_success=True,
        )
        accept, reason = should_accept_recount(first_prediction, recount_prediction)
        self.assertTrue(accept)
        self.assertEqual(reason, "recount_increased_total_with_structural_support")

    def test_rejects_recount_when_direction_flips(self) -> None:
        first_prediction = PassagePrediction(domain="kenai-val", clip_id="clip_e", left_count=0, right_count=5, parse_success=True)
        recount_prediction = PassagePrediction(domain="kenai-val", clip_id="clip_e", left_count=3, right_count=0, parse_success=True)
        accept, reason = should_accept_recount(first_prediction, recount_prediction)
        self.assertFalse(accept)
        self.assertEqual(reason, "recount_flipped_direction")


if __name__ == "__main__":
    unittest.main()

