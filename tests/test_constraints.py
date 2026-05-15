from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.eval.constraints import audit_event_hypotheses, audit_prediction_constraints
from ecp_mllm.types import DirectionalCounts, EventHypothesis, PassagePrediction


class ConstraintAuditTests(unittest.TestCase):
    def test_overlapping_events_are_merged(self) -> None:
        hypotheses = [
            EventHypothesis(
                hypothesis_id="h1",
                timestamp_start_sec=10.0,
                timestamp_end_sec=20.0,
                direction="right",
                count=3,
                representation="sff3c",
                source="test",
            ),
            EventHypothesis(
                hypothesis_id="h2",
                timestamp_start_sec=15.0,
                timestamp_end_sec=22.0,
                direction="right",
                count=2,
                representation="sff3c",
                source="test",
            ),
        ]
        audit = audit_event_hypotheses(hypotheses, final_counts=DirectionalCounts(left=0, right=3))
        self.assertTrue(audit.applied)
        self.assertTrue(audit.repaired)
        self.assertEqual(audit.corrected_event_count, 1)
        self.assertTrue(any(item.code == "overlapping_same_direction_events" for item in audit.findings))

    def test_global_counts_are_clamped_to_supported_events(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip-001",
            left_count=0,
            right_count=9,
            candidate_passages=[
                {
                    "timestamp_start_sec": 20.0,
                    "timestamp_end_sec": 30.0,
                    "direction": "right",
                    "throughput_best_count": 4,
                }
            ],
        )
        audit = audit_prediction_constraints(prediction, representation="sff3c")
        self.assertTrue(audit.repaired)
        self.assertIsNotNone(audit.corrected_counts)
        self.assertEqual(audit.corrected_counts.right, 4)
        self.assertTrue(any(item.code == "global_count_exceeds_supported_events" for item in audit.findings))

    def test_short_precursor_is_not_merged_into_longer_dense_wave(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip-001b",
            left_count=0,
            right_count=4,
            candidate_passages=[
                {
                    "timestamp_start_sec": 48.4,
                    "timestamp_end_sec": 50.4,
                    "direction": "right",
                    "throughput_best_count": 1,
                    "peak_simultaneous_count": 1,
                    "episode_duration_sec": 2.0,
                },
                {
                    "timestamp_start_sec": 48.0,
                    "timestamp_end_sec": 60.0,
                    "direction": "right",
                    "throughput_best_count": 3,
                    "peak_simultaneous_count": 3,
                    "episode_duration_sec": 12.0,
                },
            ],
        )
        audit = audit_prediction_constraints(prediction, representation="sff3c")
        self.assertFalse(audit.repaired)
        self.assertIsNotNone(audit.corrected_counts)
        self.assertEqual(audit.corrected_counts.right, 4)
        self.assertFalse(any(item.code == "overlapping_same_direction_events" for item in audit.findings))

    def test_clean_prediction_has_no_repairs(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip-002",
            left_count=0,
            right_count=4,
            candidate_passages=[
                {
                    "timestamp_start_sec": 5.0,
                    "timestamp_end_sec": 12.0,
                    "direction": "right",
                    "throughput_best_count": 2,
                },
                {
                    "timestamp_start_sec": 30.0,
                    "timestamp_end_sec": 36.0,
                    "direction": "right",
                    "throughput_best_count": 2,
                },
            ],
        )
        audit = audit_prediction_constraints(prediction, representation="sff3c")
        self.assertTrue(audit.applied)
        self.assertFalse(audit.repaired)
        self.assertEqual(audit.corrected_counts.right, 4)

    def test_text_direction_conflict_is_flagged(self) -> None:
        prediction = PassagePrediction(
            domain="elwha",
            clip_id="clip-003",
            left_count=12,
            right_count=0,
            commentary="Three distinct waves of fish passage were observed moving from left to right over the clip.",
            evidence_summary="Visual tracking shows bright targets moving left-to-right in three temporal clusters.",
            candidate_passages=[
                {
                    "timestamp_start_sec": 36.0,
                    "timestamp_end_sec": 58.0,
                    "direction": "left",
                    "throughput_best_count": 12,
                    "evidence_note": "Targets enter from left and exit right in a continuous stream.",
                }
            ],
        )
        audit = audit_prediction_constraints(prediction, representation="sff3c")
        self.assertTrue(any(item.code == "text_direction_conflicts_with_structured_counts" for item in audit.findings))


if __name__ == "__main__":
    unittest.main()

