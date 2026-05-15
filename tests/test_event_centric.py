from __future__ import annotations

import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.agent.event_centric import (
    build_observation_stream,
    enumerate_perception_actions,
    prediction_to_event_hypotheses,
    suggest_perception_actions,
)
from ecp_mllm.types import AgentMemoryRecord, ClipRecord, InputVariant, PassagePrediction


class EventCentricTests(unittest.TestCase):
    def test_build_observation_stream_uses_available_variants(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip-001",
            asset_paths={InputVariant.RAW: None, InputVariant.SFF3C: None},  # type: ignore[arg-type]
            duration_seconds=90.0,
            framerate=5.0,
        )
        stream = build_observation_stream(clip)
        self.assertEqual(stream.domain, "kenai-val")
        self.assertEqual(stream.stream_id, "clip-001")
        self.assertIn("raw", stream.available_variants)
        self.assertIn("sff3c", stream.available_variants)

    def test_suggest_actions_penalizes_failure_memory(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="elwha",
            clip_id="clip-002",
            asset_paths={InputVariant.RAW: None, InputVariant.SFF3C: None},  # type: ignore[arg-type]
        )
        stream = build_observation_stream(clip)
        memory = [
            AgentMemoryRecord(
                domain="elwha",
                failure_type="overcount",
                preferred_representation="sff3c",
                preferred_temporal_mode="full_stream",
                preferred_reasoning_mode="global_direct",
                count=8,
            )
        ]
        actions = suggest_perception_actions(stream, memory, limit=2)
        self.assertTrue(actions)
        self.assertNotEqual(actions[0].action_id, "sff3c:global")

    def test_enumerate_actions_includes_custom_branches(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip-002b",
            asset_paths={InputVariant.SFF3C: None},  # type: ignore[arg-type]
        )
        stream = build_observation_stream(clip)
        actions = enumerate_perception_actions(stream)
        action_ids = {action.action_id for action in actions}
        self.assertIn("sff3c:custom_trickle_reduction", action_ids)
        self.assertIn("sff3c:custom_high_throughput", action_ids)
        self.assertIn("sff3c:custom_low_count", action_ids)

    def test_prediction_projects_to_event_hypotheses(self) -> None:
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip-003",
            left_count=0,
            right_count=4,
            confidence=0.8,
            candidate_passages=[
                {
                    "timestamp_start_sec": 10.0,
                    "timestamp_end_sec": 16.0,
                    "direction": "right",
                    "throughput_best_count": 3,
                    "peak_simultaneous_count": 2,
                    "wave_count": 2,
                },
                {
                    "timestamp_start_sec": 30.0,
                    "timestamp_end_sec": 35.0,
                    "direction": "right",
                    "estimated_count": 1,
                },
            ],
        )
        hypotheses = prediction_to_event_hypotheses(prediction, representation="sff3c")
        self.assertEqual(len(hypotheses), 2)
        self.assertEqual(hypotheses[0].count, 3)
        self.assertEqual(hypotheses[0].representation, "sff3c")
        self.assertEqual(hypotheses[1].count, 1)


if __name__ == "__main__":
    unittest.main()


