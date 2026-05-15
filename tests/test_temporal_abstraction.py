from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

import _bootstrap  # noqa: F401

from ecp_mllm.agent.temporal_abstraction import propose_temporal_windows


class TemporalAbstractionTests(unittest.TestCase):
    def test_motion_energy_produces_event_windows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            video_path = Path(temp_dir) / "synthetic.mp4"
            writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (64, 64))
            self.assertTrue(writer.isOpened())
            for frame_index in range(50):
                frame = np.zeros((64, 64, 3), dtype=np.uint8)
                if 10 <= frame_index < 18:
                    x = 5 + (frame_index - 10) * 4
                    cv2.rectangle(frame, (x, 20), (x + 8, 28), (255, 255, 255), -1)
                if 32 <= frame_index < 40:
                    x = 10 + (frame_index - 32) * 3
                    cv2.rectangle(frame, (x, 35), (x + 8, 43), (255, 255, 255), -1)
                writer.write(frame)
            writer.release()

            proposals = propose_temporal_windows(
                video_path,
                min_duration_sec=0.8,
                merge_gap_sec=0.4,
                smoothing_sec=0.4,
                max_windows=4,
                representation="sff3c",
            )

        self.assertGreaterEqual(len(proposals), 2)
        self.assertTrue(all(item.timestamp_end_sec > item.timestamp_start_sec for item in proposals))
        self.assertTrue(all(item.source == "motion_energy" for item in proposals))


if __name__ == "__main__":
    unittest.main()


