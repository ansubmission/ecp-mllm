from __future__ import annotations

from pathlib import Path
import json
import tempfile
import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.experiments.prompt_batch import BatchClipSpec, load_batch_specs, write_batch_summary


class PromptBatchTests(unittest.TestCase):
    def test_load_batch_specs_reads_bucketed_clip_list(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "clips.json"
            path.write_text(
                json.dumps(
                    [
                        {"bucket": "1-2", "clip_id": "clip_a"},
                        {"bucket": ">10", "clip_id": "clip_b"},
                    ]
                ),
                encoding="utf-8",
            )
            specs = load_batch_specs(path)
            self.assertEqual(specs, [BatchClipSpec(bucket="1-2", clip_id="clip_a"), BatchClipSpec(bucket=">10", clip_id="clip_b")])

    def test_write_batch_summary_renders_status_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "summary.md"
            payload = {
                "name": "batch_run",
                "domain": "kenai-val",
                "variant": "sff3c",
                "model": "qwen3.5-plus",
                "transport": "stitched_mp4",
                "total_clips": 2,
                "completed_count": 1,
                "results": [
                    {
                        "status": "completed",
                        "bucket": "1-2",
                        "clip_key": "kenai-val/clip_a",
                        "ground_truth": {"right_count": 1},
                        "prediction": {"right_count": 1, "latency_sec": 12.5, "commentary": "one fish"},
                        "total_abs_error": 0,
                    },
                    {
                        "status": "failed",
                        "bucket": ">10",
                        "clip_key": "kenai-val/clip_b",
                        "ground_truth": {"right_count": 15},
                        "error": "timeout",
                    },
                ],
            }
            write_batch_summary(path, payload)
            text = path.read_text(encoding="utf-8")
            self.assertIn("`kenai-val/clip_a`", text)
            self.assertIn("completed", text)
            self.assertIn("failed", text)
            self.assertIn("timeout", text)


if __name__ == "__main__":
    unittest.main()

