from __future__ import annotations

import unittest
from argparse import Namespace

import _bootstrap  # noqa: F401

from ecp_mllm.experiments.repeated_event_agent_batch import _aggregate, _build_event_agent_command, _run_summary_metrics


class RepeatedEventAgentBatchTests(unittest.TestCase):
    def test_run_summary_metrics_extracts_selection_rates(self) -> None:
        payload = {
            "results": [
                {
                    "status": "completed",
                    "abs_error_left": 0,
                    "abs_error_right": 1,
                    "ground_truth": {"total_count": 5},
                    "selected_action": "sff3c:proposal_guided",
                    "prediction": {"parse_success": True, "latency_sec": 10.0},
                },
                {
                    "status": "completed",
                    "abs_error_left": 0,
                    "abs_error_right": 2,
                    "ground_truth": {"total_count": 5},
                    "selected_action": "sff3c:global",
                    "prediction": {"parse_success": False, "latency_sec": 20.0},
                },
            ]
        }
        metrics = _run_summary_metrics(payload)
        self.assertEqual(metrics["total_abs_error"], 3.0)
        self.assertAlmostEqual(metrics["nmae"], 0.3)
        self.assertAlmostEqual(metrics["parse_rate"], 0.5)
        self.assertAlmostEqual(metrics["mean_latency_sec"], 15.0)
        self.assertAlmostEqual(metrics["proposal_selected_rate"], 0.5)
        self.assertAlmostEqual(metrics["global_selected_rate"], 0.5)
        self.assertAlmostEqual(metrics["custom_selected_rate"], 0.0)

    def test_aggregate_summarizes_metric_distribution(self) -> None:
        summary = _aggregate(
            [
                {"nmae": 0.2, "proposal_selected_rate": 1.0},
                {"nmae": 0.4, "proposal_selected_rate": 0.0},
            ]
        )
        self.assertAlmostEqual(summary["nmae"]["mean"], 0.3)
        self.assertAlmostEqual(summary["nmae"]["std"], 0.1)
        self.assertEqual(summary["proposal_selected_rate"]["min"], 0.0)
        self.assertEqual(summary["proposal_selected_rate"]["max"], 1.0)

    def test_build_event_agent_command_forwards_flagged_repeat_runs(self) -> None:
        args = Namespace(
            settings="config/local.toml",
            domain="kenai-val",
            variant="sff3c",
            model="qwen3.5-plus",
            transport="stitched_mp4",
            stitch_fps=5.0,
            stitch_height=0,
            stitch_crf=30,
            clips_path="config/prompt_dev/kenai_val_holdout_8clips_v1.json",
            name="repeated_proto",
            prompt_text=None,
            repeats=2,
            max_clips=3,
            max_proposals=4,
            proposal_min_duration_sec=1.0,
            proposal_merge_gap_sec=1.0,
            proposal_min_video_sec=8.0,
            flagged_repeat_runs=2,
            stop_on_error=False,
        )
        command = _build_event_agent_command(args, "repeated_proto__run01")
        self.assertIn("--flagged-repeat-runs", command)
        index = command.index("--flagged-repeat-runs")
        self.assertEqual(command[index + 1], "2")
        self.assertIn("--max-clips", command)
        self.assertIn("--model", command)


if __name__ == "__main__":
    unittest.main()

