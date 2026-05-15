from __future__ import annotations

import argparse
import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.experiments.repeated_thermal_event_agent_batch import _aggregate, _build_command, _run_summary_metrics


class RepeatedThermalEventAgentBatchTests(unittest.TestCase):
    def test_run_summary_metrics_reads_summary_metrics(self) -> None:
        payload = {
            "metrics": {
                "binary_accuracy": 0.75,
                "binary_f1": 0.6,
                "binary_balanced_accuracy": 0.7,
                "coarse_macro_f1": 0.55,
                "event_window_recall": 0.4,
                "event_window_mean_tiou": 0.3,
                "center_zone_entry_f1": 0.5,
                "center_zone_first_entry_mae_sec": 1.2,
                "abstention_rate": 0.1,
                "mean_latency_sec": 12.0,
            }
        }
        metrics = _run_summary_metrics(payload)
        self.assertAlmostEqual(metrics["binary_accuracy"], 0.75)
        self.assertAlmostEqual(metrics["coarse_macro_f1"], 0.55)
        self.assertAlmostEqual(metrics["event_window_mean_tiou"], 0.3)
        self.assertAlmostEqual(metrics["center_zone_entry_f1"], 0.5)

    def test_aggregate_computes_mean_and_std(self) -> None:
        summary = _aggregate(
            [
                {"binary_accuracy": 0.5, "coarse_macro_f1": 0.2},
                {"binary_accuracy": 0.7, "coarse_macro_f1": 0.4},
            ]
        )
        self.assertAlmostEqual(summary["binary_accuracy"]["mean"], 0.6)
        self.assertAlmostEqual(summary["coarse_macro_f1"]["max"], 0.4)

    def test_build_command_normalizes_all_split(self) -> None:
        args = argparse.Namespace(
            settings="config/local.toml",
            split="",
            variant="thermal_filtered",
            model=None,
            clips_path="clips.json",
            pack=None,
            name="thermal_repeat",
            prompt_text=None,
            repeats=2,
            max_clips=2,
            include_calibration=False,
            stitch_height=360,
            stitch_crf=30,
            max_proposals=4,
            proposal_min_duration_sec=2.0,
            proposal_overlap_ratio=0.25,
            proposal_source="raw_video",
            stop_on_error=False,
        )
        command = _build_command(args, "thermal_repeat__run01")
        split_index = command.index("--split")
        self.assertEqual(command[split_index + 1], "all")


if __name__ == "__main__":
    unittest.main()
