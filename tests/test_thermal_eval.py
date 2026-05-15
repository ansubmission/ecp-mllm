from __future__ import annotations

from pathlib import Path
import unittest

import _bootstrap  # noqa: F401

from ecp_mllm.eval.thermal_eval import evaluate_thermal_predictions
from ecp_mllm.types import ClipRecord, InputVariant, ThermalEventWindow, ThermalPrediction


class ThermalEvalTests(unittest.TestCase):
    def _clip(self, clip_id: str, coarse_label: str, is_false_positive: bool, *, start: float | None = None, end: float | None = None) -> ClipRecord:
        truth_windows = []
        if start is not None and end is not None:
            truth_windows.append({"start_sec": start, "end_sec": end, "label": coarse_label, "track_id": f"t_{clip_id}"})
        truth_center_zone_entered = start is not None and end is not None
        return ClipRecord(
            dataset="nz_thermal",
            domain="loc_a",
            clip_id=clip_id,
            asset_paths={InputVariant.THERMAL_FILTERED: Path("clip.mp4")},
            duration_seconds=8.0,
            metadata={
                "coarse_label": coarse_label,
                "is_false_positive": is_false_positive,
                "truth_event_windows": truth_windows,
                "truth_center_zone_entered": truth_center_zone_entered,
                "truth_center_zone_first_entry_sec": start if truth_center_zone_entered else None,
                "truth_center_zone_dwell_sec": (end - start) if truth_center_zone_entered and start is not None and end is not None else None,
            },
        )

    def test_eval_reports_binary_and_event_metrics(self) -> None:
        clips = [
            self._clip("clip_bird", "bird", False, start=1.0, end=4.0),
            self._clip("clip_fp", "false_positive", True),
        ]
        predictions = [
            ThermalPrediction(
                domain="loc_a",
                clip_id="clip_bird",
                clip_labels=["bird"],
                coarse_label="bird",
                animal_present=True,
                false_positive_score=0.1,
                center_zone_entered=True,
                center_zone_first_entry_sec=1.4,
                center_zone_dwell_sec=2.0,
                event_windows=[ThermalEventWindow(timestamp_start_sec=1.5, timestamp_end_sec=3.8, label="bird", confidence=0.9)],
                event_labels=["bird"],
                confidence=0.9,
                abstain=False,
                latency_sec=2.0,
            ),
            ThermalPrediction(
                domain="loc_a",
                clip_id="clip_fp",
                clip_labels=["false_positive"],
                coarse_label="false_positive",
                animal_present=False,
                false_positive_score=0.95,
                event_windows=[],
                event_labels=[],
                confidence=0.8,
                abstain=False,
                latency_sec=1.0,
            ),
        ]
        report = evaluate_thermal_predictions(clips, predictions)
        self.assertAlmostEqual(report.binary_accuracy, 1.0)
        self.assertAlmostEqual(report.binary_f1, 1.0)
        self.assertAlmostEqual(report.coarse_macro_f1, 1.0)
        self.assertIsNotNone(report.animal_event_window_recall)
        self.assertGreater(float(report.animal_event_window_recall or 0.0), 0.0)
        self.assertGreater(float(report.animal_event_window_mean_tiou or 0.0), 0.0)
        self.assertAlmostEqual(float(report.animal_event_count_mae or 0.0), 0.0)
        self.assertAlmostEqual(float(report.center_zone_entry_accuracy or 0.0), 1.0)
        self.assertGreaterEqual(float(report.center_zone_entry_f1 or 0.0), 0.99)
        self.assertGreaterEqual(float(report.center_zone_first_entry_mae_sec or 0.0), 0.0)
        self.assertAlmostEqual(float(report.multi_entity_accuracy or 0.0), 1.0)
        self.assertAlmostEqual(float(report.animal_event_label_accuracy or 0.0), 1.0)
        self.assertAlmostEqual(report.mean_latency_sec, 1.5)


if __name__ == "__main__":
    unittest.main()
