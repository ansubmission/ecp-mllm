from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import _bootstrap  # noqa: F401

from ecp_mllm.config import LocalSettings, PathsConfig, QwenSettings
from ecp_mllm.experiments.thermal_common import (
    aggregate_thermal_predictions,
    build_thermal_run_metrics,
    derive_thermal_proposals,
    materialize_thermal_variant,
    materialize_thermal_window,
)
from ecp_mllm.types import ClipRecord, InputVariant, ThermalEventWindow, ThermalPrediction


class ThermalEventAgentBatchTests(unittest.TestCase):
    def _clip(self) -> ClipRecord:
        return ClipRecord(
            dataset="nz_thermal",
            domain="loc_a",
            clip_id="clip_001",
            asset_paths={
                InputVariant.THERMAL_FILTERED: Path("filtered.mp4"),
                InputVariant.THERMAL_NORMALIZED: Path("normalized.mp4"),
            },
            framerate=9.0,
            duration_seconds=12.0,
            metadata={
                "active_intervals": [(0.0, 4.0), (6.0, 12.0)],
                "truth_event_windows": [{"start_sec": 1.0, "end_sec": 3.0, "label": "bird", "track_id": "t1"}],
                "truth_center_zone_entered": True,
                "truth_center_zone_first_entry_sec": 1.25,
                "truth_center_zone_dwell_sec": 1.5,
                "coarse_label": "bird",
                "is_false_positive": False,
            },
        )

    def test_raw_video_proposals_follow_active_intervals(self) -> None:
        proposals = derive_thermal_proposals(
            self._clip(),
            proposal_source="raw_video",
            max_proposals=4,
            min_duration_sec=2.0,
            overlap_ratio=0.25,
        )
        self.assertGreaterEqual(len(proposals), 2)
        self.assertEqual(proposals[0]["start_sec"], 0.0)
        self.assertEqual(proposals[0]["source"], "raw_video")

    def test_track_oracle_proposals_use_truth_windows(self) -> None:
        proposals = derive_thermal_proposals(self._clip(), proposal_source="track_oracle", max_proposals=4)
        self.assertEqual(len(proposals), 1)
        self.assertEqual(proposals[0]["source"], "track_oracle")
        self.assertLessEqual(float(proposals[0]["start_sec"]), 1.0)
        self.assertGreaterEqual(float(proposals[0]["end_sec"]), 3.0)
        self.assertGreaterEqual(float(proposals[0]["end_sec"]) - float(proposals[0]["start_sec"]), 4.0)

    def test_aggregate_prefers_local_animal_over_global_false_positive(self) -> None:
        clip = self._clip()
        global_prediction = ThermalPrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            clip_labels=["false_positive"],
            coarse_label="false_positive",
            animal_present=False,
            false_positive_score=0.8,
            event_windows=[],
            event_labels=[],
            confidence=0.4,
            abstain=False,
            latency_sec=1.0,
        )
        local_predictions = [
            {
                "proposal": {"proposal_id": "window-01", "start_sec": 1.0, "end_sec": 4.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["bird"],
                    coarse_label="bird",
                    animal_present=True,
                    false_positive_score=0.1,
                    center_zone_entered=True,
                    center_zone_first_entry_sec=0.5,
                    center_zone_dwell_sec=1.25,
                    event_windows=[ThermalEventWindow(timestamp_start_sec=0.0, timestamp_end_sec=2.0, label="bird", confidence=0.9)],
                    event_labels=["bird"],
                    confidence=0.9,
                    abstain=False,
                    latency_sec=1.5,
                ),
            }
        ]
        final_prediction = aggregate_thermal_predictions(clip, global_prediction, local_predictions)
        self.assertEqual(final_prediction.coarse_label, "bird")
        self.assertLess(float(final_prediction.false_positive_score or 1.0), 0.5)
        self.assertEqual(len(final_prediction.event_windows), 1)
        self.assertTrue(bool(final_prediction.center_zone_entered))
        self.assertAlmostEqual(float(final_prediction.center_zone_first_entry_sec or 0.0), 1.5)

    def test_aggregate_suppresses_ambiguous_other_when_false_positive_support_is_strong(self) -> None:
        clip = self._clip()
        global_prediction = ThermalPrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            clip_labels=[],
            coarse_label="other",
            animal_present=None,
            false_positive_score=None,
            event_windows=[],
            event_labels=[],
            confidence=0.85,
            abstain=False,
            latency_sec=1.0,
        )
        local_predictions = [
            {
                "proposal": {"proposal_id": "window-01", "start_sec": 0.0, "end_sec": 6.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["false_positive"],
                    coarse_label="false_positive",
                    animal_present=False,
                    false_positive_score=0.95,
                    event_windows=[],
                    event_labels=["false_positive"],
                    confidence=0.9,
                    abstain=False,
                    latency_sec=1.0,
                ),
            },
            {
                "proposal": {"proposal_id": "window-02", "start_sec": 6.0, "end_sec": 12.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=[],
                    coarse_label="other",
                    animal_present=None,
                    false_positive_score=None,
                    event_windows=[],
                    event_labels=[],
                    confidence=0.95,
                    abstain=False,
                    latency_sec=1.0,
                ),
            },
        ]
        final_prediction = aggregate_thermal_predictions(clip, global_prediction, local_predictions)
        self.assertEqual(final_prediction.coarse_label, "false_positive")
        self.assertFalse(bool(final_prediction.animal_present))
        self.assertEqual(final_prediction.event_windows, [])

    def test_aggregate_keeps_global_positive_event_when_locals_are_false_positive(self) -> None:
        clip = self._clip()
        global_prediction = ThermalPrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            clip_labels=["other"],
            coarse_label="other",
            animal_present=True,
            false_positive_score=0.05,
            event_windows=[ThermalEventWindow(timestamp_start_sec=1.0, timestamp_end_sec=6.0, label="other", confidence=0.85)],
            event_labels=["other"],
            confidence=0.85,
            abstain=False,
            latency_sec=1.0,
        )
        local_predictions = [
            {
                "proposal": {"proposal_id": "window-01", "start_sec": 0.0, "end_sec": 4.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["false_positive"],
                    coarse_label="false_positive",
                    animal_present=False,
                    false_positive_score=0.95,
                    event_windows=[],
                    event_labels=["false_positive"],
                    confidence=0.9,
                    abstain=False,
                    latency_sec=1.0,
                ),
            },
            {
                "proposal": {"proposal_id": "window-02", "start_sec": 4.0, "end_sec": 8.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["false_positive"],
                    coarse_label="false_positive",
                    animal_present=False,
                    false_positive_score=0.95,
                    event_windows=[],
                    event_labels=["false_positive"],
                    confidence=0.95,
                    abstain=False,
                    latency_sec=1.0,
                ),
            },
        ]
        final_prediction = aggregate_thermal_predictions(clip, global_prediction, local_predictions)
        self.assertEqual(final_prediction.coarse_label, "other")
        self.assertTrue(bool(final_prediction.animal_present))
        self.assertEqual(len(final_prediction.event_windows), 1)

    def test_aggregate_treats_coherent_other_locals_as_positive_animal_evidence(self) -> None:
        clip = self._clip()
        global_prediction = ThermalPrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            clip_labels=[],
            coarse_label="other",
            animal_present=None,
            false_positive_score=None,
            event_windows=[],
            event_labels=[],
            confidence=0.4,
            abstain=False,
            latency_sec=1.0,
            commentary="A single coherent bright target drifts slowly rightward; persistence and coherence suggest a biological target.",
            evidence_summary="Single bright target visible from 1.0s to 9.0s with slight rightward displacement.",
        )
        local_predictions = [
            {
                "proposal": {"proposal_id": "window-01", "start_sec": 0.0, "end_sec": 6.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=[],
                    coarse_label="other",
                    animal_present=None,
                    false_positive_score=None,
                    event_windows=[],
                    event_labels=[],
                    confidence=0.6,
                    abstain=False,
                    latency_sec=1.0,
                    commentary="One coherent bright target moving rightward; persistence distinguishes it from random noise.",
                    evidence_summary="Single target observed from 1.0s to 6.0s with coherent motion.",
                ),
            },
            {
                "proposal": {"proposal_id": "window-02", "start_sec": 6.0, "end_sec": 12.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["false_positive"],
                    coarse_label="false_positive",
                    animal_present=False,
                    false_positive_score=0.95,
                    event_windows=[],
                    event_labels=["false_positive"],
                    confidence=0.3,
                    abstain=False,
                    latency_sec=1.0,
                    commentary="Weak static clutter only.",
                    evidence_summary="Brief stationary hotspot.",
                ),
            },
        ]
        final_prediction = aggregate_thermal_predictions(clip, global_prediction, local_predictions)
        self.assertEqual(final_prediction.coarse_label, "other")
        self.assertTrue(bool(final_prediction.animal_present))
        self.assertGreaterEqual(len(final_prediction.event_windows), 1)

    def test_aggregate_suppresses_global_other_when_only_strong_false_positive_locals_exist(self) -> None:
        clip = self._clip()
        global_prediction = ThermalPrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            clip_labels=["other"],
            coarse_label="other",
            animal_present=True,
            false_positive_score=0.05,
            event_windows=[ThermalEventWindow(timestamp_start_sec=0.0, timestamp_end_sec=12.0, label="other", confidence=0.75)],
            event_labels=["other"],
            confidence=0.5,
            abstain=False,
            latency_sec=1.0,
            commentary="One distinct target moves steadily through the scene.",
            evidence_summary="Single bright target visible from 0.0s to 12.0s, showing net displacement.",
        )
        local_predictions = [
            {
                "proposal": {"proposal_id": "window-01", "start_sec": 0.0, "end_sec": 6.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["false_positive"],
                    coarse_label="false_positive",
                    animal_present=False,
                    false_positive_score=0.95,
                    event_windows=[],
                    event_labels=["false_positive"],
                    confidence=0.95,
                    abstain=False,
                    latency_sec=1.0,
                    commentary="No moving targets detected. Static bright spot only.",
                    evidence_summary="No coherent motion; stationary clutter persists.",
                ),
            },
            {
                "proposal": {"proposal_id": "window-02", "start_sec": 6.0, "end_sec": 12.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["false_positive"],
                    coarse_label="false_positive",
                    animal_present=False,
                    false_positive_score=0.95,
                    event_windows=[],
                    event_labels=["false_positive"],
                    confidence=0.9,
                    abstain=False,
                    latency_sec=1.0,
                    commentary="No translational motion. Fixed bright spot remains stationary.",
                    evidence_summary="Static scene with no moving targets.",
                ),
            },
        ]
        final_prediction = aggregate_thermal_predictions(clip, global_prediction, local_predictions)
        self.assertEqual(final_prediction.coarse_label, "false_positive")
        self.assertFalse(bool(final_prediction.animal_present))
        self.assertEqual(final_prediction.event_windows, [])

    def test_aggregate_creates_global_fallback_window_for_positive_other(self) -> None:
        clip = self._clip()
        global_prediction = ThermalPrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            clip_labels=[],
            coarse_label="other",
            animal_present=None,
            false_positive_score=None,
            event_windows=[],
            event_labels=[],
            confidence=0.6,
            abstain=False,
            latency_sec=1.0,
            commentary="A biological target remains visible and drifts slowly through the scene.",
            evidence_summary="Single coherent target visible from 0.0s to 10.0s.",
        )
        final_prediction = aggregate_thermal_predictions(clip, global_prediction, [])
        self.assertEqual(final_prediction.coarse_label, "other")
        self.assertTrue(bool(final_prediction.animal_present))
        self.assertEqual(len(final_prediction.event_windows), 1)
        self.assertAlmostEqual(float(final_prediction.event_windows[0].timestamp_start_sec or 0.0), 6.0)
        self.assertAlmostEqual(float(final_prediction.event_windows[0].timestamp_end_sec or 0.0), 12.0)

    def test_aggregate_offsets_center_zone_entry_from_local_window(self) -> None:
        clip = self._clip()
        global_prediction = ThermalPrediction(
            domain=clip.domain,
            clip_id=clip.clip_id,
            clip_labels=["false_positive"],
            coarse_label="false_positive",
            animal_present=False,
            false_positive_score=0.9,
            center_zone_entered=False,
            event_windows=[],
            event_labels=[],
            confidence=0.4,
            abstain=False,
            latency_sec=1.0,
        )
        local_predictions = [
            {
                "proposal": {"proposal_id": "window-01", "start_sec": 6.0, "end_sec": 10.0, "source": "raw_video"},
                "prediction": ThermalPrediction(
                    domain=clip.domain,
                    clip_id=clip.clip_id,
                    clip_labels=["other"],
                    coarse_label="other",
                    animal_present=True,
                    false_positive_score=0.2,
                    center_zone_entered=True,
                    center_zone_first_entry_sec=1.25,
                    center_zone_dwell_sec=1.5,
                    event_windows=[ThermalEventWindow(timestamp_start_sec=0.5, timestamp_end_sec=2.0, label="other", confidence=0.8)],
                    event_labels=["other"],
                    confidence=0.8,
                    abstain=False,
                    latency_sec=1.0,
                    commentary="One coherent animal target crosses the middle of the frame.",
                    evidence_summary="Biological target enters the center zone around 7.25s.",
                ),
            }
        ]
        final_prediction = aggregate_thermal_predictions(clip, global_prediction, local_predictions)
        self.assertTrue(bool(final_prediction.center_zone_entered))
        self.assertAlmostEqual(float(final_prediction.center_zone_first_entry_sec or 0.0), 7.25)
        self.assertAlmostEqual(float(final_prediction.center_zone_dwell_sec or 0.0), 1.5)

    def test_materialize_dual_variant_updates_clip_asset_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            filtered = root / "filtered.mp4"
            normalized = root / "normalized.mp4"
            filtered.write_bytes(b"filtered")
            normalized.write_bytes(b"normalized")
            clip = ClipRecord(
                dataset="nz_thermal",
                domain="loc_a",
                clip_id="clip_001",
                asset_paths={
                    InputVariant.THERMAL_FILTERED: filtered,
                    InputVariant.THERMAL_NORMALIZED: normalized,
                },
            )
            settings = LocalSettings(paths=PathsConfig(output_root=root / "outputs"), qwen=QwenSettings(provider="mock"))
            with mock.patch("ecp_mllm.experiments.thermal_common.compose_side_by_side_mp4") as patched:
                def _fake_compose(left, right, out, target_height, crf):
                    output = Path(out)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"dual")
                    return output

                patched.side_effect = _fake_compose
                updated_clip, media_path = materialize_thermal_variant(
                    settings,
                    clip,
                    InputVariant.THERMAL_DUAL,
                    stitch_height=240,
                    stitch_crf=28,
                )
            self.assertIsNotNone(media_path)
            self.assertEqual(updated_clip.get_asset_path(InputVariant.THERMAL_DUAL), media_path)
            self.assertTrue(media_path.exists())

    def test_materialize_variant_with_center_zone_overlay_updates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            filtered = root / "filtered.mp4"
            filtered.write_bytes(b"filtered")
            clip = ClipRecord(
                dataset="nz_thermal",
                domain="loc_a",
                clip_id="clip_001",
                asset_paths={InputVariant.THERMAL_FILTERED: filtered},
            )
            settings = LocalSettings(paths=PathsConfig(output_root=root / "outputs"), qwen=QwenSettings(provider="mock"))
            with mock.patch("ecp_mllm.experiments.thermal_common.overlay_center_zone_mp4") as patched:
                def _fake_overlay(source, out, dual_layout, crf):
                    output = Path(out)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"overlay")
                    return output

                patched.side_effect = _fake_overlay
                updated_clip, media_path = materialize_thermal_variant(
                    settings,
                    clip,
                    InputVariant.THERMAL_FILTERED,
                    zone_overlay=True,
                )
            self.assertIsNotNone(media_path)
            self.assertNotEqual(media_path, filtered)
            self.assertEqual(updated_clip.get_asset_path(InputVariant.THERMAL_FILTERED), media_path)
            self.assertEqual(updated_clip.metadata.get("zone_overlay", {}).get("type"), "center")

    def test_materialize_window_with_center_zone_overlay_returns_overlay_video(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            filtered = root / "filtered.mp4"
            filtered.write_bytes(b"filtered")
            clip = ClipRecord(
                dataset="nz_thermal",
                domain="loc_a",
                clip_id="clip_001",
                asset_paths={InputVariant.THERMAL_FILTERED: filtered},
            )
            settings = LocalSettings(paths=PathsConfig(output_root=root / "outputs"), qwen=QwenSettings(provider="mock"))
            with mock.patch("ecp_mllm.experiments.thermal_common.trim_mp4") as patched_trim, mock.patch(
                "ecp_mllm.experiments.thermal_common.overlay_center_zone_mp4"
            ) as patched_overlay:
                def _fake_trim(source, out, start_sec, end_sec, crf):
                    output = Path(out)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"trim")
                    return output

                def _fake_overlay(source, out, dual_layout, crf):
                    output = Path(out)
                    output.parent.mkdir(parents=True, exist_ok=True)
                    output.write_bytes(b"overlay")
                    return output

                patched_trim.side_effect = _fake_trim
                patched_overlay.side_effect = _fake_overlay
                media_path = materialize_thermal_window(
                    settings,
                    clip,
                    InputVariant.THERMAL_FILTERED,
                    start_sec=1.0,
                    end_sec=3.0,
                    label="window-01",
                    zone_overlay=True,
                )
            self.assertTrue(media_path.exists())
            self.assertIn("centerzone", media_path.name)

    def test_run_metrics_recompute_from_completed_rows(self) -> None:
        clip = self._clip()
        state = {
            "results": [
                {
                    "status": "completed",
                    "clip_key": clip.key.value,
                    "prediction": {
                        "clip_labels": ["bird"],
                        "coarse_label": "bird",
                        "animal_present": True,
                        "false_positive_score": 0.2,
                        "center_zone_entered": True,
                        "center_zone_first_entry_sec": 1.25,
                        "center_zone_dwell_sec": 1.5,
                        "event_windows": [{"timestamp_start_sec": 1.0, "timestamp_end_sec": 3.0, "label": "bird"}],
                        "event_labels": ["bird"],
                        "confidence": 0.9,
                        "abstain": False,
                        "latency_sec": 1.0,
                        "parse_success": True,
                        "prompt_id": "thermal",
                    },
                }
            ]
        }
        metrics = build_thermal_run_metrics(state, {clip.key.value: clip})
        self.assertAlmostEqual(float(metrics["binary_accuracy"] or 0.0), 1.0)
        self.assertAlmostEqual(float(metrics["coarse_macro_f1"] or 0.0), 1.0)
        self.assertAlmostEqual(float(metrics["center_zone_entry_accuracy"] or 0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
