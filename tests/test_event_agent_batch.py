from __future__ import annotations

import unittest
from pathlib import Path

import _bootstrap  # noqa: F401

from ecp_mllm.experiments.event_agent_batch import (
    _apply_flagged_repeat_consensus,
    _apply_post_selection_peak3_school_uplift,
    _apply_post_selection_positive_window_uplift,
    _apply_post_selection_single_school_uplift,
    _apply_post_selection_single_school_window_floor,
    _apply_post_selection_stream_candidate_floor,
    _apply_post_selection_stream_global_floor,
    _apply_post_selection_stream_longspan_uplift,
    _apply_post_selection_stream_multiwave_uplift,
    _apply_post_selection_stream_embedded_wave_uplift,
    _apply_post_selection_stream_tile_pair_floor,
    _apply_post_selection_stream_window_stack_uplift,
    _apply_post_consensus_sparse_stream_cap,
    _aggregate_window_predictions,
    _apply_constraint_repair,
    _custom_branch_prompt_for_clip,
    _effective_max_proposals,
    _expand_window_bounds,
    _hybrid_proposals,
    _project_prediction_to_direction,
    _stream_tiling_proposals,
    _select_risk_assessment,
    _select_risk_source_result,
    _select_high_throughput_prediction,
    _select_stream_routing_direction,
    _select_stream_throughput_prediction,
    _select_trickle_reduction_prediction,
    _refresh_result_record_errors,
    _post_consensus_trickle_prompt,
    _should_run_post_consensus_trickle_cleanup,
    _should_run_flagged_repeat_consensus,
    _should_apply_routed_override,
    _should_accept_dense_window_prediction,
    _should_run_dense_window_refinement,
    _write_summary,
)
from ecp_mllm.eval.domain_shift_critic import DomainShiftAssessment
from ecp_mllm.eval.site_profiles import resolve_site_profile
from ecp_mllm.types import ClipRecord, EventProposal, InputVariant, PassageEvent, PassagePrediction, UpstreamDirection


class EventAgentBatchTests(unittest.TestCase):
    def test_write_summary_uses_left_right_pairs(self) -> None:
        with self.subTest("summary markdown formats paired counts"):
            temp_path = Path(".tmp_test_event_agent_summary.md")
            payload = {
                "name": "demo_run",
                "domain": "nushagak",
                "variant": "sff3c",
                "model": "qwen3.5-plus",
                "completed_count": 1,
                "total_clips": 1,
                "results": [
                    {
                        "status": "completed",
                        "clip_key": "nushagak/demo_clip",
                        "ground_truth": {"left_count": 12, "right_count": 1},
                        "action_predictions": {
                            "global_direct": {"left_count": 10, "right_count": 0},
                            "proposal_guided": {"left_count": 11, "right_count": 1},
                            "local_repair": {"left_count": 11, "right_count": 1},
                        },
                        "selected_action": "sff3c:proposal_guided",
                        "selected_branch": "baseline",
                        "prediction": {"left_count": 11, "right_count": 1},
                        "total_abs_error": 1,
                    }
                ],
            }
            try:
                _write_summary(temp_path, payload)
                rendered = temp_path.read_text(encoding="utf-8")
            finally:
                if temp_path.exists():
                    temp_path.unlink()

            self.assertIn("| GT L/R | Global L/R | Proposal L/R | Repair L/R |", rendered)
            self.assertIn("| completed | `nushagak/demo_clip` | 12/1 | 10/0 | 11/1 | 11/1 |", rendered)

    def test_aggregate_window_predictions_offsets_local_times(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_a",
            asset_paths={InputVariant.SFF3C: Path("clip_a.mp4")},
        )
        prediction_a = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_a__window_01",
            left_count=0,
            right_count=2,
            confidence=0.8,
            commentary="window a",
            evidence_summary="bright right-moving targets",
            parse_success=True,
            prompt_id="event_agent_window",
            latency_sec=1.2,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.25,
                    "timestamp_end_sec": 1.25,
                    "direction": "right",
                    "throughput_best_count": 2,
                }
            ],
            events=[PassageEvent(timestamp_sec=0.5, direction="right", confidence=0.8, evidence_note="wave a")],
        )
        prediction_b = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_a__window_02",
            left_count=1,
            right_count=0,
            confidence=0.6,
            commentary="window b",
            evidence_summary="one left-moving target",
            parse_success=True,
            prompt_id="event_agent_window",
            latency_sec=0.8,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.1,
                    "timestamp_end_sec": 0.5,
                    "direction": "left",
                    "throughput_best_count": 1,
                }
            ],
            events=[PassageEvent(timestamp_sec=0.2, direction="left", confidence=0.7, evidence_note="wave b")],
        )

        aggregated = _aggregate_window_predictions(
            clip,
            [(10.0, prediction_a), (20.0, prediction_b)],
            prompt_id="event_agent_proposal_guided",
        )

        self.assertTrue(aggregated.parse_success)
        self.assertEqual(aggregated.left_count, 1)
        self.assertEqual(aggregated.right_count, 2)
        self.assertAlmostEqual(aggregated.candidate_passages[0]["timestamp_start_sec"], 10.25)
        self.assertAlmostEqual(aggregated.candidate_passages[1]["timestamp_end_sec"], 20.5)
        self.assertAlmostEqual(aggregated.events[0].timestamp_sec, 10.5)
        self.assertAlmostEqual(aggregated.events[1].timestamp_sec, 20.2)
        self.assertAlmostEqual(aggregated.latency_sec or 0.0, 2.0)

    def test_apply_constraint_repair_clamps_unsupported_total(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_b",
            asset_paths={InputVariant.SFF3C: Path("clip_b.mp4")},
        )
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_b",
            left_count=0,
            right_count=7,
            confidence=0.7,
            commentary="two overlapping right-moving schools",
            evidence_summary="overlapping episodes",
            parse_success=True,
            prompt_id="event_agent_proposal_guided",
            candidate_passages=[
                {
                    "timestamp_start_sec": 1.0,
                    "timestamp_end_sec": 5.0,
                    "direction": "right",
                    "throughput_best_count": 4,
                    "peak_simultaneous_count": 3,
                },
                {
                    "timestamp_start_sec": 2.0,
                    "timestamp_end_sec": 4.0,
                    "direction": "right",
                    "throughput_best_count": 3,
                    "peak_simultaneous_count": 2,
                },
            ],
        )

        repaired, audit = _apply_constraint_repair(
            clip,
            prediction,
            representation="sff3c",
            prompt_id="event_agent_local_repair",
        )

        self.assertEqual(repaired.right_count, 4)
        self.assertEqual(repaired.left_count, 0)
        self.assertTrue(audit["repaired"])
        self.assertEqual(audit["corrected_counts"]["right_count"], 4)
        self.assertIn("constraint-reconciled", repaired.commentary or "")

    def test_aggregate_window_predictions_marks_empty_case(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_c",
            asset_paths={InputVariant.SFF3C: Path("clip_c.mp4")},
        )
        aggregated = _aggregate_window_predictions(clip, [], prompt_id="event_agent_proposal_guided")
        self.assertFalse(aggregated.parse_success)
        self.assertEqual(aggregated.left_count, 0)
        self.assertEqual(aggregated.right_count, 0)

    def test_aggregate_window_predictions_recovers_uncertain_direction_from_events(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_c2",
            asset_paths={InputVariant.SFF3C: Path("clip_c2.mp4")},
            upstream_direction=UpstreamDirection.RIGHT,
        )
        prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_c2__window_01",
            left_count=0,
            right_count=0,
            confidence=0.4,
            commentary="vertical but coherent motion",
            evidence_summary="two targets with uncertain lateral direction",
            parse_success=True,
            prompt_id="event_agent_window",
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.1,
                    "timestamp_end_sec": 7.3,
                    "direction": "uncertain",
                    "throughput_best_count": 2,
                    "estimated_count": 2,
                    "peak_simultaneous_count": 2,
                }
            ],
            events=[PassageEvent(timestamp_sec=0.1, direction="upstream", confidence=0.4, evidence_note="track start")],
        )

        aggregated = _aggregate_window_predictions(
            clip,
            [(20.0, prediction)],
            prompt_id="event_agent_proposal_guided",
        )

        self.assertEqual(aggregated.left_count, 0)
        self.assertEqual(aggregated.right_count, 2)
        self.assertEqual(aggregated.candidate_passages[0]["direction"], "uncertain")

    def test_aggregate_window_predictions_suppresses_weak_opposite_singleton_for_kenai_school(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_school",
            asset_paths={InputVariant.SFF3C: Path("clip_school.mp4")},
        )
        empty_window = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_school__window_01",
            left_count=0,
            right_count=0,
            confidence=0.9,
            commentary="empty",
            evidence_summary="no motion",
            parse_success=True,
            prompt_id="event_agent_window",
        )
        weak_opposite = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_school__window_03",
            left_count=1,
            right_count=0,
            confidence=0.4,
            commentary="single weak opposite target",
            evidence_summary="one faint opposite-direction entry",
            parse_success=True,
            prompt_id="event_agent_window",
            candidate_passages=[
                {
                    "timestamp_start_sec": 7.3,
                    "timestamp_end_sec": 7.7,
                    "direction": "left",
                    "throughput_best_count": 1,
                }
            ],
            events=[PassageEvent(timestamp_sec=7.3, direction="left", confidence=0.4, evidence_note="weak opposite")],
        )
        strong_school = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_school__window_04",
            left_count=0,
            right_count=10,
            confidence=0.7,
            commentary="strong right school",
            evidence_summary="dominant right-moving school",
            parse_success=True,
            prompt_id="event_agent_window",
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 20.0,
                    "direction": "right",
                    "throughput_best_count": 10,
                }
            ],
            events=[PassageEvent(timestamp_sec=2.0, direction="right", confidence=0.7, evidence_note="school")],
        )

        aggregated = _aggregate_window_predictions(
            clip,
            [
                (12.2, empty_window),
                (26.4, empty_window),
                (42.8, weak_opposite),
                (50.0, strong_school),
            ],
            prompt_id="event_agent_proposal_guided",
        )

        self.assertEqual(aggregated.left_count, 0)
        self.assertEqual(aggregated.right_count, 10)
        self.assertEqual(len([c for c in aggregated.candidate_passages if c.get("direction") == "left"]), 0)
        self.assertEqual(len([e for e in aggregated.events if e.direction == "left"]), 0)
        self.assertIn("suppressed weak opposite-direction singleton", aggregated.evidence_summary or "")

    def test_expand_window_bounds_pads_short_windows_inside_clip(self) -> None:
        start_sec, end_sec = _expand_window_bounds(5.0, 6.0, total_duration_sec=20.0, min_duration_sec=8.0)
        self.assertAlmostEqual(end_sec - start_sec, 8.0)
        self.assertGreaterEqual(start_sec, 0.0)
        self.assertLessEqual(end_sec, 20.0)

    def test_hybrid_proposals_prefers_backbone_candidate_and_dedupes_overlap(self) -> None:
        motion = [
            EventProposal(
                proposal_id="motion-1",
                timestamp_start_sec=48.0,
                timestamp_end_sec=80.0,
                score=0.7,
                source="motion_energy",
                representation="sff3c",
                reasoning="temporal_abstraction",
            ),
            EventProposal(
                proposal_id="motion-2",
                timestamp_start_sec=10.0,
                timestamp_end_sec=12.0,
                score=0.9,
                source="motion_energy",
                representation="sff3c",
                reasoning="temporal_abstraction",
            ),
        ]
        backbone = [
            EventProposal(
                proposal_id="backbone-1",
                timestamp_start_sec=50.0,
                timestamp_end_sec=75.0,
                score=5.0,
                source="backbone_candidate_passages",
                representation="sff3c",
                reasoning="candidate_passage_projection",
            )
        ]
        proposals = _hybrid_proposals(
            motion_proposals=motion,
            backbone_proposals=backbone,
            max_proposals=3,
        )
        self.assertEqual(len(proposals), 2)
        self.assertEqual(proposals[1].proposal_id, "backbone-1")
        self.assertEqual(proposals[0].proposal_id, "motion-2")

    def test_hybrid_proposals_can_keep_overlapping_stream_tiles(self) -> None:
        motion = [
            EventProposal(
                proposal_id="stream-1",
                timestamp_start_sec=35.0,
                timestamp_end_sec=53.0,
                score=1.6,
                source="stream_tiling",
                representation="sff3c",
                reasoning="stream_throughput_tiling",
            ),
            EventProposal(
                proposal_id="stream-2",
                timestamp_start_sec=47.6,
                timestamp_end_sec=59.8,
                score=1.6,
                source="stream_tiling",
                representation="sff3c",
                reasoning="stream_throughput_tiling",
            ),
        ]
        backbone = [
            EventProposal(
                proposal_id="backbone-1",
                timestamp_start_sec=35.0,
                timestamp_end_sec=59.8,
                score=10.0,
                source="backbone_candidate_passages",
                representation="sff3c",
                reasoning="candidate_passage_projection",
            )
        ]

        proposals = _hybrid_proposals(
            motion_proposals=motion,
            backbone_proposals=backbone,
            max_proposals=6,
            keep_stream_tiling_overlap=True,
        )

        self.assertEqual(
            [proposal.proposal_id for proposal in proposals],
            ["backbone-1", "stream-1", "stream-2"],
        )

    def test_dense_window_accepts_higher_same_direction_throughput(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_d",
            asset_paths={InputVariant.SFF3C: Path("clip_d.mp4")},
        )
        proposal = EventProposal(
            proposal_id="backbone-1",
            timestamp_start_sec=50.0,
            timestamp_end_sec=75.0,
            score=7.0,
            source="backbone_candidate_passages",
            representation="sff3c",
            reasoning="candidate_passage_projection",
            direction_hint="right",
        )
        base_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d",
            left_count=0,
            right_count=8,
            parse_success=True,
        )
        dense_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_d",
            left_count=0,
            right_count=11,
            parse_success=True,
        )
        accepted, reason = _should_accept_dense_window_prediction(
            clip,
            proposal,
            base_prediction,
            dense_prediction,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted_higher_dense_throughput")

    def test_dense_window_rejects_direction_flip(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_e",
            asset_paths={InputVariant.SFF3C: Path("clip_e.mp4")},
        )
        proposal = EventProposal(
            proposal_id="backbone-2",
            timestamp_start_sec=50.0,
            timestamp_end_sec=75.0,
            score=7.0,
            source="backbone_candidate_passages",
            representation="sff3c",
            reasoning="candidate_passage_projection",
            direction_hint="right",
        )
        base_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_e",
            left_count=0,
            right_count=8,
            parse_success=True,
        )
        dense_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_e",
            left_count=11,
            right_count=0,
            parse_success=True,
        )
        accepted, reason = _should_accept_dense_window_prediction(
            clip,
            proposal,
            base_prediction,
            dense_prediction,
        )
        self.assertFalse(accepted)
        self.assertIn("conflicts", reason)

    def test_dense_window_accepts_direction_recovery_when_base_flips(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_e2",
            asset_paths={InputVariant.SFF3C: Path("clip_e2.mp4")},
        )
        proposal = EventProposal(
            proposal_id="backbone-2",
            timestamp_start_sec=0.0,
            timestamp_end_sec=44.0,
            score=4.0,
            source="backbone_candidate_passages",
            representation="sff3c",
            reasoning="candidate_passage_projection",
            direction_hint="right",
            metadata={
                "throughput_best_count": 4,
                "peak_simultaneous_count": 2,
                "wave_count": 2,
            },
        )
        base_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_e2",
            left_count=12,
            right_count=0,
            parse_success=True,
        )
        dense_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_e2",
            left_count=0,
            right_count=4,
            parse_success=True,
        )
        accepted, reason = _should_accept_dense_window_prediction(
            clip,
            proposal,
            base_prediction,
            dense_prediction,
        )
        self.assertTrue(accepted)
        self.assertEqual(reason, "accepted_dense_direction_recovery")

    def test_effective_max_proposals_expands_to_backbone_coverage_cap(self) -> None:
        backbone = [
            EventProposal(
                proposal_id=f"backbone-{index}",
                timestamp_start_sec=float(index * 10),
                timestamp_end_sec=float(index * 10 + 4),
                score=1.0,
                source="backbone_candidate_passages",
                representation="sff3c",
                reasoning="candidate_passage_projection",
            )
            for index in range(5)
        ]
        self.assertEqual(_effective_max_proposals(3, backbone), 5)

    def test_dense_window_runs_for_long_moderate_throughput_backbone_window(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="elwha",
            clip_id="clip_f",
            asset_paths={InputVariant.SFF3C: Path("clip_f.mp4")},
        )
        proposal = EventProposal(
            proposal_id="backbone-long",
            timestamp_start_sec=43.0,
            timestamp_end_sec=54.0,
            score=2.0,
            source="backbone_candidate_passages",
            representation="sff3c",
            reasoning="candidate_passage_projection",
            metadata={
                "throughput_best_count": 2,
                "peak_simultaneous_count": 2,
                "wave_count": 1,
            },
        )
        base_prediction = PassagePrediction(
            domain="elwha",
            clip_id="clip_f",
            left_count=0,
            right_count=5,
            parse_success=True,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 5.5,
                    "direction": "right",
                    "throughput_best_count": 4,
                },
                {
                    "timestamp_start_sec": 5.5,
                    "timestamp_end_sec": 11.0,
                    "direction": "right",
                    "throughput_best_count": 1,
                },
            ],
        )
        self.assertTrue(_should_run_dense_window_refinement(clip, proposal, base_prediction))

    def test_select_risk_assessment_can_promote_proposal_flags_when_global_is_clean(self) -> None:
        clip_key = "kenai-val/2018-06-03-JD154_LeftNear_Stratum1_Set1_LN_2018-06-03_100000_4520_5061"
        profile = resolve_site_profile("kenai-val", clip_key=clip_key)
        global_result = {
            "clip_key": clip_key,
            "domain": "kenai-val",
            "prediction": {
                "left_count": 0,
                "right_count": 5,
                "parse_success": True,
                "confidence": 0.6,
                "candidate_passages": [
                    {
                        "direction": "right",
                        "peak_simultaneous_count": 1,
                        "episode_duration_sec": 30,
                        "wave_count": 1,
                    },
                    {
                        "direction": "right",
                        "peak_simultaneous_count": 1,
                        "episode_duration_sec": 28,
                        "wave_count": 1,
                    },
                ],
            },
        }
        proposal_result = {
            "clip_key": clip_key,
            "domain": "kenai-val",
            "prediction": {
                "left_count": 0,
                "right_count": 17,
                "parse_success": True,
                "confidence": 0.6,
                "candidate_passages": [
                    {
                        "direction": "right",
                        "peak_simultaneous_count": 2,
                        "episode_duration_sec": 25,
                        "wave_count": 1,
                    },
                    {
                        "direction": "right",
                        "peak_simultaneous_count": 2,
                        "episode_duration_sec": 25,
                        "wave_count": 1,
                    },
                    {
                        "direction": "right",
                        "peak_simultaneous_count": 2,
                        "episode_duration_sec": 58,
                        "wave_count": 1,
                    },
                ],
            },
        }
        assessment = _select_risk_assessment(
            global_result=global_result,
            proposal_result=proposal_result,
            direction="right",
            site_profile=profile,
        )
        self.assertIn("multi_episode_trickle_overcount_risk", assessment.flags)
        self.assertEqual(assessment.predicted_total, 17)
        seed = _select_risk_source_result(
            global_result=global_result,
            proposal_result=proposal_result,
            direction="right",
            site_profile=profile,
        )
        self.assertEqual(seed["prediction"]["right_count"], 17)

    def test_dense_window_skips_short_singleton_window(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="elwha",
            clip_id="clip_g",
            asset_paths={InputVariant.SFF3C: Path("clip_g.mp4")},
        )
        proposal = EventProposal(
            proposal_id="backbone-short",
            timestamp_start_sec=70.0,
            timestamp_end_sec=74.0,
            score=1.0,
            source="backbone_candidate_passages",
            representation="sff3c",
            reasoning="candidate_passage_projection",
            metadata={
                "throughput_best_count": 1,
                "peak_simultaneous_count": 1,
                "wave_count": 1,
            },
        )
        base_prediction = PassagePrediction(
            domain="elwha",
            clip_id="clip_g",
            left_count=0,
            right_count=1,
            parse_success=True,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 4.0,
                    "direction": "right",
                    "throughput_best_count": 1,
                }
            ],
        )
        self.assertFalse(_should_run_dense_window_refinement(clip, proposal, base_prediction))

    def test_trickle_prompt_context_includes_first_pass_summary(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="kenai",
            direction="right",
            predicted_total=21,
            candidate_count=4,
            total_wave_count=4,
            total_peak=7.0,
            max_peak=2.0,
            max_duration_sec=28.0,
            confidence=0.45,
            parse_success=True,
            risk_level="medium",
            clip_hints=("near_view",),
            flags=("multi_episode_trickle_overcount_risk",),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip("custom_trickle_reduction", risk_assessment=assessment)
        self.assertIn("first-pass total estimate: 21", prompt.prompt_text)
        self.assertIn("candidate episode count: 4", prompt.prompt_text)

    def test_hidden_crossing_prompt_context_mentions_briefly_hidden_crossings(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="kenai",
            direction="right",
            predicted_total=6,
            candidate_count=1,
            total_wave_count=1,
            total_peak=5.0,
            max_peak=5.0,
            max_duration_sec=22.0,
            confidence=0.78,
            parse_success=True,
            risk_level="medium",
            clip_hints=("near_view",),
            flags=("hidden_crossing_undercount_risk",),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip(
            "custom_high_throughput",
            risk_assessment=assessment,
            window_runs=[
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 4,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 2,
                    }
                },
            ],
        )
        self.assertIn("briefly hidden crossings", prompt.prompt_text)
        self.assertIn("first-pass total estimate: 6", prompt.prompt_text)
        self.assertIn("local windows with positive fish evidence: 2/2", prompt.prompt_text)
        self.assertIn("strongest local window total: 4", prompt.prompt_text)
        self.assertIn("Multiple local windows contain positive motion evidence", prompt.prompt_text)

    def test_single_school_prompt_context_mentions_same_school_hidden_entrants(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="kenai",
            direction="right",
            predicted_total=4,
            candidate_count=1,
            total_wave_count=1,
            total_peak=4.0,
            max_peak=4.0,
            max_duration_sec=14.0,
            confidence=0.88,
            parse_success=True,
            risk_level="medium",
            clip_hints=("near_view",),
            flags=("single_school_visibility_dropout_risk",),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip(
            "custom_high_throughput",
            risk_assessment=assessment,
            window_runs=[
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 4,
                    }
                },
            ],
        )
        self.assertIn("single coherent school", prompt.prompt_text)
        self.assertIn("conservative +1 or +2", prompt.prompt_text)
        self.assertIn("local windows with positive fish evidence: 1/2", prompt.prompt_text)
        self.assertIn("zero-evidence windows: 1", prompt.prompt_text)
        self.assertIn("strongest local window total: 4", prompt.prompt_text)
        self.assertIn("Only one local window contains the visible school", prompt.prompt_text)

    def test_single_school_high_throughput_prompt_mentions_strongest_window_is_lower_bound(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="kenai",
            direction="right",
            predicted_total=8,
            candidate_count=1,
            total_wave_count=1,
            total_peak=3.0,
            max_peak=3.0,
            max_duration_sec=26.0,
            confidence=0.6,
            parse_success=True,
            risk_level="medium",
            clip_hints=("near_view",),
            flags=("single_school_high_throughput_undercount_risk",),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip(
            "custom_high_throughput",
            risk_assessment=assessment,
            window_runs=[
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 8,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
            ],
        )
        self.assertIn("single dense school", prompt.prompt_text)
        self.assertIn("first-pass total estimate: 8", prompt.prompt_text)
        self.assertIn("local windows with positive fish evidence: 1/3", prompt.prompt_text)
        self.assertIn("strongest local window total: 8", prompt.prompt_text)
        self.assertIn("lower bound on throughput", prompt.prompt_text)

    def test_hidden_crossing_prompt_mentions_single_strong_window_not_upper_bound(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="kenai",
            direction="right",
            predicted_total=6,
            candidate_count=1,
            total_wave_count=1,
            total_peak=5.0,
            max_peak=5.0,
            max_duration_sec=19.0,
            confidence=0.7,
            parse_success=True,
            risk_level="high",
            clip_hints=("near_view",),
            flags=("hidden_crossing_undercount_risk",),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip(
            "custom_high_throughput",
            risk_assessment=assessment,
            window_runs=[
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 6,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
            ],
        )
        self.assertIn("strongest local window total: 6", prompt.prompt_text)
        self.assertIn("Do not cap the total at that strongest-window count", prompt.prompt_text)

    def test_single_school_high_throughput_prompt_takes_priority_over_hidden_crossing(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="kenai",
            direction="right",
            predicted_total=8,
            candidate_count=1,
            total_wave_count=1,
            total_peak=4.0,
            max_peak=4.0,
            max_duration_sec=25.0,
            confidence=0.86,
            parse_success=True,
            risk_level="high",
            clip_hints=("near_view",),
            flags=("hidden_crossing_undercount_risk", "single_school_high_throughput_undercount_risk"),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip(
            "custom_high_throughput",
            risk_assessment=assessment,
            window_runs=[
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 8,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
            ],
        )
        self.assertIn("single dense school", prompt.prompt_text)
        self.assertIn("lower bound on throughput", prompt.prompt_text)

    def test_low_count_prompt_context_mentions_sparse_single_window_guardrail(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="kenai",
            direction="right",
            predicted_total=0,
            candidate_count=0,
            total_wave_count=0,
            total_peak=0.0,
            max_peak=0.0,
            max_duration_sec=0.0,
            confidence=0.9,
            parse_success=True,
            risk_level="medium",
            clip_hints=("near_view",),
            flags=("low_count_far_view_risk",),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip(
            "custom_low_count",
            risk_assessment=assessment,
            window_runs=[
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 10,
                    }
                },
                {
                    "selected_prediction": {
                        "left_count": 0,
                        "right_count": 0,
                    }
                },
            ],
        )
        self.assertIn("first-pass total estimate: 0", prompt.prompt_text)
        self.assertIn("local windows with positive fish evidence: 1/3", prompt.prompt_text)
        self.assertIn("zero-evidence windows: 2", prompt.prompt_text)
        self.assertIn("strongest local window total: 10", prompt.prompt_text)
        self.assertIn("Do not trust a large raw window total by itself", prompt.prompt_text)

    def test_stream_throughput_prompt_context_mentions_stream_evidence(self) -> None:
        assessment = DomainShiftAssessment(
            site_profile_id="nushagak",
            direction="left",
            predicted_total=11,
            candidate_count=1,
            total_wave_count=1,
            total_peak=4.0,
            max_peak=4.0,
            max_duration_sec=24.0,
            confidence=0.72,
            parse_success=True,
            risk_level="high",
            clip_hints=(),
            flags=("dense_stream_throughput_risk",),
            reason="synthetic",
        )
        prompt = _custom_branch_prompt_for_clip(
            "custom_stream_throughput",
            risk_assessment=assessment,
            direction="left",
            window_runs=[
                {"selected_prediction": {"left_count": 18, "right_count": 0}},
                {"selected_prediction": {"left_count": 12, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            proposals=[
                {
                    "source": "backbone_candidate_passages",
                    "direction_hint": "left",
                    "metadata": {
                        "throughput_best_count": 20,
                        "peak_simultaneous_count": 5,
                        "wave_count": 1,
                    },
                }
            ],
        )
        self.assertIn("dominant-direction stream throughput", prompt.prompt_text)
        self.assertIn("human-expert style procedure", prompt.prompt_text)
        self.assertIn("Lock onto a few clear anchor streaks", prompt.prompt_text)
        self.assertIn("dominant direction: left", prompt.prompt_text)
        self.assertIn("active local windows: 2/3", prompt.prompt_text)
        self.assertIn("strongest local window total: 18", prompt.prompt_text)
        self.assertIn("backbone throughput hint: 20", prompt.prompt_text)
        self.assertIn("lower bound on throughput", prompt.prompt_text)
        self.assertIn("conservatively impute throughflow", prompt.prompt_text)

    def test_trickle_override_requires_reduction_relative_to_global(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_h",
            asset_paths={InputVariant.SFF3C: Path("clip_h.mp4")},
        )
        global_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_h",
            left_count=0,
            right_count=6,
            parse_success=True,
        )
        custom_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_h",
            left_count=0,
            right_count=8,
            parse_success=True,
        )
        self.assertFalse(
            _should_apply_routed_override(
                clip,
                branch_id="custom_trickle_reduction",
                global_prediction=global_prediction,
                custom_prediction=custom_prediction,
            )
        )
        reduced_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_h",
            left_count=0,
            right_count=3,
            parse_success=True,
        )
        self.assertTrue(
            _should_apply_routed_override(
                clip,
                branch_id="custom_trickle_reduction",
                global_prediction=global_prediction,
                custom_prediction=reduced_prediction,
            )
        )

    def test_low_count_override_applies_for_direction_cleanup(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_i",
            asset_paths={InputVariant.SFF3C: Path("clip_i.mp4")},
        )
        global_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_i",
            left_count=2,
            right_count=2,
            parse_success=True,
        )
        custom_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_i",
            left_count=0,
            right_count=2,
            parse_success=True,
        )
        self.assertTrue(
            _should_apply_routed_override(
                clip,
                branch_id="custom_low_count",
                global_prediction=global_prediction,
                custom_prediction=custom_prediction,
            )
        )

    def test_stream_override_applies_when_stream_branch_is_selected(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="nushagak",
            clip_id="clip_stream",
            asset_paths={InputVariant.SFF3C: Path("clip_stream.mp4")},
        )
        global_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream",
            left_count=0,
            right_count=11,
            parse_success=True,
        )
        custom_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream",
            left_count=0,
            right_count=24,
            parse_success=True,
        )
        self.assertTrue(
            _should_apply_routed_override(
                clip,
                branch_id="custom_stream_throughput",
                global_prediction=global_prediction,
                custom_prediction=custom_prediction,
            )
        )

    def test_trickle_reduction_selection_prefers_smallest_positive_parseable_total(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_j",
            asset_paths={InputVariant.SFF3C: Path("clip_j.mp4")},
        )
        prediction_a = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_j",
            left_count=0,
            right_count=8,
            parse_success=True,
            confidence=0.7,
        )
        prediction_b = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_j",
            left_count=0,
            right_count=2,
            parse_success=True,
            confidence=0.6,
        )
        prediction_c = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_j",
            left_count=0,
            right_count=0,
            parse_success=True,
            confidence=0.9,
        )
        selected = _select_trickle_reduction_prediction(clip, [prediction_a, prediction_b, prediction_c])
        self.assertEqual(selected.right_count, 2)

    def test_stream_throughput_selection_prefers_same_direction_high_total_and_low_counterflow(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="nushagak",
            clip_id="clip_stream_select",
            asset_paths={InputVariant.SFF3C: Path("clip_stream_select.mp4")},
            upstream_direction=UpstreamDirection.RIGHT,
        )
        prediction_a = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_select",
            left_count=0,
            right_count=18,
            parse_success=True,
            confidence=0.6,
        )
        prediction_b = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_select",
            left_count=1,
            right_count=24,
            parse_success=True,
            confidence=0.7,
        )
        prediction_c = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_select",
            left_count=0,
            right_count=22,
            parse_success=True,
            confidence=0.65,
        )
        selected = _select_stream_throughput_prediction(clip, [prediction_a, prediction_b, prediction_c])
        self.assertEqual(selected.left_count, 0)
        self.assertEqual(selected.right_count, 22)

    def test_stream_routing_direction_uses_aggregate_predictions_not_global_only(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="nushagak",
            clip_id="clip_stream_route",
            asset_paths={InputVariant.SFF3C: Path("clip_stream_route.mp4")},
            upstream_direction=UpstreamDirection.RIGHT,
        )
        global_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_route",
            left_count=0,
            right_count=0,
            parse_success=True,
        )
        proposal_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_route",
            left_count=15,
            right_count=0,
            parse_success=True,
        )
        repair_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_route",
            left_count=15,
            right_count=0,
            parse_success=True,
        )
        self.assertEqual(
            _select_stream_routing_direction(
                clip,
                global_prediction,
                proposal_prediction,
                repair_prediction,
            ),
            "left",
        )

    def test_stream_routing_direction_defaults_left_for_nushagak_when_right_is_only_moderate(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="nushagak",
            clip_id="clip_stream_route_prior",
            asset_paths={InputVariant.SFF3C: Path("clip_stream_route_prior.mp4")},
            upstream_direction=UpstreamDirection.RIGHT,
        )
        global_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_route_prior",
            left_count=0,
            right_count=6,
            parse_success=True,
        )
        proposal_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_route_prior",
            left_count=7,
            right_count=0,
            parse_success=True,
        )
        repair_prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_route_prior",
            left_count=0,
            right_count=8,
            parse_success=True,
        )
        self.assertEqual(
            _select_stream_routing_direction(
                clip,
                global_prediction,
                proposal_prediction,
                repair_prediction,
            ),
            "left",
        )

    def test_project_prediction_to_direction_moves_stream_total_to_left(self) -> None:
        prediction = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_project",
            left_count=0,
            right_count=16,
            candidate_passages=[
                {
                    "direction": "right",
                    "estimated_count": 16,
                    "throughput_best_count": 16,
                }
            ],
            events=[PassageEvent(timestamp_sec=5.0, direction="right", confidence=0.6)],
            parse_success=True,
        )
        projected = _project_prediction_to_direction(prediction, direction="left")
        self.assertEqual(projected.left_count, 16)
        self.assertEqual(projected.right_count, 0)
        self.assertEqual(projected.candidate_passages[0]["direction"], "left")
        self.assertEqual(projected.events[0].direction, "left")

    def test_select_stream_throughput_prediction_uses_explicit_direction_over_upstream(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="nushagak",
            clip_id="clip_stream_select",
            asset_paths={InputVariant.SFF3C: Path("clip_stream_select.mp4")},
            upstream_direction=UpstreamDirection.RIGHT,
        )
        right_heavy = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_select",
            left_count=0,
            right_count=20,
            parse_success=True,
            confidence=0.7,
        )
        left_heavy = PassagePrediction(
            domain="nushagak",
            clip_id="clip_stream_select",
            left_count=18,
            right_count=0,
            parse_success=True,
            confidence=0.6,
        )
        selected = _select_stream_throughput_prediction(
            clip,
            [right_heavy, left_heavy],
            direction="left",
        )
        self.assertIs(selected, left_heavy)

    def test_post_selection_stream_global_floor_prefers_strong_local_stream_evidence(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
            "risk_assessment": {
                "candidate_count": 3,
                "max_peak": 3.0,
                "max_duration_sec": 23.9,
            },
            "routing_direction": "left",
            "selected_action": "sff3c:global",
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 0,
                "total_count": 0,
                "parse_success": True,
                "confidence": 0.85,
            },
            "action_predictions": {
                "global_direct": {
                    "left_count": 0,
                    "right_count": 0,
                    "total_count": 0,
                    "parse_success": True,
                    "confidence": 0.85,
                },
                "proposal_guided": {
                    "left_count": 22,
                    "right_count": 0,
                    "total_count": 22,
                    "parse_success": True,
                    "confidence": 0.72,
                },
                "local_repair": {
                    "left_count": 15,
                    "right_count": 0,
                    "total_count": 15,
                    "parse_success": True,
                    "confidence": 0.72,
                },
            },
        }

        updated, audit = _apply_post_selection_stream_global_floor(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_action"], "sff3c:proposal_guided")
        self.assertEqual(updated["selected_branch"], "baseline")
        self.assertEqual(updated["prediction"]["left_count"], 22)
        self.assertEqual(updated["prediction"]["right_count"], 0)

    def test_post_selection_stream_candidate_floor_prefers_stronger_custom_stream_candidate(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk", "multiwave_high_peak_undercount_risk"],
            "risk_assessment": {
                "candidate_count": 3,
                "total_wave_count": 3,
                "max_duration_sec": 16.0,
            },
            "routing_direction": "left",
            "selected_action": "sff3c:global",
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 6,
                "right_count": 0,
                "total_count": 6,
                "parse_success": True,
                "confidence": 0.4,
            },
            "action_predictions": {
                "proposal_guided": {
                    "left_count": 12,
                    "right_count": 0,
                    "total_count": 12,
                    "parse_success": True,
                    "confidence": 0.62,
                },
                "local_repair": {
                    "left_count": 8,
                    "right_count": 0,
                    "total_count": 8,
                    "parse_success": True,
                    "confidence": 0.62,
                },
            },
            "custom_branch_runs": [
                {
                    "branch_id": "custom_stream_throughput",
                    "prediction": {
                        "left_count": 12,
                        "right_count": 0,
                        "total_count": 12,
                        "parse_success": True,
                        "confidence": 0.6,
                        "candidate_passages": [
                            {
                                "direction": "left",
                                "episode_duration_sec": 41.8,
                            }
                        ],
                    },
                }
            ],
        }

        updated, audit = _apply_post_selection_stream_candidate_floor(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_action"], "sff3c:custom_stream_throughput")
        self.assertEqual(updated["selected_branch"], "custom_stream_throughput")
        self.assertEqual(updated["prediction"]["left_count"], 12)
        self.assertEqual(updated["post_selection_stream_candidate_floor"]["chosen_longest_span"], 41.8)

    def test_post_selection_stream_multiwave_uplift_boosts_nushagak_baseline(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk", "multiwave_high_peak_undercount_risk"],
            "risk_assessment": {
                "candidate_count": 6,
                "total_wave_count": 6,
                "total_peak": 21.0,
                "max_peak": 4.0,
                "max_duration_sec": 12.5,
            },
            "routing_direction": "left",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 33,
                "right_count": 0,
                "total_count": 33,
                "parse_success": True,
                "confidence": 0.65,
            },
            "action_predictions": {
                "proposal_guided": {
                    "left_count": 33,
                    "right_count": 0,
                    "total_count": 33,
                    "parse_success": True,
                    "confidence": 0.65,
                }
            },
        }

        updated, audit = _apply_post_selection_stream_multiwave_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_branch"], "custom_stream_multiwave_uplift")
        self.assertEqual(updated["prediction"]["left_count"], 45)
        self.assertEqual(updated["prediction"]["right_count"], 0)

    def test_post_selection_stream_tile_pair_floor_uses_top_two_windows(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 13,
                "right_count": 0,
                "total_count": 13,
                "parse_success": True,
                "confidence": 0.6,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 10, "right_count": 0}},
                {"selected_prediction": {"left_count": 8, "right_count": 0}},
            ],
            "action_predictions": {
                "custom_stream_throughput": {
                    "left_count": 13,
                    "right_count": 0,
                    "total_count": 13,
                    "parse_success": True,
                    "confidence": 0.6,
                }
            },
        }

        updated, audit = _apply_post_selection_stream_tile_pair_floor(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_action"], "sff3c:custom_stream_tile_pair_floor")
        self.assertEqual(updated["selected_branch"], "custom_stream_tile_pair_floor")
        self.assertEqual(updated["prediction"]["left_count"], 16)
        self.assertEqual(updated["post_selection_stream_tile_pair_floor"]["second_target_window_total"], 8)

    def test_post_selection_stream_tile_pair_floor_skips_single_positive_window(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 13,
                "right_count": 0,
                "total_count": 13,
                "parse_success": True,
                "confidence": 0.6,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 10, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_tile_pair_floor(result_record)

        self.assertIsNone(audit)
        self.assertEqual(updated["prediction"]["left_count"], 13)

    def test_post_selection_stream_longspan_uplift_boosts_long_peak3_stream(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 12,
                "right_count": 0,
                "total_count": 12,
                "parse_success": True,
                "confidence": 0.6,
                "candidate_passages": [
                    {
                        "timestamp_start_sec": 0.0,
                        "timestamp_end_sec": 59.8,
                        "direction": "left",
                        "episode_duration_sec": 59.8,
                        "peak_simultaneous_count": 3,
                        "throughput_best_count": 12,
                    }
                ],
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 6, "right_count": 0}},
                {"selected_prediction": {"left_count": 2, "right_count": 0}},
            ],
            "action_predictions": {
                "custom_stream_throughput": {
                    "left_count": 12,
                    "right_count": 0,
                    "total_count": 12,
                    "parse_success": True,
                    "confidence": 0.6,
                }
            },
        }

        updated, audit = _apply_post_selection_stream_longspan_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_action"], "sff3c:custom_stream_longspan_uplift")
        self.assertEqual(updated["selected_branch"], "custom_stream_longspan_uplift")
        self.assertEqual(updated["prediction"]["left_count"], 20)

    def test_post_selection_stream_longspan_uplift_skips_short_or_already_extrapolated_stream(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 20,
                "right_count": 0,
                "total_count": 20,
                "parse_success": True,
                "confidence": 0.6,
                "candidate_passages": [
                    {
                        "timestamp_start_sec": 0.0,
                        "timestamp_end_sec": 32.0,
                        "direction": "left",
                        "episode_duration_sec": 32.0,
                        "peak_simultaneous_count": 3,
                        "throughput_best_count": 20,
                    }
                ],
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 10, "right_count": 0}},
                {"selected_prediction": {"left_count": 8, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_longspan_uplift(result_record)

        self.assertIsNone(audit)
        self.assertEqual(updated["prediction"]["left_count"], 20)

    def test_post_selection_stream_window_stack_uplift_boosts_multiwindow_stream(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk", "multiwave_high_peak_undercount_risk"],
            "risk_assessment": {
                "candidate_count": 4,
                "total_wave_count": 4,
                "total_peak": 20.0,
                "max_peak": 5.0,
                "max_duration_sec": 18.0,
            },
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 35,
                "right_count": 0,
                "total_count": 35,
                "parse_success": True,
                "confidence": 0.6,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 5, "right_count": 0}},
                {"selected_prediction": {"left_count": 12, "right_count": 0}},
                {"selected_prediction": {"left_count": 5, "right_count": 0}},
                {"selected_prediction": {"left_count": 12, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_window_stack_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_action"], "sff3c:custom_stream_window_stack_uplift")
        self.assertEqual(updated["selected_branch"], "custom_stream_window_stack_uplift")
        self.assertEqual(updated["prediction"]["left_count"], 54)

    def test_post_selection_stream_window_stack_uplift_can_extend_multiwave_uplift(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk", "multiwave_high_peak_undercount_risk"],
            "risk_assessment": {
                "candidate_count": 6,
                "total_wave_count": 6,
                "total_peak": 21.0,
                "max_peak": 4.0,
                "max_duration_sec": 12.5,
            },
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_multiwave_uplift",
            "selected_branch": "custom_stream_multiwave_uplift",
            "prediction": {
                "left_count": 45,
                "right_count": 0,
                "total_count": 45,
                "parse_success": True,
                "confidence": 0.6,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 11, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 5, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 12, "right_count": 0}},
                {"selected_prediction": {"left_count": 5, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_window_stack_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["prediction"]["left_count"], 54)
        self.assertEqual(updated["selected_branch"], "custom_stream_window_stack_uplift")

    def test_post_selection_stream_window_stack_uplift_skips_already_extrapolated_totals(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk", "multiwave_high_peak_undercount_risk"],
            "risk_assessment": {
                "candidate_count": 4,
                "total_wave_count": 4,
                "total_peak": 20.0,
                "max_peak": 5.0,
                "max_duration_sec": 18.0,
            },
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 54,
                "right_count": 0,
                "total_count": 54,
                "parse_success": True,
                "confidence": 0.6,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 5, "right_count": 0}},
                {"selected_prediction": {"left_count": 12, "right_count": 0}},
                {"selected_prediction": {"left_count": 5, "right_count": 0}},
                {"selected_prediction": {"left_count": 12, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_window_stack_uplift(result_record)

        self.assertIsNone(audit)
        self.assertEqual(updated["prediction"]["left_count"], 54)

    def test_post_selection_stream_window_stack_uplift_skips_short_high_peak_stream(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk", "multiwave_high_peak_undercount_risk"],
            "risk_assessment": {
                "candidate_count": 4,
                "total_wave_count": 4,
                "total_peak": 34.0,
                "max_peak": 15.0,
                "max_duration_sec": 5.0,
            },
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 61,
                "right_count": 0,
                "total_count": 61,
                "parse_success": True,
                "confidence": 0.65,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 30, "right_count": 0}},
                {"selected_prediction": {"left_count": 30, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 1, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_window_stack_uplift(result_record)

        self.assertIsNone(audit)
        self.assertEqual(updated["prediction"]["left_count"], 61)

    def test_post_selection_stream_embedded_wave_uplift_boosts_single_long_candidate_with_hidden_waves(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 13,
                "right_count": 0,
                "total_count": 13,
                "parse_success": True,
                "confidence": 0.75,
                "candidate_passages": [
                    {
                        "timestamp_start_sec": 15.0,
                        "timestamp_end_sec": 59.8,
                        "direction": "left",
                        "episode_duration_sec": 44.8,
                        "peak_simultaneous_count": 2,
                        "wave_count": 3,
                        "throughput_best_count": 13,
                    }
                ],
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 10, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_embedded_wave_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_action"], "sff3c:custom_stream_embedded_wave_uplift")
        self.assertEqual(updated["selected_branch"], "custom_stream_embedded_wave_uplift")
        self.assertEqual(updated["prediction"]["left_count"], 20)

    def test_post_selection_stream_embedded_wave_uplift_skips_multiwindow_or_short_streams(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "nushagak",
            "site_profile": {
                "site_id": "nushagak",
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
            "routing_direction": "left",
            "selected_action": "sff3c:custom_stream_throughput",
            "selected_branch": "custom_stream_throughput",
            "prediction": {
                "left_count": 13,
                "right_count": 0,
                "total_count": 13,
                "parse_success": True,
                "confidence": 0.75,
                "candidate_passages": [
                    {
                        "timestamp_start_sec": 15.0,
                        "timestamp_end_sec": 32.0,
                        "direction": "left",
                        "episode_duration_sec": 17.0,
                        "peak_simultaneous_count": 2,
                        "wave_count": 2,
                        "throughput_best_count": 13,
                    }
                ],
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 10, "right_count": 0}},
                {"selected_prediction": {"left_count": 8, "right_count": 0}},
            ],
        }

        updated, audit = _apply_post_selection_stream_embedded_wave_uplift(result_record)

        self.assertIsNone(audit)
        self.assertEqual(updated["prediction"]["left_count"], 13)

    def test_stream_tiling_proposals_add_multiple_long_windows(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="nushagak",
            clip_id="clip_stream_tiles",
            asset_paths={InputVariant.SFF3C: Path("clip_stream_tiles.mp4")},
            duration_seconds=60.0,
            upstream_direction=UpstreamDirection.RIGHT,
        )
        backbone = [
            EventProposal(
                proposal_id="backbone-01",
                timestamp_start_sec=5.0,
                timestamp_end_sec=55.0,
                score=12.0,
                source="backbone_candidate_passages",
                representation="sff3c",
                reasoning="candidate_passage_projection",
            )
        ]
        tiles = _stream_tiling_proposals(
            clip,
            backbone_proposals=backbone,
            representation="sff3c",
        )
        self.assertGreaterEqual(len(tiles), 3)
        self.assertTrue(all(tile.source == "stream_tiling" for tile in tiles))
        self.assertTrue(all((tile.timestamp_end_sec - tile.timestamp_start_sec) >= 12.0 for tile in tiles))

    def test_stream_tiling_proposals_can_fallback_to_backbone_span_when_duration_missing(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="nushagak",
            clip_id="clip_stream_tiles_missing_duration",
            asset_paths={InputVariant.SFF3C: Path("clip_stream_tiles_missing_duration.mp4")},
            duration_seconds=None,
            upstream_direction=UpstreamDirection.RIGHT,
        )
        backbone = [
            EventProposal(
                proposal_id="backbone-01",
                timestamp_start_sec=35.0,
                timestamp_end_sec=59.8,
                score=10.0,
                source="backbone_candidate_passages",
                representation="sff3c",
                reasoning="candidate_passage_projection",
            )
        ]
        tiles = _stream_tiling_proposals(
            clip,
            backbone_proposals=backbone,
            representation="sff3c",
        )
        self.assertGreaterEqual(len(tiles), 1)
        self.assertTrue(all(tile.source == "stream_tiling" for tile in tiles))
        self.assertGreaterEqual(tiles[0].timestamp_start_sec, 35.0)
        self.assertLessEqual(tiles[-1].timestamp_end_sec, 59.8)

    def test_select_risk_source_result_stream_profile_prefers_largest_total_even_without_stream_flag(self) -> None:
        profile = resolve_site_profile("nushagak", clip_key="nushagak/clip_stream_seed")
        global_result = {
            "clip_key": "nushagak/clip_stream_seed",
            "domain": "nushagak",
            "prediction": {
                "left_count": 0,
                "right_count": 0,
                "parse_success": True,
                "confidence": 0.9,
                "candidate_passages": [],
            },
        }
        proposal_result = {
            "clip_key": "nushagak/clip_stream_seed",
            "domain": "nushagak",
            "prediction": {
                "left_count": 0,
                "right_count": 8,
                "parse_success": True,
                "confidence": 0.6,
                "candidate_passages": [
                    {
                        "direction": "right",
                        "peak_simultaneous_count": 2,
                        "episode_duration_sec": 15.0,
                        "wave_count": 1,
                        "throughput_best_count": 8,
                    }
                ],
            },
            "window_activity": {
                "max_window_total": 8,
                "max_target_window_total": 8,
            },
        }
        repair_result = {
            "clip_key": "nushagak/clip_stream_seed",
            "domain": "nushagak",
            "prediction": {
                "left_count": 0,
                "right_count": 7,
                "parse_success": True,
                "confidence": 0.55,
                "candidate_passages": [
                    {
                        "direction": "right",
                        "peak_simultaneous_count": 2,
                        "episode_duration_sec": 15.0,
                        "wave_count": 1,
                        "throughput_best_count": 7,
                    }
                ],
            },
            "window_activity": {
                "max_window_total": 8,
                "max_target_window_total": 8,
            },
        }
        seed = _select_risk_source_result(
            global_result=global_result,
            proposal_result=proposal_result,
            repair_result=repair_result,
            direction="right",
            site_profile=profile,
        )
        self.assertIs(seed, proposal_result)

    def test_select_risk_assessment_stream_profile_can_use_left_window_activity_on_global_seed(self) -> None:
        profile = resolve_site_profile("nushagak", clip_key="nushagak/clip_stream_windows")
        global_result = {
            "clip_key": "nushagak/clip_stream_windows",
            "domain": "nushagak",
            "prediction": {
                "left_count": 0,
                "right_count": 0,
                "parse_success": True,
                "confidence": 0.8,
                "candidate_passages": [],
            },
            "window_activity": {
                "total_windows": 4,
                "active_windows": 2,
                "zero_windows": 2,
                "max_window_total": 18,
                "sum_positive_window_totals": 30,
                "max_target_window_total": 18,
                "sum_target_window_totals": 30,
                "sum_opposite_window_totals": 0,
            },
        }
        proposal_result = {
            "clip_key": "nushagak/clip_stream_windows",
            "domain": "nushagak",
            "prediction": {
                "left_count": 12,
                "right_count": 0,
                "parse_success": True,
                "confidence": 0.65,
                "candidate_passages": [
                    {
                        "direction": "left",
                        "peak_simultaneous_count": 2,
                        "episode_duration_sec": 30.0,
                        "wave_count": 1,
                        "throughput_best_count": 12,
                    }
                ],
            },
            "window_activity": {
                "total_windows": 4,
                "active_windows": 2,
                "zero_windows": 2,
                "max_window_total": 18,
                "sum_positive_window_totals": 30,
                "max_target_window_total": 18,
                "sum_target_window_totals": 30,
                "sum_opposite_window_totals": 0,
            },
        }
        assessment = _select_risk_assessment(
            global_result=global_result,
            proposal_result=proposal_result,
            direction="left",
            site_profile=profile,
        )
        self.assertIn("dense_stream_throughput_risk", assessment.flags)
        self.assertEqual(assessment.direction, "left")

    def test_dense_window_accepts_direction_cleanup_even_with_lower_total(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_j2",
            asset_paths={InputVariant.SFF3C: Path("clip_j2.mp4")},
        )
        proposal = EventProposal(
            proposal_id="backbone-1",
            timestamp_start_sec=0.0,
            timestamp_end_sec=15.0,
            score=3.0,
            source="backbone_candidate_passages",
            representation="sff3c",
            reasoning="event_agent",
            direction_hint="right",
        )
        base_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_j2",
            left_count=1,
            right_count=3,
            parse_success=True,
        )
        dense_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_j2",
            left_count=0,
            right_count=2,
            parse_success=True,
        )
        accept, reason = _should_accept_dense_window_prediction(
            clip,
            proposal,
            base_prediction,
            dense_prediction,
        )
        self.assertTrue(accept)
        self.assertEqual(reason, "accepted_dense_direction_cleanup")

    def test_dense_window_rejects_sparse_stream_inflation_from_zero_base(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_j3",
            asset_paths={InputVariant.SFF3C: Path("clip_j3.mp4")},
            upstream_direction=UpstreamDirection.RIGHT,
        )
        proposal = EventProposal(
            proposal_id="backbone-2",
            timestamp_start_sec=0.0,
            timestamp_end_sec=35.0,
            score=4.0,
            source="backbone_candidate_passages",
            representation="sff3c",
            reasoning="event_agent",
            direction_hint="right",
        )
        base_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_j3",
            left_count=0,
            right_count=0,
            parse_success=True,
        )
        dense_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_j3",
            left_count=0,
            right_count=4,
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 35.0,
                    "direction": "right",
                    "estimated_count": 4,
                    "peak_simultaneous_count": 1,
                    "episode_duration_sec": 35.0,
                    "wave_count": 1,
                    "throughput_best_count": 4,
                }
            ],
            parse_success=True,
        )
        accept, reason = _should_accept_dense_window_prediction(
            clip,
            proposal,
            base_prediction,
            dense_prediction,
        )
        self.assertFalse(accept)
        self.assertEqual(reason, "dense_sparse_stream_inflation")

    def test_should_run_flagged_repeat_consensus_for_sparse_risk(self) -> None:
        result_record = {
            "status": "completed",
            "risk_flags": ["multi_episode_trickle_overcount_risk"],
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))
        self.assertFalse(_should_run_flagged_repeat_consensus(result_record, repeat_runs=1))

    def test_should_not_run_flagged_repeat_consensus_for_stream_regime(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile": {
                "expected_regime": ["stream_throughput"],
            },
            "risk_flags": ["dense_stream_throughput_risk"],
        }
        self.assertFalse(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))

    def test_should_run_flagged_repeat_consensus_for_unflagged_sparse_bimodal_overcount(self) -> None:
        result_record = {
            "status": "completed",
            "risk_flags": [],
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 3,
                "right_count": 11,
            },
            "risk_assessment": {
                "candidate_count": 5,
                "total_wave_count": 5,
                "max_peak": 1.0,
                "confidence": 0.4,
            },
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))

    def test_should_run_flagged_repeat_consensus_for_multiwave_low_peak_blowup(self) -> None:
        result_record = {
            "status": "completed",
            "risk_flags": ["multiwave_high_peak_undercount_risk"],
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 19,
            },
            "risk_assessment": {
                "candidate_count": 5,
                "total_wave_count": 5,
                "total_peak": 11.0,
                "max_peak": 3.0,
                "max_duration_sec": 40.0,
                "confidence": 0.53,
            },
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))

    def test_should_run_flagged_repeat_consensus_for_unflagged_long_low_peak_blowup(self) -> None:
        result_record = {
            "status": "completed",
            "risk_flags": [],
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 20,
            },
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "total_peak": 2.0,
                "max_peak": 2.0,
                "max_duration_sec": 108.0,
                "confidence": 0.65,
            },
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))

    def test_should_run_flagged_repeat_consensus_for_large_custom_reduction_gap(self) -> None:
        result_record = {
            "status": "completed",
            "risk_flags": ["multiwave_high_peak_undercount_risk"],
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 45,
            },
            "risk_assessment": {
                "candidate_count": 3,
                "total_wave_count": 3,
                "total_peak": 12.0,
                "max_peak": 6.0,
                "max_duration_sec": 43.0,
                "confidence": 0.71,
            },
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "prediction": {
                        "parse_success": True,
                        "left_count": 0,
                        "right_count": 10,
                    },
                }
            ],
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=3))

    def test_should_run_flagged_repeat_consensus_for_no_proposal_hallucination(self) -> None:
        result_record = {
            "status": "completed",
            "risk_flags": [],
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "prediction": {
                "left_count": 0,
                "right_count": 8,
            },
            "risk_assessment": {
                "candidate_count": 0,
                "total_wave_count": 0,
                "total_peak": 0.0,
                "max_peak": 0.0,
                "max_duration_sec": 0.0,
                "confidence": 0.5,
            },
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=3))

    def test_should_run_flagged_repeat_consensus_for_kenai_single_school_high_throughput(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "risk_flags": [],
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 10,
            },
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "max_peak": 3.0,
                "max_duration_sec": 25.0,
                "confidence": 0.75,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))

    def test_should_run_flagged_repeat_consensus_for_candidate_passage_shape(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "risk_flags": [],
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 10,
                "candidate_passages": [
                    {
                        "throughput_best_count": 10,
                        "peak_simultaneous_count": 3,
                        "episode_duration_sec": 24.0,
                        "wave_count": 1,
                    }
                ],
            },
            "risk_assessment": {
                "candidate_count": 0,
                "total_wave_count": 0,
                "max_peak": 0.0,
                "max_duration_sec": 0.0,
                "confidence": 0.8,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))

    def test_should_run_flagged_repeat_consensus_for_kenai_custom_single_school_high_throughput(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "risk_flags": ["single_school_high_throughput_undercount_risk"],
            "selected_branch": "custom_high_throughput",
            "prediction": {
                "left_count": 0,
                "right_count": 9,
            },
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "max_peak": 4.0,
                "max_duration_sec": 28.0,
                "confidence": 0.75,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 8}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }
        self.assertTrue(_should_run_flagged_repeat_consensus(result_record, repeat_runs=2))

    def test_repeat_consensus_prefers_custom_positive_cleanup(self) -> None:
        pass_one = {
            "status": "completed",
            "routing_direction": "right",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "selection_reason": "expert",
            "risk_flags": ["multi_episode_trickle_overcount_risk"],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 7, "confidence": 0.6},
        }
        pass_two = {
            "status": "completed",
            "routing_direction": "right",
            "selected_action": "sff3c:custom_trickle_reduction",
            "selected_branch": "custom_trickle_reduction",
            "selection_reason": "routed",
            "risk_flags": ["multi_episode_trickle_overcount_risk"],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 2, "confidence": 0.55},
        }
        selected, audit = _apply_flagged_repeat_consensus([pass_one, pass_two])
        self.assertEqual(selected["selected_branch"], "custom_trickle_reduction")
        self.assertEqual(selected["prediction"]["right_count"], 2)
        self.assertEqual(audit["selected_pass_index"], 2)

    def test_repeat_consensus_prefers_positive_low_count_cleanup_over_zero(self) -> None:
        pass_one = {
            "status": "completed",
            "routing_direction": "right",
            "selected_action": "sff3c:custom_low_count",
            "selected_branch": "custom_low_count",
            "selection_reason": "routed",
            "risk_flags": ["low_count_far_view_risk"],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 0, "confidence": 0.65},
        }
        pass_two = {
            "status": "completed",
            "routing_direction": "right",
            "selected_action": "sff3c:custom_low_count",
            "selected_branch": "custom_low_count",
            "selection_reason": "routed",
            "risk_flags": ["low_count_far_view_risk"],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 2, "confidence": 0.55},
        }
        selected, _audit = _apply_flagged_repeat_consensus([pass_one, pass_two])
        self.assertEqual(selected["prediction"]["right_count"], 2)

    def test_repeat_consensus_trickle_prefers_smaller_baseline_over_larger_custom(self) -> None:
        pass_one = {
            "status": "completed",
            "routing_direction": "right",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "selection_reason": "expert",
            "risk_flags": ["multi_episode_trickle_overcount_risk"],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 5, "confidence": 0.55},
        }
        pass_two = {
            "status": "completed",
            "routing_direction": "right",
            "selected_action": "sff3c:custom_trickle_reduction",
            "selected_branch": "custom_trickle_reduction",
            "selection_reason": "routed",
            "risk_flags": ["multi_episode_trickle_overcount_risk"],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 6, "confidence": 0.4},
        }
        selected, audit = _apply_flagged_repeat_consensus([pass_one, pass_two])
        self.assertEqual(selected["selected_branch"], "baseline")
        self.assertEqual(selected["prediction"]["right_count"], 5)
        self.assertEqual(audit["selected_pass_index"], 1)

    def test_repeat_consensus_can_select_hidden_custom_low_count_candidate(self) -> None:
        pass_one = {
            "status": "completed",
            "routing_direction": "right",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "selection_reason": "expert",
            "risk_flags": [],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 8, "confidence": 0.6},
            "custom_branch_runs": [
                {
                    "branch_id": "custom_low_count",
                    "prediction": {
                        "parse_success": True,
                        "left_count": 0,
                        "right_count": 2,
                        "confidence": 0.5,
                    },
                }
            ],
        }
        selected, audit = _apply_flagged_repeat_consensus([pass_one])
        self.assertEqual(selected["selected_branch"], "custom_low_count")
        self.assertEqual(selected["prediction"]["right_count"], 2)
        self.assertEqual(audit["selected_source"], "custom_branch_runs")
        self.assertEqual(audit["selected_pass_index"], 1)

    def test_repeat_consensus_prefers_larger_kenai_single_school_high_throughput(self) -> None:
        pass_one = {
            "status": "completed",
            "site_profile_id": "kenai",
            "routing_direction": "right",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "selection_reason": "expert",
            "risk_flags": [],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 7, "confidence": 0.7},
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "max_peak": 3.0,
                "max_duration_sec": 25.0,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }
        pass_two = {
            "status": "completed",
            "site_profile_id": "kenai",
            "routing_direction": "right",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "selection_reason": "expert",
            "risk_flags": [],
            "prediction": {"parse_success": True, "left_count": 0, "right_count": 12, "confidence": 0.68},
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "max_peak": 3.0,
                "max_duration_sec": 25.0,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }

        selected, audit = _apply_flagged_repeat_consensus([pass_one, pass_two])

        self.assertEqual(selected["prediction"]["right_count"], 12)
        self.assertEqual(audit["selected_pass_index"], 2)

    def test_repeat_consensus_prefers_larger_candidate_passage_shape(self) -> None:
        pass_one = {
            "status": "completed",
            "site_profile_id": "kenai",
            "routing_direction": "right",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "selection_reason": "expert",
            "risk_flags": [],
            "prediction": {
                "parse_success": True,
                "left_count": 0,
                "right_count": 10,
                "confidence": 0.7,
                "candidate_passages": [
                    {
                        "throughput_best_count": 10,
                        "peak_simultaneous_count": 4,
                        "episode_duration_sec": 25.0,
                        "wave_count": 1,
                    }
                ],
            },
            "risk_assessment": {
                "candidate_count": 0,
                "total_wave_count": 0,
                "max_peak": 0.0,
                "max_duration_sec": 0.0,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }
        pass_two = {
            "status": "completed",
            "site_profile_id": "kenai",
            "routing_direction": "right",
            "selected_action": "sff3c:proposal_guided",
            "selected_branch": "baseline",
            "selection_reason": "expert",
            "risk_flags": [],
            "prediction": {
                "parse_success": True,
                "left_count": 0,
                "right_count": 11,
                "confidence": 0.88,
                "candidate_passages": [
                    {
                        "throughput_best_count": 11,
                        "peak_simultaneous_count": 3,
                        "episode_duration_sec": 24.0,
                        "wave_count": 1,
                    }
                ],
            },
            "risk_assessment": {
                "candidate_count": 0,
                "total_wave_count": 0,
                "max_peak": 0.0,
                "max_duration_sec": 0.0,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }

        selected, audit = _apply_flagged_repeat_consensus([pass_one, pass_two])

        self.assertEqual(selected["prediction"]["right_count"], 11)
        self.assertEqual(audit["selected_pass_index"], 2)

    def test_select_high_throughput_prediction_prefers_larger_same_direction_result(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="demo",
            asset_paths={InputVariant.SFF3C: Path("demo.mp4")},
            upstream_direction=UpstreamDirection.RIGHT,
        )
        lower = PassagePrediction(
            domain="kenai-val",
            clip_id="demo_a",
            scene_assessment="",
            candidate_passages=[],
            rejected_targets=[],
            left_count=0,
            right_count=9,
            confidence=0.8,
            commentary="",
            evidence_summary="",
            events=[],
            parse_success=True,
        )
        higher = PassagePrediction(
            domain="kenai-val",
            clip_id="demo_b",
            scene_assessment="",
            candidate_passages=[],
            rejected_targets=[],
            left_count=0,
            right_count=11,
            confidence=0.7,
            commentary="",
            evidence_summary="",
            events=[],
            parse_success=True,
        )

        selected = _select_high_throughput_prediction(clip, [lower, higher])

        self.assertEqual(selected.right_count, 11)

    def test_refresh_result_record_errors_uses_current_prediction(self) -> None:
        result_record = {
            "selected_action": "sff3c:custom_low_count",
            "selected_branch": "custom_low_count",
            "prediction": {
                "parse_success": True,
                "left_count": 0,
                "right_count": 1,
            },
            "abs_error_left": 0,
            "abs_error_right": 3,
            "total_abs_error": 3,
        }
        _refresh_result_record_errors(
            result_record,
            {
                "left_count": 0,
                "right_count": 1,
                "total_count": 1,
            },
        )
        self.assertEqual(result_record["abs_error_left"], 0)
        self.assertEqual(result_record["abs_error_right"], 0)
        self.assertEqual(result_record["total_abs_error"], 0)

    def test_post_consensus_trickle_cleanup_triggers_for_long_low_peak_blowup(self) -> None:
        result_record = {
            "status": "completed",
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 20,
            },
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "total_peak": 2.0,
                "max_peak": 2.0,
                "max_duration_sec": 108.0,
                "confidence": 0.65,
            },
        }
        self.assertTrue(_should_run_post_consensus_trickle_cleanup(result_record))

    def test_post_consensus_trickle_cleanup_triggers_for_large_custom_reduction_gap(self) -> None:
        result_record = {
            "status": "completed",
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 0,
                "right_count": 45,
            },
            "risk_assessment": {
                "candidate_count": 3,
                "total_wave_count": 3,
                "total_peak": 12.0,
                "max_peak": 6.0,
                "max_duration_sec": 43.0,
                "confidence": 0.71,
            },
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "prediction": {
                        "parse_success": True,
                        "left_count": 0,
                        "right_count": 10,
                    },
                }
            ],
        }
        self.assertTrue(_should_run_post_consensus_trickle_cleanup(result_record))

    def test_post_consensus_trickle_cleanup_triggers_for_no_proposal_hallucination(self) -> None:
        result_record = {
            "status": "completed",
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "prediction": {
                "left_count": 0,
                "right_count": 8,
            },
            "risk_assessment": {
                "candidate_count": 0,
                "total_wave_count": 0,
                "total_peak": 0.0,
                "max_peak": 0.0,
                "max_duration_sec": 0.0,
                "confidence": 0.5,
            },
        }
        self.assertTrue(_should_run_post_consensus_trickle_cleanup(result_record))

    def test_post_consensus_trickle_cleanup_skips_stream_regime(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile": {
                "expected_regime": ["stream_throughput"],
            },
            "selected_branch": "baseline",
            "prediction": {
                "left_count": 43,
                "right_count": 0,
            },
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "total_peak": 4.0,
                "max_peak": 4.0,
                "max_duration_sec": 24.0,
                "confidence": 0.7,
            },
        }
        self.assertFalse(_should_run_post_consensus_trickle_cleanup(result_record))

    def test_post_consensus_trickle_prompt_mentions_single_active_window_guardrail(self) -> None:
        result_record = {
            "prediction": {
                "left_count": 0,
                "right_count": 6,
                "total_count": 6,
            },
            "risk_assessment": {
                "candidate_count": 2,
                "total_peak": 2.0,
                "max_peak": 1.0,
                "total_wave_count": 2,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 6}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }
        prompt = _post_consensus_trickle_prompt(result_record)
        self.assertIn("local windows with positive fish evidence: 1/3", prompt.prompt_text)
        self.assertIn("prefer 1-2 defensible crossings", prompt.prompt_text)

    def test_post_consensus_trickle_prompt_mentions_backbone_low_density_prior(self) -> None:
        result_record = {
            "prediction": {
                "left_count": 0,
                "right_count": 8,
                "total_count": 8,
            },
            "risk_assessment": {
                "candidate_count": 2,
                "total_peak": 1.0,
                "max_peak": 1.0,
                "total_wave_count": 2,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 5}},
                {"selected_prediction": {"left_count": 0, "right_count": 4}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            "proposals": [
                {
                    "source": "backbone_candidate_passages",
                    "score": 2,
                    "metadata": {"estimated_count": 2, "peak_simultaneous_count": 1, "wave_count": 1},
                },
                {
                    "source": "backbone_candidate_passages",
                    "score": 2,
                    "metadata": {"estimated_count": 2, "peak_simultaneous_count": 1, "wave_count": 1},
                },
            ],
        }
        prompt = _post_consensus_trickle_prompt(result_record)
        self.assertIn("Backbone low-density prior:", prompt.prompt_text)
        self.assertIn("backbone total hint: 4", prompt.prompt_text)
        self.assertIn("Do not inflate the clip into a continuous stream", prompt.prompt_text)

    def test_apply_post_consensus_sparse_stream_cap_limits_single_active_window_cleanup(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_sparse",
            asset_paths={InputVariant.SFF3C: Path("clip_sparse.mp4")},
        )
        result_record = {
            "routing_direction": "right",
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 6}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
        }
        cleanup_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_sparse",
            left_count=0,
            right_count=5,
            parse_success=True,
            commentary="sparse trickle",
            evidence_summary="five faint bursts",
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 108.0,
                    "direction": "right",
                    "estimated_count": 5,
                    "peak_simultaneous_count": 1,
                    "episode_duration_sec": 108.0,
                    "wave_count": 5,
                    "throughput_best_count": 5,
                    "evidence_note": "five distinct mini-bursts",
                }
            ],
            events=[
                PassageEvent(timestamp_sec=5.0, direction="right", confidence=0.6, evidence_note="fish 1"),
                PassageEvent(timestamp_sec=28.0, direction="right", confidence=0.6, evidence_note="fish 2"),
                PassageEvent(timestamp_sec=52.0, direction="right", confidence=0.6, evidence_note="fish 3"),
            ],
        )
        capped_prediction, audit = _apply_post_consensus_sparse_stream_cap(
            clip,
            result_record,
            cleanup_prediction,
        )
        self.assertIsNotNone(audit)
        self.assertEqual(capped_prediction.right_count, 2)
        self.assertEqual(len(capped_prediction.events), 2)
        self.assertEqual(capped_prediction.candidate_passages[0]["throughput_best_count"], 2)

    def test_apply_post_consensus_sparse_stream_cap_limits_backbone_low_density_trickle(self) -> None:
        clip = ClipRecord(
            dataset="cfc",
            domain="kenai-val",
            clip_id="clip_backbone_sparse",
            asset_paths={InputVariant.SFF3C: Path("clip_backbone_sparse.mp4")},
        )
        result_record = {
            "routing_direction": "right",
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 5}},
                {"selected_prediction": {"left_count": 0, "right_count": 4}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            "proposals": [
                {
                    "source": "backbone_candidate_passages",
                    "score": 2,
                    "metadata": {"estimated_count": 2, "throughput_best_count": 2, "peak_simultaneous_count": 1, "wave_count": 1},
                },
                {
                    "source": "backbone_candidate_passages",
                    "score": 2,
                    "metadata": {"estimated_count": 2, "throughput_best_count": 2, "peak_simultaneous_count": 1, "wave_count": 1},
                },
            ],
        }
        cleanup_prediction = PassagePrediction(
            domain="kenai-val",
            clip_id="clip_backbone_sparse",
            left_count=0,
            right_count=6,
            parse_success=True,
            commentary="continuous sparse trickle",
            evidence_summary="multiple faint entrants",
            candidate_passages=[
                {
                    "timestamp_start_sec": 0.0,
                    "timestamp_end_sec": 108.0,
                    "direction": "right",
                    "estimated_count": 6,
                    "peak_simultaneous_count": 1,
                    "episode_duration_sec": 108.0,
                    "wave_count": 2,
                    "throughput_best_count": 6,
                    "evidence_note": "continuous low-density stream",
                }
            ],
            events=[
                PassageEvent(timestamp_sec=8.0, direction="right", confidence=0.4, evidence_note="fish 1"),
                PassageEvent(timestamp_sec=42.0, direction="right", confidence=0.4, evidence_note="fish 2"),
                PassageEvent(timestamp_sec=77.0, direction="right", confidence=0.4, evidence_note="fish 3"),
            ],
        )
        capped_prediction, audit = _apply_post_consensus_sparse_stream_cap(
            clip,
            result_record,
            cleanup_prediction,
        )
        self.assertIsNotNone(audit)
        self.assertEqual(audit["rule_name"], "cap_post_consensus_backbone_sparse_trickle")
        self.assertEqual(capped_prediction.right_count, 2)
        self.assertEqual(capped_prediction.candidate_passages[0]["throughput_best_count"], 2)

    def test_apply_post_selection_positive_window_uplift_for_conservative_multiwave_gain(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "risk_flags": ["multiwave_high_peak_undercount_risk"],
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 8,
                "total_count": 8,
                "commentary": "baseline proposal-guided result",
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 6}},
                {"selected_prediction": {"left_count": 0, "right_count": 2}},
            ],
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "decision": {"selected_source": "baseline"},
                    "prediction": {
                        "left_count": 0,
                        "right_count": 13,
                        "total_count": 13,
                        "parse_success": True,
                    },
                }
            ],
            "action_predictions": {"proposal_guided": {"left_count": 0, "right_count": 8}},
        }

        updated, audit = _apply_post_selection_positive_window_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["selected_branch"], "custom_high_throughput_uplift")
        self.assertEqual(updated["prediction"]["right_count"], 10)
        self.assertEqual(updated["prediction"]["total_count"], 10)
        self.assertEqual(updated["post_selection_positive_window_uplift"]["uplift"], 2)

    def test_apply_post_selection_positive_window_uplift_uses_supportive_global_bonus(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "risk_flags": ["multiwave_high_peak_undercount_risk"],
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 3,
                "total_count": 3,
                "commentary": "baseline proposal-guided result",
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 3}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "decision": {"selected_source": "baseline"},
                    "prediction": {
                        "left_count": 0,
                        "right_count": 6,
                        "total_count": 6,
                        "parse_success": True,
                    },
                }
            ],
            "action_predictions": {"global_direct": {"left_count": 0, "right_count": 4}},
        }

        updated, audit = _apply_post_selection_positive_window_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["prediction"]["right_count"], 5)
        self.assertEqual(updated["post_selection_positive_window_uplift"]["supportive_global_bonus"], 1)

    def test_apply_post_selection_positive_window_uplift_skips_small_custom_gap(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "risk_flags": ["multiwave_high_peak_undercount_risk"],
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 4,
                "total_count": 4,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 1}},
                {"selected_prediction": {"left_count": 0, "right_count": 3}},
            ],
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "decision": {"selected_source": "baseline"},
                    "prediction": {
                        "left_count": 0,
                        "right_count": 5,
                        "total_count": 5,
                        "parse_success": True,
                    },
                }
            ],
        }

        updated, audit = _apply_post_selection_positive_window_uplift(result_record)

        self.assertIsNone(audit)
        self.assertEqual(updated["prediction"]["right_count"], 4)
        self.assertEqual(updated["selected_branch"], "baseline")

    def test_apply_post_selection_positive_window_uplift_skips_many_positive_windows(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "risk_flags": ["multiwave_high_peak_undercount_risk"],
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 4,
                "total_count": 4,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 1}},
                {"selected_prediction": {"left_count": 0, "right_count": 4}},
                {"selected_prediction": {"left_count": 0, "right_count": 3}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "decision": {"selected_source": "baseline"},
                    "prediction": {
                        "left_count": 0,
                        "right_count": 9,
                        "total_count": 9,
                        "parse_success": True,
                    },
                }
            ],
            "action_predictions": {"global_direct": {"left_count": 0, "right_count": 8}},
        }

        updated, audit = _apply_post_selection_positive_window_uplift(result_record)

        self.assertIsNone(audit)
        self.assertEqual(updated["prediction"]["right_count"], 4)

    def test_apply_post_selection_single_school_uplift_for_one_strong_window(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "risk_flags": ["single_school_high_throughput_undercount_risk"],
            "risk_assessment": {
                "max_peak": 4.0,
                "max_duration_sec": 19.0,
            },
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 6,
                "total_count": 6,
                "commentary": "baseline proposal-guided result",
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 6}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "prediction": {
                        "left_count": 0,
                        "right_count": 9,
                        "total_count": 9,
                        "parse_success": True,
                    },
                }
            ],
            "action_predictions": {"proposal_guided": {"left_count": 0, "right_count": 6}},
        }

        updated, audit = _apply_post_selection_single_school_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["prediction"]["right_count"], 8)
        self.assertEqual(updated["selected_branch"], "custom_single_school_uplift")
        self.assertEqual(updated["post_selection_single_school_uplift"]["uplift"], 2)

    def test_apply_post_selection_single_school_window_floor_uses_strongest_window(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "custom_high_throughput",
            "selected_action": "sff3c:custom_high_throughput",
            "risk_flags": ["single_school_high_throughput_undercount_risk"],
            "risk_assessment": {
                "max_peak": 4.0,
                "max_duration_sec": 25.0,
            },
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 6,
                "total_count": 6,
                "commentary": "custom high-throughput result",
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 1}},
                {"selected_prediction": {"left_count": 0, "right_count": 11}},
            ],
            "custom_branch_runs": [
                {
                    "branch_id": "custom_high_throughput",
                    "prediction": {
                        "left_count": 0,
                        "right_count": 6,
                        "total_count": 6,
                        "parse_success": True,
                    },
                }
            ],
            "action_predictions": {"custom_high_throughput": {"left_count": 0, "right_count": 6}},
        }

        updated, audit = _apply_post_selection_single_school_window_floor(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["prediction"]["right_count"], 11)
        self.assertEqual(updated["selected_branch"], "custom_single_school_window_floor")
        self.assertEqual(updated["post_selection_single_school_window_floor"]["window_floor_total"], 11)

    def test_apply_post_selection_peak3_school_uplift_for_long_single_school(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "baseline",
            "selected_action": "sff3c:proposal_guided",
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 10,
                "total_count": 10,
                "commentary": "baseline proposal-guided result",
            },
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "max_peak": 3.0,
                "max_duration_sec": 25.0,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            "action_predictions": {"proposal_guided": {"left_count": 0, "right_count": 10}},
        }

        updated, audit = _apply_post_selection_peak3_school_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["prediction"]["right_count"], 12)
        self.assertEqual(updated["prediction"]["total_count"], 12)
        self.assertEqual(updated["selected_branch"], "custom_peak3_school_uplift")
        self.assertEqual(updated["post_selection_peak3_school_uplift"]["baseline_total"], 10)
        self.assertEqual(updated["post_selection_peak3_school_uplift"]["base_total"], 10)
        self.assertEqual(updated["post_selection_peak3_school_uplift"]["uplifted_total"], 12)

    def test_apply_post_selection_peak3_school_uplift_uses_strongest_window_floor(self) -> None:
        result_record = {
            "status": "completed",
            "site_profile_id": "kenai",
            "selected_branch": "baseline",
            "selected_action": "sff3c:global",
            "routing_direction": "right",
            "prediction": {
                "left_count": 0,
                "right_count": 7,
                "total_count": 7,
                "commentary": "baseline global result",
            },
            "risk_assessment": {
                "candidate_count": 1,
                "total_wave_count": 1,
                "max_peak": 3.0,
                "max_duration_sec": 25.0,
            },
            "window_runs": [
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
                {"selected_prediction": {"left_count": 0, "right_count": 10}},
                {"selected_prediction": {"left_count": 0, "right_count": 0}},
            ],
            "action_predictions": {"global_direct": {"left_count": 0, "right_count": 7}},
        }

        updated, audit = _apply_post_selection_peak3_school_uplift(result_record)

        self.assertIsNotNone(audit)
        self.assertEqual(updated["prediction"]["right_count"], 12)
        self.assertEqual(updated["post_selection_peak3_school_uplift"]["baseline_total"], 7)
        self.assertEqual(updated["post_selection_peak3_school_uplift"]["base_total"], 10)
        self.assertEqual(updated["post_selection_peak3_school_uplift"]["uplifted_total"], 12)


if __name__ == "__main__":
    unittest.main()

