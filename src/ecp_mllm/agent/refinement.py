from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..eval.counting import evaluate_predictions
from ..qwen.base import QwenClient
from ..qwen.parsing import parse_passage_prediction
from ..qwen.prompting import build_prompt, build_revised_prompt
from ..types import ClipKey, ClipRecord, EvalReport, PassagePrediction, PromptRevision, WeakCountRecord


@dataclass(frozen=True)
class VariantEvaluation:
    prompt: PromptRevision
    report: EvalReport
    predictions: dict[ClipKey, PassagePrediction]


class PromptCritic:
    @staticmethod
    def _trim(value: str | None, limit: int = 220) -> str | None:
        if value is None:
            return None
        text = " ".join(value.strip().split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def critique(self, report: EvalReport, predictions: dict[ClipKey, PassagePrediction]) -> str:
        worst_clips = sorted(report.clip_results, key=lambda item: item.clip_mae, reverse=True)[:3]
        lines = [
            "Keep the output strict JSON.",
            "Prefer conservative counts when evidence is weak.",
            "Do not double-count partial crossings or stationary fish.",
            "Use commentary and evidence_summary to explain observable motion cues, not hidden reasoning.",
        ]
        for clip in worst_clips:
            prediction = predictions.get(ClipKey(clip.domain, clip.clip_id))
            lines.append(
                f"For {clip.domain}/{clip.clip_id}, truth was left={clip.truth.left}, right={clip.truth.right} "
                f"while prediction was left={clip.predicted.left}, right={clip.predicted.right}."
            )
            if clip.abs_error_left > clip.abs_error_right:
                lines.append(f"Re-examine leftward passages for {clip.domain}/{clip.clip_id}.")
            elif clip.abs_error_right > 0:
                lines.append(f"Re-examine rightward passages for {clip.domain}/{clip.clip_id}.")
            if prediction is not None:
                commentary = self._trim(prediction.commentary)
                evidence = self._trim(prediction.evidence_summary)
                if commentary:
                    lines.append(f"Model commentary for {clip.domain}/{clip.clip_id}: {commentary}")
                if evidence:
                    lines.append(f"Model evidence summary for {clip.domain}/{clip.clip_id}: {evidence}")
        return "\n".join(lines)


class RefinementAgent:
    def __init__(self, client: QwenClient, critic: PromptCritic | None = None) -> None:
        self.client = client
        self.critic = critic or PromptCritic()

    def evaluate_prompt(
        self,
        clips: Iterable[ClipRecord],
        labels: dict[ClipKey, WeakCountRecord],
        prompt: PromptRevision,
        variant,
    ) -> VariantEvaluation:
        materialized_clips = list(clips)
        predictions: dict[ClipKey, PassagePrediction] = {}
        for clip in materialized_clips:
            if clip.get_asset_path(variant) is None:
                continue
            _ = build_prompt(clip, variant, prompt)
            response = self.client.infer(clip, variant, prompt)
            predictions[clip.key] = parse_passage_prediction(
                raw_text=response.text,
                domain=clip.domain,
                clip_id=clip.clip_id,
                prompt_id=prompt.prompt_id,
                latency_sec=response.latency_sec,
                usage_metadata=response.usage_metadata,
                estimated_cost_usd=response.estimated_cost_usd,
            )
        report = evaluate_predictions(predictions=predictions, labels=labels, clip_index={clip.key: clip for clip in materialized_clips})
        return VariantEvaluation(prompt=prompt, report=report, predictions=predictions)

    def optimize_prompt(
        self,
        train_clips: Iterable[ClipRecord],
        labels: dict[ClipKey, WeakCountRecord],
        base_prompt_text: str,
        variant,
        max_revisions: int,
    ) -> tuple[PromptRevision, list[VariantEvaluation]]:
        history: list[VariantEvaluation] = []
        current_prompt = PromptRevision(version=0, prompt_id="base", prompt_text=base_prompt_text)
        best_eval: VariantEvaluation | None = None
        materialized_clips = list(train_clips)
        for revision_index in range(max_revisions + 1):
            evaluation = self.evaluate_prompt(materialized_clips, labels, current_prompt, variant)
            history.append(evaluation)
            scored_prompt = PromptRevision(
                version=current_prompt.version,
                prompt_id=current_prompt.prompt_id,
                prompt_text=current_prompt.prompt_text,
                critique=current_prompt.critique,
                metrics={
                    "mae": evaluation.report.overall_mae,
                    "rmse": evaluation.report.overall_rmse,
                    "nmae": evaluation.report.overall_nmae,
                    "parse_rate": evaluation.report.parse_rate,
                },
            )
            if best_eval is None or evaluation.report.overall_mae < best_eval.report.overall_mae:
                best_eval = VariantEvaluation(prompt=scored_prompt, report=evaluation.report, predictions=evaluation.predictions)
            if revision_index == max_revisions:
                break
            critique = self.critic.critique(evaluation.report, evaluation.predictions)
            current_prompt = build_revised_prompt(scored_prompt, critique, revision_index + 1)
        if best_eval is None:
            raise RuntimeError("No training evaluations were produced")
        return best_eval.prompt, history
