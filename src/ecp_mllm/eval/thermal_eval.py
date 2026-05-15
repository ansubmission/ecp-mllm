from __future__ import annotations

from typing import Iterable, Sequence

from ..thermal import THERMAL_COARSE_LABELS, collapse_thermal_labels
from ..types import ClipRecord, ThermalClipEvalResult, ThermalEvalReport, ThermalEventWindow, ThermalPrediction


def _prediction_is_false_positive(prediction: ThermalPrediction, threshold: float = 0.5) -> bool:
    if prediction.coarse_label == "false_positive":
        return True
    if prediction.false_positive_score is None:
        return False
    return float(prediction.false_positive_score) >= float(threshold)


def _normalize_predicted_coarse_label(prediction: ThermalPrediction, threshold: float = 0.5) -> str:
    if _prediction_is_false_positive(prediction, threshold=threshold):
        return "false_positive"
    if prediction.coarse_label in THERMAL_COARSE_LABELS:
        return str(prediction.coarse_label)
    collapsed = collapse_thermal_labels(prediction.clip_labels)
    for label in collapsed:
        if label != "false_positive":
            return label
    return "other"


def temporal_iou(
    start_a: float | None,
    end_a: float | None,
    start_b: float | None,
    end_b: float | None,
) -> float:
    if None in {start_a, end_a, start_b, end_b}:
        return 0.0
    inter = max(0.0, min(float(end_a), float(end_b)) - max(float(start_a), float(start_b)))
    union = max(float(end_a), float(end_b)) - min(float(start_a), float(start_b))
    if union <= 0:
        return 0.0
    return inter / union


def truth_event_windows(clip: ClipRecord, *, include_false_positive: bool = False) -> list[ThermalEventWindow]:
    windows: list[ThermalEventWindow] = []
    for item in clip.metadata.get("truth_event_windows") or []:
        if not isinstance(item, dict):
            continue
        label = str(item.get("label") or clip.metadata.get("coarse_label") or "other")
        normalized_label = collapse_thermal_labels([label])
        coarse_label = normalized_label[0] if normalized_label else "other"
        if coarse_label == "false_positive" and not include_false_positive:
            continue
        windows.append(
            ThermalEventWindow(
                timestamp_start_sec=float(item["start_sec"]) if item.get("start_sec") is not None else None,
                timestamp_end_sec=float(item["end_sec"]) if item.get("end_sec") is not None else None,
                label=coarse_label,
                source="track",
                metadata={"track_id": item.get("track_id")},
            )
        )
    return windows


def predicted_animal_event_windows(prediction: ThermalPrediction) -> list[ThermalEventWindow]:
    return [
        window
        for window in prediction.event_windows
        if collapse_thermal_labels([window.label or "other"])[0] != "false_positive"
    ]


def truth_center_zone_summary(clip: ClipRecord) -> dict[str, float | bool | None]:
    return {
        "entered": bool(clip.metadata.get("truth_center_zone_entered")),
        "first_entry_sec": float(clip.metadata["truth_center_zone_first_entry_sec"])
        if clip.metadata.get("truth_center_zone_first_entry_sec") is not None
        else None,
        "dwell_sec": float(clip.metadata["truth_center_zone_dwell_sec"])
        if clip.metadata.get("truth_center_zone_dwell_sec") is not None
        else None,
    }


def _macro_f1(truth_labels: Sequence[str], pred_labels: Sequence[str]) -> tuple[float, dict[str, float]]:
    labels = sorted(set(truth_labels) | set(pred_labels)) or list(THERMAL_COARSE_LABELS)
    per_class_f1: dict[str, float] = {}
    per_class_recall: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for truth, pred in zip(truth_labels, pred_labels) if truth == label and pred == label)
        fp = sum(1 for truth, pred in zip(truth_labels, pred_labels) if truth != label and pred == label)
        fn = sum(1 for truth, pred in zip(truth_labels, pred_labels) if truth == label and pred != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        per_class_recall[label] = recall
        per_class_f1[label] = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    macro = sum(per_class_f1.values()) / len(per_class_f1) if per_class_f1 else 0.0
    return macro, per_class_recall


def evaluate_thermal_predictions(
    clips: Iterable[ClipRecord],
    predictions: Iterable[ThermalPrediction],
    *,
    false_positive_threshold: float = 0.5,
    event_iou_threshold: float = 0.1,
) -> ThermalEvalReport:
    clip_by_key = {clip.key.value: clip for clip in clips}
    prediction_by_key = {prediction.key.value: prediction for prediction in predictions}

    clip_results: list[ThermalClipEvalResult] = []
    truth_binary: list[bool] = []
    pred_binary: list[bool] = []
    truth_coarse: list[str] = []
    pred_coarse: list[str] = []
    animal_event_count_errors: list[int] = []
    animal_event_recalls: list[float] = []
    animal_event_precisions: list[float] = []
    animal_event_tious: list[float] = []
    animal_event_label_accuracies: list[float] = []
    center_zone_entry_truth: list[bool] = []
    center_zone_entry_pred: list[bool] = []
    center_zone_first_entry_errors: list[float] = []
    center_zone_dwell_errors: list[float] = []
    multi_entity_corrects: list[float] = []
    latencies: list[float] = []
    abstentions = 0

    for key, clip in clip_by_key.items():
        prediction = prediction_by_key.get(key)
        if prediction is None:
            prediction = ThermalPrediction(
                domain=clip.domain,
                clip_id=clip.clip_id,
                coarse_label="false_positive" if bool(clip.metadata.get("is_false_positive")) else "other",
                false_positive_score=1.0 if bool(clip.metadata.get("is_false_positive")) else 0.0,
                parse_success=False,
                prompt_id="missing",
            )
        truth_fp = bool(clip.metadata.get("is_false_positive"))
        pred_fp = _prediction_is_false_positive(prediction, threshold=false_positive_threshold)
        truth_label = str(clip.metadata.get("coarse_label") or "other")
        pred_label = _normalize_predicted_coarse_label(prediction, threshold=false_positive_threshold)
        truth_windows = truth_event_windows(clip, include_false_positive=False)
        pred_windows = predicted_animal_event_windows(prediction)
        truth_zone = truth_center_zone_summary(clip)
        pred_zone_entered = bool(prediction.center_zone_entered) if prediction.center_zone_entered is not None else False
        pred_zone_first_entry_sec = float(prediction.center_zone_first_entry_sec) if prediction.center_zone_first_entry_sec is not None else None
        pred_zone_dwell_sec = float(prediction.center_zone_dwell_sec) if prediction.center_zone_dwell_sec is not None else None
        truth_zone_first_entry_sec = float(truth_zone["first_entry_sec"]) if truth_zone["first_entry_sec"] is not None else None
        truth_zone_dwell_sec = float(truth_zone["dwell_sec"]) if truth_zone["dwell_sec"] is not None else None

        truth_event_count = len(truth_windows)
        pred_event_count = len(pred_windows)
        event_count_error = abs(pred_event_count - truth_event_count)
        truth_multi_entity = truth_event_count > 1
        predicted_multi_entity = pred_event_count > 1
        multi_entity_correct = truth_multi_entity == predicted_multi_entity
        animal_event_count_errors.append(event_count_error)
        multi_entity_corrects.append(1.0 if multi_entity_correct else 0.0)
        center_zone_entry_truth.append(bool(truth_zone["entered"]))
        center_zone_entry_pred.append(pred_zone_entered)
        center_zone_first_entry_error: float | None = None
        center_zone_dwell_error: float | None = None
        if truth_zone["entered"] and pred_zone_entered and truth_zone_first_entry_sec is not None and pred_zone_first_entry_sec is not None:
            center_zone_first_entry_error = abs(pred_zone_first_entry_sec - truth_zone_first_entry_sec)
            center_zone_first_entry_errors.append(center_zone_first_entry_error)
        if truth_zone["entered"] and pred_zone_entered and truth_zone_dwell_sec is not None and pred_zone_dwell_sec is not None:
            center_zone_dwell_error = abs(pred_zone_dwell_sec - truth_zone_dwell_sec)
            center_zone_dwell_errors.append(center_zone_dwell_error)

        recall_value: float | None = None
        precision_value: float | None = None
        tiou_value: float | None = None
        label_accuracy_value: float | None = None
        if truth_windows:
            best_scores: list[float] = []
            recalled = 0
            label_correct = 0
            for truth_window in truth_windows:
                scored_predictions = [
                    (
                        temporal_iou(
                            truth_window.timestamp_start_sec,
                            truth_window.timestamp_end_sec,
                            pred_window.timestamp_start_sec,
                            pred_window.timestamp_end_sec,
                        ),
                        pred_window,
                    )
                    for pred_window in pred_windows
                ]
                best, best_window = max(scored_predictions, key=lambda item: item[0], default=(0.0, None))
                best_scores.append(best)
                if best >= event_iou_threshold:
                    recalled += 1
                    truth_event_label = collapse_thermal_labels([truth_window.label or "other"])[0]
                    pred_event_label = collapse_thermal_labels([best_window.label or "other"])[0] if best_window is not None else "other"
                    if truth_event_label == pred_event_label:
                        label_correct += 1
            recall_value = recalled / len(truth_windows)
            tiou_value = sum(best_scores) / len(best_scores)
            label_accuracy_value = label_correct / len(truth_windows)
            animal_event_recalls.append(recall_value)
            animal_event_tious.append(tiou_value)
            animal_event_label_accuracies.append(label_accuracy_value)
        if pred_windows:
            matched_predictions = 0
            for pred_window in pred_windows:
                best = max(
                    (
                        temporal_iou(
                            truth_window.timestamp_start_sec,
                            truth_window.timestamp_end_sec,
                            pred_window.timestamp_start_sec,
                            pred_window.timestamp_end_sec,
                        )
                        for truth_window in truth_windows
                    ),
                    default=0.0,
                )
                if best >= event_iou_threshold:
                    matched_predictions += 1
            precision_value = matched_predictions / len(pred_windows)
            animal_event_precisions.append(precision_value)

        if prediction.abstain:
            abstentions += 1
        latency = float(prediction.latency_sec or 0.0)
        latencies.append(latency)
        truth_binary.append(truth_fp)
        pred_binary.append(pred_fp)
        truth_coarse.append(truth_label)
        pred_coarse.append(pred_label)
        clip_results.append(
            ThermalClipEvalResult(
                domain=clip.domain,
                clip_id=clip.clip_id,
                truth_false_positive=truth_fp,
                predicted_false_positive=pred_fp,
                truth_coarse_label=truth_label,
                predicted_coarse_label=pred_label,
                binary_correct=truth_fp == pred_fp,
                coarse_correct=truth_label == pred_label,
                truth_animal_event_count=truth_event_count,
                predicted_animal_event_count=pred_event_count,
                animal_event_count_error=event_count_error,
                truth_multi_entity=truth_multi_entity,
                predicted_multi_entity=predicted_multi_entity,
                multi_entity_correct=multi_entity_correct,
                animal_event_recall=recall_value,
                animal_event_precision=precision_value,
                animal_event_mean_tiou=tiou_value,
                animal_event_label_accuracy=label_accuracy_value,
                truth_center_zone_entered=bool(truth_zone["entered"]),
                predicted_center_zone_entered=pred_zone_entered,
                center_zone_entry_correct=bool(truth_zone["entered"]) == pred_zone_entered,
                truth_center_zone_first_entry_sec=truth_zone_first_entry_sec,
                predicted_center_zone_first_entry_sec=pred_zone_first_entry_sec,
                center_zone_first_entry_error_sec=center_zone_first_entry_error,
                truth_center_zone_dwell_sec=truth_zone_dwell_sec,
                predicted_center_zone_dwell_sec=pred_zone_dwell_sec,
                center_zone_dwell_error_sec=center_zone_dwell_error,
                abstain=prediction.abstain,
                latency_sec=latency,
            )
        )

    tp = sum(1 for truth, pred in zip(truth_binary, pred_binary) if truth and pred)
    fp = sum(1 for truth, pred in zip(truth_binary, pred_binary) if not truth and pred)
    fn = sum(1 for truth, pred in zip(truth_binary, pred_binary) if truth and not pred)
    tn = sum(1 for truth, pred in zip(truth_binary, pred_binary) if not truth and not pred)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    binary_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    binary_accuracy = sum(1 for truth, pred in zip(truth_binary, pred_binary) if truth == pred) / len(truth_binary) if truth_binary else 0.0
    binary_balanced_accuracy = (recall + specificity) / 2.0
    coarse_macro_f1, per_class_recall = _macro_f1(truth_coarse, pred_coarse)
    zone_tp = sum(1 for truth, pred in zip(center_zone_entry_truth, center_zone_entry_pred) if truth and pred)
    zone_fp = sum(1 for truth, pred in zip(center_zone_entry_truth, center_zone_entry_pred) if not truth and pred)
    zone_fn = sum(1 for truth, pred in zip(center_zone_entry_truth, center_zone_entry_pred) if truth and not pred)
    zone_precision = zone_tp / (zone_tp + zone_fp) if (zone_tp + zone_fp) > 0 else 0.0
    zone_recall = zone_tp / (zone_tp + zone_fn) if (zone_tp + zone_fn) > 0 else 0.0
    center_zone_entry_f1 = (
        (2 * zone_precision * zone_recall / (zone_precision + zone_recall))
        if (zone_precision + zone_recall) > 0
        else 0.0
    )
    center_zone_entry_accuracy = (
        sum(1 for truth, pred in zip(center_zone_entry_truth, center_zone_entry_pred) if truth == pred) / len(center_zone_entry_truth)
        if center_zone_entry_truth
        else None
    )

    return ThermalEvalReport(
        binary_accuracy=binary_accuracy,
        binary_f1=binary_f1,
        binary_balanced_accuracy=binary_balanced_accuracy,
        coarse_macro_f1=coarse_macro_f1,
        per_class_recall=per_class_recall,
        animal_event_count_mae=(sum(animal_event_count_errors) / len(animal_event_count_errors)) if animal_event_count_errors else None,
        animal_event_window_recall=(sum(animal_event_recalls) / len(animal_event_recalls)) if animal_event_recalls else None,
        animal_event_window_precision=(sum(animal_event_precisions) / len(animal_event_precisions)) if animal_event_precisions else None,
        animal_event_window_mean_tiou=(sum(animal_event_tious) / len(animal_event_tious)) if animal_event_tious else None,
        animal_event_label_accuracy=(sum(animal_event_label_accuracies) / len(animal_event_label_accuracies)) if animal_event_label_accuracies else None,
        center_zone_entry_accuracy=center_zone_entry_accuracy,
        center_zone_entry_f1=center_zone_entry_f1,
        center_zone_first_entry_mae_sec=(sum(center_zone_first_entry_errors) / len(center_zone_first_entry_errors)) if center_zone_first_entry_errors else None,
        center_zone_dwell_mae_sec=(sum(center_zone_dwell_errors) / len(center_zone_dwell_errors)) if center_zone_dwell_errors else None,
        multi_entity_accuracy=(sum(multi_entity_corrects) / len(multi_entity_corrects)) if multi_entity_corrects else None,
        abstention_rate=(abstentions / len(clip_results)) if clip_results else 0.0,
        mean_latency_sec=(sum(latencies) / len(latencies)) if latencies else 0.0,
        clip_results=tuple(clip_results),
    )
