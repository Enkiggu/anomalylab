"""Threshold optimization and business cost-based tuning for AnomalyLab."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from anomalylab.evaluation.metrics import calculate_metrics


@dataclass
class ThresholdTuningReport:
    best_threshold: float
    tuning_strategy: str
    target_metric_value: float
    precision_at_threshold: float
    recall_at_threshold: float
    f1_at_threshold: float
    fpr_at_threshold: float
    fp_per_10k_at_threshold: float
    cost_at_threshold: float
    threshold_sweep: list[dict[str, Any]]


def optimize_threshold(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    strategy: str = "target_recall",
    target_recall: float = 0.90,
    cost_fp: float = 1.0,
    cost_fn: float = 20.0,
    num_steps: int = 100,
) -> ThresholdTuningReport:
    """
    Search candidate thresholds in [0.01, 0.99] to find optimal operating point.

    Strategies:
    - 'target_recall': Find threshold achieving Recall >= target_recall with minimum FPR.
    - 'f1_optimal': Find threshold maximizing F1 score.
    - 'cost_optimal': Find threshold minimizing Expected Cost = (cost_fp * FP) + (cost_fn * FN).
    """
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_scores_arr = np.asarray(y_scores, dtype=np.float32)

    candidate_thresholds = np.linspace(0.01, 0.99, num_steps)
    sweep_results = []

    best_thresh = 0.5
    best_cost = float("inf")
    best_f1 = -1.0
    best_fpr_for_recall = float("inf")

    for thresh in candidate_thresholds:
        res = calculate_metrics(y_true_arr, y_scores_arr, threshold=thresh)
        cost = (cost_fp * res.fp) + (cost_fn * res.fn)

        sweep_item = {
            "threshold": round(float(thresh), 4),
            "precision": round(res.precision, 4),
            "recall": round(res.recall, 4),
            "f1": round(res.f1, 4),
            "fpr": round(res.false_positive_rate, 6),
            "fp_per_10k": round(res.fp_per_10k, 2),
            "tp": res.tp,
            "fp": res.fp,
            "tn": res.tn,
            "fn": res.fn,
            "cost": round(float(cost), 2),
        }
        sweep_results.append(sweep_item)

        if strategy == "target_recall":
            if res.recall >= target_recall:
                # Prioritize lower FPR while maintaining target recall
                if res.false_positive_rate < best_fpr_for_recall or (
                    res.false_positive_rate == best_fpr_for_recall and thresh > best_thresh
                ):
                    best_fpr_for_recall = res.false_positive_rate
                    best_thresh = float(thresh)

        elif strategy == "f1_optimal":
            if res.f1 > best_f1:
                best_f1 = res.f1
                best_thresh = float(thresh)

        elif strategy == "cost_optimal":
            if cost < best_cost:
                best_cost = cost
                best_thresh = float(thresh)

    # If target_recall was not met at all by any threshold, pick max recall point
    if strategy == "target_recall" and best_fpr_for_recall == float("inf"):
        max_rec_item = max(sweep_results, key=lambda x: x["recall"])
        best_thresh = max_rec_item["threshold"]

    # Final metrics at chosen threshold
    final_res = calculate_metrics(y_true_arr, y_scores_arr, threshold=best_thresh)
    final_cost = (cost_fp * final_res.fp) + (cost_fn * final_res.fn)

    target_val = (
        final_res.recall
        if strategy == "target_recall"
        else final_res.f1
        if strategy == "f1_optimal"
        else final_cost
    )

    return ThresholdTuningReport(
        best_threshold=round(best_thresh, 4),
        tuning_strategy=strategy,
        target_metric_value=round(target_val, 4),
        precision_at_threshold=round(final_res.precision, 4),
        recall_at_threshold=round(final_res.recall, 4),
        f1_at_threshold=round(final_res.f1, 4),
        fpr_at_threshold=round(final_res.false_positive_rate, 6),
        fp_per_10k_at_threshold=round(final_res.fp_per_10k, 2),
        cost_at_threshold=round(final_cost, 2),
        threshold_sweep=sweep_results,
    )
