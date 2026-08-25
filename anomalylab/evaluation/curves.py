"""Curve generation utilities (Precision-Recall, ROC, Threshold vs Metric)."""

from typing import Any

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve


def generate_evaluation_curves(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    max_points: int = 50,
) -> dict[str, Any]:
    """
    Generate downsampled, JSON-serializable PR and ROC curve data for dashboard rendering.
    """
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_scores_arr = np.asarray(y_scores, dtype=np.float32)

    # Precision-Recall Curve
    precisions, recalls, pr_thresholds = precision_recall_curve(y_true_arr, y_scores_arr)

    # ROC Curve
    fprs, tprs, roc_thresholds = roc_curve(y_true_arr, y_scores_arr)

    def downsample(x_arr: np.ndarray, y_arr: np.ndarray, n: int) -> list[dict[str, float]]:
        if len(x_arr) <= n:
            indices = np.arange(len(x_arr))
        else:
            indices = np.linspace(0, len(x_arr) - 1, n, dtype=int)
        return [{"x": round(float(x_arr[i]), 4), "y": round(float(y_arr[i]), 4)} for i in indices]

    pr_points = downsample(recalls, precisions, max_points)
    roc_points = downsample(fprs, tprs, max_points)

    # Score distribution histogram (20 bins)
    hist_normal, bin_edges = np.histogram(
        y_scores_arr[~y_true_arr], bins=20, range=(0.0, 1.0), density=True
    )
    hist_anomaly, _ = np.histogram(
        y_scores_arr[y_true_arr], bins=20, range=(0.0, 1.0), density=True
    )

    score_distribution = []
    for i in range(len(hist_normal)):
        score_distribution.append(
            {
                "bin_start": round(float(bin_edges[i]), 2),
                "bin_end": round(float(bin_edges[i + 1]), 2),
                "normal_density": round(float(hist_normal[i]), 4),
                "anomaly_density": round(float(hist_anomaly[i]), 4),
            }
        )

    return {
        "pr_curve": pr_points,
        "roc_curve": roc_points,
        "score_distribution": score_distribution,
    }
