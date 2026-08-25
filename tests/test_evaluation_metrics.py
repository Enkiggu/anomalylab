"""Tests for imbalanced evaluation metrics and threshold tuning."""

import numpy as np

from anomalylab.evaluation.metrics import calculate_metrics
from anomalylab.evaluation.threshold import optimize_threshold


def test_metrics_calculation_controlled_confusion_matrix():
    """Verify precision, recall, FPR, and FNR calculations on known binary vectors."""
    # 90 Normal, 10 Anomalies
    y_true = np.array([False] * 90 + [True] * 10)
    # Perfect scores
    y_scores = np.array([0.1] * 90 + [0.9] * 10)

    res = calculate_metrics(y_true, y_scores, threshold=0.5)

    assert res.precision == 1.0
    assert res.recall == 1.0
    assert res.f1 == 1.0
    assert res.false_positive_rate == 0.0
    assert res.fp_per_10k == 0.0
    assert res.tp == 10
    assert res.tn == 90
    assert res.fp == 0
    assert res.fn == 0


def test_threshold_optimizer_target_recall():
    """Verify threshold optimization finds cutoff meeting target recall."""
    rng = np.random.default_rng(42)
    y_true = np.array([False] * 200 + [True] * 20)
    # Scores with slight overlap
    y_scores = np.concatenate(
        [
            rng.uniform(0.1, 0.6, size=200),
            rng.uniform(0.5, 0.95, size=20),
        ]
    )

    report = optimize_threshold(y_true, y_scores, strategy="target_recall", target_recall=0.90)

    assert report.recall_at_threshold >= 0.90
    assert 0.01 <= report.best_threshold <= 0.99
    assert report.cost_at_threshold > 0.0
