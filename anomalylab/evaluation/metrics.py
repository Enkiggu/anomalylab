"""Comprehensive evaluation metrics for imbalanced anomaly detection."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class ConfidenceInterval:
    mean: float
    ci_lower: float
    ci_upper: float


@dataclass
class EvaluationResult:
    threshold: float
    precision: float
    recall: float
    f1: float
    false_positive_rate: float
    false_negative_rate: float
    pr_auc: float
    roc_auc: float
    fp_per_1k: float
    fp_per_10k: float
    fp_per_100k: float
    projected_daily_fp_1m_events: float
    tp: int
    fp: int
    tn: int
    fn: int
    total_samples: int
    normal_samples: int
    anomaly_samples: int
    anomaly_rate: float
    per_anomaly_recall: dict[str, float] | None = None
    confidence_intervals: dict[str, ConfidenceInterval] | None = None

    def to_dict(self) -> dict[str, Any]:
        ci_dict = {}
        if self.confidence_intervals:
            for k, v in self.confidence_intervals.items():
                ci_dict[k] = {
                    "mean": round(v.mean, 4),
                    "ci_lower": round(v.ci_lower, 4),
                    "ci_upper": round(v.ci_upper, 4),
                }

        return {
            "threshold": round(self.threshold, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "false_positive_rate": round(self.false_positive_rate, 6),
            "false_negative_rate": round(self.false_negative_rate, 6),
            "pr_auc": round(self.pr_auc, 4),
            "roc_auc": round(self.roc_auc, 4),
            "fp_per_1k": round(self.fp_per_1k, 2),
            "fp_per_10k": round(self.fp_per_10k, 2),
            "fp_per_100k": round(self.fp_per_100k, 2),
            "projected_daily_fp_1m_events": round(self.projected_daily_fp_1m_events, 1),
            "confusion_matrix": {
                "tp": self.tp,
                "fp": self.fp,
                "tn": self.tn,
                "fn": self.fn,
            },
            "class_distribution": {
                "total_samples": self.total_samples,
                "normal_samples": self.normal_samples,
                "anomaly_samples": self.anomaly_samples,
                "anomaly_rate": round(self.anomaly_rate, 6),
            },
            "per_anomaly_recall": self.per_anomaly_recall or {},
            "confidence_intervals": ci_dict,
        }


def calculate_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5,
    anomaly_types: np.ndarray | list[str] | None = None,
    compute_bootstrap: bool = False,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> EvaluationResult:
    """
    Calculate full suite of imbalanced anomaly metrics at given threshold.
    """
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_scores_arr = np.asarray(y_scores, dtype=np.float32)
    y_pred = y_scores_arr >= threshold

    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred, labels=[False, True]).ravel()

    precision = float(precision_score(y_true_arr, y_pred, zero_division=0))
    recall = float(recall_score(y_true_arr, y_pred, zero_division=0))
    f1 = float(f1_score(y_true_arr, y_pred, zero_division=0))

    fpr = float(fp / max(1, fp + tn))
    fnr = float(fn / max(1, fn + tp))

    fp_per_1k = fpr * 1000.0
    fp_per_10k = fpr * 10000.0
    fp_per_100k = fpr * 100000.0
    # Mathematical projection of daily workload under 1,000,000 normal events/day
    projected_daily_fp = fpr * 1000000.0

    try:
        pr_auc = float(average_precision_score(y_true_arr, y_scores_arr))
    except Exception:
        pr_auc = 0.0

    try:
        roc_auc = float(roc_auc_score(y_true_arr, y_scores_arr))
    except Exception:
        roc_auc = 0.5

    # Per-anomaly-type breakdown
    per_anomaly_recall = None
    if anomaly_types is not None and len(anomaly_types) == len(y_true_arr):
        types_arr = np.asarray(anomaly_types)
        unique_types = [t for t in np.unique(types_arr) if t not in ["normal", "False", False]]
        per_anomaly_recall = {}
        for a_type in unique_types:
            mask = types_arr == a_type
            if np.sum(mask) > 0:
                type_caught = np.sum(y_pred[mask])
                per_anomaly_recall[str(a_type)] = round(float(type_caught / np.sum(mask)), 4)

    ci_map = None
    if compute_bootstrap and len(y_true_arr) > 10:
        ci_map = _compute_bootstrap_ci(
            y_true_arr, y_scores_arr, threshold=threshold, n_bootstrap=n_bootstrap, seed=seed
        )

    normal_count = int(tn + fp)
    anomaly_count = int(tp + fn)
    total_count = normal_count + anomaly_count

    return EvaluationResult(
        threshold=threshold,
        precision=precision,
        recall=recall,
        f1=f1,
        false_positive_rate=fpr,
        false_negative_rate=fnr,
        pr_auc=pr_auc,
        roc_auc=roc_auc,
        fp_per_1k=fp_per_1k,
        fp_per_10k=fp_per_10k,
        fp_per_100k=fp_per_100k,
        projected_daily_fp_1m_events=projected_daily_fp,
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        total_samples=total_count,
        normal_samples=normal_count,
        anomaly_samples=anomaly_count,
        anomaly_rate=float(anomaly_count / max(1, total_count)),
        per_anomaly_recall=per_anomaly_recall,
        confidence_intervals=ci_map,
    )


def _compute_bootstrap_ci(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> dict[str, ConfidenceInterval]:
    rng = np.random.default_rng(seed)
    n = len(y_true)

    precisions = []
    recalls = []
    f1s = []
    pr_aucs = []
    fprs = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt_sample = y_true[idx]
        ys_sample = y_scores[idx]

        if np.sum(yt_sample) == 0:
            continue

        yp_sample = ys_sample >= threshold

        precisions.append(precision_score(yt_sample, yp_sample, zero_division=0))
        recalls.append(recall_score(yt_sample, yp_sample, zero_division=0))
        f1s.append(f1_score(yt_sample, yp_sample, zero_division=0))

        tn_s, fp_s, _, _ = confusion_matrix(yt_sample, yp_sample, labels=[False, True]).ravel()
        fprs.append(fp_s / max(1, fp_s + tn_s))

        try:
            pr_aucs.append(average_precision_score(yt_sample, ys_sample))
        except Exception:
            pass

    def get_ci(arr: list[float]) -> ConfidenceInterval:
        if not arr:
            return ConfidenceInterval(0.0, 0.0, 0.0)
        return ConfidenceInterval(
            mean=float(np.mean(arr)),
            ci_lower=float(np.percentile(arr, 2.5)),
            ci_upper=float(np.percentile(arr, 97.5)),
        )

    return {
        "precision": get_ci(precisions),
        "recall": get_ci(recalls),
        "f1": get_ci(f1s),
        "pr_auc": get_ci(pr_aucs),
        "false_positive_rate": get_ci(fprs),
    }
