"""Statistical and rule-based baseline anomaly detection models for AnomalyLab."""

from typing import Any

import numpy as np

from anomalylab.models.base import BaseAnomalyModel


class ZScoreStatisticalBaseline(BaseAnomalyModel):
    """
    Robust Z-Score Statistical Baseline.
    Uses Median and Median Absolute Deviation (MAD) on training data:
    z_i = |x_i - median_i| / (1.4826 * MAD_i + 1e-6)
    Anomaly score is the normalized combination of top-k feature z-scores.
    """

    def __init__(
        self,
        model_id: str = "statistical_baseline",
        threshold: float = 0.65,
        top_k: int = 3,
        method: str = "robust_mad",
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.top_k = top_k
        self.method = method
        self.medians: np.ndarray | None = None
        self.mads: np.ndarray | None = None

    def fit(
        self, X: np.ndarray, feature_names: list[str] | None = None
    ) -> "ZScoreStatisticalBaseline":
        X_arr = np.asarray(X, dtype=np.float32)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        self.medians = np.median(X_arr, axis=0)
        deviations = np.abs(X_arr - self.medians)
        # Normal scale factor 1.4826 for standard deviation consistency under normality
        self.mads = 1.4826 * np.median(deviations, axis=0)
        # Prevent division by zero
        self.mads = np.where(self.mads < 1e-5, 1.0, self.mads)
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted or self.medians is None or self.mads is None:
            raise ValueError("Model is not fitted. Call fit() before score_samples().")

        X_arr = np.asarray(X, dtype=np.float32)
        # Compute z-scores for all features
        z_scores = np.abs(X_arr - self.medians) / self.mads

        # Sort z-scores descending per row and average top-k
        k = min(self.top_k, z_scores.shape[1])
        top_z = np.sort(z_scores, axis=1)[:, -k:]
        mean_top_z = np.mean(top_z, axis=1)

        # Smooth squashing to [0, 1] range: 1 / (1 + exp(- (z - 3.0)))
        scores = 1.0 / (1.0 + np.exp(-0.8 * (mean_top_z - 3.0)))
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model_type": "statistical_baseline",
            "method": self.method,
            "top_k": self.top_k,
            "threshold": self.threshold,
        }


class RuleBaseline(BaseAnomalyModel):
    """
    Deterministic domain heuristic security rule baseline.
    Flags events with obvious brute-force, high velocity, or extreme off-hours novelty.
    """

    def __init__(self, model_id: str = "rule_baseline", threshold: float = 0.5):
        super().__init__(model_id=model_id, threshold=threshold)

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "RuleBaseline":
        if feature_names is not None:
            self.feature_names = list(feature_names)
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        X_arr = np.asarray(X, dtype=np.float32)
        n_samples = X_arr.shape[0]
        scores = np.zeros(n_samples, dtype=np.float32)

        feat_idx = {name: i for i, name in enumerate(self.feature_names)}

        # Rule 1: High failed logins
        if "failed_logins_5m" in feat_idx:
            idx = feat_idx["failed_logins_5m"]
            scores = np.maximum(scores, np.where(X_arr[:, idx] >= 10.0, 0.95, 0.0))

        # Rule 2: High velocity events
        if "events_per_user_5m" in feat_idx:
            idx = feat_idx["events_per_user_5m"]
            scores = np.maximum(scores, np.where(X_arr[:, idx] >= 80.0, 0.85, 0.0))

        # Rule 3: Extreme off-hours with new source IP
        if "distance_from_typical_hour" in feat_idx and "is_new_source_ip" in feat_idx:
            h_idx = feat_idx["distance_from_typical_hour"]
            ip_idx = feat_idx["is_new_source_ip"]
            off_hours_new_ip = (X_arr[:, h_idx] >= 8.0) & (X_arr[:, ip_idx] > 0.5)
            scores = np.maximum(scores, np.where(off_hours_new_ip, 0.80, 0.0))

        # Rule 4: High failure ratio
        if "failure_ratio" in feat_idx:
            f_idx = feat_idx["failure_ratio"]
            scores = np.maximum(scores, np.where(X_arr[:, f_idx] >= 0.75, 0.70, 0.0))

        return scores

    def get_params(self) -> dict[str, Any]:
        return {
            "model_type": "rule_baseline",
            "threshold": self.threshold,
        }
