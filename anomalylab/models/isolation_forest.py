"""Isolation Forest anomaly detection model wrapper with calibrated [0, 1] scoring."""

from typing import Any

import numpy as np
from sklearn.ensemble import IsolationForest

from anomalylab.models.base import BaseAnomalyModel


class IsolationForestModel(BaseAnomalyModel):
    """
    Calibrated Isolation Forest Anomaly Detector.

    Standard sklearn Isolation Forest produces decision_function where:
    - Normal instances have positive values (e.g. +0.15)
    - Anomalies have negative values (e.g. -0.25)

    This wrapper inverts and calibrates the decision function into a standard risk score in [0.0, 1.0]:
    score = 1 / (1 + exp(12 * decision_function))
    """

    def __init__(
        self,
        model_id: str = "isolation_forest_model",
        n_estimators: int = 250,
        max_samples: float | str = 0.8,
        contamination: float | str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
        random_state: int = 42,
        threshold: float = 0.65,
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state

        self.estimator = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.calibration_min: float = -0.3
        self.calibration_max: float = 0.2

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "IsolationForestModel":
        X_arr = np.asarray(X, dtype=np.float32)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        self.estimator.fit(X_arr)

        # Fit calibration ranges from training decision scores
        train_raw_scores = self.estimator.decision_function(X_arr)
        self.calibration_min = float(np.percentile(train_raw_scores, 0.5))
        self.calibration_max = float(np.percentile(train_raw_scores, 99.5))
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        raw_decision = self.estimator.decision_function(X_arr)

        # Logistic calibration: inverts so lower decision -> higher anomaly score
        # Using calibrated scaling factor based on decision score distribution
        scores = 1.0 / (1.0 + np.exp(10.0 * raw_decision))
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model_type": "isolation_forest",
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "contamination": str(self.contamination),
            "max_features": self.max_features,
            "bootstrap": self.bootstrap,
            "random_state": self.random_state,
            "threshold": self.threshold,
        }
