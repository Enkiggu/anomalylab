"""Local Outlier Factor (LOF) novelty detection model wrapper."""

from typing import Any

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from anomalylab.models.base import BaseAnomalyModel


class LocalOutlierFactorModel(BaseAnomalyModel):
    """
    Local Outlier Factor (LOF) Novelty Detector.
    Uses novelty=True to enable out-of-sample prediction and score_samples.
    """

    def __init__(
        self,
        model_id: str = "lof_model",
        n_neighbors: int = 35,
        contamination: float = 0.02,
        algorithm: str = "auto",
        metric: str = "minkowski",
        threshold: float = 0.60,
    ):
        super().__init__(model_id=model_id, threshold=threshold)
        self.n_neighbors = n_neighbors
        self.contamination = contamination
        self.algorithm = algorithm
        self.metric = metric

        self.estimator = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            algorithm=self.algorithm,
            metric=self.metric,
            novelty=True,
            n_jobs=-1,
        )

    def fit(
        self, X: np.ndarray, feature_names: list[str] | None = None
    ) -> "LocalOutlierFactorModel":
        X_arr = np.asarray(X, dtype=np.float32)
        if feature_names is not None:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        self.estimator.fit(X_arr)
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model is not fitted. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        # decision_function gives negative values for outliers, positive for inliers
        raw_decision = self.estimator.decision_function(X_arr)

        # Logistic calibration into [0, 1]
        scores = 1.0 / (1.0 + np.exp(4.0 * raw_decision))
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {
            "model_type": "local_outlier_factor",
            "n_neighbors": self.n_neighbors,
            "contamination": self.contamination,
            "algorithm": self.algorithm,
            "metric": self.metric,
            "threshold": self.threshold,
        }
