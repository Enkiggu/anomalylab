"""Base interface and abstract class for AnomalyLab anomaly detection models."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import joblib
import numpy as np


class BaseAnomalyModel(ABC):
    """
    Abstract Base Class for all anomaly detection algorithms in AnomalyLab.

    Standard Invariants:
    1. Anomaly scores are normalized: higher score in [0.0, 1.0] means higher risk / more anomalous.
    2. fit(X) operates in unsupervised mode (y is ignored or optional for interface compatibility).
    3. predict(X, threshold) uses calibrated threshold.
    """

    def __init__(self, model_id: str = "base_model", threshold: float = 0.5):
        self.model_id = model_id
        self.threshold = threshold
        self.is_fitted = False
        self.feature_names: list[str] = []

    @abstractmethod
    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "BaseAnomalyModel":
        """Fit the anomaly detection model on feature matrix X (N x D)."""
        pass

    @abstractmethod
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Compute continuous anomaly scores for samples in X.
        Returns 1D array of shape (N,) with scores in [0.0, 1.0].
        """
        pass

    def predict(self, X: np.ndarray, threshold: float | None = None) -> np.ndarray:
        """
        Predict binary anomaly flags (True = anomaly, False = normal).
        """
        thresh = threshold if threshold is not None else self.threshold
        scores = self.score_samples(X)
        return (scores >= thresh).astype(bool)

    @abstractmethod
    def get_params(self) -> dict[str, Any]:
        """Return model hyperparameters as a dictionary."""
        pass

    def save(self, path: str | Path) -> None:
        """Serialize model artifact to disk."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, target)

    @classmethod
    def load(cls, path: str | Path) -> "BaseAnomalyModel":
        """Deserialize model artifact from disk."""
        loaded = joblib.load(path)
        if not isinstance(loaded, BaseAnomalyModel):
            raise TypeError(f"Loaded object is {type(loaded)}, expected BaseAnomalyModel")
        return loaded
