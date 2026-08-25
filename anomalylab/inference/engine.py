"""Production inference engine with warm-up, vectorized batch execution, and explainability."""

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from typing import Any

import numpy as np

from anomalylab.explainability.engine import ExplainabilityEngine
from anomalylab.features.contract import FeatureContract, get_security_feature_contract_v1
from anomalylab.inference.parity import FeatureParityTransformer
from anomalylab.inference.registry import ModelRegistry
from anomalylab.models.base import BaseAnomalyModel
from anomalylab.models.manifest import ModelManifest, ModelStage


class RiskBand(StrEnum):
    NORMAL = "NORMAL"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ReferenceRiskModel(BaseAnomalyModel):
    """Deterministic artifact-free baseline used when no trained model is installed."""

    def __init__(self, feature_names: list[str], threshold: float = 0.60):
        super().__init__(model_id="reference-risk-v1", threshold=threshold)
        self.feature_names = feature_names
        self.is_fitted = True

    def fit(self, X: np.ndarray, feature_names: list[str] | None = None) -> "ReferenceRiskModel":
        if feature_names is not None:
            self.feature_names = list(feature_names)
        self.is_fitted = True
        return self

    def score_samples(self, X: np.ndarray) -> np.ndarray:
        values = np.asarray(X, dtype=np.float32)
        indexes = {name: index for index, name in enumerate(self.feature_names)}

        def column(name: str) -> np.ndarray:
            return values[:, indexes[name]]

        scores = (
            0.30 * np.clip(column("failed_logins_1h") / 30.0, 0.0, 1.0)
            + 0.15 * np.clip(column("failure_ratio"), 0.0, 1.0)
            + 0.12 * np.clip(column("is_new_source_ip"), 0.0, 1.0)
            + 0.10 * np.clip(column("is_new_device"), 0.0, 1.0)
            + 0.12 * np.clip(column("distance_from_typical_hour") / 12.0, 0.0, 1.0)
            + 0.12 * np.clip(column("events_per_user_5m") / 50.0, 0.0, 1.0)
            + 0.09 * np.clip(column("distinct_users_per_ip_10m") / 10.0, 0.0, 1.0)
        )
        return np.clip(scores, 0.0, 1.0).astype(np.float32)

    def get_params(self) -> dict[str, Any]:
        return {"model_type": "reference_risk", "threshold": self.threshold}


def compute_risk_band(score: float, is_anomaly: bool, is_privileged: bool = False) -> RiskBand:
    """Map anomaly score and entity sensitivity into operational risk band."""
    effective_score = score + (0.08 if is_privileged else 0.0)

    if effective_score >= 0.88:
        return RiskBand.CRITICAL
    elif effective_score >= 0.72:
        return RiskBand.HIGH
    elif effective_score >= 0.55:
        return RiskBand.MEDIUM
    elif effective_score >= 0.40:
        return RiskBand.LOW
    return RiskBand.NORMAL


class ProductionInferenceEngine:
    """
    High-throughput, thread-safe anomaly inference service.
    Loads active champion model once into memory and provides hot reload.
    """

    def __init__(self, models_dir: str = "models", max_workers: int = 4):
        self.registry = ModelRegistry(models_dir=models_dir)
        self.contract: FeatureContract = get_security_feature_contract_v1()
        self.transformer = FeatureParityTransformer(contract=self.contract)
        self.explainability = ExplainabilityEngine()

        self.current_manifest: ModelManifest | None = None
        self.current_model: BaseAnomalyModel | None = None
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.is_ready = False

    def load_champion_model(self) -> ModelManifest:
        """Load or reload the current production model from registry."""
        manifest = self.registry.get_production_model()
        if not manifest:
            return self.initialize_reference_model()

        print(
            f"Loading champion model: {manifest.model_version} ({manifest.algorithm}) from {manifest.artifact_path}"
        )
        model = BaseAnomalyModel.load(manifest.artifact_path)

        self.current_manifest = manifest
        self.current_model = model
        self.warmup()
        self.is_ready = True
        return manifest

    def initialize_reference_model(self) -> ModelManifest:
        """Initialize a deterministic baseline without persisting model artifacts."""
        model = ReferenceRiskModel(feature_names=self.contract.feature_names)
        manifest = ModelManifest(
            model_id=model.model_id,
            model_version="reference-v1",
            algorithm="deterministic_risk_baseline",
            stage=ModelStage.PRODUCTION,
            dataset_version="runtime-defaults-v1",
            feature_version=self.contract.version,
            dataset_sha256="not-applicable",
            threshold=model.threshold,
            feature_names=self.contract.feature_names,
            parameters=model.get_params(),
            artifact_path="built-in",
            description="Artifact-free startup baseline for local and evaluation environments.",
        )

        self.current_manifest = manifest
        self.current_model = model
        self.warmup()
        self.is_ready = True
        return manifest

    def warmup(self, n_samples: int = 10) -> None:
        """Run synthetic warmup inference pass to initialize CPU vectorization pipelines."""
        if self.current_model is None:
            return

        dummy_event = {
            "login_hour": 14,
            "event_outcome": "SUCCESS",
            "failed_attempts": 0,
            "request_count": 5,
            "bytes_sent": 1200,
            "bytes_received": 4500,
            "session_duration": 180,
            "is_privileged_user": False,
            "is_new_source": False,
            "is_new_device": False,
        }
        vector, _ = self.transformer.transform_single(dummy_event)
        batch = np.repeat(vector, n_samples, axis=0)
        _ = self.current_model.score_samples(batch)
        print(f"Inference engine warmup completed with {n_samples} vectors.")

    def predict_single(self, event: dict[str, Any]) -> dict[str, Any]:
        """Synchronous single-event prediction with explainability signals."""
        if not self.is_ready or self.current_model is None or self.current_manifest is None:
            raise RuntimeError("Inference engine not ready: No active model loaded.")

        start_t = time.perf_counter()
        vector, feature_map = self.transformer.transform_single(event)

        scores = self.current_model.score_samples(vector)
        score = float(scores[0])
        threshold = float(self.current_manifest.threshold)
        is_anomaly = bool(score >= threshold)

        is_priv = bool(event.get("is_privileged_user", event.get("isPrivilegedUser", False)))
        risk_band = compute_risk_band(score, is_anomaly, is_privileged=is_priv)

        signals = self.explainability.explain_sample(
            features=feature_map,
            anomaly_score=score,
            threshold=threshold,
            top_k=4,
        )

        latency_ms = (time.perf_counter() - start_t) * 1000.0

        return {
            "modelVersion": self.current_manifest.model_version,
            "algorithm": self.current_manifest.algorithm,
            "anomalyScore": round(score, 4),
            "isAnomaly": is_anomaly,
            "threshold": round(threshold, 4),
            "riskBand": risk_band.value,
            "contributingSignals": [s.to_dict() for s in signals],
            "featureValues": {k: round(v, 2) for k, v in feature_map.items()},
            "latencyMs": round(latency_ms, 3),
        }

    async def predict_single_async(self, event: dict[str, Any]) -> dict[str, Any]:
        """Non-blocking async wrapper executing inference in thread pool."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.predict_single, event)

    def predict_batch(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Vectorized batch prediction for high-throughput batch evaluation."""
        if not self.is_ready or self.current_model is None or self.current_manifest is None:
            raise RuntimeError("Inference engine not ready: No active model loaded.")

        start_t = time.perf_counter()
        batch_size = len(events)

        if batch_size == 0:
            return {
                "modelVersion": self.current_manifest.model_version,
                "predictions": [],
                "batchSize": 0,
                "latencyMs": 0.0,
            }

        vectors, maps = self.transformer.transform_batch(events)
        scores = self.current_model.score_samples(vectors)
        threshold = float(self.current_manifest.threshold)

        results = []
        for i in range(batch_size):
            score = float(scores[i])
            is_anomaly = bool(score >= threshold)
            is_priv = bool(
                events[i].get("is_privileged_user", events[i].get("isPrivilegedUser", False))
            )
            risk_band = compute_risk_band(score, is_anomaly, is_privileged=is_priv)

            signals = (
                self.explainability.explain_sample(
                    features=maps[i],
                    anomaly_score=score,
                    threshold=threshold,
                    top_k=3,
                )
                if is_anomaly or score > 0.5
                else []
            )

            results.append(
                {
                    "index": i,
                    "anomalyScore": round(score, 4),
                    "isAnomaly": is_anomaly,
                    "riskBand": risk_band.value,
                    "contributingSignals": [s.to_dict() for s in signals],
                }
            )

        total_latency_ms = (time.perf_counter() - start_t) * 1000.0
        avg_per_item_ms = total_latency_ms / max(1, batch_size)

        return {
            "modelVersion": self.current_manifest.model_version,
            "algorithm": self.current_manifest.algorithm,
            "threshold": round(threshold, 4),
            "batchSize": batch_size,
            "anomalyCount": sum(1 for r in results if r["isAnomaly"]),
            "predictions": results,
            "totalLatencyMs": round(total_latency_ms, 2),
            "avgLatencyPerItemMs": round(avg_per_item_ms, 4),
            "throughputPerSec": round((batch_size / (total_latency_ms / 1000.0)), 1)
            if total_latency_ms > 0
            else 0.0,
        }

    async def predict_batch_async(self, events: list[dict[str, Any]]) -> dict[str, Any]:
        """Non-blocking async wrapper for batch prediction."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self.executor, self.predict_batch, events)
