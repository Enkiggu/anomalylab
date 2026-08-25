"""Tests for anomaly detection models, baselines, and serialization."""

import tempfile

import numpy as np
import pytest

from anomalylab.models.base import BaseAnomalyModel
from anomalylab.models.baseline import ZScoreStatisticalBaseline
from anomalylab.models.isolation_forest import IsolationForestModel
from anomalylab.models.manifest import ModelManifest, ModelStage
from anomalylab.models.promotion import ModelPromotionPolicy, PromotionPolicyViolation


@pytest.fixture
def synthetic_features():
    rng = np.random.default_rng(42)
    # Normal data: centered around mean 0
    normal = rng.normal(loc=0.0, scale=1.0, size=(200, 10))
    # Injected anomalies: extreme values
    anomalies = rng.normal(loc=8.0, scale=2.0, size=(10, 10))
    X = np.vstack([normal, anomalies]).astype(np.float32)
    return X


def test_statistical_baseline_scoring(synthetic_features):
    model = ZScoreStatisticalBaseline(threshold=0.6)
    model.fit(synthetic_features[:150])

    scores = model.score_samples(synthetic_features)
    assert scores.shape == (210,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    # Anomaly scores on extreme points should be significantly higher
    normal_mean = float(np.mean(scores[:200]))
    anomaly_mean = float(np.mean(scores[200:]))
    assert anomaly_mean > normal_mean


def test_isolation_forest_normalized_scoring(synthetic_features):
    model = IsolationForestModel(n_estimators=50, random_state=42)
    model.fit(synthetic_features[:150])

    scores = model.score_samples(synthetic_features)
    assert scores.shape == (210,)
    assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    normal_mean = float(np.mean(scores[:200]))
    anomaly_mean = float(np.mean(scores[200:]))
    assert anomaly_mean > normal_mean


def test_model_serialization(synthetic_features):
    model = IsolationForestModel(n_estimators=30, random_state=42)
    model.fit(synthetic_features)

    scores_orig = model.score_samples(synthetic_features[:10])

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    model.save(tmp_path)
    loaded_model = BaseAnomalyModel.load(tmp_path)
    scores_loaded = loaded_model.score_samples(synthetic_features[:10])

    assert np.allclose(scores_orig, scores_loaded, atol=1e-5)


def test_model_promotion_policy_enforcement():
    """Verify that promotion policy enforces validation gates and blocks sub-standard models."""
    policy = ModelPromotionPolicy(
        min_validation_recall=0.85,
        max_validation_fpr=0.05,
        min_validation_precision=0.40,
    )

    # Compliant model (e.g. Isolation Forest)
    good_manifest = ModelManifest(
        model_id="iforest_test",
        model_version="iforest_test_v1",
        algorithm="isolation_forest",
        stage=ModelStage.CANDIDATE,
        dataset_version="security_v2",
        feature_version="v2",
        dataset_sha256="dummy",
        threshold=0.63,
        feature_names=["f1", "f2"],
        validation_metrics={"recall": 0.91, "precision": 0.55, "false_positive_rate": 0.015},
        artifact_path="dummy.joblib",
    )
    passed, violations = policy.evaluate(good_manifest)
    assert passed is True
    assert len(violations) == 0

    # Policy enforcement should allow promotion to production
    policy.enforce_promotion(good_manifest, ModelStage.PRODUCTION)

    # Sub-standard model with excessive FPR (e.g. LOF or uncalibrated baseline)
    bad_manifest = ModelManifest(
        model_id="bad_model_test",
        model_version="bad_model_v1",
        algorithm="local_outlier_factor",
        stage=ModelStage.CANDIDATE,
        dataset_version="security_v2",
        feature_version="v2",
        dataset_sha256="dummy",
        threshold=0.15,
        feature_names=["f1", "f2"],
        validation_metrics={"recall": 0.90, "precision": 0.03, "false_positive_rate": 0.75},
        artifact_path="dummy.joblib",
    )
    passed_bad, violations_bad = policy.evaluate(bad_manifest)
    assert passed_bad is False
    assert len(violations_bad) >= 2

    # Attempting to promote bad model should raise PromotionPolicyViolation
    with pytest.raises(PromotionPolicyViolation):
        policy.enforce_promotion(bad_manifest, ModelStage.PRODUCTION)


def test_preprocessor_fitting_isolation(synthetic_features):
    """Verify that statistical baseline parameters remain strictly frozen after fitting on train partition."""
    X_train = synthetic_features[:100]
    X_test = synthetic_features[100:]

    model = ZScoreStatisticalBaseline(threshold=0.5)
    model.fit(X_train)

    frozen_medians = np.copy(model.medians)
    frozen_mads = np.copy(model.mads)

    # Score test data multiple times
    _ = model.score_samples(X_test)
    _ = model.score_samples(X_test)

    # Parameters must be 100% invariant
    assert np.array_equal(model.medians, frozen_medians)
    assert np.array_equal(model.mads, frozen_mads)
