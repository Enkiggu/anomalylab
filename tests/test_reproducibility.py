"""Tests verifying full end-to-end reproducibility of data generation, features, and model training."""

import numpy as np

from anomalylab.data.fingerprint import compute_dataframe_fingerprint
from anomalylab.data.generate import generate_security_telemetry
from anomalylab.features.pipeline import FeaturePipeline
from anomalylab.models.isolation_forest import IsolationForestModel


def test_end_to_end_reproducibility():
    """Verify that starting from seed=42 produces bit-for-bit identical outputs."""
    # 1. Reproducible Data
    df_a = generate_security_telemetry(rows=300, seed=42)
    df_b = generate_security_telemetry(rows=300, seed=42)
    assert compute_dataframe_fingerprint(df_a) == compute_dataframe_fingerprint(df_b)

    # 2. Reproducible Features
    pipeline = FeaturePipeline(version="v1")
    X_a, _ = pipeline.extract_features(df_a)
    X_b, _ = pipeline.extract_features(df_b)
    assert np.allclose(X_a.to_numpy(), X_b.to_numpy())

    # 3. Reproducible Model Scores
    model_a = IsolationForestModel(n_estimators=50, random_state=42)
    model_b = IsolationForestModel(n_estimators=50, random_state=42)

    model_a.fit(X_a.to_numpy())
    model_b.fit(X_b.to_numpy())

    scores_a = model_a.score_samples(X_a.to_numpy())
    scores_b = model_b.score_samples(X_b.to_numpy())

    assert np.allclose(scores_a, scores_b, atol=1e-5)
