"""Tests for statistical drift detection engine."""

import numpy as np

from anomalylab.drift.detector import calculate_feature_drift, calculate_psi


def test_psi_identical_distributions():
    """Verify PSI is ~0 for identical distributions."""
    rng = np.random.default_rng(42)
    sample_a = rng.normal(loc=10.0, scale=2.0, size=1000)
    sample_b = rng.normal(loc=10.0, scale=2.0, size=1000)

    psi = calculate_psi(sample_a, sample_b)
    assert psi < 0.05


def test_psi_shifted_distributions():
    """Verify PSI detects significant shift (> 0.25)."""
    rng = np.random.default_rng(42)
    sample_a = rng.normal(loc=10.0, scale=2.0, size=1000)
    sample_b = rng.normal(loc=25.0, scale=2.0, size=1000)  # Heavy shift

    psi = calculate_psi(sample_a, sample_b)
    assert psi > 0.25


def test_feature_drift_detector():
    rng = np.random.default_rng(42)
    ref = rng.normal(loc=5.0, scale=1.0, size=500)
    shifted = rng.normal(loc=12.0, scale=3.0, size=500)

    res = calculate_feature_drift(ref, shifted, feature_name="test_feature")
    assert res["status"] in ["WARNING", "DRIFT_DETECTED"]
    assert res["ks_pvalue"] < 0.001
