"""Tests for explainability engine and non-causal message generation."""

from anomalylab.explainability.engine import ExplainabilityEngine


def test_explainability_identifies_dominant_signals():
    engine = ExplainabilityEngine()

    # Event with heavy failed logins and new IP
    features = {
        "failed_logins_5m": 25.0,  # Huge deviation
        "failed_logins_1h": 35.0,
        "is_new_source_ip": 1.0,
        "login_hour": 14.0,  # Normal
        "distance_from_typical_hour": 1.0,  # Normal
    }

    signals = engine.explain_sample(features, anomaly_score=0.92, threshold=0.72, top_k=3)

    assert len(signals) >= 2
    top_feature_names = [s.feature for s in signals]
    assert "failed_logins_5m" in top_feature_names or "failed_logins_1h" in top_feature_names

    # Check non-causal language requirement (must not assert certainty/causation)
    for s in signals:
        assert "caused" not in s.message.lower()
        assert len(s.message) > 10
