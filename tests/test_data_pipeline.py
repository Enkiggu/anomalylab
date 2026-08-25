"""Unit tests for synthetic data generation, quality validation, and fingerprinting."""

import polars as pl
import pytest

from anomalylab.data.fingerprint import compute_dataframe_fingerprint
from anomalylab.data.generate import generate_security_telemetry
from anomalylab.data.validator import DataQualityValidator, DataValidationError


def test_synthetic_data_generation_deterministic():
    """Verify that same seed and config produce identical datasets."""
    df1 = generate_security_telemetry(rows=500, seed=42)
    df2 = generate_security_telemetry(rows=500, seed=42)

    assert len(df1) == 500
    assert len(df2) == 500
    assert df1.equals(df2)

    fp1 = compute_dataframe_fingerprint(df1)
    fp2 = compute_dataframe_fingerprint(df2)
    assert fp1 == fp2


def test_synthetic_data_ground_truth_anomalies():
    """Verify anomaly rate and anomaly types injected."""
    df = generate_security_telemetry(rows=1000, anomaly_rate=0.05, seed=123)
    anomaly_count = int(df["is_anomaly"].sum())

    assert anomaly_count == 50
    unique_types = set(df.filter(pl.col("is_anomaly"))["anomaly_type"].to_list())
    assert len(unique_types) > 1
    assert "normal" not in unique_types


def test_data_quality_validator_passes_valid_data():
    """Verify DataQualityValidator on pristine generated data."""
    df = generate_security_telemetry(rows=500, seed=42)
    validator = DataQualityValidator()
    report = validator.validate(df, dataset_version="test-v1")

    assert report.overall_status == "PASSED"
    assert report.total_records == 500
    assert len(report.checks) >= 5
    assert all(c.passed for c in report.checks)


def test_data_quality_validator_fails_corrupted_data():
    """Verify validator catches corrupted ranges or invalid types."""
    df = generate_security_telemetry(rows=500, seed=42)

    # Corrupt login_hour
    corrupted_df = df.with_columns(pl.lit(99).alias("login_hour"))
    validator = DataQualityValidator()

    with pytest.raises(DataValidationError):
        validator.validate(corrupted_df, dataset_version="corrupted-v1")
