"""Verification of critical ML Leakage Invariants (1, 2, 3, and 4)."""

import numpy as np
import polars as pl

from anomalylab.data.generate import generate_security_telemetry
from anomalylab.features.pipeline import FeaturePipeline
from anomalylab.inference.parity import FeatureParityTransformer
from anomalylab.training.split import time_based_split


def test_invariant_1_train_val_test_disjoint_time_split():
    """
    INVARIANT 1: Train, Validation, and Test datasets are strictly disjoint in time.
    max(Train.timestamp) < min(Val.timestamp) <= max(Val.timestamp) < min(Test.timestamp)
    """
    df = generate_security_telemetry(rows=1000, seed=42)
    train_df, val_df, test_df = time_based_split(df, train_ratio=0.6, val_ratio=0.2)

    assert len(train_df) > 0
    assert len(val_df) > 0
    assert len(test_df) > 0
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    train_max_ts = train_df["timestamp"].max()
    val_min_ts = val_df["timestamp"].min()
    val_max_ts = val_df["timestamp"].max()
    test_min_ts = test_df["timestamp"].min()

    assert train_max_ts <= val_min_ts, (
        f"Temporal overlap between Train ({train_max_ts}) and Val ({val_min_ts})"
    )
    assert val_max_ts <= test_min_ts, (
        f"Temporal overlap between Val ({val_max_ts}) and Test ({test_min_ts})"
    )


def test_invariant_2_zero_lookahead_leakage_in_windows():
    """
    INVARIANT 2: Time-window features only aggregate historical events <= T.
    Adding future events must NOT alter the feature values of past events.
    """
    pipeline = FeaturePipeline(version="v2")
    df = generate_security_telemetry(rows=500, seed=42)

    # Process first 250 rows
    first_half = df.head(250)
    X_first_half, _ = pipeline.extract_features(first_half)

    # Process full 500 rows and inspect the first 250 rows
    X_full, _ = pipeline.extract_features(df)
    X_full_first_half = X_full.head(250)

    # Vector difference must be zero (no future lookahead altering past features)
    diff = X_first_half.to_numpy() - X_full_first_half.to_numpy()
    max_diff = float(abs(diff).max())
    assert max_diff < 1e-5, f"Future lookahead detected! Max feature difference: {max_diff}"


def test_invariant_3_target_labels_never_enter_features():
    """
    INVARIANT 3: Target labels ('is_anomaly', 'anomaly_type') strictly excluded from feature matrix.
    """
    pipeline = FeaturePipeline(version="v2")
    df = generate_security_telemetry(rows=500, seed=42)
    X_df, meta_df = pipeline.extract_features(df)

    forbidden_cols = ["is_anomaly", "anomaly_type", "user_id", "host_id", "source_ip", "timestamp"]
    for col in forbidden_cols:
        assert col not in X_df.columns, (
            f"Target or identifier column '{col}' leaked into feature matrix!"
        )

    # Ensure all feature contract columns are purely numeric
    for col in X_df.columns:
        assert X_df[col].dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64], (
            f"Non-numeric feature dtype in '{col}': {X_df[col].dtype}"
        )


def test_invariant_4_training_serving_feature_parity():
    """
    INVARIANT 4: Single-event online feature transformation exactly matches vectorized batch parity.
    """
    transformer = FeatureParityTransformer()

    raw_event = {
        "userId": "user_0042.demo",
        "loginHour": 14,
        "eventOutcome": "SUCCESS",
        "failedAttempts": 2,
        "requestCount": 5,
        "bytesSent": 2500,
        "bytesReceived": 10000,
        "sessionDuration": 180,
        "isPrivilegedUser": True,
        "newSourceIp": False,
        "newDevice": False,
    }

    # Transform single event
    vec_single, _ = transformer.transform_single(raw_event)

    # Transform batch containing identical event
    matrix_batch, _ = transformer.transform_batch([raw_event, raw_event])

    assert vec_single.shape == (1, 15)
    assert matrix_batch.shape == (2, 15)
    assert np.allclose(vec_single[0], matrix_batch[0], atol=1e-5)
    assert np.allclose(matrix_batch[0], matrix_batch[1], atol=1e-5)
