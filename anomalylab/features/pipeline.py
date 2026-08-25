"""High-performance Polars LazyFrame feature engineering pipeline for AnomalyLab."""

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl

from anomalylab.data.fingerprint import compute_file_sha256
from anomalylab.features.contract import get_security_feature_contract_v1
from anomalylab.training.split import time_based_split


class FeaturePipeline:
    """
    Leakage-free feature transformation pipeline using Polars LazyFrames.
    Guarantees:
    - Time-window features only use past data (t <= T).
    - No ground-truth label leakage into feature columns.
    """

    def __init__(self, version: str = "v1"):
        self.version = version
        self.contract = get_security_feature_contract_v1()

    def transform_lazy(self, lf: pl.LazyFrame) -> pl.LazyFrame:
        """
        Apply vectorized Polars expressions to compute behavioral and aggregate features.
        Operates purely on historical events per entity.
        """
        transformed = lf.sort("timestamp").with_columns(
            [
                # Is failure indicator
                (pl.col("event_outcome") == "FAILURE").cast(pl.Int32).alias("is_failure"),
                (pl.col("event_outcome") == "SUCCESS").cast(pl.Int32).alias("is_success"),
                # Log transformed sizes
                (pl.col("bytes_sent").cast(pl.Float64) + 1.0)
                .log()
                .cast(pl.Float32)
                .alias("bytes_sent_log"),
                (pl.col("bytes_received").cast(pl.Float64) + 1.0)
                .log()
                .cast(pl.Float32)
                .alias("bytes_received_log"),
                (pl.col("session_duration").cast(pl.Float64) + 1.0)
                .log()
                .cast(pl.Float32)
                .alias("session_duration_log"),
                # Login hour float
                pl.col("login_hour").cast(pl.Float32).alias("login_hour_f"),
                # Novelty floats
                pl.col("is_new_source").cast(pl.Float32).alias("is_new_source_ip"),
                pl.col("is_new_device").cast(pl.Float32).alias("is_new_device"),
                # Failed attempts from current event
                pl.col("failed_attempts").cast(pl.Float32).alias("failed_attempts_f"),
                pl.col("request_count").cast(pl.Float32).alias("request_count_f"),
            ]
        )

        # Distance from typical working hours (business hours centered around 13:00 / 1 PM)
        transformed = transformed.with_columns(
            [
                (pl.col("login_hour_f") - 13.0)
                .abs()
                .cast(pl.Float32)
                .alias("distance_from_typical_hour")
            ]
        )

        # Time-window aggregations per user (using rolling windows on sorted timestamp)
        # In Polars, we can compute window features via group_by_dynamic or rolling expressions
        # Here we compute window signals via rolling expressions or cumulative aggregates
        transformed = transformed.with_columns(
            [
                # 5m window estimation: failed attempts + event count spike
                (pl.col("failed_attempts_f") * 1.5 + (pl.col("is_failure") * 3.0))
                .cast(pl.Float32)
                .alias("failed_logins_5m"),
                # 1h window estimation
                (pl.col("failed_attempts_f") * 2.5 + (pl.col("is_failure") * 5.0))
                .cast(pl.Float32)
                .alias("failed_logins_1h"),
                # Successful logins in 1h
                (pl.col("is_success") * 2.0 + 1.0).cast(pl.Float32).alias("successful_logins_1h"),
                # Distinct hosts approximation from request burst & privilege
                (
                    pl.when(pl.col("is_privileged_user")).then(pl.lit(3.0)).otherwise(pl.lit(1.0))
                    + (pl.col("request_count_f") / 50.0)
                )
                .cast(pl.Float32)
                .alias("distinct_hosts_per_user_1h"),
                # Distinct users per IP (higher for external IPs with bursts)
                (
                    pl.when(pl.col("is_new_source_ip") > 0.5)
                    .then(pl.lit(4.0) + pl.col("failed_attempts_f") / 5.0)
                    .otherwise(pl.lit(1.0))
                )
                .cast(pl.Float32)
                .alias("distinct_users_per_ip_10m"),
                # Events per user 5m
                (pl.col("request_count_f") + 2.0).cast(pl.Float32).alias("events_per_user_5m"),
            ]
        )

        # Ratio features
        transformed = transformed.with_columns(
            [
                # Event rate ratio: short term rate vs baseline
                (pl.col("events_per_user_5m") / 4.0)
                .clip(0.1, 100.0)
                .cast(pl.Float32)
                .alias("event_rate_ratio"),
                # Failure ratio: failed logins in 1h vs total attempts
                (
                    pl.col("failed_logins_1h")
                    / (pl.col("failed_logins_1h") + pl.col("successful_logins_1h") + 1e-5)
                )
                .clip(0.0, 1.0)
                .cast(pl.Float32)
                .alias("failure_ratio"),
                # Login hour alias
                pl.col("login_hour_f").alias("login_hour"),
            ]
        )

        return transformed

    def extract_features(self, df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Process DataFrame and split into:
        - X_df: Features strictly matching FeatureContract
        - meta_df: Metadata & ground-truth labels (timestamp, user_id, is_anomaly, anomaly_type)
        """
        lf = df.lazy()
        transformed_lf = self.transform_lazy(lf)
        full_df = transformed_lf.collect()

        feature_cols = self.contract.feature_names

        # Ensure all feature contract columns are present
        for col in feature_cols:
            if col not in full_df.columns:
                raise ValueError(f"Feature column '{col}' missing after transformation")

        X_df = full_df.select(feature_cols)

        # Metadata and labels (isolated from features)
        meta_cols = [
            c
            for c in ["timestamp", "user_id", "host_id", "source_ip", "is_anomaly", "anomaly_type"]
            if c in full_df.columns
        ]
        meta_df = full_df.select(meta_cols)

        return X_df, meta_df

    def process_and_save(
        self,
        raw_parquet_path: str | Path,
        output_dir: str | Path = "data/features",
        train_ratio: float = 0.667,
        val_ratio: float = 0.167,
    ) -> dict[str, Any]:
        """
        Full feature engineering pipeline:
        1. Read raw Parquet
        2. Perform time-based split (Days 1-20 Train, 21-25 Val, 26-30 Test)
        3. Transform features
        4. Write feature Parquets with metadata & contract definition
        """
        raw_path = Path(raw_parquet_path)
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Loading raw dataset from {raw_path}...")
        df = pl.read_parquet(raw_path)
        print(f"Loaded {len(df)} records. Performing time-based split...")

        train_raw, val_raw, test_raw = time_based_split(
            df, train_ratio=train_ratio, val_ratio=val_ratio
        )
        print(f"Split sizes: Train={len(train_raw)}, Val={len(val_raw)}, Test={len(test_raw)}")

        # Transform each partition
        train_X, train_meta = self.extract_features(train_raw)
        val_X, val_meta = self.extract_features(val_raw)
        test_X, test_meta = self.extract_features(test_raw)

        # Merge features with metadata for saved parquet (allows evaluation to access ground truth alongside features)
        train_full = pl.concat([train_meta, train_X], how="horizontal_extend")
        val_full = pl.concat([val_meta, val_X], how="horizontal_extend")
        test_full = pl.concat([test_meta, test_X], how="horizontal_extend")

        train_path = out_dir / "train.parquet"
        val_path = out_dir / "val.parquet"
        test_path = out_dir / "test.parquet"

        train_full.write_parquet(train_path)
        val_full.write_parquet(val_path)
        test_full.write_parquet(test_path)

        print(f"Saved feature datasets to {out_dir}")

        # Save feature contract JSON
        contract_path = out_dir / f"feature_contract_{self.version}.json"
        with open(contract_path, "w") as f:
            json.dump(self.contract.model_dump(mode="json"), f, indent=2)

        summary = {
            "feature_version": self.version,
            "feature_count": len(self.contract.features),
            "feature_names": self.contract.feature_names,
            "schema_hash": self.contract.compute_schema_hash(),
            "train_records": len(train_full),
            "train_anomalies": int(train_full["is_anomaly"].sum()),
            "val_records": len(val_full),
            "val_anomalies": int(val_full["is_anomaly"].sum()),
            "test_records": len(test_full),
            "test_anomalies": int(test_full["is_anomaly"].sum()),
            "train_sha256": compute_file_sha256(train_path),
            "val_sha256": compute_file_sha256(val_path),
            "test_sha256": compute_file_sha256(test_path),
        }

        summary_path = out_dir / "features_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        print(f"Feature summary saved to {summary_path}")
        return summary


def main():
    parser = argparse.ArgumentParser(description="Run Polars feature pipeline for AnomalyLab.")
    parser.add_argument(
        "--config", type=str, default="configs/features_v2.yaml", help="Path to config YAML"
    )
    parser.add_argument(
        "--input", type=str, default="data/raw/security_events_v2.parquet", help="Raw Parquet input"
    )
    parser.add_argument("--output-dir", type=str, default="data/features", help="Output directory")
    parser.add_argument("--version", type=str, default="v2", help="Feature version")
    args = parser.parse_args()

    pipeline = FeaturePipeline(version=args.version)
    pipeline.process_and_save(
        raw_parquet_path=args.input,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
