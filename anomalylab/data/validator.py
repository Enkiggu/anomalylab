"""Data quality validation engine for AnomalyLab."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from anomalylab.data.schemas import DataQualityCheckResult, DataQualityReport


class DataValidationError(Exception):
    """Raised when data quality validation fails."""

    pass


class DataQualityValidator:
    """Validates raw and processed telemetry datasets before training."""

    REQUIRED_SECURITY_COLUMNS = {
        "timestamp": pl.Datetime,
        "user_id": pl.String,
        "host_id": pl.String,
        "source_ip": pl.String,
        "event_type": pl.String,
        "event_outcome": pl.String,
        "login_hour": pl.Int32,
        "failed_attempts": pl.Int32,
        "request_count": pl.Int32,
        "bytes_sent": pl.Int64,
        "bytes_received": pl.Int64,
        "session_duration": pl.Int32,
        "is_privileged_user": pl.Boolean,
        "is_anomaly": pl.Boolean,
        "anomaly_type": pl.String,
    }

    VALID_EVENT_OUTCOMES = {"SUCCESS", "FAILURE"}
    VALID_EVENT_TYPES = {
        "login",
        "api_access",
        "file_transfer",
        "database_query",
        "privilege_escalation",
    }

    def __init__(self, max_null_percentage: float = 0.0, allow_empty: bool = False):
        self.max_null_percentage = max_null_percentage
        self.allow_empty = allow_empty

    def validate(self, df: pl.DataFrame, dataset_version: str = "unknown") -> DataQualityReport:
        """Run all data quality checks on the dataframe."""
        checks: list[DataQualityCheckResult] = []
        overall_passed = True

        # Check 1: Non-empty dataset
        row_count = len(df)
        if row_count == 0 and not self.allow_empty:
            checks.append(
                DataQualityCheckResult(
                    check_name="non_empty_check",
                    passed=False,
                    details={"row_count": 0},
                    message="Dataset contains zero rows.",
                )
            )
            overall_passed = False
        else:
            checks.append(
                DataQualityCheckResult(
                    check_name="non_empty_check",
                    passed=True,
                    details={"row_count": row_count},
                    message=f"Dataset has {row_count} records.",
                )
            )

        # Check 2: Required columns existence and compatibility
        missing_cols = []
        for col_name, expected_dtype in self.REQUIRED_SECURITY_COLUMNS.items():
            if col_name not in df.columns:
                missing_cols.append(col_name)

        if missing_cols:
            checks.append(
                DataQualityCheckResult(
                    check_name="schema_columns_check",
                    passed=False,
                    details={"missing_columns": missing_cols},
                    message=f"Missing required columns: {missing_cols}",
                )
            )
            overall_passed = False
        else:
            checks.append(
                DataQualityCheckResult(
                    check_name="schema_columns_check",
                    passed=True,
                    details={"columns_found": len(self.REQUIRED_SECURITY_COLUMNS)},
                    message="All required columns are present.",
                )
            )

        # Check 3: Null checks
        null_counts = {}
        null_percentages = {}
        exceeded_null_cols = []

        for col in df.columns:
            n_null = int(df[col].null_count())
            pct_null = float(n_null / max(1, row_count))
            null_counts[col] = n_null
            null_percentages[col] = pct_null
            if pct_null > self.max_null_percentage:
                exceeded_null_cols.append((col, pct_null))

        if exceeded_null_cols:
            checks.append(
                DataQualityCheckResult(
                    check_name="null_values_check",
                    passed=False,
                    details={"exceeded_columns": exceeded_null_cols},
                    message=f"Columns exceeded max null threshold ({self.max_null_percentage}): {exceeded_null_cols}",
                )
            )
            overall_passed = False
        else:
            checks.append(
                DataQualityCheckResult(
                    check_name="null_values_check",
                    passed=True,
                    details={
                        "max_null_pct": max(null_percentages.values()) if null_percentages else 0.0
                    },
                    message="Null percentage check passed.",
                )
            )

        # Check 4: Value range validity (login_hour, non-negative numbers)
        range_errors = []
        if "login_hour" in df.columns:
            min_h = int(df["login_hour"].min()) if row_count > 0 else 0
            max_h = int(df["login_hour"].max()) if row_count > 0 else 0
            if min_h < 0 or max_h > 23:
                range_errors.append(f"login_hour out of bounds [0, 23]: min={min_h}, max={max_h}")

        for num_col in [
            "failed_attempts",
            "request_count",
            "bytes_sent",
            "bytes_received",
            "session_duration",
        ]:
            if num_col in df.columns and row_count > 0:
                min_val = float(df[num_col].min())
                if min_val < 0:
                    range_errors.append(f"{num_col} contains negative values: min={min_val}")

        if range_errors:
            checks.append(
                DataQualityCheckResult(
                    check_name="value_ranges_check",
                    passed=False,
                    details={"range_errors": range_errors},
                    message=f"Value range check failed: {'; '.join(range_errors)}",
                )
            )
            overall_passed = False
        else:
            checks.append(
                DataQualityCheckResult(
                    check_name="value_ranges_check",
                    passed=True,
                    details={},
                    message="Value ranges are within physical and business bounds.",
                )
            )

        # Check 5: Infinite or NaN values
        infinite_cols = []
        for col in df.columns:
            if df[col].dtype in [pl.Float32, pl.Float64]:
                has_nan = bool(df[col].is_nan().any())
                has_inf = bool(df[col].is_infinite().any())
                if has_nan or has_inf:
                    infinite_cols.append(col)

        if infinite_cols:
            checks.append(
                DataQualityCheckResult(
                    check_name="numerical_finite_check",
                    passed=False,
                    details={"infinite_or_nan_cols": infinite_cols},
                    message=f"Found NaN or Infinite values in columns: {infinite_cols}",
                )
            )
            overall_passed = False
        else:
            checks.append(
                DataQualityCheckResult(
                    check_name="numerical_finite_check",
                    passed=True,
                    details={},
                    message="All numerical values are finite.",
                )
            )

        # Check 6: Categorical domain validity
        category_errors = []
        if "event_outcome" in df.columns and row_count > 0:
            distinct_outcomes = set(df["event_outcome"].unique().to_list())
            unknown_outcomes = distinct_outcomes - self.VALID_EVENT_OUTCOMES
            if unknown_outcomes:
                category_errors.append(f"Unknown event outcomes: {unknown_outcomes}")

        if category_errors:
            checks.append(
                DataQualityCheckResult(
                    check_name="categorical_domains_check",
                    passed=False,
                    details={"category_errors": category_errors},
                    message=f"Categorical domain violations: {category_errors}",
                )
            )
            overall_passed = False
        else:
            checks.append(
                DataQualityCheckResult(
                    check_name="categorical_domains_check",
                    passed=True,
                    details={},
                    message="Categorical values conform to domain schema.",
                )
            )

        # Anomaly label summary
        anomaly_count = int(df["is_anomaly"].sum()) if "is_anomaly" in df.columns else 0
        anomaly_ratio = float(anomaly_count / max(1, row_count))
        anomaly_breakdown = {}
        if "anomaly_type" in df.columns:
            counts = df.group_by("anomaly_type").len().to_dicts()
            anomaly_breakdown = {d["anomaly_type"]: int(d["len"]) for d in counts}

        data_types = {col: str(dtype) for col, dtype in df.schema.items()}

        report = DataQualityReport(
            dataset_version=dataset_version,
            total_records=row_count,
            checked_at=datetime.now(UTC),
            overall_status="PASSED" if overall_passed else "FAILED",
            null_counts=null_counts,
            null_percentages=null_percentages,
            data_types=data_types,
            checks=checks,
            anomaly_count=anomaly_count,
            anomaly_ratio=anomaly_ratio,
            anomaly_breakdown=anomaly_breakdown,
        )

        if not overall_passed:
            failed_msgs = [c.message for c in checks if not c.passed]
            raise DataValidationError(f"Data quality validation failed: {'; '.join(failed_msgs)}")

        return report


def main():
    parser = argparse.ArgumentParser(description="Validate data quality for AnomalyLab dataset.")
    parser.add_argument("--input", type=str, required=True, help="Path to input Parquet file")
    parser.add_argument(
        "--output-report", type=str, default=None, help="Path to write JSON quality report"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset not found at {input_path}")

    df = pl.read_parquet(input_path)
    validator = DataQualityValidator()

    try:
        report = validator.validate(df, dataset_version=input_path.stem)
        print(f"Data Quality Status: {report.overall_status}")
        print(f"Total Records: {report.total_records}")
        print(f"Anomalies: {report.anomaly_count} ({report.anomaly_ratio:.2%})")
        print(f"Checks Passed: {sum(1 for c in report.checks if c.passed)}/{len(report.checks)}")

        if args.output_report:
            out_p = Path(args.output_report)
            out_p.parent.mkdir(parents=True, exist_ok=True)
            with open(out_p, "w") as f:
                json.dump(report.model_dump(mode="json"), f, indent=2)
            print(f"Report saved to {out_p}")
    except DataValidationError as e:
        print(f"VALIDATION FAILED: {e}")
        exit(1)


if __name__ == "__main__":
    main()
