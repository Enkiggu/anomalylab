"""Statistical data and prediction drift detection engine for AnomalyLab."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp

from anomalylab.features.contract import get_security_feature_contract_v1
from anomalylab.models.base import BaseAnomalyModel


def calculate_psi(expected: np.ndarray, actual: np.ndarray, num_bins: int = 10) -> float:
    """
    Calculate Population Stability Index (PSI) between reference and current distribution.
    PSI = sum((actual% - expected%) * ln(actual% / expected%))
    """
    if len(expected) == 0 or len(actual) == 0:
        return 0.0

    # Determine quantile bins from expected distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(expected, percentiles)
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Handle duplicate bin edges
    unique_edges = np.unique(bin_edges)
    if len(unique_edges) < 3:
        unique_edges = np.linspace(
            min(expected.min(), actual.min()), max(expected.max(), actual.max()), num_bins + 1
        )
        unique_edges[0] = -np.inf
        unique_edges[-1] = np.inf

    expected_counts, _ = np.histogram(expected, bins=unique_edges)
    actual_counts, _ = np.histogram(actual, bins=unique_edges)

    # Convert to fractions with small epsilon to prevent log(0)
    eps = 1e-4
    expected_pct = (expected_counts + eps) / (len(expected) + eps * len(expected_counts))
    actual_pct = (actual_counts + eps) / (len(actual) + eps * len(actual_counts))

    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(max(0.0, psi_val))


def calculate_feature_drift(
    ref_series: np.ndarray,
    curr_series: np.ndarray,
    feature_name: str,
    psi_warning: float = 0.10,
    psi_critical: float = 0.25,
) -> dict[str, Any]:
    """Compute KS-test, PSI, and JS-distance for a single numerical feature."""
    ks_res = ks_2samp(ref_series, curr_series)
    ks_stat = float(ks_res.statistic)
    ks_pvalue = float(ks_res.pvalue)

    psi_val = calculate_psi(ref_series, curr_series, num_bins=10)

    # JS Distance on 20-bin histogram
    min_v = min(ref_series.min(), curr_series.min())
    max_v = max(ref_series.max(), curr_series.max())
    bins = np.linspace(min_v, max_v, 20)
    p_hist, _ = np.histogram(ref_series, bins=bins, density=True)
    q_hist, _ = np.histogram(curr_series, bins=bins, density=True)
    js_dist = float(jensenshannon(p_hist + 1e-6, q_hist + 1e-6))

    if psi_val >= psi_critical:
        status = "DRIFT_DETECTED"
    elif psi_val >= psi_warning or ks_pvalue < 0.01:
        status = "WARNING"
    else:
        status = "NO_DRIFT"

    return {
        "feature": feature_name,
        "status": status,
        "psi": round(psi_val, 4),
        "ks_statistic": round(ks_stat, 4),
        "ks_pvalue": round(ks_pvalue, 6),
        "jensen_shannon_distance": round(js_dist, 4),
        "reference_mean": round(float(np.mean(ref_series)), 4),
        "current_mean": round(float(np.mean(curr_series)), 4),
        "reference_std": round(float(np.std(ref_series)), 4),
        "current_std": round(float(np.std(curr_series)), 4),
    }


class DriftDetector:
    """End-to-end dataset and prediction score drift monitor."""

    def __init__(
        self,
        psi_warning: float = 0.10,
        psi_critical: float = 0.25,
    ):
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.contract = get_security_feature_contract_v1()

    def run_drift_analysis(
        self,
        reference_df: pl.DataFrame,
        current_df: pl.DataFrame,
        model: BaseAnomalyModel | None = None,
        report_id: str = "drift_run",
    ) -> dict[str, Any]:
        """Perform feature distribution drift checks and anomaly score distribution drift checks."""
        feature_names = self.contract.feature_names
        feature_reports = []

        drifting_features_count = 0
        warning_features_count = 0

        for feat in feature_names:
            if feat in reference_df.columns and feat in current_df.columns:
                ref_vals = reference_df[feat].to_numpy()
                curr_vals = current_df[feat].to_numpy()

                report = calculate_feature_drift(
                    ref_vals,
                    curr_vals,
                    feature_name=feat,
                    psi_warning=self.psi_warning,
                    psi_critical=self.psi_critical,
                )
                feature_reports.append(report)

                if report["status"] == "DRIFT_DETECTED":
                    drifting_features_count += 1
                elif report["status"] == "WARNING":
                    warning_features_count += 1

        # Score distribution drift if model provided
        score_drift_info = None
        if model is not None and model.is_fitted:
            ref_X = reference_df.select(feature_names).to_numpy()
            curr_X = current_df.select(feature_names).to_numpy()

            ref_scores = model.score_samples(ref_X)
            curr_scores = model.score_samples(curr_X)

            score_psi = calculate_psi(ref_scores, curr_scores, num_bins=10)
            score_ks = ks_2samp(ref_scores, curr_scores)

            score_status = (
                "DRIFT_DETECTED"
                if score_psi >= self.psi_critical
                else "WARNING"
                if score_psi >= self.psi_warning
                else "NO_DRIFT"
            )

            score_drift_info = {
                "status": score_status,
                "psi": round(score_psi, 4),
                "ks_statistic": round(float(score_ks.statistic), 4),
                "ks_pvalue": round(float(score_ks.pvalue), 6),
                "reference_mean_score": round(float(np.mean(ref_scores)), 4),
                "current_mean_score": round(float(np.mean(curr_scores)), 4),
            }

        # Overall status
        if drifting_features_count >= 2 or (
            score_drift_info and score_drift_info["status"] == "DRIFT_DETECTED"
        ):
            overall_status = "DRIFT_DETECTED"
        elif drifting_features_count >= 1 or warning_features_count >= 2:
            overall_status = "WARNING"
        else:
            overall_status = "NO_DRIFT"

        result = {
            "report_id": report_id,
            "timestamp": datetime.now(UTC).isoformat(),
            "overall_status": overall_status,
            "reference_records": len(reference_df),
            "current_records": len(current_df),
            "features_evaluated": len(feature_reports),
            "drifting_features_count": drifting_features_count,
            "warning_features_count": warning_features_count,
            "features": feature_reports,
            "score_drift": score_drift_info,
        }

        return result


def main():
    parser = argparse.ArgumentParser(description="Run statistical drift detection for AnomalyLab.")
    parser.add_argument(
        "--reference",
        type=str,
        default="data/features/train.parquet",
        help="Reference Parquet path",
    )
    parser.add_argument(
        "--current", type=str, default="data/features/test.parquet", help="Current Parquet path"
    )
    parser.add_argument(
        "--model-path", type=str, default=None, help="Optional trained model path for score drift"
    )
    parser.add_argument(
        "--output-dir", type=str, default="monitoring/drift", help="Output directory"
    )
    args = parser.parse_args()

    ref_df = pl.read_parquet(args.reference)
    curr_df = pl.read_parquet(args.current)

    model = None
    if args.model_path and Path(args.model_path).exists():
        model = BaseAnomalyModel.load(args.model_path)

    detector = DriftDetector()
    report = detector.run_drift_analysis(ref_df, curr_df, model=model)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_file = out_dir / "latest_drift_report.json"

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    print("\n--- Drift Analysis Result ---")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Features Evaluated: {report['features_evaluated']}")
    print(f"Drifting Features: {report['drifting_features_count']}")
    print(f"Warning Features:  {report['warning_features_count']}")
    if report.get("score_drift"):
        print(
            f"Score Drift PSI:   {report['score_drift']['psi']} ({report['score_drift']['status']})"
        )
    print(f"Report saved to {report_file}")


if __name__ == "__main__":
    main()
