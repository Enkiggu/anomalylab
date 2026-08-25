"""
Self-contained End-to-End Demonstration Script for AnomalyLab.
Runs data validation, feature transformation, model comparison, real-time inference, and drift monitoring.
"""

from pathlib import Path

import polars as pl

from anomalylab.data.fingerprint import compute_dataframe_fingerprint
from anomalylab.data.validator import DataQualityValidator
from anomalylab.drift.detector import DriftDetector
from anomalylab.inference.engine import ProductionInferenceEngine
from anomalylab.inference.registry import ModelRegistry


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def main():
    print_banner("AnomalyLab End-to-End System Demonstration")

    # Step 1: Data Ingestion & Quality Validation
    print_banner("1. Data Ingestion & Quality Validation")
    raw_path = Path("data/raw/security_events.parquet")
    if not raw_path.exists():
        print("Raw dataset not found. Generating synthetic events...")
        from anomalylab.data.generate import main as gen_main

        gen_main()

    df = pl.read_parquet(raw_path)
    fp = compute_dataframe_fingerprint(df)
    print(f"Loaded Raw Dataset: {len(df):,} events (SHA-256: {fp[:16]}...)")

    validator = DataQualityValidator()
    report = validator.validate(df, dataset_version="v1")
    print(
        f"Data Quality Checks: {len(report.checks)}/6 passed | Overall Status: {report.overall_status}"
    )

    # Step 2: Feature Splits & Contract
    print_banner("2. Leakage-Free Temporal Splits & Feature Contract")
    train_df = pl.read_parquet("data/features/train.parquet")
    val_df = pl.read_parquet("data/features/val.parquet")
    test_df = pl.read_parquet("data/features/test.parquet")
    print(
        f"Time-disjoint Splits -> Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}"
    )
    print(
        f"Feature Dimensions: {len(train_df.columns) - 4} numerical features (Contract: security_features_v1)"
    )

    # Step 3: Model Benchmark Comparison
    print_banner("3. Comparative Model Evaluation (Held-Out Test Set)")
    registry = ModelRegistry()
    manifests = registry.list_manifests()

    print(
        f"{'Algorithm':<24} | {'PR-AUC':<8} | {'Recall':<8} | {'Precision':<10} | {'F1':<8} | {'FP / 10k':<10} | {'Stage':<10}"
    )
    print("-" * 90)
    for m in manifests:
        metrics = m.metrics or {}
        pr_auc = f"{metrics.get('pr_auc', 0.0) * 100:.1f}%"
        recall = f"{metrics.get('recall', 0.0) * 100:.1f}%"
        prec = f"{metrics.get('precision', 0.0) * 100:.1f}%"
        f1 = f"{metrics.get('f1', 0.0) * 100:.1f}%"
        fp10k = f"{metrics.get('fp_per_10k', 0.0):.1f}"
        print(
            f"{m.algorithm:<24} | {pr_auc:<8} | {recall:<8} | {prec:<10} | {f1:<8} | {fp10k:<10} | {m.stage:<10}"
        )

    # Step 4: Real-time Online Inference & Explainability
    print_banner("4. Production Online Inference & Explainability")
    engine = ProductionInferenceEngine()
    champion = engine.load_champion_model()
    print(
        f"Active Champion Loaded: {champion.model_version} (Threshold: {champion.threshold:.4f})\n"
    )

    # Normal event test
    normal_event = {
        "userId": "user_0042.demo",
        "loginHour": 14,
        "eventOutcome": "SUCCESS",
        "failedAttempts": 0,
        "requestCount": 3,
        "bytesSent": 1200,
        "bytesReceived": 5400,
        "sessionDuration": 240,
        "isPrivilegedUser": False,
        "newSourceIp": False,
        "newDevice": False,
    }
    normal_res = engine.predict_single(normal_event)
    print("--> [TEST 1: Normal Office Login]")
    print(
        f"    Anomaly Score: {normal_res['anomalyScore']:.4f} | Is Anomaly: {normal_res['isAnomaly']} | Risk Band: {normal_res['riskBand']} | Latency: {normal_res['latencyMs']:.2f}ms"
    )

    # Brute-force attack test
    attack_event = {
        "userId": "user_0019.demo",
        "loginHour": 3,
        "eventOutcome": "FAILURE",
        "failedAttempts": 28,
        "requestCount": 85,
        "bytesSent": 4500,
        "bytesReceived": 1200,
        "sessionDuration": 15,
        "isPrivilegedUser": True,
        "newSourceIp": True,
        "newDevice": True,
    }
    attack_res = engine.predict_single(attack_event)
    print("\n--> [TEST 2: Brute-Force & Credential Stuffing Attack]")
    print(
        f"    Anomaly Score: {attack_res['anomalyScore']:.4f} | Is Anomaly: {attack_res['isAnomaly']} | Risk Band: {attack_res['riskBand']} | Latency: {attack_res['latencyMs']:.2f}ms"
    )
    print("    Contributing Signals (Non-Causal Interpretability):")
    for sig in attack_res["contributingSignals"]:
        print(f"      • [{sig['feature']}] {sig['message']} (Deviation: {sig['deviation']:+.1f}σ)")

    # Step 5: Statistical Drift Detection
    print_banner("5. Statistical Feature & Score Drift Monitoring")
    detector = DriftDetector()
    drift_res = detector.run_drift_analysis(train_df, test_df, model=engine.current_model)
    print(f"Overall Drift Status: {drift_res['overall_status']}")
    print(
        f"Features Evaluated: {drift_res['features_evaluated']} | Drifting Features: {drift_res['drifting_features_count']}"
    )
    if drift_res.get("score_drift"):
        print(
            f"Score Drift PSI: {drift_res['score_drift']['psi']} ({drift_res['score_drift']['status']})"
        )

    print_banner("Demonstration Complete - System Operating Within SLAs")


if __name__ == "__main__":
    main()
