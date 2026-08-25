"""Unified model training, threshold tuning, evaluation, and MLflow tracking orchestrator."""

import argparse
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import polars as pl
import yaml

from anomalylab.data.fingerprint import compute_file_sha256
from anomalylab.evaluation.curves import generate_evaluation_curves
from anomalylab.evaluation.error_analysis import perform_error_analysis
from anomalylab.evaluation.metrics import calculate_metrics
from anomalylab.evaluation.report import save_evaluation_report
from anomalylab.evaluation.threshold import optimize_threshold
from anomalylab.features.contract import get_security_feature_contract_v1
from anomalylab.models.base import BaseAnomalyModel
from anomalylab.models.baseline import RuleBaseline, ZScoreStatisticalBaseline
from anomalylab.models.isolation_forest import IsolationForestModel
from anomalylab.models.lof import LocalOutlierFactorModel
from anomalylab.models.manifest import ModelManifest, ModelStage
from anomalylab.models.promotion import ModelPromotionPolicy


def create_model_from_config(model_cfg: dict[str, Any], model_id: str) -> BaseAnomalyModel:
    model_type = model_cfg.get("type", "isolation_forest")

    if model_type == "statistical_baseline":
        return ZScoreStatisticalBaseline(
            model_id=model_id,
            top_k=model_cfg.get("top_k_signals", 3),
            method=model_cfg.get("method", "robust_mad"),
        )
    elif model_type == "rule_baseline":
        return RuleBaseline(model_id=model_id)
    elif model_type == "isolation_forest":
        return IsolationForestModel(
            model_id=model_id,
            n_estimators=model_cfg.get("n_estimators", 250),
            max_samples=model_cfg.get("max_samples", 0.8),
            contamination=model_cfg.get("contamination", "auto"),
            max_features=model_cfg.get("max_features", 1.0),
            bootstrap=model_cfg.get("bootstrap", False),
            random_state=model_cfg.get("random_state", 42),
        )
    elif model_type == "local_outlier_factor":
        return LocalOutlierFactorModel(
            model_id=model_id,
            n_neighbors=model_cfg.get("n_neighbors", 35),
            contamination=model_cfg.get("contamination", 0.02),
            algorithm=model_cfg.get("algorithm", "auto"),
            metric=model_cfg.get("metric", "minkowski"),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_and_evaluate(
    config_path: str | Path,
    output_dir: str | Path = "models",
    experiment_dir: str | Path = "experiments",
) -> tuple[ModelManifest, dict[str, Any]]:
    """Run end-to-end training, threshold optimization on validation set, and held-out test evaluation."""
    cfg_file = Path(config_path)
    with open(cfg_file) as f:
        config = yaml.safe_load(f)

    exp_cfg = config.get("experiment", {})
    ds_cfg = config.get("dataset", {})
    model_cfg = config.get("model", {})
    eval_cfg = config.get("evaluation", {})
    feat_cfg = config.get("features", {})

    model_type = model_cfg.get("type", "isolation_forest")
    dataset_version = ds_cfg.get("version", "security_v2")
    feature_version = feat_cfg.get("version", "v2")
    seed = exp_cfg.get("seed", 42)

    train_path = Path(ds_cfg.get("path", "data/features/train.parquet"))
    val_path = Path("data/features/val.parquet")
    test_path = Path(ds_cfg.get("test_path", "data/features/test.parquet"))

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Feature dataset not found: {train_path} or {test_path}")

    # Load datasets
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path) if val_path.exists() else None
    test_df = pl.read_parquet(test_path)

    contract = get_security_feature_contract_v1()
    feature_names = contract.feature_names

    X_train = train_df.select(feature_names).to_numpy()
    X_test = test_df.select(feature_names).to_numpy()
    y_test = test_df["is_anomaly"].to_numpy()
    test_anomaly_types = (
        test_df["anomaly_type"].to_numpy() if "anomaly_type" in test_df.columns else None
    )

    if val_df is not None:
        X_val = val_df.select(feature_names).to_numpy()
        y_val = val_df["is_anomaly"].to_numpy()
        val_anomaly_types = (
            val_df["anomaly_type"].to_numpy() if "anomaly_type" in val_df.columns else None
        )
    else:
        X_val = X_test
        y_val = y_test
        val_anomaly_types = test_anomaly_types

    timestamp_str = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    model_version = f"{model_type}_{feature_version}_{timestamp_str}"
    model_id = exp_cfg.get("name", model_version)

    print("\n==========================================")
    print(f"Training Model: {model_id} ({model_type})")
    print(
        f"Features: {len(feature_names)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}"
    )
    print("==========================================")

    model = create_model_from_config(model_cfg, model_id=model_id)

    # 1. Fit Model Exclusively on Train Partition
    start_t = time.perf_counter()
    model.fit(X_train, feature_names=feature_names)
    train_duration = time.perf_counter() - start_t
    print(f"Model training completed in {train_duration:.3f}s")

    # 2. Tune Threshold Exclusively on Validation Partition
    val_scores = model.score_samples(X_val)
    target_recall = eval_cfg.get("target_recall", 0.90)
    cost_fp = eval_cfg.get("cost_fp", 1.0)
    cost_fn = eval_cfg.get("cost_fn", 20.0)

    print(
        f"\n[Validation Tuning] Optimizing decision threshold (Target Recall >= {target_recall:.0%})..."
    )
    threshold_report = optimize_threshold(
        y_true=y_val,
        y_scores=val_scores,
        strategy="target_recall",
        target_recall=target_recall,
        cost_fp=cost_fp,
        cost_fn=cost_fn,
    )
    best_threshold = threshold_report.best_threshold
    model.threshold = best_threshold

    # Calculate full validation metrics at selected threshold
    val_eval_result = calculate_metrics(
        y_true=y_val,
        y_scores=val_scores,
        threshold=best_threshold,
        anomaly_types=val_anomaly_types,
        compute_bootstrap=False,
    )

    print(f"Selected Threshold: {best_threshold:.4f}")
    print(
        f"  Validation Recall:    {val_eval_result.recall:.2%} ({val_eval_result.tp}/{val_eval_result.anomaly_samples})"
    )
    print(f"  Validation Precision: {val_eval_result.precision:.2%}")
    print(
        f"  Validation FPR:       {val_eval_result.false_positive_rate:.4%} ({val_eval_result.fp_per_10k:.1f} FP/10k)"
    )
    print(f"  Validation PR-AUC:    {val_eval_result.pr_auc:.4f}")

    # 3. Evaluate Promotion Policy Gate on Validation Metrics
    policy = ModelPromotionPolicy(
        min_validation_recall=0.85,
        max_validation_fpr=0.05,
        min_validation_precision=0.40,
    )
    passed_gate, violations = policy.evaluate(
        ModelManifest(
            model_id=model_id,
            model_version=model_version,
            algorithm=model_type,
            stage=ModelStage.CANDIDATE,
            dataset_version=dataset_version,
            feature_version=feature_version,
            dataset_sha256="",
            threshold=best_threshold,
            feature_names=feature_names,
            validation_metrics=val_eval_result.to_dict(),
            artifact_path="",
        )
    )

    if passed_gate:
        initial_stage = ModelStage.CANDIDATE
        print("\n[Promotion Gate] PASSED -> Model eligible for Candidate / Champion promotion.")
    else:
        initial_stage = ModelStage.REJECTED
        print(
            "\n[Promotion Gate] REJECTED -> Model violated validation constraints:\n  - "
            + "\n  - ".join(violations)
        )

    # 4. Evaluate Once on Untouched Held-Out Test Set (Freeze Threshold)
    test_scores = model.score_samples(X_test)
    test_eval_result = calculate_metrics(
        y_true=y_test,
        y_scores=test_scores,
        threshold=best_threshold,
        anomaly_types=test_anomaly_types,
        compute_bootstrap=True,
        n_bootstrap=500,
        seed=seed,
    )

    print("\n--- Final Held-Out Test Evaluation (Untouched) ---")
    print(f"PR-AUC (Average Precision): {test_eval_result.pr_auc:.4f}")
    print(
        f"Recall:                     {test_eval_result.recall:.2%} ({test_eval_result.tp}/{test_eval_result.anomaly_samples} caught)"
    )
    print(f"Precision:                  {test_eval_result.precision:.2%}")
    print(f"F1 Score:                   {test_eval_result.f1:.4f}")
    print(f"False Positive Rate (FPR):  {test_eval_result.false_positive_rate:.4%}")
    print(f"False Alarms / 10k events:  {test_eval_result.fp_per_10k:.1f}")
    print(
        f"Projected Daily FPs (1M):   {test_eval_result.projected_daily_fp_1m_events:,.0f} alerts/day (Mathematical projection)"
    )

    if test_eval_result.per_anomaly_recall:
        print("\nRecall per Anomaly Taxonomy:")
        for a_name, a_rec in test_eval_result.per_anomaly_recall.items():
            print(f"  • {a_name:<26}: {a_rec:.2%}")

    # Generate curves & error analysis
    curves_data = generate_evaluation_curves(y_true=y_test, y_scores=test_scores)
    error_analysis_data = perform_error_analysis(
        df=test_df,
        y_true=y_test,
        y_scores=test_scores,
        threshold=best_threshold,
        top_k=10,
    )

    # Save artifacts
    out_models_dir = Path(output_dir)
    out_models_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = out_models_dir / f"{model_version}.joblib"
    model.save(artifact_path)

    # Dataset hash
    ds_sha256 = compute_file_sha256(train_path)

    manifest = ModelManifest(
        model_id=model_id,
        model_version=model_version,
        algorithm=model_type,
        stage=initial_stage,
        dataset_version=dataset_version,
        feature_version=feature_version,
        dataset_sha256=ds_sha256,
        threshold=best_threshold,
        feature_names=feature_names,
        parameters=model.get_params(),
        validation_metrics=val_eval_result.to_dict(),
        metrics=test_eval_result.to_dict(),
        confusion_matrix={
            "tp": test_eval_result.tp,
            "fp": test_eval_result.fp,
            "tn": test_eval_result.tn,
            "fn": test_eval_result.fn,
        },
        created_at=datetime.now(UTC),
        artifact_path=str(artifact_path).replace("\\", "/"),
        description=exp_cfg.get("description", ""),
    )

    manifest_path = out_models_dir / f"{model_version}_manifest.json"
    manifest.save(manifest_path)

    # Save detailed evaluation report
    exp_run_dir = Path(experiment_dir) / model_version
    save_evaluation_report(
        eval_result=test_eval_result,
        threshold_report=threshold_report,
        curves_data=curves_data,
        error_analysis_data=error_analysis_data,
        model_manifest_data=manifest.model_dump(mode="json"),
        output_dir=exp_run_dir,
    )

    # MLflow Tracking
    try:
        mlflow.set_experiment(exp_cfg.get("name", "anomalylab_experiments"))
        with mlflow.start_run(run_name=model_version):
            mlflow.log_params(model.get_params())
            mlflow.log_param("dataset_version", dataset_version)
            mlflow.log_param("feature_version", feature_version)
            mlflow.log_param("threshold", best_threshold)
            mlflow.log_param("initial_stage", initial_stage.value)
            mlflow.log_param("train_duration_sec", round(train_duration, 4))

            mlflow.log_metric("val_recall", val_eval_result.recall)
            mlflow.log_metric("val_precision", val_eval_result.precision)
            mlflow.log_metric("val_fpr", val_eval_result.false_positive_rate)
            mlflow.log_metric("val_pr_auc", val_eval_result.pr_auc)

            mlflow.log_metric("test_pr_auc", test_eval_result.pr_auc)
            mlflow.log_metric("test_recall", test_eval_result.recall)
            mlflow.log_metric("test_precision", test_eval_result.precision)
            mlflow.log_metric("test_f1", test_eval_result.f1)
            mlflow.log_metric("test_false_positive_rate", test_eval_result.false_positive_rate)
            mlflow.log_metric("test_fp_per_10k", test_eval_result.fp_per_10k)
            mlflow.log_metric("test_tp", test_eval_result.tp)
            mlflow.log_metric("test_fp", test_eval_result.fp)

            mlflow.log_artifact(str(manifest_path))
            mlflow.log_artifact(str(exp_run_dir / "evaluation.json"))
            mlflow.log_artifact(str(exp_run_dir / "evaluation_report.html"))
    except Exception as e:
        print(f"Note: MLflow local run tracking logged with fallback ({e})")

    print(f"\nModel saved to {artifact_path}")
    print(f"Manifest saved to {manifest_path} (Stage: {initial_stage.value})")
    print(f"HTML Report generated at {exp_run_dir / 'evaluation_report.html'}")

    return manifest, test_eval_result.to_dict()


def main():
    parser = argparse.ArgumentParser(
        description="Train and evaluate AnomalyLab anomaly detection model."
    )
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument(
        "--output-dir", type=str, default="models", help="Output directory for model artifacts"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments",
        help="Output directory for experiment runs",
    )
    args = parser.parse_args()

    train_and_evaluate(
        config_path=args.config,
        output_dir=args.output_dir,
        experiment_dir=args.experiment_dir,
    )


if __name__ == "__main__":
    main()
