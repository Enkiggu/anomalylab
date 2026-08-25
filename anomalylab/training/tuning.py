"""Optuna hyperparameter tuning integration for AnomalyLab models."""

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
import yaml

from anomalylab.evaluation.metrics import calculate_metrics
from anomalylab.evaluation.threshold import optimize_threshold
from anomalylab.features.contract import get_security_feature_contract_v1
from anomalylab.models.isolation_forest import IsolationForestModel


def objective(
    trial: optuna.Trial,
    X_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> float:
    # Hyperparameter search space
    n_estimators = trial.suggest_int("n_estimators", 100, 350, step=50)
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0, step=0.1)
    max_features = trial.suggest_float("max_features", 0.6, 1.0, step=0.1)

    model = IsolationForestModel(
        n_estimators=n_estimators,
        max_samples=max_samples,
        max_features=max_features,
        random_state=42,
    )
    model.fit(X_train, feature_names=feature_names)

    val_scores = model.score_samples(X_val)
    report = optimize_threshold(
        y_true=y_val, y_scores=val_scores, strategy="target_recall", target_recall=0.90
    )

    # Maximize PR-AUC while penalizing high FPR
    res = calculate_metrics(y_true=y_val, y_scores=val_scores, threshold=report.best_threshold)
    score = res.pr_auc - (2.0 * res.false_positive_rate)
    return float(score)


def run_tuning(
    n_trials: int = 15,
    train_path: str = "data/features/train.parquet",
    val_path: str = "data/features/val.parquet",
    output_config: str = "configs/isolation_forest_tuned.yaml",
) -> dict[str, Any]:
    """Run Optuna study and save optimal hyperparameters."""
    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)

    contract = get_security_feature_contract_v1()
    feature_names = contract.feature_names

    X_train = train_df.select(feature_names).to_numpy()
    X_val = val_df.select(feature_names).to_numpy()
    y_val = val_df["is_anomaly"].to_numpy()

    study = optuna.create_study(direction="maximize", study_name="iforest_tuning")
    study.optimize(lambda t: objective(t, X_train, X_val, y_val, feature_names), n_trials=n_trials)

    print("\n--- Best Optuna Trial ---")
    print(f"Trial Number: {study.best_trial.number}")
    print(f"Best Objective Score: {study.best_value:.4f}")
    print(f"Best Parameters: {study.best_params}")

    tuned_config = {
        "experiment": {
            "name": "isolation_forest_tuned_security_v1",
            "description": "Optuna hyperparameter tuned Isolation Forest",
            "seed": 42,
        },
        "dataset": {
            "version": "security-2026-v1",
            "path": train_path,
            "test_path": "data/features/test.parquet",
        },
        "features": {
            "version": "v1",
        },
        "model": {
            "type": "isolation_forest",
            "n_estimators": study.best_params["n_estimators"],
            "max_samples": study.best_params["max_samples"],
            "max_features": study.best_params["max_features"],
            "contamination": "auto",
            "random_state": 42,
        },
        "evaluation": {
            "target_recall": 0.90,
            "max_fpr": 0.015,
            "cost_fp": 1.0,
            "cost_fn": 20.0,
        },
    }

    out_p = Path(output_config)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w") as f:
        yaml.dump(tuned_config, f, sort_keys=False)

    print(f"Tuned configuration saved to {out_p}")
    return study.best_params


def main():
    parser = argparse.ArgumentParser(description="Run Optuna hyperparameter optimization.")
    parser.add_argument("--trials", type=int, default=10, help="Number of tuning trials")
    args = parser.parse_args()

    run_tuning(n_trials=args.trials)


if __name__ == "__main__":
    main()
