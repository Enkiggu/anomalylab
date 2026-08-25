"""Multi-seed stability benchmark evaluator for AnomalyLab."""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from anomalylab.evaluation.metrics import calculate_metrics
from anomalylab.evaluation.threshold import optimize_threshold
from anomalylab.features.contract import get_security_feature_contract_v1
from anomalylab.models.isolation_forest import IsolationForestModel


def run_multiseed_study(
    seeds: list[int] = [42, 123, 777, 2026, 9001],
    train_path: str = "data/features/train.parquet",
    val_path: str = "data/features/val.parquet",
    test_path: str = "data/features/test.parquet",
    output_path: str = "experiments/multiseed_stability.json",
) -> dict:
    """Evaluate Isolation Forest performance and threshold stability across multiple deterministic random seeds."""
    print("\n==========================================")
    print(f"Running Multi-Seed Stability Benchmark across {len(seeds)} seeds: {seeds}")
    print("==========================================")

    train_df = pl.read_parquet(train_path)
    val_df = pl.read_parquet(val_path)
    test_df = pl.read_parquet(test_path)

    contract = get_security_feature_contract_v1()
    feature_names = contract.feature_names

    X_train = train_df.select(feature_names).to_numpy()
    X_val = val_df.select(feature_names).to_numpy()
    y_val = val_df["is_anomaly"].to_numpy()
    X_test = test_df.select(feature_names).to_numpy()
    y_test = test_df["is_anomaly"].to_numpy()
    test_anomaly_types = (
        test_df["anomaly_type"].to_numpy() if "anomaly_type" in test_df.columns else None
    )

    seed_results = []
    pr_aucs = []
    recalls = []
    precisions = []
    f1s = []
    fprs = []
    thresholds = []

    for s in seeds:
        model = IsolationForestModel(
            model_id=f"iforest_seed_{s}",
            n_estimators=250,
            max_samples=0.8,
            contamination="auto",
            random_state=s,
        )
        model.fit(X_train, feature_names=feature_names)

        # Tune threshold on validation only
        val_scores = model.score_samples(X_val)
        thresh_rep = optimize_threshold(
            y_true=y_val, y_scores=val_scores, strategy="target_recall", target_recall=0.90
        )
        best_t = thresh_rep.best_threshold
        model.threshold = best_t

        # Evaluate once on test
        test_scores = model.score_samples(X_test)
        metrics = calculate_metrics(
            y_true=y_test,
            y_scores=test_scores,
            threshold=best_t,
            anomaly_types=test_anomaly_types,
            compute_bootstrap=False,
        )

        res = {
            "seed": s,
            "threshold": round(best_t, 4),
            "val_recall": round(thresh_rep.recall_at_threshold, 4),
            "test_pr_auc": round(metrics.pr_auc, 4),
            "test_recall": round(metrics.recall, 4),
            "test_precision": round(metrics.precision, 4),
            "test_f1": round(metrics.f1, 4),
            "test_fpr": round(metrics.false_positive_rate, 6),
            "test_fp_per_10k": round(metrics.fp_per_10k, 2),
        }
        seed_results.append(res)
        pr_aucs.append(metrics.pr_auc)
        recalls.append(metrics.recall)
        precisions.append(metrics.precision)
        f1s.append(metrics.f1)
        fprs.append(metrics.false_positive_rate)
        thresholds.append(best_t)

        print(
            f"Seed {s:5d} -> PR-AUC: {metrics.pr_auc:.4f} | Recall: {metrics.recall:.2%} | Precision: {metrics.precision:.2%} | F1: {metrics.f1:.4f} | FPR: {metrics.false_positive_rate:.4%} | Thresh: {best_t:.4f}"
        )

    summary = {
        "seeds": seeds,
        "seed_runs": seed_results,
        "aggregates": {
            "pr_auc": {
                "mean": round(float(np.mean(pr_aucs)), 4),
                "std": round(float(np.std(pr_aucs)), 4),
                "min": round(float(np.min(pr_aucs)), 4),
                "max": round(float(np.max(pr_aucs)), 4),
            },
            "recall": {
                "mean": round(float(np.mean(recalls)), 4),
                "std": round(float(np.std(recalls)), 4),
                "min": round(float(np.min(recalls)), 4),
                "max": round(float(np.max(recalls)), 4),
            },
            "precision": {
                "mean": round(float(np.mean(precisions)), 4),
                "std": round(float(np.std(precisions)), 4),
                "min": round(float(np.min(precisions)), 4),
                "max": round(float(np.max(precisions)), 4),
            },
            "f1": {
                "mean": round(float(np.mean(f1s)), 4),
                "std": round(float(np.std(f1s)), 4),
                "min": round(float(np.min(f1s)), 4),
                "max": round(float(np.max(f1s)), 4),
            },
            "false_positive_rate": {
                "mean": round(float(np.mean(fprs)), 6),
                "std": round(float(np.std(fprs)), 6),
                "min": round(float(np.min(fprs)), 6),
                "max": round(float(np.max(fprs)), 6),
            },
            "threshold": {
                "mean": round(float(np.mean(thresholds)), 4),
                "std": round(float(np.std(thresholds)), 4),
                "min": round(float(np.min(thresholds)), 4),
                "max": round(float(np.max(thresholds)), 4),
            },
        },
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n--- Multi-Seed Summary (Mean ± Std) ---")
    print(
        f"PR-AUC:    {summary['aggregates']['pr_auc']['mean']:.4f} ± {summary['aggregates']['pr_auc']['std']:.4f} [min: {summary['aggregates']['pr_auc']['min']:.4f}, max: {summary['aggregates']['pr_auc']['max']:.4f}]"
    )
    print(
        f"Recall:    {summary['aggregates']['recall']['mean']:.2%} ± {summary['aggregates']['recall']['std']:.2%} [min: {summary['aggregates']['recall']['min']:.2%}, max: {summary['aggregates']['recall']['max']:.2%}]"
    )
    print(
        f"Precision: {summary['aggregates']['precision']['mean']:.2%} ± {summary['aggregates']['precision']['std']:.2%} [min: {summary['aggregates']['precision']['min']:.2%}, max: {summary['aggregates']['precision']['max']:.2%}]"
    )
    print(
        f"F1:        {summary['aggregates']['f1']['mean']:.4f} ± {summary['aggregates']['f1']['std']:.4f}"
    )
    print(f"Report saved to {out_file}\n")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Run multi-seed stability study for AnomalyLab.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 777, 2026, 9001])
    parser.add_argument("--output", type=str, default="experiments/multiseed_stability.json")
    args = parser.parse_args()

    run_multiseed_study(seeds=args.seeds, output_path=args.output)


if __name__ == "__main__":
    main()
