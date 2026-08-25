"""Experiment reproduction module."""

import argparse
from pathlib import Path

import yaml

from anomalylab.data.fingerprint import compute_file_sha256
from anomalylab.models.manifest import ModelManifest
from anomalylab.training.train import train_and_evaluate


def reproduce_from_manifest(manifest_path: str | Path) -> bool:
    """
    Given a model manifest, verify data hashes, reload parameters, and verify deterministic reproducibility.
    """
    manifest = ModelManifest.load(manifest_path)
    print(
        f"Reproducing experiment for model: {manifest.model_id} (Version: {manifest.model_version})"
    )

    # Check data hash
    train_path = Path("data/features/train.parquet")
    if train_path.exists():
        current_sha256 = compute_file_sha256(train_path)
        if current_sha256 == manifest.dataset_sha256:
            print(f"Dataset SHA-256 matches: {current_sha256[:12]}...")
        else:
            print(
                f"Warning: Dataset SHA-256 differs (Original: {manifest.dataset_sha256[:12]}..., Current: {current_sha256[:12]}...)"
            )

    temp_cfg = {
        "experiment": {
            "name": f"reproduced_{manifest.model_id}",
            "description": f"Reproduced run of {manifest.model_version}",
            "seed": manifest.parameters.get("random_state", 42),
        },
        "dataset": {
            "version": manifest.dataset_version,
            "path": "data/features/train.parquet",
            "test_path": "data/features/test.parquet",
        },
        "features": {
            "version": manifest.feature_version,
        },
        "model": {
            "type": manifest.algorithm,
            **manifest.parameters,
        },
        "evaluation": {
            "target_recall": 0.90,
            "cost_fp": 1.0,
            "cost_fn": 20.0,
        },
    }

    temp_cfg_path = Path("experiments") / f"reproduce_{manifest.model_version}.yaml"
    temp_cfg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(temp_cfg_path, "w") as f:
        yaml.dump(temp_cfg, f)

    new_manifest, metrics = train_and_evaluate(temp_cfg_path)

    print("\n--- Reproduction Verification ---")
    orig_pr_auc = manifest.metrics.get("pr_auc", 0.0)
    new_pr_auc = metrics.get("pr_auc", 0.0)
    diff = abs(orig_pr_auc - new_pr_auc)
    print(
        f"Original PR-AUC: {orig_pr_auc:.4f} | Reproduced PR-AUC: {new_pr_auc:.4f} (Delta: {diff:.4f})"
    )

    return diff < 0.01


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce AnomalyLab experiment from model manifest."
    )
    parser.add_argument("--manifest", type=str, required=True, help="Path to model manifest JSON")
    args = parser.parse_args()

    success = reproduce_from_manifest(args.manifest)
    if success:
        print("REPRODUCTION VERIFIED: Metrics match original run within tolerance.")
    else:
        print("REPRODUCTION MISMATCH: Discrepancy observed.")


if __name__ == "__main__":
    main()
