"""Authoritative report and Markdown table generator from empirical experiment artifacts."""

import argparse

from anomalylab.inference.registry import ModelRegistry


def generate_markdown_benchmark_table(models_dir: str = "models") -> str:
    """Generate Markdown comparison table directly from saved ModelManifests."""
    registry = ModelRegistry(models_dir=models_dir)
    manifests = registry.list_manifests()

    if not manifests:
        return "_No model manifests found._"

    lines = [
        "| Algorithm | Dataset | Features | Val Recall | Val Precision | Val FPR | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test FPR | FP / 10k | Threshold | Stage |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]

    for m in manifests:
        val_m = m.validation_metrics or {}
        test_m = m.metrics or {}

        val_rec = f"{val_m.get('recall', 0.0) * 100:.1f}%" if val_m else "N/A"
        val_prec = f"{val_m.get('precision', 0.0) * 100:.1f}%" if val_m else "N/A"
        val_fpr = f"{val_m.get('false_positive_rate', 0.0) * 100:.2f}%" if val_m else "N/A"

        test_pr_auc = f"{test_m.get('pr_auc', 0.0) * 100:.2f}%"
        test_rec = f"{test_m.get('recall', 0.0) * 100:.2f}%"
        test_prec = f"{test_m.get('precision', 0.0) * 100:.2f}%"
        test_f1 = f"{test_m.get('f1', 0.0):.4f}"
        test_fpr = f"{test_m.get('false_positive_rate', 0.0) * 100:.2f}%"
        fp_10k = f"{test_m.get('fp_per_10k', 0.0):.1f}"

        stage_str = (
            f"**{m.stage.value.upper()}**" if m.stage == "production" else m.stage.value.upper()
        )

        lines.append(
            f"| **{m.algorithm}** | `{m.dataset_version}` | `{m.feature_version}` | {val_rec} | {val_prec} | {val_fpr} | {test_pr_auc} | {test_rec} | {test_prec} | {test_f1} | {test_fpr} | {fp_10k} | `{m.threshold:.4f}` | {stage_str} |"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate Markdown benchmark tables from model artifacts."
    )
    parser.add_argument("--models-dir", type=str, default="models")
    args = parser.parse_args()

    table = generate_markdown_benchmark_table(models_dir=args.models_dir)
    print("\n--- Generated Authoritative Benchmark Table ---\n")
    print(table)
    print("\n")


if __name__ == "__main__":
    main()
