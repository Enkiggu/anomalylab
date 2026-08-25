"""Model Promotion Policy and Validation Gate Evaluator for AnomalyLab."""

from dataclasses import dataclass

from anomalylab.models.manifest import ModelManifest, ModelStage


class PromotionPolicyViolation(Exception):
    """Raised when a model fails validation policy constraints during promotion."""

    pass


@dataclass
class ModelPromotionPolicy:
    """
    Defines quantitative validation performance gates required for promotion to Champion/Candidate.

    Evaluated strictly against VALIDATION metrics (never held-out test data).
    """

    min_validation_recall: float = 0.85
    max_validation_fpr: float = 0.05
    min_validation_precision: float = 0.40
    max_inference_latency_p95_ms: float = 50.0

    def evaluate(self, manifest: ModelManifest) -> tuple[bool, list[str]]:
        """
        Evaluate model manifest against validation criteria.
        Returns (passed, list_of_violations).
        """
        violations = []
        metrics = manifest.validation_metrics or manifest.metrics or {}

        val_recall = metrics.get("recall", 0.0)
        val_fpr = metrics.get("false_positive_rate", 1.0)
        val_precision = metrics.get("precision", 0.0)

        if val_recall < self.min_validation_recall:
            violations.append(
                f"Validation Recall ({val_recall:.2%}) below required minimum ({self.min_validation_recall:.2%})"
            )

        if val_fpr > self.max_validation_fpr:
            violations.append(
                f"Validation FPR ({val_fpr:.2%}) exceeds maximum permitted ({self.max_validation_fpr:.2%})"
            )

        if val_precision < self.min_validation_precision:
            violations.append(
                f"Validation Precision ({val_precision:.2%}) below required minimum ({self.min_validation_precision:.2%})"
            )

        passed = len(violations) == 0
        return passed, violations

    def enforce_promotion(self, manifest: ModelManifest, target_stage: ModelStage) -> None:
        """Enforce validation gates before allowing promotion to CANDIDATE or PRODUCTION."""
        if target_stage in [ModelStage.PRODUCTION, ModelStage.CANDIDATE]:
            passed, violations = self.evaluate(manifest)
            if not passed:
                raise PromotionPolicyViolation(
                    f"Model '{manifest.model_version}' failed promotion gate to '{target_stage.value}':\n- "
                    + "\n- ".join(violations)
                )
