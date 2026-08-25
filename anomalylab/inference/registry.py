"""Model Registry managing model lifecycle, stage promotions, and active champion resolution."""

from datetime import UTC, datetime
from pathlib import Path

from anomalylab.models.manifest import ModelManifest, ModelStage
from anomalylab.models.promotion import ModelPromotionPolicy


class ModelRegistry:
    """
    Manages model artifacts, manifests, and stage lifecycle:
    (EXPERIMENTAL -> CANDIDATE -> PRODUCTION, REJECTED, ARCHIVED)
    """

    def __init__(
        self,
        models_dir: str | Path = "models",
        promotion_policy: ModelPromotionPolicy | None = None,
    ):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.policy = promotion_policy or ModelPromotionPolicy()

    def list_manifests(self) -> list[ModelManifest]:
        """List all model manifests in registry sorted by creation date."""
        manifests = []
        for path in self.models_dir.glob("*_manifest.json"):
            try:
                manifests.append(ModelManifest.load(path))
            except Exception as e:
                print(f"Warning: Failed to load manifest {path}: {e}")
        return sorted(manifests, key=lambda m: m.created_at, reverse=True)

    def get_manifest(self, model_version: str) -> ModelManifest | None:
        """Get manifest by exact model version or ID."""
        for path in self.models_dir.glob("*_manifest.json"):
            try:
                m = ModelManifest.load(path)
                if m.model_version == model_version or m.model_id == model_version:
                    return m
            except Exception:
                continue
        return None

    def get_production_model(self) -> ModelManifest | None:
        """
        Resolve the active PRODUCTION champion model.
        Returns the newest model explicitly in PRODUCTION stage.
        """
        manifests = self.list_manifests()
        for m in manifests:
            if m.stage == ModelStage.PRODUCTION:
                return m

        # Fallback to best performing non-rejected candidate
        candidates = [m for m in manifests if m.stage == ModelStage.CANDIDATE]
        if candidates:
            best = max(
                candidates, key=lambda m: m.validation_metrics.get("f1", m.metrics.get("f1", 0.0))
            )
            return best
        return None

    def promote_model(
        self,
        model_version: str,
        target_stage: ModelStage,
        bypass_policy: bool = False,
    ) -> ModelManifest:
        """
        Promote or transition model to a target stage.
        Enforces quantitative validation gates before allowing CANDIDATE or PRODUCTION.
        """
        manifest = self.get_manifest(model_version)
        if not manifest:
            raise FileNotFoundError(f"Model manifest '{model_version}' not found in registry")

        # Validation gate check
        if not bypass_policy:
            self.policy.enforce_promotion(manifest, target_stage)

        # If promoting to production, archive any previous production model
        if target_stage == ModelStage.PRODUCTION:
            for other_path in self.models_dir.glob("*_manifest.json"):
                try:
                    other = ModelManifest.load(other_path)
                    if (
                        other.model_version != manifest.model_version
                        and other.stage == ModelStage.PRODUCTION
                    ):
                        other.stage = ModelStage.ARCHIVED
                        other.save(other_path)
                except Exception:
                    pass

        manifest.stage = target_stage
        manifest.promoted_at = datetime.now(UTC)

        manifest_path = self.models_dir / f"{manifest.model_version}_manifest.json"
        manifest.save(manifest_path)
        print(f"Model {manifest.model_version} promoted to stage: {target_stage.value}")
        return manifest
