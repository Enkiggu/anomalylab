"""Model manifest schemas and metadata management for AnomalyLab."""

import hashlib
import json
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class ModelStage(StrEnum):
    PRODUCTION = "production"
    CANDIDATE = "candidate"
    EXPERIMENTAL = "experimental"
    REJECTED = "rejected"
    ARCHIVED = "archived"


class ModelManifest(BaseModel):
    """Production model manifest tracking complete metadata, lineage, and metrics."""

    model_id: str
    model_version: str
    algorithm: str
    stage: ModelStage = Field(default=ModelStage.CANDIDATE)
    dataset_version: str
    feature_version: str
    dataset_sha256: str
    threshold: float
    feature_names: list[str]
    parameters: dict[str, Any] = Field(default_factory=dict)
    validation_metrics: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, Any] = Field(default_factory=dict)
    confusion_matrix: dict[str, int] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    promoted_at: datetime | None = None
    artifact_path: str
    manifest_sha256: str = ""
    description: str = ""

    def compute_manifest_hash(self) -> str:
        """Compute SHA-256 hash of manifest contents."""
        data = self.model_dump(mode="json", exclude={"manifest_sha256"})
        norm = json.dumps(data, sort_keys=True)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_sha256 = self.compute_manifest_hash()
        with open(target, "w") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "ModelManifest":
        with open(path) as f:
            data = json.load(f)
        return cls.model_validate(data)
