"""Model registry and lifecycle management endpoints."""

from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from anomalylab.inference.registry import ModelRegistry
from anomalylab.models.manifest import ModelStage

router = APIRouter(prefix="/api/v1/models", tags=["Model Registry"])


class PromoteRequest(BaseModel):
    stage: ModelStage


def get_engine():
    from apps.api.main import inference_engine

    return inference_engine


@router.get("", response_model=list[dict[str, Any]])
def list_models():
    """List all registered models with their metadata, metrics, and deployment stage."""
    registry = ModelRegistry()
    manifests = registry.list_manifests()
    if not manifests:
        engine = get_engine()
        if engine and engine.current_manifest:
            manifests = [engine.current_manifest]
    return [m.model_dump(mode="json") for m in manifests]


@router.get("/current", response_model=dict[str, Any])
def get_current_production_model():
    """Retrieve metadata and metrics for the active champion production model."""
    engine = get_engine()
    if not engine or not engine.current_manifest:
        registry = ModelRegistry()
        m = registry.get_production_model()
        if not m:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "MODEL_NOT_FOUND", "message": "No model found"},
            )
        return m.model_dump(mode="json")
    return engine.current_manifest.model_dump(mode="json")


@router.get("/{model_version}", response_model=dict[str, Any])
def get_model_details(model_version: str):
    """Retrieve full manifest for a specific model version."""
    registry = ModelRegistry()
    manifest = registry.get_manifest(model_version)
    if not manifest:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "MODEL_NOT_FOUND", "message": f"Model '{model_version}' not found."},
        )
    return manifest.model_dump(mode="json")


@router.post("/{model_version}/promote", response_model=dict[str, Any])
def promote_model(model_version: str, payload: PromoteRequest):
    """Promote model to candidate, staging, production, or archived. Hot-reloads API if promoted to production."""
    registry = ModelRegistry()
    try:
        updated_manifest = registry.promote_model(model_version, payload.stage)

        # Hot-reload if promoted to production
        if payload.stage == ModelStage.PRODUCTION:
            engine = get_engine()
            if engine:
                engine.load_champion_model()
                print(f"Hot-reloaded engine with promoted model: {model_version}")

        return updated_manifest.model_dump(mode="json")
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "MODEL_NOT_FOUND", "message": f"Model '{model_version}' not found."},
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "PROMOTION_FAILED", "message": str(e)},
        )
