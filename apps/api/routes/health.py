"""Health and readiness check endpoints."""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel

from anomalylab.inference.engine import ProductionInferenceEngine

router = APIRouter(tags=["Health"])


class HealthStatus(BaseModel):
    status: str
    service: str = "anomalylab-api"
    modelLoaded: bool
    activeModelVersion: str | None = None
    algorithm: str | None = None


def get_engine() -> ProductionInferenceEngine:
    from apps.api.main import inference_engine

    return inference_engine


@router.get("/health/live", response_model=dict)
def liveness():
    """Liveness probe to confirm the server process is responsive."""
    return {"status": "LIVE", "service": "anomalylab-api"}


@router.get("/health/ready", response_model=HealthStatus)
def readiness():
    """Readiness probe verifying model and metadata access are ready for traffic."""
    engine = get_engine()
    if not engine or not engine.is_ready or engine.current_manifest is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={
                "code": "MODEL_NOT_READY",
                "message": "Inference engine is not ready or model is not loaded.",
            },
        )

    return HealthStatus(
        status="READY",
        modelLoaded=True,
        activeModelVersion=engine.current_manifest.model_version,
        algorithm=engine.current_manifest.algorithm,
    )
