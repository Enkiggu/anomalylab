"""Inference endpoints for single and vectorized batch prediction."""

from fastapi import APIRouter, HTTPException, status
from prometheus_client import Counter, Histogram

from apps.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)

router = APIRouter(prefix="/api/v1", tags=["Inference"])

# Prometheus Metrics
PREDICTIONS_COUNTER = Counter(
    "anomalylab_predictions_total",
    "Total count of anomaly predictions made",
    ["model_version", "risk_band"],
)
ANOMALIES_COUNTER = Counter(
    "anomalylab_anomalies_total",
    "Total count of confirmed anomalies detected",
    ["model_version"],
)
INFERENCE_LATENCY = Histogram(
    "anomalylab_prediction_latency_seconds",
    "Time spent computing model inference in seconds",
    ["model_version"],
    buckets=(0.001, 0.005, 0.010, 0.025, 0.050, 0.100, 0.250, 0.500, 1.0),
)


def get_engine():
    from apps.api.main import inference_engine

    return inference_engine


@router.post("/predict", response_model=PredictResponse)
async def predict_single_event(payload: PredictRequest):
    """Real-time online inference on a single incoming security telemetry event."""
    engine = get_engine()
    if not engine or not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MODEL_NOT_READY", "message": "Production model is not loaded."},
        )

    try:
        event_dict = payload.model_dump()
        result = await engine.predict_single_async(event_dict)

        # Track Prometheus metrics
        m_version = result["modelVersion"]
        risk_band = result["riskBand"]
        PREDICTIONS_COUNTER.labels(model_version=m_version, risk_band=risk_band).inc()
        if result["isAnomaly"]:
            ANOMALIES_COUNTER.labels(model_version=m_version).inc()
        INFERENCE_LATENCY.labels(model_version=m_version).observe(result["latencyMs"] / 1000.0)

        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "INVALID_INPUT", "message": str(e)},
        )


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_events(payload: BatchPredictRequest):
    """Vectorized high-throughput batch inference for bulk telemetry logs."""
    engine = get_engine()
    if not engine or not engine.is_ready:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"code": "MODEL_NOT_READY", "message": "Production model is not loaded."},
        )

    try:
        events_list = [ev.model_dump() for ev in payload.events]
        result = await engine.predict_batch_async(events_list)

        m_version = result["modelVersion"]
        for p in result["predictions"]:
            PREDICTIONS_COUNTER.labels(model_version=m_version, risk_band=p["riskBand"]).inc()
            if p["isAnomaly"]:
                ANOMALIES_COUNTER.labels(model_version=m_version).inc()

        INFERENCE_LATENCY.labels(model_version=m_version).observe(result["totalLatencyMs"] / 1000.0)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "BATCH_EXECUTION_ERROR", "message": str(e)},
        )
