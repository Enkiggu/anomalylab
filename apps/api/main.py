"""FastAPI application entrypoint for AnomalyLab."""

import time
import uuid
from contextlib import asynccontextmanager
from datetime import UTC, datetime

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from anomalylab.inference.engine import ProductionInferenceEngine
from apps.api.routes import datasets, drift, evaluations, health, metrics, models, predict

inference_engine: ProductionInferenceEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup & shutdown lifespan."""
    global inference_engine
    print("\n--- Initializing AnomalyLab Inference Service ---")
    inference_engine = ProductionInferenceEngine(models_dir="models")
    try:
        manifest = inference_engine.load_champion_model()
        print(f"Production champion model loaded: {manifest.model_version} ({manifest.algorithm})")
    except Exception as e:
        print(f"Warning on startup: Could not load production model: {e}")

    yield

    print("Shutting down AnomalyLab Inference Service...")
    if inference_engine and inference_engine.executor:
        inference_engine.executor.shutdown(wait=False)


app = FastAPI(
    title="AnomalyLab ML Analytics & Inference API",
    description="Production-grade anomaly detection, model registry, explainability, and statistical drift monitoring API.",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def structured_logging_middleware(request: Request, call_next):
    """Structured request logging middleware tracking latency and request IDs."""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    request.state.request_id = request_id

    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time-Ms"] = f"{duration_ms:.2f}"
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Unified error handling returning structured error models."""
    req_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "code": "INTERNAL_SERVER_ERROR",
            "message": str(exc),
            "requestId": req_id,
            "timestamp": datetime.now(UTC).isoformat(),
        },
    )


# Attach API Routers
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)
app.include_router(evaluations.router)
app.include_router(drift.router)
app.include_router(datasets.router)
app.include_router(metrics.router)


@app.get("/")
def root():
    return {
        "service": "AnomalyLab ML Platform API",
        "status": "ONLINE",
        "docs": "/docs",
        "health": "/health/ready",
        "version": "1.0.0",
    }
