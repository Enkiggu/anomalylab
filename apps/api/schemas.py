"""Pydantic request and response schemas for FastAPI endpoints."""

from typing import Any

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    timestamp: str | None = None
    userId: str = Field(default="user_0001.demo")
    hostId: str = Field(default="HOST-DEMO-001")
    sourceIp: str = Field(default="192.0.2.10")
    eventType: str = Field(default="login")
    eventOutcome: str = Field(default="SUCCESS")
    loginHour: int = Field(default=12, ge=0, le=23)
    failedAttempts: int = Field(default=0, ge=0)
    requestCount: int = Field(default=1, ge=1)
    bytesSent: int = Field(default=1200, ge=0)
    bytesReceived: int = Field(default=4500, ge=0)
    sessionDuration: int = Field(default=180, ge=0)
    isPrivilegedUser: bool = Field(default=False)
    newSourceIp: bool = Field(default=False)
    newDevice: bool = Field(default=False)


class BatchPredictRequest(BaseModel):
    events: list[PredictRequest] = Field(min_length=1, max_length=5000)


class ContributingSignalResponse(BaseModel):
    feature: str
    observedValue: float
    baselineValue: float
    deviation: float
    contributionWeight: float
    message: str


class PredictResponse(BaseModel):
    modelVersion: str
    algorithm: str
    anomalyScore: float
    isAnomaly: bool
    threshold: float
    riskBand: str
    contributingSignals: list[ContributingSignalResponse]
    featureValues: dict[str, float]
    latencyMs: float


class BatchPredictResponse(BaseModel):
    modelVersion: str
    algorithm: str
    threshold: float
    batchSize: int
    anomalyCount: int
    predictions: list[dict[str, Any]]
    totalLatencyMs: float
    avgLatencyPerItemMs: float
    throughputPerSec: float


class ErrorResponse(BaseModel):
    code: str
    message: str
    requestId: str
    timestamp: str
