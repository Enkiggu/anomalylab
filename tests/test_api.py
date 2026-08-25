"""Integration tests for FastAPI inference endpoints."""

import pytest
from fastapi.testclient import TestClient

from apps.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_api_health_liveness(client):
    res = client.get("/health/live")
    assert res.status_code == 200
    assert res.json()["status"] == "LIVE"


def test_api_health_readiness(client):
    res = client.get("/health/ready")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "READY"
    assert data["modelLoaded"] is True
    assert "activeModelVersion" in data


def test_api_predict_single_normal_event(client):
    payload = {
        "userId": "user_0001.demo",
        "loginHour": 14,
        "eventOutcome": "SUCCESS",
        "failedAttempts": 0,
        "requestCount": 2,
        "bytesSent": 500,
        "bytesReceived": 2000,
        "sessionDuration": 120,
        "isPrivilegedUser": False,
        "newSourceIp": False,
        "newDevice": False,
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "anomalyScore" in data
    assert 0.0 <= data["anomalyScore"] <= 1.0
    assert "riskBand" in data
    assert "latencyMs" in data


def test_api_predict_single_anomalous_event(client):
    payload = {
        "userId": "user_0001.demo",
        "loginHour": 3,
        "eventOutcome": "FAILURE",
        "failedAttempts": 25,
        "requestCount": 80,
        "bytesSent": 50000,
        "bytesReceived": 10000,
        "sessionDuration": 10,
        "isPrivilegedUser": True,
        "newSourceIp": True,
        "newDevice": True,
    }
    res = client.post("/api/v1/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["anomalyScore"] > 0.60
    assert len(data["contributingSignals"]) > 0


def test_api_batch_prediction(client):
    payload = {
        "events": [
            {"userId": f"user_{i:04d}.demo", "failedAttempts": i % 5, "requestCount": 3}
            for i in range(25)
        ]
    }
    res = client.post("/api/v1/predict/batch", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["batchSize"] == 25
    assert len(data["predictions"]) == 25
    assert data["throughputPerSec"] > 0.0


def test_api_models_endpoint(client):
    res = client.get("/api/v1/models")
    assert res.status_code == 200
    data = res.json()
    assert isinstance(data, list)
    assert len(data) >= 1


def test_api_prometheus_metrics(client):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "anomalylab_predictions_total" in res.text
