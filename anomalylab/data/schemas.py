"""Canonical schemas for AnomalyLab data entities, quality reports, and dataset manifests."""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AnomalyType(StrEnum):
    NORMAL = "normal"
    BRUTE_FORCE = "brute_force_pattern"
    UNUSUAL_LOGIN_TIME = "unusual_login_time"
    NEW_DEVICE_LOGIN = "new_device_login"
    HIGH_VELOCITY = "high_velocity_activity"
    CREDENTIAL_STUFFING = "credential_stuffing"
    UNUSUAL_AMOUNT = "unusual_transaction_amount"
    NEW_COUNTRY = "new_country_activity"


class SecurityEvent(BaseModel):
    """Synthetic security telemetry event schema."""

    timestamp: datetime
    user_id: str
    host_id: str
    source_ip: str
    event_type: str = Field(
        description="e.g. login, api_access, privilege_escalation, file_transfer"
    )
    event_outcome: str = Field(description="SUCCESS or FAILURE")
    login_hour: int = Field(ge=0, le=23)
    device_type: str = Field(description="workstation, server, mobile, unknown")
    country_code: str = Field(default="US")
    failed_attempts: int = Field(default=0, ge=0)
    request_count: int = Field(default=1, ge=1)
    bytes_sent: int = Field(default=0, ge=0)
    bytes_received: int = Field(default=0, ge=0)
    session_duration: int = Field(default=0, ge=0, description="Duration in seconds")
    is_privileged_user: bool = Field(default=False)
    is_new_device: bool = Field(default=False)
    is_new_source: bool = Field(default=False)

    # Hidden ground-truth metadata (NEVER fed directly into model feature matrix)
    is_anomaly: bool = Field(default=False)
    anomaly_type: AnomalyType = Field(default=AnomalyType.NORMAL)


class TransactionEvent(BaseModel):
    """Synthetic financial transaction event schema."""

    timestamp: datetime
    customer_id: str
    transaction_id: str
    amount: float = Field(gt=0)
    currency: str = Field(default="USD")
    merchant_category: str
    country: str = Field(default="US")
    device_id: str
    payment_method: str = Field(default="card")
    transaction_hour: int = Field(ge=0, le=23)
    account_age_days: int = Field(ge=0)
    transactions_last_10m: int = Field(default=1, ge=0)
    transactions_last_24h: int = Field(default=1, ge=0)
    average_amount_30d: float = Field(default=50.0, ge=0)
    is_new_device: bool = Field(default=False)
    is_new_country: bool = Field(default=False)

    # Ground truth
    is_anomaly: bool = Field(default=False)
    anomaly_type: AnomalyType = Field(default=AnomalyType.NORMAL)


class DataQualityCheckResult(BaseModel):
    check_name: str
    passed: bool
    details: dict[str, Any] = Field(default_factory=dict)
    message: str


class DataQualityReport(BaseModel):
    dataset_version: str
    total_records: int
    checked_at: datetime
    overall_status: str  # PASSED or FAILED
    null_counts: dict[str, int]
    null_percentages: dict[str, float]
    data_types: dict[str, str]
    checks: list[DataQualityCheckResult]
    anomaly_count: int
    anomaly_ratio: float
    anomaly_breakdown: dict[str, int]


class DatasetManifest(BaseModel):
    dataset_id: str
    dataset_version: str
    generator_version: str
    scenario: str
    config_hash: str
    data_sha256: str
    row_count: int
    column_count: int
    columns: list[str]
    created_at: datetime
    seed: int
    start_date: str
    end_date: str
    anomaly_count: int
    anomaly_rate: float
    raw_path: str
    train_path: str | None = None
    val_path: str | None = None
    test_path: str | None = None
