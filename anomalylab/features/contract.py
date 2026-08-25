"""Feature schema contracts and validation for AnomalyLab."""

import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field


class FeatureDefinition(BaseModel):
    name: str
    dtype: str  # float32, int32, bool, etc.
    nullable: bool = False
    min_value: float | None = None
    max_value: float | None = None
    description: str = ""


class FeatureContract(BaseModel):
    version: str
    scenario: str
    features: list[FeatureDefinition]
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def compute_schema_hash(self) -> str:
        """Compute deterministic hash of feature contract schema."""
        raw = json.dumps([f.model_dump() for f in self.features], sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def validate_record(self, record: dict[str, Any]) -> tuple[bool, list[str]]:
        """Validate a single inference record against feature contract."""
        errors = []
        for feat in self.features:
            val = record.get(feat.name)
            if val is None:
                if not feat.nullable:
                    errors.append(f"Missing required feature: '{feat.name}'")
                continue

            # Numeric range check
            if isinstance(val, (int, float)):
                if feat.min_value is not None and val < feat.min_value:
                    errors.append(f"Feature '{feat.name}' value {val} < min {feat.min_value}")
                if feat.max_value is not None and val > feat.max_value:
                    errors.append(f"Feature '{feat.name}' value {val} > max {feat.max_value}")

        return len(errors) == 0, errors


def get_security_feature_contract_v1() -> FeatureContract:
    """Return standard v1 feature contract for security anomaly detection."""
    features = [
        FeatureDefinition(
            name="failed_logins_5m",
            dtype="float32",
            min_value=0.0,
            max_value=500.0,
            description="Failed logins in past 5m",
        ),
        FeatureDefinition(
            name="failed_logins_1h",
            dtype="float32",
            min_value=0.0,
            max_value=2000.0,
            description="Failed logins in past 1h",
        ),
        FeatureDefinition(
            name="successful_logins_1h",
            dtype="float32",
            min_value=0.0,
            max_value=2000.0,
            description="Successful logins in past 1h",
        ),
        FeatureDefinition(
            name="distinct_users_per_ip_10m",
            dtype="float32",
            min_value=0.0,
            max_value=500.0,
            description="Distinct users observed on this IP in 10m",
        ),
        FeatureDefinition(
            name="distinct_hosts_per_user_1h",
            dtype="float32",
            min_value=0.0,
            max_value=200.0,
            description="Distinct hosts accessed by user in 1h",
        ),
        FeatureDefinition(
            name="events_per_user_5m",
            dtype="float32",
            min_value=0.0,
            max_value=2000.0,
            description="Total events for user in 5m",
        ),
        FeatureDefinition(
            name="is_new_source_ip",
            dtype="float32",
            min_value=0.0,
            max_value=1.0,
            description="1.0 if source IP never seen for user before",
        ),
        FeatureDefinition(
            name="is_new_device",
            dtype="float32",
            min_value=0.0,
            max_value=1.0,
            description="1.0 if device type never seen for user before",
        ),
        FeatureDefinition(
            name="login_hour",
            dtype="float32",
            min_value=0.0,
            max_value=23.0,
            description="Hour of day (0-23)",
        ),
        FeatureDefinition(
            name="distance_from_typical_hour",
            dtype="float32",
            min_value=0.0,
            max_value=12.0,
            description="Distance from user typical work hours",
        ),
        FeatureDefinition(
            name="event_rate_ratio",
            dtype="float32",
            min_value=0.0,
            max_value=100.0,
            description="Short-term to long-term event rate ratio",
        ),
        FeatureDefinition(
            name="failure_ratio",
            dtype="float32",
            min_value=0.0,
            max_value=1.0,
            description="Ratio of failures to total attempts in 1h",
        ),
        FeatureDefinition(
            name="bytes_sent_log",
            dtype="float32",
            min_value=0.0,
            max_value=30.0,
            description="Natural log of bytes sent + 1",
        ),
        FeatureDefinition(
            name="bytes_received_log",
            dtype="float32",
            min_value=0.0,
            max_value=30.0,
            description="Natural log of bytes received + 1",
        ),
        FeatureDefinition(
            name="session_duration_log",
            dtype="float32",
            min_value=0.0,
            max_value=20.0,
            description="Natural log of session duration + 1",
        ),
    ]
    return FeatureContract(
        version="v1",
        scenario="security",
        features=features,
        metadata={"created_by": "AnomalyLab", "feature_count": len(features)},
    )
