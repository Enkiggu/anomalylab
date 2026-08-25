"""Training-serving parity transformer for real-time and batch inference."""

import math
from typing import Any

import numpy as np

from anomalylab.features.contract import FeatureContract, get_security_feature_contract_v1


class FeatureParityTransformer:
    """
    Transforms raw incoming telemetry event dictionaries into the exact numerical feature vector
    expected by the trained models, guaranteeing training-serving parity.
    """

    def __init__(self, contract: FeatureContract | None = None):
        self.contract = contract or get_security_feature_contract_v1()

    def transform_single(self, event: dict[str, Any]) -> tuple[np.ndarray, dict[str, float]]:
        """
        Transform a single raw event into (1 x D) feature vector and feature dict.
        """
        # Extract fields with safe defaults
        login_hour = float(event.get("login_hour", event.get("loginHour", 12)))
        outcome = str(event.get("event_outcome", event.get("eventOutcome", "SUCCESS"))).upper()
        is_failure = 1.0 if outcome == "FAILURE" else 0.0
        is_success = 1.0 if outcome == "SUCCESS" else 0.0

        failed_attempts = float(event.get("failed_attempts", event.get("failedAttempts", 0)))
        req_count = float(event.get("request_count", event.get("requestCount", 1)))
        bytes_sent = float(event.get("bytes_sent", event.get("bytesSent", 0)))
        bytes_recv = float(event.get("bytes_received", event.get("bytesReceived", 0)))
        sess_dur = float(event.get("session_duration", event.get("sessionDuration", 0)))

        is_priv = bool(event.get("is_privileged_user", event.get("isPrivilegedUser", False)))
        is_new_ip = (
            1.0 if bool(event.get("is_new_source", event.get("newSourceIp", False))) else 0.0
        )
        is_new_dev = 1.0 if bool(event.get("is_new_device", event.get("newDevice", False))) else 0.0

        # Exact match of training expressions
        bytes_sent_log = float(math.log(max(0.0, bytes_sent) + 1.0))
        bytes_received_log = float(math.log(max(0.0, bytes_recv) + 1.0))
        session_duration_log = float(math.log(max(0.0, sess_dur) + 1.0))
        distance_from_typical_hour = float(abs(login_hour - 13.0))

        failed_logins_5m = float(
            event.get(
                "failed_logins_5m",
                event.get("failedLogins5m", failed_attempts * 1.5 + (is_failure * 3.0)),
            )
        )
        failed_logins_1h = float(
            event.get(
                "failed_logins_1h",
                event.get("failedLogins1h", failed_attempts * 2.5 + (is_failure * 5.0)),
            )
        )
        successful_logins_1h = float(
            event.get(
                "successful_logins_1h", event.get("successfulLogins1h", is_success * 2.0 + 1.0)
            )
        )

        distinct_hosts_per_user_1h = float(
            event.get(
                "distinct_hosts_per_user_1h",
                event.get("distinctHosts1h", (3.0 if is_priv else 1.0) + (req_count / 50.0)),
            )
        )
        distinct_users_per_ip_10m = float(
            event.get(
                "distinct_users_per_ip_10m",
                event.get(
                    "distinctUsers10m", (4.0 + failed_attempts / 5.0) if is_new_ip > 0.5 else 1.0
                ),
            )
        )
        events_per_user_5m = float(
            event.get("events_per_user_5m", event.get("eventsPerUser5m", req_count + 2.0))
        )

        event_rate_ratio = float(min(100.0, max(0.1, events_per_user_5m / 4.0)))
        failure_ratio = float(
            min(1.0, max(0.0, failed_logins_1h / (failed_logins_1h + successful_logins_1h + 1e-5)))
        )

        feature_map = {
            "failed_logins_5m": failed_logins_5m,
            "failed_logins_1h": failed_logins_1h,
            "successful_logins_1h": successful_logins_1h,
            "distinct_users_per_ip_10m": distinct_users_per_ip_10m,
            "distinct_hosts_per_user_1h": distinct_hosts_per_user_1h,
            "events_per_user_5m": events_per_user_5m,
            "is_new_source_ip": is_new_ip,
            "is_new_device": is_new_dev,
            "login_hour": login_hour,
            "distance_from_typical_hour": distance_from_typical_hour,
            "event_rate_ratio": event_rate_ratio,
            "failure_ratio": failure_ratio,
            "bytes_sent_log": bytes_sent_log,
            "bytes_received_log": bytes_received_log,
            "session_duration_log": session_duration_log,
        }

        # Vector ordered exactly per FeatureContract
        vector = np.array(
            [[feature_map[name] for name in self.contract.feature_names]], dtype=np.float32
        )
        return vector, feature_map

    def transform_batch(
        self, events: list[dict[str, Any]]
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Vectorized transformation of batch events."""
        vectors = []
        maps = []
        for ev in events:
            v, m = self.transform_single(ev)
            vectors.append(v[0])
            maps.append(m)
        return np.array(vectors, dtype=np.float32), maps
