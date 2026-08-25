"""Explainability engine for calculating feature deviations and non-causal contributing signals."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ContributingSignal:
    feature: str
    observed_value: float
    baseline_value: float
    deviation_zscore: float
    contribution_weight: float
    message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature,
            "observedValue": round(self.observed_value, 2),
            "baselineValue": round(self.baseline_value, 2),
            "deviation": round(self.deviation_zscore, 2),
            "contributionWeight": round(self.contribution_weight, 4),
            "message": self.message,
        }


class ExplainabilityEngine:
    """
    Computes feature deviations against reference population baselines.
    Produces interpretable, non-causal signal breakdowns without misleading claims of root causation.
    """

    DEFAULT_BASELINES = {
        "failed_logins_5m": {"mean": 0.25, "std": 0.8},
        "failed_logins_1h": {"mean": 0.50, "std": 1.2},
        "successful_logins_1h": {"mean": 2.80, "std": 1.5},
        "distinct_users_per_ip_10m": {"mean": 1.05, "std": 0.3},
        "distinct_hosts_per_user_1h": {"mean": 1.10, "std": 0.4},
        "events_per_user_5m": {"mean": 5.50, "std": 3.0},
        "is_new_source_ip": {"mean": 0.02, "std": 0.14},
        "is_new_device": {"mean": 0.03, "std": 0.17},
        "login_hour": {"mean": 13.0, "std": 3.5},
        "distance_from_typical_hour": {"mean": 2.10, "std": 1.8},
        "event_rate_ratio": {"mean": 1.35, "std": 0.7},
        "failure_ratio": {"mean": 0.04, "std": 0.12},
        "bytes_sent_log": {"mean": 7.0, "std": 1.2},
        "bytes_received_log": {"mean": 8.5, "std": 1.4},
        "session_duration_log": {"mean": 5.5, "std": 1.1},
    }

    def __init__(self, baselines: dict[str, dict[str, float]] | None = None):
        self.baselines = baselines or self.DEFAULT_BASELINES

    def explain_sample(
        self,
        features: dict[str, float],
        anomaly_score: float,
        threshold: float,
        top_k: int = 4,
    ) -> list[ContributingSignal]:
        """Generate top contributing signals ranked by statistical deviation."""
        signals = []

        for feat_name, val in features.items():
            base = self.baselines.get(feat_name, {"mean": 0.0, "std": 1.0})
            mean = base["mean"]
            std = max(1e-4, base["std"])

            z_score = float((val - mean) / std)
            abs_z = abs(z_score)

            # We focus on deviations that elevate anomaly risk
            if abs_z > 1.2:
                msg = self._generate_signal_message(feat_name, val, mean, z_score)
                signals.append(
                    ContributingSignal(
                        feature=feat_name,
                        observed_value=float(val),
                        baseline_value=float(mean),
                        deviation_zscore=float(z_score),
                        contribution_weight=float(abs_z),
                        message=msg,
                    )
                )

        # Sort by deviation descending and take top_k
        signals.sort(key=lambda s: s.contribution_weight, reverse=True)

        # Normalize weights
        total_weight = sum(s.contribution_weight for s in signals[:top_k])
        if total_weight > 0:
            for s in signals[:top_k]:
                s.contribution_weight = s.contribution_weight / total_weight

        return signals[:top_k]

    def _generate_signal_message(
        self, feature: str, observed: float, baseline: float, z_score: float
    ) -> str:
        ratio = observed / max(0.1, baseline)

        if feature == "failed_logins_5m":
            return (
                f"Failed login volume ({observed:.0f}) is {ratio:.1f}× above historical baseline."
            )
        elif feature == "failed_logins_1h":
            return f"Cumulative 1-hour authentication failures ({observed:.0f}) elevated significantly."
        elif feature == "events_per_user_5m":
            return f"Burst event velocity ({observed:.0f} events/5m) is {ratio:.1f}× expected rate."
        elif feature == "is_new_source_ip" and observed > 0.5:
            return "Source IP was not previously observed in this user's connection profile."
        elif feature == "is_new_device" and observed > 0.5:
            return "Login attempted from an unauthenticated device type."
        elif feature == "distance_from_typical_hour" and observed >= 5.0:
            return f"Event occurred outside normal operational schedule ({z_score:+.1f} std dev shift)."
        elif feature == "failure_ratio" and observed >= 0.5:
            return f"High failure ratio ({observed:.1%}) observed over the preceding window."
        elif feature == "bytes_sent_log" and z_score > 2.5:
            return f"Outbound data volume is significantly larger than typical session baseline ({z_score:+.1f}σ)."
        elif feature == "distinct_users_per_ip_10m" and observed > 2.0:
            return (
                f"Multiple distinct user accounts ({observed:.0f}) originating from same IP subnet."
            )
        elif feature == "distinct_hosts_per_user_1h" and observed > 2.0:
            return f"User accessed {observed:.0f} distinct infrastructure hosts in short sequence."
        else:
            return f"Signal '{feature}' observed at {observed:.2f} (historical baseline: {baseline:.2f}, {z_score:+.1f}σ)."
