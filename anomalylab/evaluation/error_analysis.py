"""Error analysis engine for deep inspection of False Positives and False Negatives."""

from typing import Any

import numpy as np
import polars as pl


def perform_error_analysis(
    df: pl.DataFrame,
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float,
    top_k: int = 10,
) -> dict[str, Any]:
    """
    Perform deep inspection on:
    1. Top False Positives (Normal events with high anomaly scores)
    2. Top False Negatives (True anomalies with low anomaly scores)
    """
    y_true_arr = np.asarray(y_true, dtype=bool)
    y_scores_arr = np.asarray(y_scores, dtype=np.float32)
    y_pred = y_scores_arr >= threshold

    is_fp = (~y_true_arr) & y_pred
    is_fn = y_true_arr & (~y_pred)

    fp_indices = np.where(is_fp)[0]
    fn_indices = np.where(is_fn)[0]

    # Sort FPs by score descending (highest confidence mistakes)
    if len(fp_indices) > 0:
        sorted_fp_idx = fp_indices[np.argsort(-y_scores_arr[fp_indices])][:top_k]
    else:
        sorted_fp_idx = []

    # Sort FNs by score ascending (lowest confidence misses)
    if len(fn_indices) > 0:
        sorted_fn_idx = fn_indices[np.argsort(y_scores_arr[fn_indices])][:top_k]
    else:
        sorted_fn_idx = []

    def format_error_sample(idx: int, error_type: str) -> dict[str, Any]:
        row = df.row(idx, named=True)
        score = float(y_scores_arr[idx])

        # Categorize probable root cause based on feature signals
        category = _categorize_error(row, score, threshold, error_type)

        # Extract prominent feature values
        key_features = {
            "failed_logins_5m": float(row.get("failed_logins_5m", 0.0)),
            "failed_logins_1h": float(row.get("failed_logins_1h", 0.0)),
            "events_per_user_5m": float(row.get("events_per_user_5m", 0.0)),
            "is_new_source_ip": float(row.get("is_new_source_ip", 0.0)),
            "is_new_device": float(row.get("is_new_device", 0.0)),
            "login_hour": float(row.get("login_hour", 0.0)),
            "distance_from_typical_hour": float(row.get("distance_from_typical_hour", 0.0)),
            "bytes_sent_log": float(row.get("bytes_sent_log", 0.0)),
        }

        return {
            "index": int(idx),
            "error_type": error_type,
            "category": category,
            "anomaly_score": round(score, 4),
            "threshold": round(threshold, 4),
            "ground_truth_label": bool(y_true_arr[idx]),
            "ground_truth_type": str(row.get("anomaly_type", "normal")),
            "user_id": str(row.get("user_id", "unknown")),
            "source_ip": str(row.get("source_ip", "unknown")),
            "host_id": str(row.get("host_id", "unknown")),
            "key_features": key_features,
            "explanation": _generate_error_explanation(category, key_features, score, error_type),
        }

    top_fps = [format_error_sample(i, "FALSE_POSITIVE") for i in sorted_fp_idx]
    top_fns = [format_error_sample(i, "FALSE_NEGATIVE") for i in sorted_fn_idx]

    # Category aggregates
    fp_category_counts: dict[str, int] = {}
    for fp in top_fps:
        fp_category_counts[fp["category"]] = fp_category_counts.get(fp["category"], 0) + 1

    fn_category_counts: dict[str, int] = {}
    for fn in top_fns:
        fn_category_counts[fn["category"]] = fn_category_counts.get(fn["category"], 0) + 1

    return {
        "total_false_positives": int(np.sum(is_fp)),
        "total_false_negatives": int(np.sum(is_fn)),
        "top_false_positives": top_fps,
        "top_false_negatives": top_fns,
        "fp_categories": fp_category_counts,
        "fn_categories": fn_category_counts,
    }


def _categorize_error(row: dict[str, Any], score: float, threshold: float, error_type: str) -> str:
    if error_type == "FALSE_POSITIVE":
        if (
            float(row.get("is_new_device", 0.0)) > 0.5
            or float(row.get("is_new_source_ip", 0.0)) > 0.5
        ):
            return "new_legitimate_device_or_ip"
        if float(row.get("distance_from_typical_hour", 0.0)) >= 6.0:
            return "legitimate_off_hours_work"
        if float(row.get("events_per_user_5m", 0.0)) >= 20.0:
            return "legitimate_batch_automation"
        return "rare_normal_traffic_variation"
    else:  # FALSE_NEGATIVE
        if float(row.get("failed_logins_5m", 0.0)) <= 3.0:
            return "low_volume_stealth_anomaly"
        if float(row.get("distance_from_typical_hour", 0.0)) <= 2.0:
            return "business_hours_mimicry"
        return "near_baseline_subtle_drift"


def _generate_error_explanation(
    category: str, features: dict[str, float], score: float, error_type: str
) -> str:
    if error_type == "FALSE_POSITIVE":
        if category == "new_legitimate_device_or_ip":
            return "Normal user connected from a previously unobserved IP/device, driving novelty signals above threshold."
        elif category == "legitimate_off_hours_work":
            return "Legitimate activity occurred outside typical working hours, inflating distance-from-baseline."
        elif category == "legitimate_batch_automation":
            return "Benign bulk script/query created temporary event volume spike."
        return "Statistical outlier within normal operational variance."
    else:
        if category == "low_volume_stealth_anomaly":
            return "Slow-and-low attack volume blended into normal baseline failure rate."
        elif category == "business_hours_mimicry":
            return "Attacker operated strictly within target user's typical login schedule."
        return "Subtle multi-feature anomaly with individual signals below detection sensitivity."
