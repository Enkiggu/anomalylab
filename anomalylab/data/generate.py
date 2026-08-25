"""Deterministic synthetic telemetry event generator for AnomalyLab."""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from anomalylab.data.fingerprint import (
    compute_config_hash,
    compute_file_sha256,
)
from anomalylab.data.schemas import AnomalyType, DatasetManifest


def generate_security_telemetry(
    rows: int = 50000,
    anomaly_rate: float = 0.02,
    seed: int = 42,
    start_date_str: str = "2026-08-01T00:00:00Z",
    duration_days: int = 30,
    users_count: int = 500,
    hosts_count: int = 100,
    privileged_ratio: float = 0.05,
    version: str = "security_v2",
) -> pl.DataFrame:
    """
    Generate synthetic security events deterministically with realistic, non-trivial enterprise behavioral distributions.

    In v2:
    - Normal background includes legitimate night-shift users, occasional password retry bursts (2-5 failures),
      legitimate new device/IP introductions, and scheduled high-throughput automation batch jobs.
    - Anomalies represent subtle multi-signal deviations rather than single-feature extreme outliers.
    """
    rng = np.random.default_rng(seed)

    start_dt = datetime.fromisoformat(start_date_str.replace("Z", "+00:00"))
    total_seconds = duration_days * 86400

    # User profiles & roles
    users = [f"user_{i:04d}.demo" for i in range(users_count)]
    privileged_users = set(
        rng.choice(users, size=int(users_count * privileged_ratio), replace=False)
    )

    # 8% of users are night-shift or global timezone workers (legitimate off-hours)
    night_shift_users = set(rng.choice(users, size=int(users_count * 0.08), replace=False))

    # 5% of users are automated service accounts (legitimate high event velocity)
    service_account_users = set(rng.choice(users, size=int(users_count * 0.05), replace=False))

    # User primary hosts & base IPs
    hosts = [f"HOST-DEMO-{i:03d}" for i in range(hosts_count)]
    user_primary_host = {u: rng.choice(hosts) for u in users}
    user_primary_ip = {u: f"192.0.2.{(i % 250) + 1}" for i, u in enumerate(users)}

    # Typical login window per user
    user_typical_start = {}
    user_typical_end = {}
    for u in users:
        if u in night_shift_users:
            user_typical_start[u] = int(rng.integers(21, 23))  # 21:00 or 22:00
            user_typical_end[u] = int(rng.integers(5, 8))  # 05:00 to 07:00
        else:
            user_typical_start[u] = int(rng.integers(7, 10))  # 07:00 to 09:00
            user_typical_end[u] = int(rng.integers(17, 20))  # 17:00 to 19:00

    # Generate chronological timestamps across 30 days
    event_offsets = np.sort(rng.uniform(0, total_seconds, size=rows))

    timestamps = []
    user_ids = []
    host_ids = []
    source_ips = []
    event_types = []
    event_outcomes = []
    login_hours = []
    device_types = []
    failed_attempts_list = []
    request_counts = []
    bytes_sent_list = []
    bytes_received_list = []
    session_durations = []
    is_privileged_list = []
    is_new_device_list = []
    is_new_source_list = []
    is_anomaly_list = []
    anomaly_types_list = []

    # Track seen devices and IPs per user for realistic novelty rates
    seen_user_ips: dict[str, set[str]] = {u: {user_primary_ip[u]} for u in users}
    seen_user_devices: dict[str, set[str]] = {u: {"workstation"} for u in users}

    # Anomaly indices
    anomaly_count = int(rows * anomaly_rate)
    anomaly_indices = set(rng.choice(rows, size=anomaly_count, replace=False))

    anomaly_distribution = [
        (AnomalyType.BRUTE_FORCE, 0.30),
        (AnomalyType.UNUSUAL_LOGIN_TIME, 0.25),
        (AnomalyType.NEW_DEVICE_LOGIN, 0.20),
        (AnomalyType.HIGH_VELOCITY, 0.15),
        (AnomalyType.CREDENTIAL_STUFFING, 0.10),
    ]
    anomaly_type_choices = [a[0] for a in anomaly_distribution]
    anomaly_type_probs = [a[1] for a in anomaly_distribution]

    for idx, offset in enumerate(event_offsets):
        current_time = start_dt + timedelta(seconds=float(offset))
        hour = current_time.hour

        is_anomaly = idx in anomaly_indices
        anomaly_type = (
            rng.choice(anomaly_type_choices, p=anomaly_type_probs)
            if is_anomaly
            else AnomalyType.NORMAL
        )

        user = rng.choice(users)
        is_priv = user in privileged_users
        is_service_acc = user in service_account_users
        is_night_shift = user in night_shift_users

        if not is_anomaly:
            # ---------------------------------------------------------
            # NORMAL ENTERPRISE TELEMETRY
            # ---------------------------------------------------------
            typ_start = user_typical_start[user]
            typ_end = user_typical_end[user]

            # 88% of events align with user's specific working hours; 12% represent occasional overtime / off-hours checkins
            if rng.random() < 0.88:
                if is_night_shift:
                    hour_mod = int(rng.choice([22, 23, 0, 1, 2, 3, 4, 5, 6]))
                else:
                    hour_mod = int(rng.integers(typ_start, max(typ_start + 1, typ_end)))
                current_time = current_time.replace(hour=hour_mod)
                hour = hour_mod

            host = user_primary_host[user] if rng.random() > 0.08 else rng.choice(hosts)

            # Legitimate remote/VPN or secondary IP introductions (5% probability)
            if rng.random() > 0.06:
                ip = user_primary_ip[user]
            else:
                ip = f"198.51.100.{rng.integers(1, 250)}"

            # Legitimate secondary device usage (workstation, laptop, mobile)
            dev_roll = rng.random()
            if dev_roll < 0.70:
                device = "workstation"
            elif dev_roll < 0.90:
                device = "laptop"
            else:
                device = "mobile"

            is_new_ip = ip not in seen_user_ips[user]
            is_new_dev = device not in seen_user_devices[user]

            seen_user_ips[user].add(ip)
            seen_user_devices[user].add(device)

            # Legitimate password typos / retry bursts:
            # 96% single success; 4% have 1 to 4 failed attempts before resolution
            if rng.random() < 0.96:
                outcome = "SUCCESS"
                failed_attempts = 0
            else:
                outcome = "FAILURE" if rng.random() < 0.3 else "SUCCESS"
                failed_attempts = int(rng.integers(1, 5))

            if is_service_acc:
                event_type = "api_access"
                req_count = int(rng.integers(25, 75))  # High legitimate velocity
                bytes_s = int(rng.lognormal(mean=9.0, sigma=0.8))
                bytes_r = int(rng.lognormal(mean=10.0, sigma=0.9))
                sess_dur = int(rng.integers(60, 600))
            else:
                event_type = rng.choice(
                    ["login", "api_access", "file_transfer", "database_query"],
                    p=[0.35, 0.35, 0.20, 0.10],
                )
                req_count = int(rng.integers(1, 10))
                bytes_s = int(rng.lognormal(mean=7.0, sigma=0.9))
                bytes_r = int(rng.lognormal(mean=8.5, sigma=1.0))
                sess_dur = int(rng.lognormal(mean=5.5, sigma=1.0))

        else:
            # ---------------------------------------------------------
            # REALISTIC SYNTHETIC ANOMALIES (Multi-Feature Conjunctions)
            # ---------------------------------------------------------
            if anomaly_type == AnomalyType.BRUTE_FORCE:
                # Password spraying/brute force: 6-18 failures from external untrusted IP
                outcome = "FAILURE"
                failed_attempts = int(rng.integers(6, 19))
                event_type = "login"
                ip = f"203.0.113.{rng.integers(1, 250)}"  # External subnet
                host = user_primary_host[user]
                device = "workstation" if rng.random() < 0.5 else "unknown_client"
                is_new_ip = True
                is_new_dev = True
                req_count = int(rng.integers(15, 55))
                bytes_s = int(rng.integers(400, 2500))
                bytes_r = int(rng.integers(200, 1500))
                sess_dur = int(rng.integers(2, 25))

            elif anomaly_type == AnomalyType.UNUSUAL_LOGIN_TIME:
                # Off-hours privileged exfiltration: regular daytime user accessing between 01:00-04:00 UTC
                hour_mod = int(rng.integers(1, 5))
                current_time = current_time.replace(hour=hour_mod)
                hour = hour_mod

                # Pick a daytime user specifically so distance_from_typical_hour is significant
                day_users = [u for u in users if u not in night_shift_users]
                user = rng.choice(day_users)
                is_priv = user in privileged_users or rng.random() < 0.6

                outcome = "SUCCESS"
                failed_attempts = int(rng.choice([0, 1]))
                event_type = "file_transfer"
                ip = f"198.51.100.{rng.integers(1, 250)}"
                host = rng.choice(hosts)
                device = "laptop" if rng.random() < 0.6 else "workstation"
                is_new_ip = True
                is_new_dev = False
                req_count = int(rng.integers(6, 24))
                bytes_s = int(
                    rng.lognormal(mean=11.2, sigma=0.7)
                )  # Subtly elevated (~75 KB to ~500 KB)
                bytes_r = int(rng.lognormal(mean=9.5, sigma=0.8))
                sess_dur = int(rng.integers(120, 1800))

            elif anomaly_type == AnomalyType.NEW_DEVICE_LOGIN:
                # Privilege escalation from unseen external device with immediate sensitive access
                outcome = "SUCCESS"
                failed_attempts = int(rng.choice([0, 1, 2]))
                event_type = "privilege_escalation"
                ip = f"203.0.113.{rng.integers(1, 250)}"
                host = rng.choice(hosts)
                device = "untrusted_mobile"
                is_new_ip = True
                is_new_dev = True
                req_count = int(rng.integers(10, 40))
                bytes_s = int(rng.lognormal(mean=9.5, sigma=0.9))
                bytes_r = int(rng.lognormal(mean=10.5, sigma=0.9))
                sess_dur = int(rng.integers(30, 450))

            elif anomaly_type == AnomalyType.HIGH_VELOCITY:
                # Rapid scraping / unauthorized data sweep
                outcome = "SUCCESS"
                failed_attempts = 0
                event_type = "api_access"
                ip = (
                    user_primary_ip[user]
                    if rng.random() < 0.5
                    else f"198.51.100.{rng.integers(1, 250)}"
                )
                host = rng.choice(hosts)
                device = "workstation"
                is_new_ip = ip not in seen_user_ips[user]
                is_new_dev = False
                req_count = int(
                    rng.integers(50, 160)
                )  # 5-15x normal human, overlapping with high batch
                bytes_s = int(rng.lognormal(mean=10.2, sigma=0.7))
                bytes_r = int(rng.lognormal(mean=11.0, sigma=0.7))
                sess_dur = int(rng.integers(15, 180))

            else:  # CREDENTIAL_STUFFING
                # Distributed credential probing with moderate failures
                outcome = "FAILURE" if rng.random() < 0.85 else "SUCCESS"
                failed_attempts = int(rng.integers(4, 14))
                event_type = "login"
                ip = f"203.0.113.{rng.integers(100, 200)}"
                host = rng.choice(hosts)
                device = "browser_client"
                is_new_ip = True
                is_new_dev = True
                req_count = int(rng.integers(18, 65))
                bytes_s = int(rng.integers(450, 2200))
                bytes_r = int(rng.integers(250, 1200))
                sess_dur = int(rng.integers(3, 20))

        timestamps.append(current_time)
        user_ids.append(user)
        host_ids.append(host)
        source_ips.append(ip)
        event_types.append(event_type)
        event_outcomes.append(outcome)
        login_hours.append(hour)
        device_types.append(device)
        failed_attempts_list.append(failed_attempts)
        request_counts.append(req_count)
        bytes_sent_list.append(bytes_s)
        bytes_received_list.append(bytes_r)
        session_durations.append(sess_dur)
        is_privileged_list.append(is_priv)
        is_new_device_list.append(is_new_dev)
        is_new_source_list.append(is_new_ip)
        is_anomaly_list.append(is_anomaly)
        anomaly_types_list.append(
            anomaly_type.value if isinstance(anomaly_type, AnomalyType) else str(anomaly_type)
        )

    df = pl.DataFrame(
        {
            "timestamp": timestamps,
            "user_id": user_ids,
            "host_id": host_ids,
            "source_ip": source_ips,
            "event_type": event_types,
            "event_outcome": event_outcomes,
            "login_hour": pl.Series(login_hours, dtype=pl.Int32),
            "device_type": device_types,
            "country_code": ["US"] * rows,
            "failed_attempts": pl.Series(failed_attempts_list, dtype=pl.Int32),
            "request_count": pl.Series(request_counts, dtype=pl.Int32),
            "bytes_sent": pl.Series(bytes_sent_list, dtype=pl.Int64),
            "bytes_received": pl.Series(bytes_received_list, dtype=pl.Int64),
            "session_duration": pl.Series(session_durations, dtype=pl.Int32),
            "is_privileged_user": pl.Series(is_privileged_list, dtype=pl.Boolean),
            "is_new_device": pl.Series(is_new_device_list, dtype=pl.Boolean),
            "is_new_source": pl.Series(is_new_source_list, dtype=pl.Boolean),
            "is_anomaly": pl.Series(is_anomaly_list, dtype=pl.Boolean),
            "anomaly_type": anomaly_types_list,
        }
    )

    # Sort deterministically by timestamp
    df = df.sort("timestamp")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Deterministic Security Telemetry Generator for AnomalyLab."
    )
    parser.add_argument(
        "--config", type=str, default="configs/data_security_v2.yaml", help="Path to config YAML"
    )
    parser.add_argument("--rows", type=int, default=50000, help="Number of rows")
    parser.add_argument("--anomaly-rate", type=float, default=0.02, help="Anomaly prevalence rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--version", type=str, default="security_v2", help="Dataset version identifier"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/security_events_v2.parquet",
        help="Output Parquet path",
    )
    parser.add_argument(
        "--manifest-output",
        type=str,
        default="data/raw/security_manifest_v2.json",
        help="Manifest JSON path",
    )
    args = parser.parse_args()

    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)

    rows = config.get("rows", args.rows)
    anomaly_rate = config.get("anomaly_rate", args.anomaly_rate)
    seed = config.get("seed", args.seed)
    version = config.get("version", args.version)
    start_date = config.get("start_date", "2026-08-01T00:00:00Z")
    duration_days = config.get("duration_days", 30)

    print(f"\n--- Generating AnomalyLab Synthetic Telemetry [{version}] ---")
    print(f"Rows: {rows:,} | Anomaly Rate: {anomaly_rate:.2%} | Seed: {seed}")

    df = generate_security_telemetry(
        rows=rows,
        anomaly_rate=anomaly_rate,
        seed=seed,
        start_date_str=start_date,
        duration_days=duration_days,
        version=version,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out_path)

    # Compute binary SHA-256 fingerprint
    fingerprint = compute_file_sha256(out_path)
    config_hash = compute_config_hash(
        {"rows": rows, "anomaly_rate": anomaly_rate, "seed": seed, "version": version}
    )

    manifest = DatasetManifest(
        dataset_id=f"{version}_{seed}",
        dataset_version=version,
        generator_version="2.0.0",
        scenario="enterprise_security_telemetry",
        config_hash=config_hash,
        data_sha256=fingerprint,
        row_count=len(df),
        column_count=len(df.columns),
        columns=df.columns,
        created_at=datetime.now(UTC),
        seed=seed,
        start_date=str(df["timestamp"].min()),
        end_date=str(df["timestamp"].max()),
        anomaly_count=int(df["is_anomaly"].sum()),
        anomaly_rate=float(df["is_anomaly"].sum() / len(df)),
        raw_path=str(out_path),
    )

    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest.model_dump(mode="json"), f, indent=2)

    print(
        f"Generated {len(df):,} events ({df['is_anomaly'].sum():,} anomalies, {manifest.anomaly_rate:.2%})"
    )
    print(f"Output: {out_path}")
    print(f"SHA-256 Fingerprint: {fingerprint}")
    print(f"Manifest: {manifest_path}\n")


if __name__ == "__main__":
    main()
