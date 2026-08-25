# AnomalyLab Dataset Specification & Lineage

## 1. Synthetic Security Event Generation

To evaluate anomaly detection algorithms rigorously without risking exposure of real production credentials, **AnomalyLab** incorporates a deterministic generator parameterized by fixed random seeds.

### Canonical Schema (`SecurityEvent`)

| Field | Type | Domain / Constraints | Description |
|---|---|---|---|
| `event_id` | `String` | UUIDv4 | Unique event identifier |
| `timestamp` | `Datetime` | 30-day temporal range | Event creation timestamp (UTC) |
| `user_id` | `String` | `user_0001` - `user_0100` | Account identifier |
| `host_id` | `String` | `HOST-001` - `HOST-020` | Target infrastructure server |
| `source_ip` | `String` | IPv4 Address | Originating connection IP |
| `event_type` | `String` | `login`, `file_access`, `privilege_escalation` | Action performed |
| `event_outcome`| `String` | `SUCCESS`, `FAILURE` | Authentication outcome |
| `login_hour` | `Int` | 0 – 23 | Hour of event in UTC |
| `failed_attempts`| `Int` | $\ge 0$ | Consecutive authentication failures |
| `request_count` | `Int` | $\ge 1$ | 5-minute request burst count |
| `bytes_sent` | `Float` | $\ge 0.0$ | Outbound network payload volume |
| `bytes_received` | `Float` | $\ge 0.0$ | Inbound network payload volume |
| `session_duration` | `Float` | $\ge 0.0$ | Active session duration in seconds |
| `is_privileged_user` | `Bool` | `True` / `False` | Root or administrator privilege flag |
| `is_anomaly` | `Bool` | `True` / `False` | Ground-truth anomaly indicator |
| `anomaly_type` | `String` | `normal`, `brute_force_pattern`, ... | Injected anomaly taxonomy |

---

## 2. Injected Anomaly Taxonomy

1. **`brute_force_pattern`**: 15–35 rapid login failures per 5-minute window from unseen external IP addresses targeting privileged accounts.
2. **`unusual_login_time`**: Legitimate credentials authenticated between 01:00 and 04:00 UTC (normal work baseline: 08:00–18:00 UTC).
3. **`new_device_login`**: Authentication originating from an unknown device fingerprint with anomalous user-agent signature.
4. **`high_velocity_activity`**: High event volume bursts (>80 requests/min) across multiple distinct hosts in short sequence.
5. **`credential_stuffing`**: Single source IP attempting logins against multiple distinct user accounts with mixed success/failure ratios.

---

## 3. Data Splits & Feature Contract (`v1`)

- **Train Split**: Days 1–20 (33,252 records, 701 ground-truth anomalies).
- **Validation Split**: Days 21–25 (8,449 records, 178 ground-truth anomalies).
- **Held-Out Test Split**: Days 26–30 (8,299 records, 179 ground-truth anomalies).

### Feature List (15 Contract Features)
1. `failed_logins_5m`
2. `failed_logins_1h`
3. `successful_logins_1h`
4. `distinct_users_per_ip_10m`
5. `distinct_hosts_per_user_1h`
6. `events_per_user_5m`
7. `is_new_source_ip`
8. `is_new_device`
9. `login_hour`
10. `distance_from_typical_hour`
11. `event_rate_ratio`
12. `failure_ratio`
13. `bytes_sent_log`
14. `bytes_received_log`
15. `session_duration_log`
