# Model Card: Threshold-Tuned Isolation Forest Champion (`isolation_forest_v2`)

## Model Details
- **Developer**: AnomalyLab ML Engineering Team
- **Model Date**: August 2026
- **Model Version**: `isolation_forest_v2_20260821_001323`
- **Model Type**: Ensemble Tree-Based Anomaly Detector (`IsolationForest` with Normalized Scoring & Validation-Tuned Threshold)
- **License**: Apache-2.0
- **Artifact Path**: `models/isolation_forest_v2_20260821_001323.joblib`
- **Manifest Path**: `models/isolation_forest_v2_20260821_001323_manifest.json`
- **Active Registry Stage**: `PRODUCTION` (Champion)

---

## Intended Use
- **Primary Use**: Real-time detection of brute-force attacks, credential stuffing, anomalous off-hours access, new device privilege escalations, and high-velocity scraping in enterprise telemetry streams.
- **Primary Users**: Security Operations Center (SOC) engineers, DevSecOps pipelines, automated SIEM triage systems.
- **Out-of-Scope Use**: Sole automated termination of user accounts without human-in-the-loop review; prediction on unauthenticated external traffic without historical user baselines.

---

## Factors & Feature Inputs (`v2` Feature Contract)
1. `failed_logins_5m` (5-minute rolling failures)
2. `failed_logins_1h` (1-hour rolling failures)
3. `successful_logins_1h` (1-hour successful authentications)
4. `distinct_users_per_ip_10m` (Unique account attempts per IP)
5. `distinct_hosts_per_user_1h` (Distinct infrastructure destinations)
6. `events_per_user_5m` (Burst activity velocity)
7. `is_new_source_ip` (Unseen IP indicator)
8. `is_new_device` (Unseen device indicator)
9. `login_hour` (Hour of day in UTC)
10. `distance_from_typical_hour` (Deviation from user-specific baseline working hours)
11. `event_rate_ratio` (Burst multiplier vs median)
12. `failure_ratio` (Failure proportion over sliding window)
13. `bytes_sent_log` ($\ln(\text{bytes\_sent} + 1)$)
14. `bytes_received_log` ($\ln(\text{bytes\_received} + 1)$)
15. `session_duration_log` ($\ln(\text{duration} + 1)$)

---

## Quantitative Performance Metrics

### 1. Validation Promotion Gate (Evaluated on $N = 8,451$ Validation Events)
- **Validation Recall**: $90.68\%$ ($146/161$) $\ge 85.0\%$ threshold (**PASSED**)
- **Validation Precision**: $53.87\%$ $\ge 40.0\%$ threshold (**PASSED**)
- **Validation FPR**: $1.51\%$ $\le 5.0\%$ threshold (**PASSED**)
- **Validation Selected Threshold**: `0.6336`

### 2. Untouched Held-Out Test Evaluation ($N = 8,289$ Events)
- **PR-AUC (Average Precision)**: $0.6949$ (95% Bootstrap CI: $[0.648, 0.739]$)
- **Recall @ Threshold (0.6336)**: $93.10\%$ ($162/174$ attacks caught)
- **Precision @ Threshold**: $61.13\%$ ($162/265$ alerts true positives)
- **F1 Score**: $0.7380$
- **False Positive Rate (FPR)**: $1.2693\%$
- **False Alarms per 10,000 Events**: $126.9$

### 3. Recall per Anomaly Taxonomy
- **`brute_force_pattern`**: $100.00\%$
- **`credential_stuffing`**: $100.00\%$
- **`new_device_login`**: $100.00\%$
- **`high_velocity_activity`**: $88.46\%$
- **`unusual_login_time`**: $79.07\%$

### 4. Multi-Seed Stability ($N = 5$ Seeds: 42, 123, 777, 2026, 9001)
- **PR-AUC**: $0.6830 \pm 0.0089$
- **Recall**: $93.56\% \pm 0.43\%$
- **Precision**: $59.85\% \pm 1.35\%$
- **F1 Score**: $0.7299 \pm 0.0088$
- **False Positive Rate**: $1.35\% \pm 0.08\%$

---

## Operational Workload & Alert Burden
- **Held-Out Test FPR**: $1.27\%$
- **Projected Daily False Alarms (1,000,000 Normal Events/Day)**: $\approx 12,693$ alerts/day *(Mathematical projection from test FPR)*
- **Mitigation**: Filter high-confidence alerts with `explainability` contribution flags to prioritize top 10% severe incidents for tier-1 SOC analysts.

---

## Ethical & Operational Considerations
- **Non-Causal Interpretability**: Predictions produce statistical deviation signals rather than claiming causal proof of malicious intent.
- **Night-Shift & Timezone Generalization**: Legitimate night-shift and international employees are modeled with personalized working hour baselines to prevent biased alerting on off-hours workers.
- **Drift Monitoring**: Telemetry distributions must be monitored weekly using Kolmogorov-Smirnov and Population Stability Index ($\text{PSI} \ge 0.25$ triggers automated drift warnings).
