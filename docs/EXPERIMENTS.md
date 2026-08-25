# AnomalyLab Empirical Evaluation & Experimentation

## 1. Class Imbalance & Three-Phase ML Governance

In enterprise security analytics and fraud monitoring, anomalous events represent approximately **2.0%** of total system volume.

Traditional metrics such as **Accuracy** or **ROC-AUC** are fundamentally deceptive under extreme class imbalance:
- A trivial constant model predicting `False` achieves **98.0% Accuracy**.
- High ROC-AUC can mask thousands of false alarms because the large pool of True Negatives inflates the denominator of the False Positive Rate ($FPR = \frac{FP}{FP + TN}$).

Therefore, **AnomalyLab** enforces a strict three-phase evaluation protocol:
1. **Model Fitting (Train Partition: Days 1–20, $N = 33,260$)**: Models learn representations and split boundaries strictly from historical data.
2. **Validation Policy Gate (Validation Partition: Days 21–25, $N = 8,451$)**:
   - Decision threshold $T^*$ is optimized to achieve target recall ($\text{Recall}_{\text{val}} \ge 90\%$) while minimizing cost.
   - Enforces pre-deployment validation gate: $\text{Recall}_{\text{val}} \ge 85\%$, $\text{FPR}_{\text{val}} \le 5.0\%$, $\text{Precision}_{\text{val}} \ge 40.0\%$. Models failing these criteria are classified as `REJECTED`.
3. **Untouched Evaluation (Held-Out Test Partition: Days 26–30, $N = 8,289$)**:
   - The selected threshold $T^*$ is frozen and applied once to the future test set.
   - Reported metrics include **Precision-Recall AUC (PR-AUC)**, **F1 Score**, **FPR**, and **False Alarms per 10,000 Events**.

---

## 2. Authoritative Benchmark Matrix (`security_v2` Telemetry)

Evaluated on $N = 8,289$ held-out test events (174 true anomalies, 8,115 normal events):

| Model Algorithm | Val Recall | Val FPR | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test FPR | FP / 10k Events | Frozen Threshold | Stage |
|---|---|---|---|---|---|---|---|---|---|---|
| **Threshold-Tuned Isolation Forest** | 90.68% | 1.51% | **69.49%** | **93.10%** | **61.13%** | **0.7380** | **1.27%** | **126.9** | `0.6336` | **PRODUCTION (Champion)** |
| **Robust Z-Score Baseline** | 90.06% | 12.26% | 53.72% | 88.51% | 13.46% | 0.2337 | 12.20% | 1220.0 | `0.6930` | REJECTED (FPR > 5%) |
| **Local Outlier Factor (LOF)** | 90.68% | 86.78% | 3.35% | 90.80% | 2.21% | 0.0431 | 86.27% | 8627.2 | `0.1288` | REJECTED (FPR > 5%) |

---

## 3. Scientific Analysis & Root Cause Takeaways

### 1. Isolation Forest (Production Champion)
- Captures **93.10% of true security attacks** ($162/174$) while maintaining **61.13% Precision** and an operational FPR of **1.27%** (126.9 false alarms per 10k events).
- Isolates complex multi-feature anomalies (such as off-hours logins from new devices with elevated byte volume) through recursive feature partitions.

### 2. Robust Z-Score Baseline
- In simplistic synthetic datasets (v1), extreme single-feature outliers allowed Z-score to achieve 99.99% PR-AUC.
- Under realistic enterprise overlap (`security_v2` with legitimate night shifts, password typos, and scheduled batch jobs), the Z-Score baseline achieves **53.72% PR-AUC** with an elevated **12.20% FPR** (1,220 false alarms / 10k).
- Because 12.20% FPR exceeds the 5.0% maximum policy constraint, the baseline is correctly classified as `REJECTED`.

### 3. Local Outlier Factor (LOF)
- Suffers from the "curse of dimensionality" across 15 correlated feature dimensions. Local neighborhood density estimates flatten, causing normal background points to receive artificially high anomaly scores.
- Tuning for $\ge 90\%$ recall forces the threshold so low that **86.27% of normal events are flagged**, generating 8,627 false alarms per 10k events.
- Correctly classified as `REJECTED`.

---

## 4. Multi-Seed Stability Benchmark

Isolation Forest was evaluated across 5 distinct random seeds (42, 123, 777, 2026, 9001):

| Random Seed | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test FPR | Threshold |
|---|---|---|---|---|---|---|
| **Seed 42** | 0.6949 | 93.10% | 61.13% | 0.7380 | 1.27% | 0.6336 |
| **Seed 123** | 0.6700 | 93.68% | 58.63% | 0.7212 | 1.42% | 0.6237 |
| **Seed 777** | 0.6839 | 94.25% | 58.16% | 0.7193 | 1.45% | 0.6138 |
| **Seed 2026** | 0.6764 | 93.68% | 59.71% | 0.7293 | 1.36% | 0.6237 |
| **Seed 9001** | 0.6896 | 93.10% | 61.60% | 0.7414 | 1.24% | 0.6336 |
| **Mean ± Std** | **0.6830 ± 0.0089** | **93.56% ± 0.43%** | **59.85% ± 1.35%** | **0.7299 ± 0.0088** | **1.35% ± 0.08%** | **0.6257 ± 0.0076** |

---

## 5. Operational Burden & 24-Hour SOC Workload

Under an enterprise ingestion volume of **1,000,000 normal events per day**:
- **Isolation Forest (Champion)**: $\approx 12,693$ alerts/day *(Mathematical projection from test FPR 1.27%)*
- **Z-Score Baseline**: $\approx 121,996$ alerts/day (9.6× higher burden)
- **Local Outlier Factor**: $\approx 862,723$ alerts/day (Complete operational denial of service)
