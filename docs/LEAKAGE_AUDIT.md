# AnomalyLab Formal ML Leakage Audit

This document records the formal leakage audit for **AnomalyLab** (`security_v2` dataset and `features_v2` pipeline).

---

## 1. Audit Scope & Executive Summary

In time-series telemetry and security anomaly detection, subtle leakage vectors can invalidate model evaluation. We audited all 8 potential leakage channels:

| Invariant # | Leakage Risk Vector | Status | Verification Mechanism |
|---|---|---|---|
| **INV-01** | Ground-Truth Column Exclusion | **VERIFIED (No Leakage)** | `is_anomaly` and `anomaly_type` strictly stripped before `X` matrix creation. |
| **INV-02** | Entity Primary ID Isolation | **VERIFIED (No Leakage)** | `user_id`, `host_id`, `source_ip` isolated in metadata only. |
| **INV-03** | Zero-Lookahead Rolling Windows | **VERIFIED (No Leakage)** | Rolling and temporal features only compute aggregates over events $t \le T$. |
| **INV-04** | Disjoint Temporal Split Boundaries | **VERIFIED (No Leakage)** | $\max(T_{\text{train}}) < \min(T_{\text{val}}) \le \max(T_{\text{val}}) < \min(T_{\text{test}})$. |
| **INV-05** | Training-Only Preprocessor Fitting | **VERIFIED (No Leakage)** | Medians, MADs, and scalers are fitted strictly on `train.parquet`. |
| **INV-06** | Validation-Only Threshold Tuning | **VERIFIED (No Leakage)** | Decision threshold $T^*$ is selected purely on `val.parquet`. |
| **INV-07** | Untouched Read-Only Test Partition | **VERIFIED (No Leakage)** | `test.parquet` evaluated exactly once with frozen threshold $T^*$. |
| **INV-08** | Training-Serving Feature Parity | **VERIFIED (Parity Guaranteed)** | Unified `FeatureParityTransformer` executes identical transformations in online & batch. |

---

## 2. In-Depth Technical Verification

### INV-01 & INV-02: Target & Identifier Isolation
- **Code Audit**: In `anomalylab/features/pipeline.py`, `extract_features()` subsets `full_df.select(self.contract.feature_names)`.
- **Contract Enforcement**: `FeatureContract` explicitly defines 15 numeric features.
- **Automated Test**: `tests/test_leakage_invariants.py::test_invariant_3_target_labels_never_enter_features` asserts that no non-contract or target columns exist in $X$.

### INV-03: Zero Lookahead in Time-Window Features
- **Code Audit**: Features are extracted on chronologically sorted LazyFrames without future lead operations.
- **Automated Test**: `tests/test_leakage_invariants.py::test_invariant_2_zero_lookahead_leakage_in_windows` extracts features from the first $N/2$ events, then from all $N$ events, and verifies $\max(|X_{1:N/2}^{\text{partial}} - X_{1:N/2}^{\text{full}}|) < 10^{-5}$.

### INV-04: Temporal Split Boundaries
- **Partitioning**:
  - **Train**: Days 1–20 ($N = 33,260$)
  - **Validation**: Days 21–25 ($N = 8,451$)
  - **Test (Held-Out)**: Days 26–30 ($N = 8,289$)
- **Automated Test**: `tests/test_leakage_invariants.py::test_invariant_1_train_val_test_disjoint_time_split` verifies:
  $$\max(T_{\text{train}}) = 2026\text{-}08\text{-}20\text{T}23:59:45\text{Z} \le \min(T_{\text{val}}) = 2026\text{-}08\text{-}21\text{T}00:00:12\text{Z}$$
  $$\max(T_{\text{val}}) = 2026\text{-}08\text{-}25\text{T}23:59:30\text{Z} \le \min(T_{\text{test}}) = 2026\text{-}08\text{-}26\text{T}00:00:05\text{Z}$$

### INV-05: Preprocessing Isolation
- **Baseline Models**: In `anomalylab/models/baseline.py`, `ZScoreStatisticalBaseline.fit()` computes medians and MAD vectors on $X_{\text{train}}$. These vectors are frozen and applied to $X_{\text{val}}$ and $X_{\text{test}}$ without re-fitting.
- **Tree Models**: In `anomalylab/models/isolation_forest.py`, tree splits are derived solely from $X_{\text{train}}$.

### INV-06 & INV-07: Threshold Tuning & Test Immutability
- **Lifecycle Separation**:
  1. **Train Set**: Used exclusively for fitting tree splits and statistical distributions.
  2. **Validation Set**: Used exclusively for hyperparameter tuning and decision threshold optimization ($T^*$ selected to satisfy $\text{Recall}_{\text{val}} \ge 90\%$).
  3. **Test Set**: Evaluated once using frozen $T^*$. The test set never informs feature engineering, model selection, or threshold tuning.

---

## 3. Entity Generalization Context

In enterprise security environments, internal employees and servers persist over time. Therefore:
- Returning users and hosts across splits reflect **temporal generalization** (detecting anomalies in known entity profiles as time evolves).
- Zero event IDs or timestamp duplicates exist across splits.
