# AnomalyLab Inference Performance & Latency Benchmarks

This report documents reproducible cold-startup and warm inference latency benchmarks for **AnomalyLab**'s production champion model (`Threshold-Tuned Isolation Forest`).

---

## 1. Benchmark Execution Environment

- **Operating System**: Windows 11 Enterprise (10.0.26200-SP0)
- **Host Architecture**: AMD64 / 16 Logical Processors
- **Python Runtime**: CPython 3.12.11
- **Key Dependencies**:
  - `scikit-learn`: 1.7.1
  - `polars`: 1.42.1
  - `numpy`: 2.2.3
- **Active Champion Model**: `isolation_forest_v2_20260821_001323` (250 estimators, 15 features)

---

## 2. Cold Startup Latency

Cold startup measures the total elapsed time required to deserialize the `.joblib` model artifact from disk and initialize feature transformers:

| Metric | Measured Duration | SLA Threshold | Status |
|---|---|---|---|
| **Median Model Deserialization** | **210.81 ms** | < 1,000 ms | **PASS** |
| **Engine Warmup (10 vectors)** | **24.50 ms** | < 200 ms | **PASS** |

---

## 3. Single-Event Warm Inference Latency ($N = 1,000$)

Evaluated on single JSON telemetry payloads hitting the full preprocessing and feature parity pipeline:

| Percentile | Latency (ms) | Target SLA (ms) | Status |
|---|---|---|---|
| **p50 (Median)** | **22.31 ms** | < 35.0 ms | **PASS** |
| **p95 (Upper SLA)** | **28.58 ms** | < 50.0 ms | **PASS** |
| **p99 (Tail Latency)** | **30.00 ms** | < 60.0 ms | **PASS** |
| **Mean** | **21.85 ms** | — | — |
| **Maximum Peak** | **32.56 ms** | < 100.0 ms | **PASS** |

---

## 4. Vectorized Batch Throughput & Scaling

Batch inference measures vectorization efficiency across variable workload spikes:

| Batch Size | Total Latency (ms) | Per-Event Latency (ms) | Throughput (Events/sec) |
|---|---|---|---|
| **Batch 10** | 21.56 ms | 2.156 ms | 463.7 / sec |
| **Batch 100** | 21.47 ms | 0.215 ms | 4,658.0 / sec |
| **Batch 1,000** | 48.18 ms | 0.048 ms | **20,754.9 / sec** |

---

## 5. Benchmark Reproducibility

Run this benchmark locally on any environment via:

```bash
uv run python -m anomalylab.benchmarks.inference_benchmark --iterations 1000
```
