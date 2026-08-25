# AnomalyLab — Production-Grade Anomaly Detection & ML Analytics Platform

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Polars](https://img.shields.io/badge/data-polars-ff4b4b.svg)](https://pola.rs/)
[![FastAPI](https://img.shields.io/badge/api-fastapi-009688.svg)](https://fastapi.tiangolo.com/)
[![Next.js 15](https://img.shields.io/badge/frontend-next.js%2015-black.svg)](https://nextjs.org/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

**AnomalyLab** is an end-to-end, measurable, and fully reproducible machine learning platform for detecting security intrusions and anomalous behavioral deviations in enterprise telemetry streams.

---

## 🌟 Key Engineering Highlights

- **Three-Phase ML Evaluation Lifecycle**: Strictly isolates Model Fitting (Train: Days 1–20), Validation Policy Gating & Threshold Tuning (Validation: Days 21–25), and Untouched Evaluation (Test: Days 26–30).
- **Leakage-Free Temporal Invariants**: Formally audited across 8 leakage vectors ([docs/LEAKAGE_AUDIT.md](docs/LEAKAGE_AUDIT.md)) with zero-lookahead rolling window feature extraction.
- **Quantitative Model Promotion Gate**: Enforces pre-deployment validation gates ($\text{Recall}_{\text{val}} \ge 85\%$, $\text{FPR}_{\text{val}} \le 5.0\%$, $\text{Precision}_{\text{val}} \ge 40.0\%$) before allowing promotion to Candidate or Production Champion.
- **Operational False-Positive Accounting**: Translates academic metrics into tangible SOC burden (reporting FP per 1k, 10k, 100k events and projected workload for 1,000,000 normal events/day).
- **Multi-Seed Stability & Bootstrap CIs**: Validated across 5 deterministic random seeds (42, 123, 777, 2026, 9001) and 500-sample bootstrap 95% confidence intervals.
- **High-Throughput Vectorized Inference**: Thread-safe FastAPI inference engine delivering median single-event inference of **22.3 ms** and vectorized batch throughput of **20,750 events/sec** ([docs/PERFORMANCE.md](docs/PERFORMANCE.md)).
- **Statistical Drift & Non-Causal Explainability**: Integrated Kolmogorov-Smirnov (KS-test), Population Stability Index (PSI), and reference population feature deviation scoring for SOC triage.
- **Dark-Mode Telemetry Operations Console**: Built with Next.js 15 (App Router), React 19, TypeScript, Tailwind CSS, Lucide icons, and TanStack Query.

---

## 📊 Authoritative Benchmark Matrix (`security_v2` Telemetry)

All metrics below are generated directly from reproducible run artifacts in `models/` and `experiments/`.

| Algorithm | Dataset | Features | Val Recall | Val Precision | Val FPR | Test PR-AUC | Test Recall | Test Precision | Test F1 | Test FPR | FP / 10k | Threshold | Stage |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **Threshold-Tuned Isolation Forest** | `security_v2` | `v2` | 90.7% | 53.9% | 1.51% | **69.49%** | **93.10%** | **61.13%** | **0.7380** | **1.27%** | **126.9** | `0.6336` | **PRODUCTION (Champion)** |
| **Robust Z-Score Baseline** | `security_v2` | `v2` | 90.1% | 12.5% | 12.26% | 53.72% | 88.51% | 13.46% | 0.2337 | 12.20% | 1220.0 | `0.6930` | REJECTED (FPR > 5%) |
| **Local Outlier Factor (LOF)** | `security_v2` | `v2` | 90.7% | 2.0% | 86.78% | 3.35% | 90.80% | 2.21% | 0.0431 | 86.27% | 8627.2 | `0.1288` | REJECTED (FPR > 5%) |

### Multi-Seed Stability Summary (Isolation Forest across 5 seeds: 42, 123, 777, 2026, 9001)
- **PR-AUC**: $0.6830 \pm 0.0089$ (min: $0.6700$, max: $0.6949$)
- **Recall**: $93.56\% \pm 0.43\%$ (min: $93.10\%$, max: $94.25\%$)
- **Precision**: $59.85\% \pm 1.35\%$ (min: $58.16\%$, max: $61.60\%$)
- **F1 Score**: $0.7299 \pm 0.0088$
- **False Positive Rate**: $1.35\% \pm 0.08\%$

### Operational Alert Burden (Production Champion)
- **FPR on Held-Out Test Set**: 1.27%
- **False Alarms per 10,000 Normal Events**: 126.9
- **Projected 24-Hour SOC Burden (1,000,000 normal events/day)**: $\approx 12,693$ alerts/day *(Mathematical projection: $\text{FPR} \times 10^6$)*

---

## 🚀 Quickstart

### Prerequisites
- Python 3.12+ with `uv` (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Node.js 20+ & npm

### 1. Installation & Setup
```bash
# Clone repository
git clone https://github.com/anomaly-lab/anomalylab.git
cd anomalylab

# Sync Python dependencies
uv sync

# Install Next.js frontend dependencies
cd apps/web && npm install && cd ../..
```

### 2. End-to-End Pipeline Execution
```bash
# 1. Generate deterministic synthetic enterprise telemetry (v2)
uv run python -m anomalylab.data.generate --config configs/data_security_v2.yaml

# 2. Extract features with temporal split isolation
uv run python -m anomalylab.features.pipeline --config configs/features_v2.yaml

# 3. Train all models and tune thresholds on Validation set
uv run python -m anomalylab.training.train --config configs/isolation_forest_security.yaml
uv run python -m anomalylab.training.train --config configs/baseline_model.yaml
uv run python -m anomalylab.training.train --config configs/lof_security.yaml

# 4. Run Multi-Seed Stability Benchmark
uv run python -m anomalylab.training.multiseed --seeds 42 123 777 2026 9001

# 5. Run Inference Latency & Batch Throughput Benchmark
uv run python -m anomalylab.benchmarks.inference_benchmark --iterations 1000

# 6. Execute Test Suite
uv run pytest tests/ -v
```

### 3. Launch Services
```bash
# Terminal 1: FastAPI Backend (Port 8000)
uv run uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Next.js Dashboard (Port 3000)
cd apps/web && npm run dev
```

Visit the console at `http://localhost:3000` and API docs at `http://localhost:8000/docs`.

---

## 📚 Technical Documentation Index

- [docs/LEAKAGE_AUDIT.md](docs/LEAKAGE_AUDIT.md): Formal verification of 8 temporal, target, and preprocessor leakage invariants.
- [docs/PERFORMANCE.md](docs/PERFORMANCE.md): Cold start, p50/p95/p99 single-event latency, and vectorized batch throughput benchmarks.
- [MODEL_CARD.md](MODEL_CARD.md): Production model card for Threshold-Tuned Isolation Forest.
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md): Detailed lineage of baseline, LOF, and tree models.
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md): System architecture, inference engine lifecycle, and Prometheus metrics.
- [docs/DATA.md](docs/DATA.md): Feature dictionary, window parameters, and synthetic data taxonomy.
