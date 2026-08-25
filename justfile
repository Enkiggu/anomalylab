# AnomalyLab Task Runner

set shell := ["powershell.exe", "-NoLogo", "-Command"]

default:
    @just --list

# Install dependencies and sync virtual environment
install:
    uv sync

# Generate deterministic synthetic telemetry dataset
generate-data:
    uv run python -m anomalylab.data.generate --config configs/data_security_v2.yaml

# Validate dataset quality and fingerprint
validate-data input="data/raw/security_events_v2.parquet":
    uv run python -m anomalylab.data.validator --input {{input}}

# Run Polars feature engineering pipeline
prepare-data:
    uv run python -m anomalylab.features.pipeline --config configs/features_v2.yaml

# Train baseline statistical model
train-baseline:
    uv run python -m anomalylab.training.train --config configs/baseline_model.yaml

# Train Isolation Forest model
train-iforest:
    uv run python -m anomalylab.training.train --config configs/isolation_forest_security.yaml

# Train LOF alternative model
train-lof:
    uv run python -m anomalylab.training.train --config configs/lof_security.yaml

# Train all models and produce comparative evaluation
train-all: train-baseline train-iforest train-lof

# Run Multi-Seed Stability Benchmark across 5 seeds
multiseed:
    uv run python -m anomalylab.training.multiseed --seeds 42 123 777 2026 9001

# Run inference latency and throughput benchmark
benchmark:
    uv run python -m anomalylab.benchmarks.inference_benchmark --iterations 1000

# Generate Markdown tables from authoritative model manifests
report-metrics:
    uv run python -m anomalylab.reporting.generate_readme_metrics

# Run statistical drift detection
drift ref="data/features/train.parquet" curr="data/features/test.parquet":
    uv run python -m anomalylab.drift.detector --reference {{ref}} --current {{curr}}

# Run unit and invariant tests
test:
    uv run pytest tests/ -v

# Full one-command evaluation
evaluate: train-all multiseed benchmark report-metrics

# Full one-command verification: lint, test, multiseed, and evaluate
verify: lint test multiseed benchmark

# Run linting and formatting checks
lint:
    uv run ruff check .
    uv run ruff format --check .

# Format code
format:
    uv run ruff format .
    uv run ruff check --fix .

# Start FastAPI server
serve port="8000":
    uv run uvicorn apps.api.main:app --host 0.0.0.0 --port {{port}} --reload

# Start Web dashboard
web:
    cd apps/web && npm run dev

# One-command full demo: generate data, build features, train all models, compute drift, start services
demo:
    uv run python -m scripts.demo
