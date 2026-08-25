"""Deterministic latency and throughput benchmarking harness for AnomalyLab."""

import argparse
import json
import os
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import sklearn

from anomalylab.inference.engine import ProductionInferenceEngine
from anomalylab.inference.registry import ModelRegistry
from anomalylab.models.base import BaseAnomalyModel


def collect_system_metadata() -> dict:
    """Collect runtime environment, hardware, and dependency metadata."""
    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
        "python_version": platform.python_version(),
        "sklearn_version": sklearn.__version__,
        "polars_version": pl.__version__,
        "timestamp": datetime.now(UTC).isoformat(),
    }


def run_benchmark(
    models_dir: str = "models",
    n_single_iterations: int = 1000,
    batch_sizes: list[int] = [10, 100, 1000],
    output_path: str = "experiments/inference_benchmark.json",
) -> dict:
    """Execute cold and warm inference benchmarks with detailed percentile reporting."""
    print("\n==========================================")
    print(" Running Deterministic Inference Benchmark")
    print("==========================================")

    system_meta = collect_system_metadata()
    registry = ModelRegistry(models_dir=models_dir)
    manifest = registry.get_production_model()

    if not manifest:
        manifests = registry.list_manifests()
        if manifests:
            manifest = manifests[0]
        else:
            raise RuntimeError("No model found in registry to benchmark.")

    print(f"Benchmarking Model: {manifest.model_version} ({manifest.algorithm})")
    print(
        f"System: {system_meta['os']} | Python {system_meta['python_version']} | CPU Cores: {system_meta['cpu_count']}"
    )

    # 1. Cold Startup & Model Load Benchmark
    cold_load_times = []
    for _ in range(5):
        t0 = time.perf_counter()
        _ = BaseAnomalyModel.load(manifest.artifact_path)
        cold_load_times.append((time.perf_counter() - t0) * 1000.0)

    cold_load_ms = float(np.median(cold_load_times))
    print(f"\n[Cold Startup] Median Model Deserialization: {cold_load_ms:.2f} ms")

    # 2. Initialize Engine with Warmup
    engine = ProductionInferenceEngine(models_dir=models_dir)
    engine.load_champion_model()

    sample_event = {
        "userId": "user_0042.demo",
        "loginHour": 14,
        "eventOutcome": "SUCCESS",
        "failedAttempts": 0,
        "requestCount": 3,
        "bytesSent": 1500,
        "bytesReceived": 5500,
        "sessionDuration": 240,
        "isPrivilegedUser": False,
        "newSourceIp": False,
        "newDevice": False,
    }

    # 3. Single-Event Warm Inference Latency
    print(f"\n[Single Warm Inference] Executing {n_single_iterations} iterations...")
    single_latencies = []
    for _ in range(n_single_iterations):
        t0 = time.perf_counter()
        _ = engine.predict_single(sample_event)
        single_latencies.append((time.perf_counter() - t0) * 1000.0)

    single_latencies_arr = np.array(single_latencies)
    single_p50 = float(np.percentile(single_latencies_arr, 50))
    single_p95 = float(np.percentile(single_latencies_arr, 95))
    single_p99 = float(np.percentile(single_latencies_arr, 99))
    single_max = float(np.max(single_latencies_arr))
    single_mean = float(np.mean(single_latencies_arr))

    print(f"  p50 (Median): {single_p50:.3f} ms")
    print(f"  p95:          {single_p95:.3f} ms")
    print(f"  p99 (Tail):   {single_p99:.3f} ms")
    print(f"  Max Peak:     {single_max:.3f} ms")
    print(f"  Mean:         {single_mean:.3f} ms")

    # 4. Vectorized Batch Throughput
    batch_results = {}
    print("\n[Vectorized Batch Inference]")
    for b_size in batch_sizes:
        batch_events = [sample_event for _ in range(b_size)]

        # Warmup pass
        _ = engine.predict_batch(batch_events)

        batch_times = []
        for _ in range(30):
            t0 = time.perf_counter()
            _ = engine.predict_batch(batch_events)
            batch_times.append((time.perf_counter() - t0) * 1000.0)

        median_batch_ms = float(np.median(batch_times))
        per_item_ms = median_batch_ms / b_size
        throughput_per_sec = float(b_size / (median_batch_ms / 1000.0))

        batch_results[f"batch_{b_size}"] = {
            "batch_size": b_size,
            "total_latency_ms": round(median_batch_ms, 3),
            "per_item_latency_ms": round(per_item_ms, 5),
            "throughput_events_per_sec": round(throughput_per_sec, 1),
        }
        print(
            f"  Batch {b_size:5d}: {median_batch_ms:6.2f} ms total | {per_item_ms:.4f} ms/item | {throughput_per_sec:,.1f} events/sec"
        )

    result = {
        "model_version": manifest.model_version,
        "algorithm": manifest.algorithm,
        "system_metadata": system_meta,
        "cold_load_median_ms": round(cold_load_ms, 2),
        "single_warm_inference_ms": {
            "iterations": n_single_iterations,
            "mean": round(single_mean, 3),
            "p50": round(single_p50, 3),
            "p95": round(single_p95, 3),
            "p99": round(single_p99, 3),
            "max": round(single_max, 3),
        },
        "batch_benchmarks": batch_results,
    }

    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nBenchmark results saved to {out_file}\n")
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run inference latency and throughput benchmark for AnomalyLab."
    )
    parser.add_argument("--models-dir", type=str, default="models")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--output", type=str, default="experiments/inference_benchmark.json")
    args = parser.parse_args()

    run_benchmark(
        models_dir=args.models_dir, n_single_iterations=args.iterations, output_path=args.output
    )


if __name__ == "__main__":
    main()
