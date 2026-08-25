"""Data and prediction drift monitoring endpoints."""

import json
from pathlib import Path
from typing import Any

import polars as pl
from fastapi import APIRouter, HTTPException, status

from anomalylab.drift.detector import DriftDetector

router = APIRouter(prefix="/api/v1/drift", tags=["Drift Monitoring"])


def get_engine():
    from apps.api.main import inference_engine

    return inference_engine


@router.get("/latest", response_model=dict[str, Any])
def get_latest_drift_report():
    """Retrieve the latest computed dataset & score drift report."""
    report_path = Path("monitoring/drift/latest_drift_report.json")
    if not report_path.exists():
        # Compute on demand if not existing
        return run_drift_check()
    with open(report_path) as f:
        return json.load(f)


@router.post("/run", response_model=dict[str, Any])
def run_drift_check():
    """Trigger statistical drift analysis between training baseline and test/production stream."""
    ref_path = Path("data/features/train.parquet")
    curr_path = Path("data/features/test.parquet")

    if not ref_path.exists() or not curr_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "DATASET_NOT_FOUND",
                "message": "Feature datasets not found for drift calculation.",
            },
        )

    ref_df = pl.read_parquet(ref_path)
    curr_df = pl.read_parquet(curr_path)

    engine = get_engine()
    model = engine.current_model if engine and engine.is_ready else None

    detector = DriftDetector()
    report = detector.run_drift_analysis(ref_df, curr_df, model=model)

    out_dir = Path("monitoring/drift")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "latest_drift_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
