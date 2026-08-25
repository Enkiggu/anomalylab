"""Dataset catalog and data lineage endpoints."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/api/v1/datasets", tags=["Dataset Catalog"])


@router.get("", response_model=list[dict[str, Any]])
def list_datasets():
    """List all registered dataset manifests and summaries."""
    datasets = []
    raw_dir = Path("data/raw")
    if raw_dir.exists():
        for manifest_file in raw_dir.glob("*_manifest.json"):
            try:
                with open(manifest_file) as f:
                    datasets.append(json.load(f))
            except Exception:
                continue
    return datasets


@router.get("/summary", response_model=dict[str, Any])
def get_features_summary():
    """Get time-split summary and feature counts."""
    sum_path = Path("data/features/features_summary.json")
    if not sum_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "DATASET_NOT_FOUND", "message": "Feature summary not found."},
        )
    with open(sum_path) as f:
        return json.load(f)


@router.get("/quality", response_model=dict[str, Any])
def get_quality_report():
    """Retrieve raw dataset quality report."""
    q_path = Path("data/raw/quality_report.json")
    if not q_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "REPORT_NOT_FOUND", "message": "Quality report not found."},
        )
    with open(q_path) as f:
        return json.load(f)
