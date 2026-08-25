"""Evaluation reports and experiment comparison endpoints."""

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import HTMLResponse

router = APIRouter(prefix="/api/v1/evaluations", tags=["Evaluations"])


@router.get("", response_model=list[dict[str, Any]])
def list_evaluations():
    """List all experiment evaluation reports."""
    exp_dir = Path("experiments")
    evals = []
    if exp_dir.exists():
        for sub in exp_dir.iterdir():
            eval_file = sub / "evaluation.json"
            if eval_file.exists():
                try:
                    with open(eval_file) as f:
                        evals.append(json.load(f))
                except Exception:
                    continue
    return evals


@router.get("/{model_version}", response_model=dict[str, Any])
def get_evaluation(model_version: str):
    """Retrieve detailed evaluation JSON report for a specific experiment run."""
    eval_file = Path("experiments") / model_version / "evaluation.json"
    if not eval_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "EVALUATION_NOT_FOUND",
                "message": f"No evaluation found for {model_version}",
            },
        )
    with open(eval_file) as f:
        return json.load(f)


@router.get("/{model_version}/report.html", response_class=HTMLResponse)
def get_evaluation_html(model_version: str):
    """Retrieve human-readable HTML evaluation report for sharing or printing."""
    html_file = Path("experiments") / model_version / "evaluation_report.html"
    if not html_file.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "code": "REPORT_NOT_FOUND",
                "message": f"No HTML report found for {model_version}",
            },
        )
    with open(html_file, encoding="utf-8") as f:
        return f.read()
