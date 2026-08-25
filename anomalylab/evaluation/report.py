"""Evaluation report generator producing machine-readable JSON and human-readable HTML."""

import json
from pathlib import Path
from typing import Any

from anomalylab.evaluation.metrics import EvaluationResult
from anomalylab.evaluation.threshold import ThresholdTuningReport


def generate_html_report(
    model_manifest_data: dict[str, Any],
    eval_result: EvaluationResult,
    threshold_report: ThresholdTuningReport,
    curves_data: dict[str, Any],
    error_analysis_data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Generate a self-contained HTML evaluation report."""
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)

    fp_rows_html = "".join(
        [
            f"""<tr>
            <td class="px-3 py-2 font-mono text-xs">{fp.get("user_id", "")}</td>
            <td class="px-3 py-2 text-xs">{fp.get("category", "")}</td>
            <td class="px-3 py-2 font-mono text-xs text-amber-400">{fp.get("anomaly_score", 0.0):.4f}</td>
            <td class="px-3 py-2 text-xs text-slate-300">{fp.get("explanation", "")}</td>
        </tr>"""
            for fp in error_analysis_data.get("top_false_positives", [])[:5]
        ]
    )

    fn_rows_html = "".join(
        [
            f"""<tr>
            <td class="px-3 py-2 font-mono text-xs">{fn.get("user_id", "")}</td>
            <td class="px-3 py-2 text-xs">{fn.get("ground_truth_type", "")}</td>
            <td class="px-3 py-2 font-mono text-xs text-rose-400">{fn.get("anomaly_score", 0.0):.4f}</td>
            <td class="px-3 py-2 text-xs text-slate-300">{fn.get("explanation", "")}</td>
        </tr>"""
            for fn in error_analysis_data.get("top_false_negatives", [])[:5]
        ]
    )

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>AnomalyLab Evaluation Report — {model_manifest_data.get("model_id", "Model")}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body {{ background-color: #0b0f19; color: #f8fafc; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
    </style>
</head>
<body class="p-8 max-w-6xl mx-auto space-y-8">
    <header class="border-b border-slate-800 pb-6">
        <div class="flex items-center justify-between">
            <div>
                <span class="text-xs uppercase tracking-widest text-cyan-400 font-mono">AnomalyLab ML Evaluation</span>
                <h1 class="text-3xl font-bold text-white mt-1">{model_manifest_data.get("algorithm", "Anomaly Model")} Evaluation</h1>
                <p class="text-sm text-slate-400 mt-1">Model Version: <span class="font-mono text-cyan-300">{model_manifest_data.get("model_version", "")}</span> | Dataset: <span class="font-mono text-slate-300">{model_manifest_data.get("dataset_version", "")}</span></p>
            </div>
            <div class="text-right">
                <div class="inline-flex px-3 py-1 bg-emerald-950/80 border border-emerald-500/30 text-emerald-400 text-xs font-mono rounded-full">HELD-OUT TEST SET</div>
                <div class="text-xs text-slate-500 mt-2">Evaluated at: {model_manifest_data.get("created_at", "")}</div>
            </div>
        </div>
    </header>

    <!-- Key Metrics Grid -->
    <section class="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <span class="text-xs text-slate-400 font-mono">PR-AUC (Avg Precision)</span>
            <div class="text-3xl font-bold text-cyan-400 mt-2 font-mono">{eval_result.pr_auc:.4f}</div>
            <div class="text-xs text-slate-500 mt-1">Primary imbalanced metric</div>
        </div>
        <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <span class="text-xs text-slate-400 font-mono">Recall @ Threshold</span>
            <div class="text-3xl font-bold text-emerald-400 mt-2 font-mono">{eval_result.recall:.2%}</div>
            <div class="text-xs text-slate-500 mt-1">{eval_result.tp} / {eval_result.anomaly_samples} anomalies caught</div>
        </div>
        <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <span class="text-xs text-slate-400 font-mono">Precision</span>
            <div class="text-3xl font-bold text-indigo-400 mt-2 font-mono">{eval_result.precision:.2%}</div>
            <div class="text-xs text-slate-500 mt-1">F1 Score: {eval_result.f1:.4f}</div>
        </div>
        <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-5">
            <span class="text-xs text-slate-400 font-mono">False Positive Rate</span>
            <div class="text-3xl font-bold text-amber-400 mt-2 font-mono">{eval_result.false_positive_rate:.4%}</div>
            <div class="text-xs text-slate-500 mt-1">{eval_result.fp_per_10k:.1f} false alerts / 10k normal</div>
        </div>
    </section>

    <!-- Threshold Tuning & Confusion Matrix -->
    <section class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
            <h3 class="text-sm font-semibold text-slate-200 uppercase tracking-wider font-mono">Threshold Strategy</h3>
            <div class="mt-4 space-y-3 text-sm">
                <div class="flex justify-between border-b border-slate-800/80 pb-2">
                    <span class="text-slate-400">Optimization Goal:</span>
                    <span class="font-mono text-cyan-300">{threshold_report.tuning_strategy}</span>
                </div>
                <div class="flex justify-between border-b border-slate-800/80 pb-2">
                    <span class="text-slate-400">Optimal Cutoff Threshold:</span>
                    <span class="font-mono text-white font-bold">{threshold_report.best_threshold:.4f}</span>
                </div>
                <div class="flex justify-between border-b border-slate-800/80 pb-2">
                    <span class="text-slate-400">Expected Total Cost:</span>
                    <span class="font-mono text-amber-400">{threshold_report.cost_at_threshold:.1f} (FP=1, FN=20)</span>
                </div>
                <div class="flex justify-between pb-1">
                    <span class="text-slate-400">Target Metric Value:</span>
                    <span class="font-mono text-emerald-400">{threshold_report.target_metric_value:.4f}</span>
                </div>
            </div>
        </div>

        <div class="bg-slate-900/60 border border-slate-800 rounded-xl p-6">
            <h3 class="text-sm font-semibold text-slate-200 uppercase tracking-wider font-mono">Confusion Matrix (Held-out Test)</h3>
            <div class="grid grid-cols-2 gap-3 mt-4 text-center font-mono">
                <div class="bg-emerald-950/30 border border-emerald-500/20 p-3 rounded-lg">
                    <div class="text-xs text-emerald-400">True Positives (TP)</div>
                    <div class="text-2xl font-bold text-white mt-1">{eval_result.tp}</div>
                </div>
                <div class="bg-amber-950/30 border border-amber-500/20 p-3 rounded-lg">
                    <div class="text-xs text-amber-400">False Positives (FP)</div>
                    <div class="text-2xl font-bold text-white mt-1">{eval_result.fp}</div>
                </div>
                <div class="bg-rose-950/30 border border-rose-500/20 p-3 rounded-lg">
                    <div class="text-xs text-rose-400">False Negatives (FN)</div>
                    <div class="text-2xl font-bold text-white mt-1">{eval_result.fn}</div>
                </div>
                <div class="bg-slate-800/40 border border-slate-700/30 p-3 rounded-lg">
                    <div class="text-xs text-slate-400">True Negatives (TN)</div>
                    <div class="text-2xl font-bold text-white mt-1">{eval_result.tn}</div>
                </div>
            </div>
        </div>
    </section>

    <!-- Error Analysis Tables -->
    <section class="space-y-6">
        <h2 class="text-lg font-bold text-white">Diagnostic Error Analysis</h2>

        <div class="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden">
            <div class="bg-slate-800/50 px-5 py-3 border-b border-slate-800 flex justify-between items-center">
                <span class="text-xs font-semibold text-amber-400 font-mono uppercase">Top False Positives (Highest Score Normal Events)</span>
                <span class="text-xs text-slate-400">Total FPs: {error_analysis_data.get("total_false_positives", 0)}</span>
            </div>
            <table class="w-full text-left border-collapse">
                <thead class="text-xs text-slate-400 border-b border-slate-800 bg-slate-950/40">
                    <tr>
                        <th class="px-3 py-2">User ID</th>
                        <th class="px-3 py-2">Category</th>
                        <th class="px-3 py-2">Score</th>
                        <th class="px-3 py-2">Explanation</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/60">
                    {fp_rows_html}
                </tbody>
            </table>
        </div>

        <div class="bg-slate-900/60 border border-slate-800 rounded-xl overflow-hidden">
            <div class="bg-slate-800/50 px-5 py-3 border-b border-slate-800 flex justify-between items-center">
                <span class="text-xs font-semibold text-rose-400 font-mono uppercase">Top False Negatives (Lowest Score Anomalies)</span>
                <span class="text-xs text-slate-400">Total FNs: {error_analysis_data.get("total_false_negatives", 0)}</span>
            </div>
            <table class="w-full text-left border-collapse">
                <thead class="text-xs text-slate-400 border-b border-slate-800 bg-slate-950/40">
                    <tr>
                        <th class="px-3 py-2">User ID</th>
                        <th class="px-3 py-2">Ground Truth Anomaly</th>
                        <th class="px-3 py-2">Score</th>
                        <th class="px-3 py-2">Explanation</th>
                    </tr>
                </thead>
                <tbody class="divide-y divide-slate-800/60">
                    {fn_rows_html}
                </tbody>
            </table>
        </div>
    </section>
</body>
</html>
"""
    with open(target, "w", encoding="utf-8") as f:
        f.write(html_content)


def save_evaluation_report(
    eval_result: EvaluationResult,
    threshold_report: ThresholdTuningReport,
    curves_data: dict[str, Any],
    error_analysis_data: dict[str, Any],
    model_manifest_data: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, Any]:
    """Write machine-readable JSON and human-readable HTML evaluation report."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    report_data = {
        "model_id": model_manifest_data.get("model_id"),
        "model_version": model_manifest_data.get("model_version"),
        "algorithm": model_manifest_data.get("algorithm"),
        "dataset_version": model_manifest_data.get("dataset_version"),
        "feature_version": model_manifest_data.get("feature_version"),
        "metrics": eval_result.to_dict(),
        "threshold_tuning": {
            "best_threshold": threshold_report.best_threshold,
            "strategy": threshold_report.tuning_strategy,
            "precision": threshold_report.precision_at_threshold,
            "recall": threshold_report.recall_at_threshold,
            "f1": threshold_report.f1_at_threshold,
            "fpr": threshold_report.fpr_at_threshold,
            "fp_per_10k": threshold_report.fp_per_10k_at_threshold,
            "expected_cost": threshold_report.cost_at_threshold,
        },
        "curves": curves_data,
        "error_analysis": {
            "total_false_positives": error_analysis_data.get("total_false_positives"),
            "total_false_negatives": error_analysis_data.get("total_false_negatives"),
            "top_false_positives": error_analysis_data.get("top_false_positives"),
            "top_false_negatives": error_analysis_data.get("top_false_negatives"),
            "fp_categories": error_analysis_data.get("fp_categories"),
            "fn_categories": error_analysis_data.get("fn_categories"),
        },
    }

    json_path = out_dir / "evaluation.json"
    with open(json_path, "w") as f:
        json.dump(report_data, f, indent=2)

    html_path = out_dir / "evaluation_report.html"
    generate_html_report(
        model_manifest_data=model_manifest_data,
        eval_result=eval_result,
        threshold_report=threshold_report,
        curves_data=curves_data,
        error_analysis_data=error_analysis_data,
        output_path=html_path,
    )

    return report_data
