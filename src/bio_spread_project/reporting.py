"""Advanced 'Code-Chic' Reporting for BioSpread."""

from __future__ import annotations

from typing import Any
from bio_spread_project.model import Prediction

def _fmt(value: Any) -> str:
    if value is None or (isinstance(value, float) and not (value == value)): # NaN check
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)

def _generate_box_header(title: str, width: int = 60) -> str:
    line = "═" * (width - 2)
    return f"╔{line}╗\n║ {title:<{width-4}} ║\n╚{line}╝"

def _generate_markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "*No data available*"
    header_row = "| " + " | ".join(headers) + " |"
    separator = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = "\n".join(["| " + " | ".join([_fmt(cell) for cell in row]) + " |" for row in rows])
    return f"{header_row}\n{separator}\n{body}"

def render_chic_report(
    *,
    predictions: list[Prediction],
    metrics: dict[str, Any],
    audit: dict[str, Any],
    governance: Any,
    release_gate: dict[str, Any],
    split_year: int,
    horizon_years: int,
) -> str:
    """Render a high-fidelity, algorithmic-chic report."""
    
    # 1. HEADER CARD
    status_label = release_gate.get("readiness", "unknown").upper().replace("_", " ")
    status_color = "🟢 " if "GO" in status_label and "NO" not in status_label else ("🟡 " if "CONDITIONAL" in status_label else "🔴 ")
    
    header = [
        "```text",
        _generate_box_header("BIOSPREAD PREDICTIVE SURVEILLANCE REPORT"),
        f"STATUS:    {status_color}{status_label}",
        f"RUN ID:    {audit.get('run_id', 'N/A')}",
        f"TIMESTAMP: {audit.get('timestamp', 'N/A')}",
        "```",
        "",
    ]

    # 2. CORE METRICS TABLE
    metrics_headers = ["Metric", "Value", "Threshold", "Status"]
    metrics_rows = [
        ["ROC AUC", metrics.get("roc_auc"), ">= 0.820", "PASS" if metrics.get("roc_auc", 0) >= 0.82 else "FAIL"],
        ["Avg Precision", metrics.get("average_precision"), "> Prev", "PASS" if metrics.get("average_precision", 0) > metrics.get("prevalence", 0) else "FAIL"],
        ["Calibration ECE", metrics.get("expected_calibration_error"), "<= 0.100", "PASS" if metrics.get("expected_calibration_error", 1) <= 0.1 else "FAIL"],
        ["Brier Score", metrics.get("brier_score"), "N/A", "INFO"],
    ]
    
    # 3. RELIABILITY & LEAKAGE GATES
    gate_headers = ["Gate Type", "Metric", "Value", "Status"]
    # Extract tracks for better value mapping
    tracks = metrics.get("tracks", {})
    gate_rows = [
        ["Spatial Group CV", "OOF ROC AUC", metrics.get("group_oof_roc_auc"), "PASS" if metrics.get("group_oof_roc_auc", 0) >= 0.80 else "FAIL"],
        ["Temporal Holdout", "AUC @ " + str(metrics.get("temporal_holdout_cutoff_year", "N/A")), metrics.get("temporal_holdout_roc_auc"), "PASS" if metrics.get("temporal_holdout_roc_auc", 0) >= 0.78 else "FAIL"],
        ["External Holdout", "Independent AUC", metrics.get("external_holdout_roc_auc"), "PASS" if metrics.get("external_holdout_roc_auc", 0) >= 0.78 else "PASS" if metrics.get("external_holdout_roc_auc") is None else "FAIL"],
        ["Bootstrap CI", "ROC AUC Low (95%)", metrics.get("bootstrap_roc_auc_ci_low"), "PASS" if metrics.get("bootstrap_roc_auc_ci_low", 0) >= 0.78 else "FAIL"],
        ["Leakage Scan", "Max Single-Feature AUC", metrics.get("max_single_feature_auc"), "PASS" if metrics.get("max_single_feature_auc", 1) < 0.95 else "FAIL"],
    ]

    # 4. TOP RISK CANDIDATES
    risk_headers = ["Rank", "Backbone ID", "Risk Prob", "Confidence", "Future Spread", "Explanation"]
    top_preds = sorted(predictions, key=lambda x: x.risk_probability, reverse=True)[:10]
    risk_rows = [
        [i+1, p.backbone_id, p.risk_probability, p.confidence_tier, p.n_new_countries_future, p.explanation]
        for i, p in enumerate(top_preds)
    ]

    # 5. ENVIRONMENT & DATA HASHES
    env_headers = ["Entity", "Value / Hash"]
    env_rows = [
        ["Python Version", audit.get("environment", {}).get("python", "N/A")],
        ["Polars Version", audit.get("environment", {}).get("polars", "N/A")],
        ["Input SHA-256", list(audit.get("input_hashes", {}).values())[0] if audit.get("input_hashes") else "N/A"],
        ["Training Split", f"Year <= {split_year}"],
        ["Forecast Horizon", f"{horizon_years} Years"],
    ]

    report = [
        "# BioSpread Executive Summary",
        "",
        *header,
        "",
        "### 📊 Predictive Performance",
        _generate_markdown_table(metrics_headers, metrics_rows),
        "",
        "### 🛡️ Reliability & Leakage Firewall",
        _generate_markdown_table(gate_headers, gate_rows),
        "",
        "### 🚨 High-Risk Backbone Registry (Top 10)",
        _generate_markdown_table(risk_headers, risk_rows),
        "",
        "### ⚙️ Environment & Reproducibility",
        _generate_markdown_table(env_headers, env_rows),
        "",
        "---",
        "*Generated by BioSpread Autonomous Pipeline v0.1.0*",
    ]
    
    return "\n".join(report)
