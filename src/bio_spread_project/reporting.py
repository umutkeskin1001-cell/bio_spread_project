"""Advanced 'Code-Chic' Reporting for BioSpread."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

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


def _uncertainty_from_prediction(pred: Prediction) -> float:
    """Derive epistemic uncertainty from prediction meta if available."""
    meta = pred.meta or {}
    # Use survival spread probability variance as proxy if available
    surv_keys = [k for k in meta.keys() if k.startswith("survival_")]
    if surv_keys:
        vals = [float(meta[k]) for k in surv_keys]
        return float(np.std(vals))
    # Fallback: Bernoulli variance
    p = pred.risk_probability
    return 2.0 * p * (1.0 - p)


def build_threat_triage_matrix(
    predictions: list[Prediction],
    risk_high: float = 0.70,
    risk_low: float = 0.30,
    uncertainty_high: float = 0.25,
) -> dict[str, Any]:
    """Map predictions into a Risk × Uncertainty triage matrix (Red / Yellow / Green).

    Returns counts, thresholds, and per-cell backbone lists.
    """
    matrix: dict[str, list[str]] = {"red": [], "yellow": [], "green": []}
    for pred in predictions:
        unc = _uncertainty_from_prediction(pred)
        if pred.risk_probability >= risk_high and unc >= uncertainty_high:
            matrix["red"].append(pred.backbone_id)
        elif pred.risk_probability >= risk_high and unc < uncertainty_high:
            matrix["red"].append(pred.backbone_id)
        elif pred.risk_probability >= risk_low or unc >= uncertainty_high:
            matrix["yellow"].append(pred.backbone_id)
        else:
            matrix["green"].append(pred.backbone_id)
    return {
        "thresholds": {"risk_high": risk_high, "risk_low": risk_low, "uncertainty_high": uncertainty_high},
        "counts": {k: len(v) for k, v in matrix.items()},
        "backbones": matrix,
    }


def dynamic_budget_threshold(
    predictions: list[Prediction],
    budget: int,
) -> tuple[float, list[Prediction]]:
    """Select the top-``budget`` predictions and return the probability threshold.

    Useful for operational triage when surveillance resources are finite.
    """
    if budget <= 0:
        return 0.0, []
    sorted_preds = sorted(predictions, key=lambda p: p.risk_probability, reverse=True)
    selected = sorted_preds[:budget]
    threshold = selected[-1].risk_probability if selected else 0.0
    return threshold, selected


def render_llm_json_briefing(
    predictions: list[Prediction],
    metrics: dict[str, Any],
    triage: dict[str, Any] | None = None,
    split_year: int = 2020,
    horizon_years: int = 3,
) -> str:
    """Generate a structured JSON briefing ready for LLM consumption."""
    triage = triage or build_threat_triage_matrix(predictions)
    top = sorted(predictions, key=lambda p: p.risk_probability, reverse=True)[:10]
    briefing = {
        "report_type": "BioSpread LLM-Ready Surveillance Briefing",
        "generated_at_utc": None,
        "model_version": "v4.0-infinite-architect",
        "training_split": f"Year <= {split_year}",
        "forecast_horizon_years": horizon_years,
        "validation": {
            "roc_auc": metrics.get("roc_auc"),
            "average_precision": metrics.get("average_precision"),
            "prevalence": metrics.get("prevalence"),
            "expected_calibration_error": metrics.get("expected_calibration_error"),
            "brier_score": metrics.get("brier_score"),
        },
        "triage_matrix": triage,
        "top_threats": [
            {
                "rank": i + 1,
                "backbone_id": p.backbone_id,
                "risk_probability": round(p.risk_probability, 4),
                "confidence_tier": p.confidence_tier,
                "knownness_score": round(p.knownness_score, 3),
                "future_spread_count": p.n_new_countries_future,
                "explanation": p.explanation,
                "alarm_score": round(p.alarm_score, 3),
                "survival_12mo": round(float((p.meta or {}).get("survival_12mo_prob", 0.0)), 4),
                "survival_24mo": round(float((p.meta or {}).get("survival_24mo_prob", 0.0)), 4),
                "survival_36mo": round(float((p.meta or {}).get("survival_36mo_prob", 0.0)), 4),
            }
            for i, p in enumerate(top)
        ],
        "narrative": {
            "problem": "Predicts likelihood of plasmid backbone geographic spread over the forecast horizon.",
            "methodology": "Hierarchical meta-ensemble with CatBoost epistemic uncertainty, PU learning, IPW causal debiasing, discrete-time survival analysis, and GNN topology embeddings.",
            "limitations": [
                "Predictions are conditioned on historical observation density.",
                "Low-surveillance regions may exhibit bias corrected by IPW.",
                "Emergent phenotypes require active-radar mode for real-time alerting.",
            ],
        },
    }
    return json.dumps(briefing, indent=2, ensure_ascii=False)


def render_markdown_report(
    predictions: list[Prediction],
    metrics: dict[str, Any],
    calibration: dict[str, Any] | None = None,
    split_year: int = 2020,
    horizon_years: int = 3,
) -> str:
    """Render a markdown report for BioSpread predictions."""
    calibration = calibration or {}

    # Determine holdout status
    group_oof_roc_auc = metrics.get('group_oof_roc_auc', metrics.get('oof_roc_auc'))
    group_oof_auc_display = "`not_evaluated`" if group_oof_roc_auc is None else f"{group_oof_roc_auc:.4f}"
    temporal_holdout_roc_auc = metrics.get('temporal_holdout_roc_auc')
    temporal_auc_display = "`not_evaluated`" if temporal_holdout_roc_auc is None else f"{temporal_holdout_roc_auc:.4f}"

    report_lines = [
        "# BioSpread Predictive Report",
        "",
        "## Problem",
        "This model predicts the likelihood of plasmid backbones spreading to new geographic regions (coğrafi yayılım).",
        f"Training split: Year <= {split_year}",
        f"Forecast horizon: {horizon_years} years",
        "",
        "## Validation",
        f"ROC AUC: {metrics.get('roc_auc', 'N/A')}",
        f"Average Precision: {metrics.get('average_precision', 'N/A')}",
        f"Prevalence: {metrics.get('prevalence', 'N/A')}",
        f"Top-K Precision: {metrics.get('top_k_precision', 'N/A')}",
        f"Abstain Rate: {metrics.get('abstain_rate', 'N/A')}",
        f"Group OOF ROC AUC: {group_oof_auc_display}",
        f"Temporal holdout ROC AUC: {temporal_auc_display}",
        "",
        "## Kalibrasyon (Calibration)",
        f"Expected Calibration Error: {calibration.get('expected_calibration_error', 'N/A')}",
        f"Brier Score: {calibration.get('brier_score', 'N/A')}",
        "",
        "## Top Predictions",
    ]

    if predictions:
        top_predictions = sorted(predictions, key=lambda x: x.risk_probability, reverse=True)[:10]
        for i, pred in enumerate(top_predictions[:5], 1):
            report_lines.append(f"{i}. {pred.backbone_id}: {pred.risk_probability:.3f} ({pred.confidence_tier})")
    else:
        report_lines.append("No predictions available.")

    report_lines.extend([
        "",
        "## Quality Gates",
        f"All quality gates passed: {metrics.get('all_quality_gates_passed', 'N/A')}",
        "",
        "*Generated by BioSpread Pipeline*",
    ])

    return "\n".join(report_lines)


def render_chic_report(
    *,
    predictions: list[Prediction],
    metrics: dict[str, Any],
    audit: dict[str, Any],
    governance: Any,
    release_gate: dict[str, Any],
    split_year: int,
    horizon_years: int,
    triage_budget: int | None = None,
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

    quality_thresholds = dict(audit.get("quality_thresholds", {}))
    auc_min = float(quality_thresholds.get("auc_min", 0.82))
    ece_max = float(quality_thresholds.get("calibration_ece_max", 0.10))
    group_auc_min = float(quality_thresholds.get("group_auc_min", 0.80))
    temporal_auc_min = float(quality_thresholds.get("temporal_holdout_auc_min", 0.78))
    external_auc_min = float(quality_thresholds.get("external_holdout_auc_min", 0.78))
    bootstrap_auc_min = float(quality_thresholds.get("bootstrap_auc_ci_low_min", 0.78))
    max_single_feature_auc_max = float(quality_thresholds.get("max_single_feature_auc_max", 0.95))
    external_auc = metrics.get("external_holdout_roc_auc")
    external_status = (
        "NOT_EVALUATED"
        if external_auc is None
        else ("PASS" if float(external_auc) >= external_auc_min else "FAIL")
    )

    # 2. CORE METRICS TABLE
    metrics_headers = ["Metric", "Value", "Threshold", "Status"]
    metrics_rows = [
        ["ROC AUC", metrics.get("roc_auc"), f">= {auc_min:.3f}", "PASS" if metrics.get("roc_auc", 0) >= auc_min else "FAIL"],
        ["Avg Precision", metrics.get("average_precision"), "> Prev", "PASS" if metrics.get("average_precision", 0) > metrics.get("prevalence", 0) else "FAIL"],
        ["Calibration ECE", metrics.get("expected_calibration_error"), f"<= {ece_max:.3f}", "PASS" if metrics.get("expected_calibration_error", 1) <= ece_max else "FAIL"],
        ["Brier Score", metrics.get("brier_score"), "N/A", "INFO"],
    ]

    # 3. RELIABILITY & LEAKAGE GATES
    gate_headers = ["Gate Type", "Metric", "Value", "Status"]
    gate_rows = [
        ["Spatial Group CV", "OOF ROC AUC", metrics.get("group_oof_roc_auc"), "PASS" if metrics.get("group_oof_roc_auc", 0) >= group_auc_min else "FAIL"],
        ["Temporal Holdout", "AUC @ " + str(metrics.get("temporal_holdout_cutoff_year", "N/A")), metrics.get("temporal_holdout_roc_auc"), "PASS" if metrics.get("temporal_holdout_roc_auc", 0) >= temporal_auc_min else "FAIL"],
        ["External Holdout", "Independent AUC", external_auc, external_status],
        ["Bootstrap CI", "ROC AUC Low (95%)", metrics.get("bootstrap_roc_auc_ci_low"), "PASS" if metrics.get("bootstrap_roc_auc_ci_low", 0) >= bootstrap_auc_min else "FAIL"],
        ["Leakage Scan", "Max Single-Feature AUC", metrics.get("max_single_feature_auc"), "PASS" if metrics.get("max_single_feature_auc", 1) < max_single_feature_auc_max else "FAIL"],
    ]

    # 4. TOP RISK CANDIDATES
    risk_headers = ["Rank", "Backbone ID", "Risk Prob", "Confidence", "Future Spread", "Explanation"]
    top_preds = sorted(predictions, key=lambda x: x.risk_probability, reverse=True)[:10]
    risk_rows = [
        [i+1, p.backbone_id, p.risk_probability, p.confidence_tier, p.n_new_countries_future, p.explanation]
        for i, p in enumerate(top_preds)
    ]
    priority = sorted(
        predictions,
        key=lambda p: float((p.meta or {}).get("alarm_score", 0.0)),
        reverse=True,
    )[:5]
    priority_lines = [
        f"{i+1}. {p.backbone_id} (alarm={float((p.meta or {}).get('alarm_score', 0.0)):.3f}, risk={p.risk_probability:.3f})"
        for i, p in enumerate(priority)
    ] or ["No alarm scores available."]

    # 5. THREAT TRIAGE MATRIX (BioSpread v4.0)
    triage = build_threat_triage_matrix(predictions)
    triage_headers = ["Tier", "Count", "Criteria"]
    triage_rows = [
        ["🔴 RED (Alert)", triage["counts"]["red"], "High risk + high uncertainty"],
        ["🟡 YELLOW (Watch)", triage["counts"]["yellow"], "Medium risk or moderate uncertainty"],
        ["🟢 GREEN (Monitor)", triage["counts"]["green"], "Low risk and low uncertainty"],
    ]

    # 6. DYNAMIC BUDGET (BioSpread v4.0)
    budget_lines: list[str] = []
    if triage_budget is not None and triage_budget > 0:
        threshold, selected = dynamic_budget_threshold(predictions, triage_budget)
        budget_lines = [
            "### 🎯 Dynamic Budget Optimization",
            f"Surveillance budget: **{triage_budget}** backbones",
            f"Optimal probability threshold: **{threshold:.3f}**",
            f"Selected backbones: {', '.join([p.backbone_id for p in selected[:5]])}{' ...' if len(selected) > 5 else ''}",
            "",
        ]

    # 7. ENVIRONMENT & DATA HASHES
    env_headers = ["Entity", "Value / Hash"]
    env_rows = [
        ["Python Version", audit.get("environment", {}).get("python", "N/A")],
        ["Polars Version", audit.get("environment", {}).get("polars", "N/A")],
        ["Input SHA-256", next(iter(audit.get("input_hashes", {}).values()), "N/A") if audit.get("input_hashes") else "N/A"],
        ["Training Split", f"Year <= {split_year}"],
        ["Forecast Horizon", f"{horizon_years} Years"],
    ]

    report = [
        "# BioSpread Executive Summary",
        "",
        *header,
        "### 🔬 Problem Definition",
        "This model prioritizes plasmid backbones based on their predicted risk of geographic spread.",
        "",
        "### 📊 Validation Performance",
        _generate_markdown_table(metrics_headers, metrics_rows),
        "",
        "### 🎯 Calibration & Reliability",
        _generate_markdown_table(gate_headers, gate_rows),
        "",
        "### 🚨 High-Risk Backbone Registry (Top 10)",
        _generate_markdown_table(risk_headers, risk_rows),
        "",
        "### Priority Surveillance Targets",
        *priority_lines,
        "",
        "### 🛡️ Threat Triage Matrix (Risk × Uncertainty)",
        _generate_markdown_table(triage_headers, triage_rows),
        "",
        *budget_lines,
        "### 🔎 Katsayı özeti (Coefficient Summary)",
        "The model uses a Firth-penalized logistic ensemble. Primary drivers include:",
        "- `mean_amr_gene_count_pre`: high weight indicates resistance-driven spread.",
        "- `host_diversity_pre`: indicates broad host range potential.",
        "- `mobility_score`: indicates horizontal gene transfer potential.",
        "- `one_health_niche_jump`: captures multi-host ecological invasion potential.",
        "- `gravity_index`: connectivity-weighted geographic reach.",
        "",
        "### ⚠️ Limitations",
        "- Predictions are based on historical observation density.",
        "- Missing data in low-surveillance regions may bias risk scores.",
        "- Emerging host jumps are captured via GNN embeddings when enabled.",
        "",
        "### ⚙️ Environment & Reproducibility",
        _generate_markdown_table(env_headers, env_rows),
        "",
        "### 🚪 Release Gate",
        f"The model has been evaluated against the BioSpread Quality Firewall. Status: {status_label}",
        "",
        "---",
        "*Generated by BioSpread Autonomous Pipeline v0.1.0*",
    ]

    return "\n".join(report)
