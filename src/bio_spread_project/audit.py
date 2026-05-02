"""Audit and model-card helpers for BioSpread runs."""

from __future__ import annotations

import platform
import ssl
import subprocess
from pathlib import Path
from typing import Any, cast

import numpy
import sklearn

from bio_spread_project.geo_reliability import FEATURE_COLUMNS, leakage_audit
from bio_spread_project.governance import (
    QualityThresholds,
    evaluate_quality_gate_details,
    evaluate_quality_gates,
    load_quality_thresholds,
)
from bio_spread_project.io_utils import sha256_file
from bio_spread_project.model_metrics import validation_tracks


def build_run_audit(
    *,
    input_paths: dict[str, Path],
    metrics: dict[str, Any],
    primary_model: str,
    input_mode: str,
    quality_thresholds_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a compact reproducibility and reliability audit payload."""
    auc = float(metrics.get("oof_roc_auc", metrics.get("roc_auc", 0.0)))
    average_precision = float(metrics.get("oof_average_precision", metrics.get("average_precision", 0.0)))
    prevalence = float(metrics.get("prevalence", 0.0))
    ece = float(metrics.get("expected_calibration_error", 1.0))
    bootstrap_auc_low = float(metrics.get("bootstrap_roc_auc_ci_low", auc))
    bootstrap_auc_high = float(metrics.get("bootstrap_roc_auc_ci_high", auc))
    bootstrap_ap_low = float(metrics.get("bootstrap_average_precision_ci_low", average_precision))
    bootstrap_ap_high = float(metrics.get("bootstrap_average_precision_ci_high", average_precision))
    audit = leakage_audit(FEATURE_COLUMNS if input_mode == "geo_reliability_feature_surface" else ())
    thresholds = load_quality_thresholds(quality_thresholds_path)
    quality_checks = evaluate_quality_gates(
        metrics=metrics,
        input_mode=input_mode,
        leakage_audit_passed=audit["status"] == "pass",
        thresholds=thresholds,
    )
    quality_gates = {check.name: check.passed for check in quality_checks}
    quality_gate_details = evaluate_quality_gate_details(
        metrics=metrics,
        input_mode=input_mode,
        leakage_audit_passed=audit["status"] == "pass",
        thresholds=thresholds,
    )
    max_single_feature_auc = float(metrics.get("max_single_feature_auc", 0.5))
    suspicious_feature_count = int(float(metrics.get("suspicious_feature_count", 0.0)))
    return {
        "project": "BioSpread",
        "primary_model": primary_model,
        "input_mode": input_mode,
        "input_hashes": _hash_inputs(input_paths),
        "validation": {
            "mode": metrics.get("validation_mode", "direct"),
            "roc_auc": auc,
            "average_precision": average_precision,
            "prevalence": prevalence,
            "expected_calibration_error": ece,
            "brier_score": float(metrics.get("brier_score", 0.0)),
            "n_backbones": int(float(metrics.get("n_backbones", 0.0))),
            "n_positive": int(float(metrics.get("n_positive", 0.0))),
            "bootstrap_roc_auc_ci_low": bootstrap_auc_low,
            "bootstrap_roc_auc_ci_high": bootstrap_auc_high,
            "bootstrap_average_precision_ci_low": bootstrap_ap_low,
            "bootstrap_average_precision_ci_high": bootstrap_ap_high,
            "max_single_feature_auc": max_single_feature_auc,
            "suspicious_feature_count": suspicious_feature_count,
            "top_features": metrics.get("top_features", []),
            "calibration_bins": metrics.get("calibration_bins", []),
            "tracks": validation_tracks(metrics),
        },
        "quality_gates": quality_gates,
        "quality_gate_details": quality_gate_details,
        "all_quality_gates_passed": all(quality_gates.values()),
        "quality_thresholds": _thresholds_payload(thresholds),
        "leakage_audit": audit,
        "environment": {
            "python": platform.python_version(),
            "ssl_backend": ssl.OPENSSL_VERSION,
            "numpy": numpy.__version__,
            "scikit_learn": sklearn.__version__,
            "git_commit": _git_commit(),
        },
    }


def render_model_card(*, audit: dict[str, Any], coefficient_summary: str) -> str:
    """Render a jury-friendly model card."""
    validation = audit["validation"]
    gates = audit["quality_gates"]
    threshold_auc = float(audit.get("quality_thresholds", {}).get("auc_min", 0.82))
    tracks = audit["validation"].get("tracks", {})
    group_track = tracks.get("group_oof", {})
    temporal_track = tracks.get("temporal_holdout", {})
    external_track = tracks.get("external_holdout", {})
    lines = [
        "# BioSpread Model Card",
        "",
        "## Model",
        f"- Name: `{audit.get('primary_model', 'BioSpread-Ensemble')}`",
        f"- Input mode: `{audit.get('input_mode', 'geo_reliability')}`",
        f"- Validation mode: `{validation.get('mode', validation.get('validation_mode', 'spatial_group_cv'))}`",
        "- Intended use: prioritize plasmid backbones for geographic-spread monitoring.",
        "- Not intended use: clinical diagnosis or direct public-health intervention without expert review.",
        "",
        "## Reliability",
        f"- OOF ROC AUC: `{validation.get('roc_auc', 0.0):.3f}`",
        f"- Minimum AUC target: `{threshold_auc:.2f}`",
        f"- OOF average precision: `{validation.get('average_precision', 0.0):.3f}`",
        f"- Positive prevalence: `{validation.get('prevalence', 0.0):.3f}`",
        f"- Expected calibration error: `{validation.get('expected_calibration_error', validation.get('brier_score', 0.0)):.3f}`",
        f"- Brier score: `{validation.get('brier_score', 0.0):.3f}`",
        f"- Group OOF ROC AUC: `{_metric_or_na(group_track.get('roc_auc', validation.get('roc_auc')))}`",
        f"- Temporal holdout ROC AUC: `{_metric_or_na(temporal_track.get('roc_auc', validation.get('roc_auc')))}`",
        f"- External holdout ROC AUC: `{_metric_or_na(external_track.get('roc_auc', validation.get('roc_auc')))}`",
        (
            f"- Bootstrap ROC AUC CI: "
            f"`[{validation.get('bootstrap_roc_auc_ci_low', 0.0):.3f}, {validation.get('bootstrap_roc_auc_ci_high', 0.0):.3f}]`"
        ),
        (
            f"- Bootstrap AP CI: "
            f"`[{validation.get('bootstrap_average_precision_ci_low', 0.0):.3f}, "
            f"{validation.get('bootstrap_average_precision_ci_high', 0.0):.3f}]`"
        ),
        f"- Max single-feature AUC: `{validation.get('max_single_feature_auc', 0.0):.3f}`",
        f"- Suspicious feature count: `{validation.get('suspicious_feature_count', 0)}`",
        f"- Evaluation cohort: `{validation.get('n_backbones', 0)}` backbones, `{validation.get('n_positive', 0)}` positives",
        "",
        "## Quality Gates",
    ]
    for name, passed in gates.items():
        lines.append(f"- {name}: `{'pass' if passed else 'fail'}`")
    lines.extend(
        [
            "",
            "## Leakage Guard",
            f"- Status: `{audit.get('leakage_audit', {}).get('status', 'N/A')}`",
            f"- Feature count: `{audit.get('leakage_audit', {}).get('feature_count', 0)}`",
            "- Future/outcome columns are excluded from model features.",
            "",
            "## Explanation Surface",
            f"- Top feature signal summary: `{coefficient_summary or 'not_available'}`",
            "",
            "## Reproducibility",
        ]
    )
    for name, digest in audit.get("input_hashes", {}).items():
        lines.append(f"- {name}: `{digest}`")
    lines.extend(
        [
            f"- Python: `{audit.get('environment', {}).get('python', 'unknown')}`",
            f"- NumPy: `{audit.get('environment', {}).get('numpy', 'unknown')}`",
            f"- scikit-learn: `{audit.get('environment', {}).get('scikit_learn', 'unknown')}`",
            f"- Git commit: `{audit.get('environment', {}).get('git_commit', 'unknown')}`",
            "",
        ]
    )
    return "\n".join(lines)


def _hash_inputs(input_paths: dict[str, Path]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name, path in sorted(input_paths.items()):
        if path.exists() and path.is_file():
            hashes[name] = sha256_file(path)
    return hashes


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "not_available"
    return completed.stdout.strip() or "not_available"


def _thresholds_payload(thresholds: QualityThresholds) -> dict[str, Any]:
    return {
        "auc_min": thresholds.auc_min,
        "average_precision_above_prevalence": thresholds.average_precision_above_prevalence,
        "calibration_ece_max": thresholds.calibration_ece_max,
        "bootstrap_auc_ci_low_min": thresholds.bootstrap_auc_ci_low_min,
        "bootstrap_average_precision_ci_low_above_prevalence": (
            thresholds.bootstrap_average_precision_ci_low_above_prevalence
        ),
        "group_auc_min": thresholds.group_auc_min,
        "temporal_holdout_auc_min": thresholds.temporal_holdout_auc_min,
        "external_holdout_auc_min": thresholds.external_holdout_auc_min,
        "external_holdout_required": thresholds.external_holdout_required,
        "max_single_feature_auc_max": thresholds.max_single_feature_auc_max,
        "suspicious_feature_count_max": thresholds.suspicious_feature_count_max,
    }


def _metric_or_na(value: object) -> str:
    if value is None:
        return "not_evaluated"
    return f"{float(cast(Any, value)):.3f}"
