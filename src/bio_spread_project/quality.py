from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bio_spread_project.validation import require_non_negative, require_range, to_float, to_int


@dataclass(frozen=True)
class QualityThresholds:
    auc_min: float = 0.82
    calibration_ece_max: float = 0.10
    bootstrap_auc_ci_low_min: float = 0.78
    group_auc_min: float = 0.80
    temporal_holdout_auc_min: float = 0.78
    external_holdout_auc_min: float = 0.78
    max_single_feature_auc_max: float = 0.95

    average_precision_above_prevalence: bool = True
    bootstrap_average_precision_ci_low_above_prevalence: bool = True
    external_holdout_required: bool = True
    suspicious_feature_count_max: int = 0

    def __post_init__(self) -> None:
        for name in (
            "auc_min",
            "calibration_ece_max",
            "bootstrap_auc_ci_low_min",
            "group_auc_min",
            "temporal_holdout_auc_min",
            "external_holdout_auc_min",
            "max_single_feature_auc_max",
        ):
            require_range(name, float(getattr(self, name)), minimum=0.0, maximum=1.0)
        require_non_negative("suspicious_feature_count_max", float(self.suspicious_feature_count_max))


def load_quality_thresholds(path: str | Path | None = None) -> QualityThresholds:
    if path is None:
        return QualityThresholds()

    source = Path(path)
    payload = json.loads(source.read_text(encoding="utf-8"))
    allowed = set(QualityThresholds.__dataclass_fields__.keys())
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown quality threshold keys: {', '.join(unknown)}")

    return QualityThresholds(
        auc_min=require_range("auc_min", to_float("auc_min", payload.get("auc_min", 0.82)), minimum=0.0, maximum=1.0),
        calibration_ece_max=require_range(
            "calibration_ece_max", to_float("calibration_ece_max", payload.get("calibration_ece_max", 0.10)), minimum=0.0, maximum=1.0
        ),
        bootstrap_auc_ci_low_min=require_range(
            "bootstrap_auc_ci_low_min", to_float("bootstrap_auc_ci_low_min", payload.get("bootstrap_auc_ci_low_min", 0.78)), minimum=0.0, maximum=1.0
        ),
        group_auc_min=require_range("group_auc_min", to_float("group_auc_min", payload.get("group_auc_min", 0.80)), minimum=0.0, maximum=1.0),
        temporal_holdout_auc_min=require_range(
            "temporal_holdout_auc_min", to_float("temporal_holdout_auc_min", payload.get("temporal_holdout_auc_min", 0.78)), minimum=0.0, maximum=1.0
        ),
        external_holdout_auc_min=require_range(
            "external_holdout_auc_min", to_float("external_holdout_auc_min", payload.get("external_holdout_auc_min", 0.78)), minimum=0.0, maximum=1.0
        ),
        max_single_feature_auc_max=require_range(
            "max_single_feature_auc_max", to_float("max_single_feature_auc_max", payload.get("max_single_feature_auc_max", 0.95)), minimum=0.0, maximum=1.0
        ),
        average_precision_above_prevalence=bool(payload.get("average_precision_above_prevalence", True)),
        bootstrap_average_precision_ci_low_above_prevalence=bool(
            payload.get("bootstrap_average_precision_ci_low_above_prevalence", True)
        ),
        external_holdout_required=bool(payload.get("external_holdout_required", True)),
        suspicious_feature_count_max=to_int(
            "suspicious_feature_count_max", payload.get("suspicious_feature_count_max", 0)
        ),
    )


def _validation_mode_is_cv(validation_mode: str) -> bool:
    accepted_cv_modes = {"cross_validated", "spatial_group_cv_stacked"}
    return validation_mode in accepted_cv_modes


def evaluate_quality_gates(
    *,
    metrics: dict[str, Any],
    input_mode: str,
    leakage_audit_passed: bool,
    thresholds: QualityThresholds,
) -> dict[str, bool]:
    auc = float(metrics.get("oof_roc_auc", metrics.get("roc_auc", 0.0)))
    average_precision = float(metrics.get("oof_average_precision", metrics.get("average_precision", 0.0)))
    prevalence = float(metrics.get("prevalence", 0.0))
    ece = float(metrics.get("expected_calibration_error", 1.0))
    group_auc = float(metrics.get("group_oof_roc_auc", 0.0))
    temporal_auc = float(metrics.get("temporal_holdout_roc_auc", 0.0))
    external_holdout_auc = float(metrics.get("external_holdout_roc_auc", 0.0))
    bootstrap_auc_low = float(metrics.get("bootstrap_roc_auc_ci_low", auc))
    bootstrap_ap_low = float(metrics.get("bootstrap_average_precision_ci_low", average_precision))
    max_single_feature_auc = float(metrics.get("max_single_feature_auc", 0.5))
    suspicious_feature_count = int(float(metrics.get("suspicious_feature_count", 0.0)))
    is_geo_mode = input_mode == "geo_reliability_feature_surface"

    val_mode = str(metrics.get("validation_mode", ""))
    has_group_auc = "group_oof_roc_auc" in metrics
    has_temporal_auc = "temporal_holdout_roc_auc" in metrics
    has_external_holdout_auc = "external_holdout_roc_auc" in metrics

    return {
        "cross_validated": _validation_mode_is_cv(val_mode),
        "auc_at_least_target": auc >= thresholds.auc_min,
        "average_precision_above_prevalence": (
            average_precision > prevalence if thresholds.average_precision_above_prevalence else True
        ),
        "calibration_ece_at_most_target": ece <= thresholds.calibration_ece_max,
        "bootstrap_auc_ci_low_at_least_target": bootstrap_auc_low >= thresholds.bootstrap_auc_ci_low_min,
        "bootstrap_average_precision_ci_low_above_prevalence": (
            bootstrap_ap_low > prevalence
            if thresholds.bootstrap_average_precision_ci_low_above_prevalence
            else True
        ),
        "group_auc_at_least_target": (has_group_auc and group_auc >= thresholds.group_auc_min)
        if is_geo_mode
        else True,
        "temporal_holdout_auc_at_least_target": (has_temporal_auc and temporal_auc >= thresholds.temporal_holdout_auc_min)
        if is_geo_mode
        else True,
        "external_holdout_auc_at_least_target": (
            has_external_holdout_auc and external_holdout_auc >= thresholds.external_holdout_auc_min
        )
        if is_geo_mode and thresholds.external_holdout_required
        else True,
        "leakage_audit_passed": leakage_audit_passed,
        "adversarial_leakage_scan_passed": (
            suspicious_feature_count <= thresholds.suspicious_feature_count_max
            and max_single_feature_auc < thresholds.max_single_feature_auc_max
        )
        if is_geo_mode
        else True,
    }


def evaluate_quality_gate_details(
    *,
    metrics: dict[str, Any],
    input_mode: str,
    leakage_audit_passed: bool,
    thresholds: QualityThresholds,
) -> dict[str, dict[str, Any]]:
    bool_gates = evaluate_quality_gates(
        metrics=metrics,
        input_mode=input_mode,
        leakage_audit_passed=leakage_audit_passed,
        thresholds=thresholds,
    )
    details: dict[str, dict[str, Any]] = {}
    for gate, passed in bool_gates.items():
        details[gate] = {
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "reason": "" if passed else "threshold_not_met",
        }
    return details
