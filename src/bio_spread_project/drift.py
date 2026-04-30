"""Benchmark drift evaluation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from bio_spread_project.validation import require_non_negative, require_range, to_float, to_int


@dataclass(frozen=True)
class DriftThresholds:
    roc_auc_max_drop: float = 0.03
    average_precision_max_drop: float = 0.04
    group_auc_max_drop: float = 0.04
    temporal_auc_max_drop: float = 0.04
    external_holdout_auc_max_drop: float = 0.04
    max_single_feature_auc_max_increase: float = 0.03
    suspicious_feature_count_max_increase: int = 0


def load_json(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def load_drift_thresholds(path: str | Path | None = None) -> DriftThresholds:
    if path is None:
        return DriftThresholds()
    payload = load_json(path)
    roc_auc_max_drop = require_range(
        "roc_auc_max_drop", to_float("roc_auc_max_drop", payload.get("roc_auc_max_drop", 0.03)), minimum=0.0, maximum=1.0
    )
    average_precision_max_drop = require_range(
        "average_precision_max_drop",
        to_float("average_precision_max_drop", payload.get("average_precision_max_drop", 0.04)),
        minimum=0.0,
        maximum=1.0,
    )
    group_auc_max_drop = require_range(
        "group_auc_max_drop", to_float("group_auc_max_drop", payload.get("group_auc_max_drop", 0.04)), minimum=0.0, maximum=1.0
    )
    temporal_auc_max_drop = require_range(
        "temporal_auc_max_drop",
        to_float("temporal_auc_max_drop", payload.get("temporal_auc_max_drop", 0.04)),
        minimum=0.0,
        maximum=1.0,
    )
    external_holdout_auc_max_drop = require_range(
        "external_holdout_auc_max_drop",
        to_float("external_holdout_auc_max_drop", payload.get("external_holdout_auc_max_drop", 0.04)),
        minimum=0.0,
        maximum=1.0,
    )
    max_single_feature_auc_max_increase = require_range(
        "max_single_feature_auc_max_increase",
        to_float("max_single_feature_auc_max_increase", payload.get("max_single_feature_auc_max_increase", 0.03)),
        minimum=0.0,
        maximum=1.0,
    )
    suspicious_feature_count_max_increase = to_int(
        "suspicious_feature_count_max_increase",
        payload.get("suspicious_feature_count_max_increase", 0),
    )
    require_non_negative("suspicious_feature_count_max_increase", float(suspicious_feature_count_max_increase))
    return DriftThresholds(
        roc_auc_max_drop=roc_auc_max_drop,
        average_precision_max_drop=average_precision_max_drop,
        group_auc_max_drop=group_auc_max_drop,
        temporal_auc_max_drop=temporal_auc_max_drop,
        external_holdout_auc_max_drop=external_holdout_auc_max_drop,
        max_single_feature_auc_max_increase=max_single_feature_auc_max_increase,
        suspicious_feature_count_max_increase=suspicious_feature_count_max_increase,
    )


def evaluate_drift(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: DriftThresholds,
) -> dict[str, Any]:
    current_summary = dict(current.get("validation_summary", {}))
    baseline_summary = dict(baseline.get("validation_summary", {}))

    checks = {
        "roc_auc": ("roc_auc_max_drop", thresholds.roc_auc_max_drop),
        "average_precision": ("average_precision_max_drop", thresholds.average_precision_max_drop),
        "group_oof_roc_auc": ("group_auc_max_drop", thresholds.group_auc_max_drop),
        "temporal_holdout_roc_auc": ("temporal_auc_max_drop", thresholds.temporal_auc_max_drop),
        "external_holdout_roc_auc": ("external_holdout_auc_max_drop", thresholds.external_holdout_auc_max_drop),
    }
    metric_checks: dict[str, Any] = {}
    all_passed = True
    for metric_name, (threshold_name, max_drop) in checks.items():
        if metric_name not in current_summary or metric_name not in baseline_summary:
            metric_checks[metric_name] = {"status": "not_available"}
            continue
        current_value = float(current_summary[metric_name])
        baseline_value = float(baseline_summary[metric_name])
        delta = current_value - baseline_value
        passed = delta >= -max_drop
        metric_checks[metric_name] = {
            "status": "pass" if passed else "fail",
            "current": current_value,
            "baseline": baseline_value,
            "delta": delta,
            "max_drop": max_drop,
            "threshold_name": threshold_name,
        }
        all_passed = all_passed and passed

    aux_checks: dict[str, Any] = {}
    for metric_name, max_increase in (
        ("max_single_feature_auc", thresholds.max_single_feature_auc_max_increase),
        ("suspicious_feature_count", float(thresholds.suspicious_feature_count_max_increase)),
    ):
        if metric_name not in current_summary or metric_name not in baseline_summary:
            aux_checks[metric_name] = {"status": "not_available"}
            continue
        current_value = float(current_summary[metric_name])
        baseline_value = float(baseline_summary[metric_name])
        delta = current_value - baseline_value
        passed = delta <= max_increase
        aux_checks[metric_name] = {
            "status": "pass" if passed else "fail",
            "current": current_value,
            "baseline": baseline_value,
            "delta": delta,
            "max_increase": max_increase,
        }
        all_passed = all_passed and passed

    return {
        "all_passed": all_passed,
        "metric_checks": metric_checks,
        "aux_checks": aux_checks,
    }
