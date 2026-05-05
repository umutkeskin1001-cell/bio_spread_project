from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import polars as pl

from bio_spread_project.io_utils import write_json
from bio_spread_project.runtime_policy import EnforcementPolicy
from bio_spread_project.validation import require_non_negative, require_range, to_float, to_int

ALLOWED_CHECK_STATUSES = {"pass", "fail", "not_evaluated", "blocked", "not_required"}


@dataclass(frozen=True)
class QualityThresholds:
    auc_min: float = 0.82
    calibration_ece_max: float = 0.10
    calibration_bin_gap_max: float = 0.20
    bootstrap_auc_ci_low_min: float = 0.78
    group_auc_min: float = 0.80
    temporal_holdout_auc_min: float = 0.78
    external_holdout_auc_min: float = 0.78
    max_single_feature_auc_max: float = 0.95
    average_precision_above_prevalence: bool = True
    bootstrap_average_precision_ci_low_above_prevalence: bool = True
    external_holdout_required: bool = True
    suspicious_feature_count_max: int = 0
    temporal_consistency_required: bool = True
    feature_lineage_required: bool = False
    max_unknown_lineage_count: int = 0
    max_disabled_feature_leak_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "auc_min",
            "calibration_ece_max",
            "calibration_bin_gap_max",
            "bootstrap_auc_ci_low_min",
            "group_auc_min",
            "temporal_holdout_auc_min",
            "external_holdout_auc_min",
            "max_single_feature_auc_max",
        ):
            require_range(name, float(getattr(self, name)), minimum=0.0, maximum=1.0)
        require_non_negative("suspicious_feature_count_max", float(self.suspicious_feature_count_max))
        require_non_negative("max_unknown_lineage_count", float(self.max_unknown_lineage_count))
        require_non_negative("max_disabled_feature_leak_count", float(self.max_disabled_feature_leak_count))


@dataclass(frozen=True)
class DriftThresholds:
    roc_auc_max_drop: float = 0.03
    average_precision_max_drop: float = 0.04
    group_auc_max_drop: float = 0.04
    temporal_auc_max_drop: float = 0.04
    external_holdout_auc_max_drop: float = 0.04
    max_single_feature_auc_max_increase: float = 0.03
    suspicious_feature_count_max_increase: int = 0


@dataclass(frozen=True)
class TrendThresholds:
    roc_auc_max_drop: float = 0.02
    average_precision_max_drop: float = 0.03
    min_gate_pass_rate: float = 0.90
    required_entries_for_go: int = 20


def load_quality_thresholds(path: str | Path | None = None) -> QualityThresholds:
    if path is None:
        from bio_spread_project.config_loader import project_root
        default_path = project_root() / "project_config" / "quality_thresholds.json"
        if default_path.exists():
            path = default_path
        else:
            return QualityThresholds()
    payload = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    allowed = set(QualityThresholds.__dataclass_fields__.keys())
    unknown = sorted(set(payload.keys()) - allowed)
    if unknown:
        raise ValueError(f"Unknown quality threshold keys: {', '.join(unknown)}")
    return QualityThresholds(
        auc_min=require_range("auc_min", to_float("auc_min", payload.get("auc_min", 0.82)), minimum=0.0, maximum=1.0),
        calibration_ece_max=require_range("calibration_ece_max", to_float("calibration_ece_max", payload.get("calibration_ece_max", 0.10)), minimum=0.0, maximum=1.0),
        calibration_bin_gap_max=require_range("calibration_bin_gap_max", to_float("calibration_bin_gap_max", payload.get("calibration_bin_gap_max", 0.20)), minimum=0.0, maximum=1.0),
        bootstrap_auc_ci_low_min=require_range("bootstrap_auc_ci_low_min", to_float("bootstrap_auc_ci_low_min", payload.get("bootstrap_auc_ci_low_min", 0.78)), minimum=0.0, maximum=1.0),
        group_auc_min=require_range("group_auc_min", to_float("group_auc_min", payload.get("group_auc_min", 0.80)), minimum=0.0, maximum=1.0),
        temporal_holdout_auc_min=require_range("temporal_holdout_auc_min", to_float("temporal_holdout_auc_min", payload.get("temporal_holdout_auc_min", 0.78)), minimum=0.0, maximum=1.0),
        external_holdout_auc_min=require_range("external_holdout_auc_min", to_float("external_holdout_auc_min", payload.get("external_holdout_auc_min", 0.78)), minimum=0.0, maximum=1.0),
        max_single_feature_auc_max=require_range("max_single_feature_auc_max", to_float("max_single_feature_auc_max", payload.get("max_single_feature_auc_max", 0.95)), minimum=0.0, maximum=1.0),
        average_precision_above_prevalence=bool(payload.get("average_precision_above_prevalence", True)),
        bootstrap_average_precision_ci_low_above_prevalence=bool(payload.get("bootstrap_average_precision_ci_low_above_prevalence", True)),
        external_holdout_required=bool(payload.get("external_holdout_required", True)),
        suspicious_feature_count_max=to_int("suspicious_feature_count_max", payload.get("suspicious_feature_count_max", 0)),
        temporal_consistency_required=bool(payload.get("temporal_consistency_required", True)),
        feature_lineage_required=bool(payload.get("feature_lineage_required", False)),
        max_unknown_lineage_count=to_int("max_unknown_lineage_count", payload.get("max_unknown_lineage_count", 0)),
        max_disabled_feature_leak_count=to_int("max_disabled_feature_leak_count", payload.get("max_disabled_feature_leak_count", 0)),
    )


def load_json(path: str | Path) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))


def load_drift_thresholds(path: str | Path | None = None) -> DriftThresholds:
    if path is None:
        from bio_spread_project.config_loader import project_root
        default_path = project_root() / "project_config" / "drift_thresholds.json"
        if default_path.exists():
            path = default_path
        else:
            return DriftThresholds()
    payload = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    suspicious = to_int("suspicious_feature_count_max_increase", payload.get("suspicious_feature_count_max_increase", 0))
    require_non_negative("suspicious_feature_count_max_increase", float(suspicious))
    return DriftThresholds(
        roc_auc_max_drop=require_range("roc_auc_max_drop", to_float("roc_auc_max_drop", payload.get("roc_auc_max_drop", 0.03)), minimum=0.0, maximum=1.0),
        average_precision_max_drop=require_range("average_precision_max_drop", to_float("average_precision_max_drop", payload.get("average_precision_max_drop", 0.04)), minimum=0.0, maximum=1.0),
        group_auc_max_drop=require_range("group_auc_max_drop", to_float("group_auc_max_drop", payload.get("group_auc_max_drop", 0.04)), minimum=0.0, maximum=1.0),
        temporal_auc_max_drop=require_range("temporal_auc_max_drop", to_float("temporal_auc_max_drop", payload.get("temporal_auc_max_drop", 0.04)), minimum=0.0, maximum=1.0),
        external_holdout_auc_max_drop=require_range("external_holdout_auc_max_drop", to_float("external_holdout_auc_max_drop", payload.get("external_holdout_auc_max_drop", 0.04)), minimum=0.0, maximum=1.0),
        max_single_feature_auc_max_increase=require_range("max_single_feature_auc_max_increase", to_float("max_single_feature_auc_max_increase", payload.get("max_single_feature_auc_max_increase", 0.03)), minimum=0.0, maximum=1.0),
        suspicious_feature_count_max_increase=suspicious,
    )


def load_trend_thresholds(path: str | Path | None = None) -> TrendThresholds:
    if path is None:
        from bio_spread_project.config_loader import project_root
        default_path = project_root() / "project_config" / "trend_thresholds.json"
        if default_path.exists():
            path = default_path
        else:
            return TrendThresholds()
    payload = cast(dict[str, Any], json.loads(Path(path).read_text(encoding="utf-8")))
    return TrendThresholds(
        roc_auc_max_drop=require_range("roc_auc_max_drop", to_float("roc_auc_max_drop", payload.get("roc_auc_max_drop", 0.02)), minimum=0.0, maximum=1.0),
        average_precision_max_drop=require_range("average_precision_max_drop", to_float("average_precision_max_drop", payload.get("average_precision_max_drop", 0.03)), minimum=0.0, maximum=1.0),
        min_gate_pass_rate=require_range("min_gate_pass_rate", to_float("min_gate_pass_rate", payload.get("min_gate_pass_rate", 0.90)), minimum=0.0, maximum=1.0),
        required_entries_for_go=int(require_range("required_entries_for_go", to_float("required_entries_for_go", payload.get("required_entries_for_go", 20)), minimum=2.0, maximum=1000000.0)),
    )


def _validation_mode_is_cv(validation_mode: str) -> bool:
    return validation_mode in {"cross_validated", "spatial_group_cv_stacked"}


def evaluate_quality_gate_details(
    *,
    metrics: dict[str, Any],
    input_mode: str,
    leakage_audit_passed: bool,
    thresholds: QualityThresholds,
) -> dict[str, dict[str, Any]]:
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
    temporal_consistency_status = str(metrics.get("temporal_consistency_status", "not_evaluated"))
    feature_lineage_status = str(metrics.get("feature_lineage_status", "not_evaluated"))
    feature_lineage_unknown_count = int(float(metrics.get("feature_lineage_unknown_count", 0.0)))
    disabled_feature_leak_count = int(float(metrics.get("disabled_feature_leak_count", 0.0)))
    max_bin_gap = float(metrics.get("max_calibration_bin_gap", 0.0))
    is_geo_mode = input_mode == "geo_reliability_feature_surface"
    def gate_status(name: str, passed: bool, observed: Any, target: Any) -> dict[str, Any]:
        return {
            "passed": bool(passed),
            "status": "pass" if passed else "fail",
            "observed": observed,
            "threshold": target,
            "reason": "" if passed else f"Observed {observed} does not meet required {target}"
        }

    return {
        "cross_validated": gate_status("cross_validated", _validation_mode_is_cv(str(metrics.get("validation_mode", ""))), metrics.get("validation_mode", "direct"), "cv_variants"),
        "auc_at_least_target": gate_status("auc", auc >= thresholds.auc_min, auc, thresholds.auc_min),
        "average_precision_above_prevalence": gate_status("ap", average_precision > prevalence, average_precision, prevalence) if thresholds.average_precision_above_prevalence else {"passed": True, "status": "pass", "reason": "skipped"},
        "calibration_ece_at_most_target": gate_status("ece", ece <= thresholds.calibration_ece_max, ece, thresholds.calibration_ece_max),
        "calibration_bin_gap_at_most_target": gate_status("cal_bin_gap", max_bin_gap <= thresholds.calibration_bin_gap_max, max_bin_gap, thresholds.calibration_bin_gap_max),
        "bootstrap_auc_ci_low_at_least_target": gate_status("boot_auc_low", bootstrap_auc_low >= thresholds.bootstrap_auc_ci_low_min, bootstrap_auc_low, thresholds.bootstrap_auc_ci_low_min),
        "bootstrap_average_precision_ci_low_above_prevalence": gate_status("boot_ap_low", bootstrap_ap_low > prevalence, bootstrap_ap_low, prevalence) if thresholds.bootstrap_average_precision_ci_low_above_prevalence else {"passed": True, "status": "pass", "reason": "skipped"},
        "group_auc_at_least_target": gate_status("group_auc", group_auc >= thresholds.group_auc_min, group_auc, thresholds.group_auc_min) if is_geo_mode else {"passed": True, "status": "pass", "reason": "skipped"},
        "temporal_holdout_auc_at_least_target": gate_status("temporal_auc", temporal_auc >= thresholds.temporal_holdout_auc_min, temporal_auc, thresholds.temporal_holdout_auc_min) if is_geo_mode else {"passed": True, "status": "pass", "reason": "skipped"},
        "temporal_consistency_passed": gate_status("temporal_consistency", temporal_consistency_status in {"pass", "not_evaluated"}, temporal_consistency_status, "pass_or_not_evaluated") if (is_geo_mode and thresholds.temporal_consistency_required) else {"passed": True, "status": "pass", "reason": "skipped"},
        "feature_lineage_passed": gate_status("feature_lineage", feature_lineage_status == "pass" and feature_lineage_unknown_count <= thresholds.max_unknown_lineage_count, f"status={feature_lineage_status}, unknown={feature_lineage_unknown_count}", f"status=pass, unknown<={thresholds.max_unknown_lineage_count}") if (is_geo_mode and thresholds.feature_lineage_required) else {"passed": True, "status": "pass", "reason": "skipped"},
        "disabled_feature_leak_passed": gate_status("disabled_feature_leak", disabled_feature_leak_count <= thresholds.max_disabled_feature_leak_count, disabled_feature_leak_count, thresholds.max_disabled_feature_leak_count) if is_geo_mode else {"passed": True, "status": "pass", "reason": "skipped"},
        "external_holdout_auc_at_least_target": gate_status("external_auc", external_holdout_auc >= thresholds.external_holdout_auc_min, external_holdout_auc, thresholds.external_holdout_auc_min) if (is_geo_mode and thresholds.external_holdout_required) else {"passed": True, "status": "pass", "reason": "skipped"},
        "leakage_audit_passed": {"passed": bool(leakage_audit_passed), "status": "pass" if leakage_audit_passed else "fail", "reason": "Structural leakage check failed" if not leakage_audit_passed else ""},
        "adversarial_leakage_scan_passed": gate_status("adv_scan", suspicious_feature_count <= thresholds.suspicious_feature_count_max and max_single_feature_auc < thresholds.max_single_feature_auc_max, f"count={suspicious_feature_count}, max_auc={max_single_feature_auc:.3f}", f"max_count={thresholds.suspicious_feature_count_max}, max_auc={thresholds.max_single_feature_auc_max}") if is_geo_mode else {"passed": True, "status": "pass", "reason": "skipped"},
    }


def evaluate_quality_gates(
    *,
    metrics: dict[str, Any],
    input_mode: str,
    leakage_audit_passed: bool,
    thresholds: QualityThresholds,
) -> list["CheckResult"]:
    raw = evaluate_quality_gate_details(metrics=metrics, input_mode=input_mode, leakage_audit_passed=leakage_audit_passed, thresholds=thresholds)
    return [CheckResult(name=k, passed=bool(v["passed"]), status=str(v["status"]), reason=str(v["reason"]), detail=v) for k, v in raw.items()]


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
    missing_required_metric = False
    for metric_name, (threshold_name, max_drop) in checks.items():
        if metric_name not in current_summary or metric_name not in baseline_summary:
            metric_checks[metric_name] = {"status": "not_available"}
            missing_required_metric = True
            continue
        current_value = float(current_summary[metric_name])
        baseline_value = float(baseline_summary[metric_name])
        delta = current_value - baseline_value
        passed = delta >= -max_drop
        metric_checks[metric_name] = {"status": "pass" if passed else "fail", "current": current_value, "baseline": baseline_value, "delta": delta, "max_drop": max_drop, "threshold_name": threshold_name}
        all_passed = all_passed and passed
    aux_checks: dict[str, Any] = {}
    for metric_name, max_increase in (("max_single_feature_auc", thresholds.max_single_feature_auc_max_increase), ("suspicious_feature_count", float(thresholds.suspicious_feature_count_max_increase))):
        if metric_name not in current_summary or metric_name not in baseline_summary:
            aux_checks[metric_name] = {"status": "not_available"}
            continue
        current_value = float(current_summary[metric_name])
        baseline_value = float(baseline_summary[metric_name])
        delta = current_value - baseline_value
        passed = delta <= max_increase
        aux_checks[metric_name] = {"status": "pass" if passed else "fail", "current": current_value, "baseline": baseline_value, "delta": delta, "max_increase": max_increase}
        all_passed = all_passed and passed
    if missing_required_metric:
        all_passed = False
    return {"all_passed": all_passed, "metric_checks": metric_checks, "aux_checks": aux_checks}


def evaluate_drift_checks(
    *,
    current: dict[str, Any],
    baseline: dict[str, Any],
    thresholds: DriftThresholds,
) -> list["CheckResult"]:
    payload = evaluate_drift(current=current, baseline=baseline, thresholds=thresholds)
    results: list[CheckResult] = []
    for prefix in ("metric_checks", "aux_checks"):
        for metric_name, detail in payload.get(prefix, {}).items():
            status = str(detail.get("status", "not_evaluated"))
            if status == "not_available":
                results.append(CheckResult(name=f"drift_{metric_name}", passed=False, status="blocked", reason="missing_metric", detail=detail))
            else:
                results.append(CheckResult(name=f"drift_{metric_name}", passed=status == "pass", status="pass" if status == "pass" else "fail", detail=detail))
    return results


def evaluate_model_registry_trend(
    *,
    entries: pl.DataFrame | list[dict[str, Any]],
    window_size: int,
    thresholds: TrendThresholds,
) -> dict[str, Any]:
    df = entries if isinstance(entries, pl.DataFrame) else pl.DataFrame(entries)
    resolved_window = max(2, int(window_size))
    required_entries = max(2 * resolved_window, thresholds.required_entries_for_go)
    if df.is_empty() or len(df) < required_entries:
        return {"status": "insufficient_data", "all_passed": True, "trend_evidence_sufficient": False, "required_entries": required_entries, "available_entries": len(df), "window_size": resolved_window}
    recent = df.tail(resolved_window)
    previous = df.slice(len(df) - 2 * resolved_window, resolved_window)

    def get_mean(data: pl.DataFrame, col: str) -> float:
        return float(cast(Any, data[col].mean())) if col in data.columns else 0.0

    recent_auc, prev_auc = get_mean(recent, "roc_auc"), get_mean(previous, "roc_auc")
    recent_ap, prev_ap = get_mean(recent, "average_precision"), get_mean(previous, "average_precision")
    gate_rate = float(cast(Any, recent["all_quality_gates_passed"].cast(pl.Int32).mean())) if "all_quality_gates_passed" in recent.columns else 0.0
    auc_delta, ap_delta = recent_auc - prev_auc, recent_ap - prev_ap
    auc_passed = auc_delta >= -thresholds.roc_auc_max_drop
    ap_passed = ap_delta >= -thresholds.average_precision_max_drop
    gate_passed = gate_rate >= thresholds.min_gate_pass_rate
    all_passed = auc_passed and ap_passed and gate_passed
    return {
        "status": "ok",
        "all_passed": all_passed,
        "trend_evidence_sufficient": True,
        "window_size": resolved_window,
        "available_entries": len(df),
        "checks": {
            "roc_auc": {"previous_mean": prev_auc, "recent_mean": recent_auc, "delta": auc_delta, "passed": auc_passed},
            "average_precision": {"previous_mean": prev_ap, "recent_mean": recent_ap, "delta": ap_delta, "passed": ap_passed},
            "gate_pass_rate": {"recent_rate": gate_rate, "min_required": thresholds.min_gate_pass_rate, "passed": gate_passed},
        },
    }


def load_model_registry(path: str | Path) -> pl.DataFrame:
    source = Path(path)
    if not source.exists() or source.stat().st_size == 0:
        return pl.DataFrame()
    return pl.read_ndjson(source)


def write_trend_report(path: str | Path, payload: dict[str, Any]) -> Path:
    return write_json(Path(path), payload)


def evaluate_trend_from_registry(
    *,
    entries: list[dict[str, Any]],
    thresholds: TrendThresholds,
    model_name: str,
    input_mode: str,
    window_size: int = 10,
) -> list["CheckResult"]:
    compatible = [e for e in entries if e.get("model_name") == model_name and e.get("input_mode") == input_mode]
    trend = evaluate_model_registry_trend(entries=compatible, window_size=window_size, thresholds=thresholds)
    if trend.get("status") == "insufficient_data":
        return [CheckResult(name="trend_evidence", passed=False, status="not_evaluated", reason="insufficient_data", detail=trend)]
    out: list[CheckResult] = []
    for name, detail in trend.get("checks", {}).items():
        passed = bool(detail.get("passed", False))
        out.append(CheckResult(name=f"trend_{name}", passed=passed, status="pass" if passed else "fail", detail=detail))
    return out


@dataclass(frozen=True)
class CheckResult:
    name: str
    passed: bool
    status: str
    reason: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status not in ALLOWED_CHECK_STATUSES:
            raise ValueError(f"Unsupported check status: {self.status}")


@dataclass(frozen=True)
class GovernanceReport:
    readiness: str
    blocked_by: list[str]
    quality_checks: list[CheckResult]
    drift_checks: list[CheckResult]
    trend_checks: list[CheckResult]
    policy_flags: dict[str, bool] = field(default_factory=dict)


def build_governance_report(
    *,
    quality_checks: list[CheckResult],
    drift_checks: list[CheckResult],
    trend_checks: list[CheckResult],
    policy: EnforcementPolicy,
) -> GovernanceReport:
    blocked_by: list[str] = []
    quality_failed = any(c.status == "fail" for c in quality_checks if c.status != "not_required")
    drift_failed = any(c.status == "fail" for c in drift_checks)
    drift_blocked = any(c.status == "blocked" for c in drift_checks)
    trend_failed = any(c.status == "fail" for c in trend_checks)
    trend_insufficient = any(c.status == "not_evaluated" for c in trend_checks)
    if quality_failed:
        blocked_by.append("quality_gates")
    if drift_blocked:
        blocked_by.append("drift_evidence_missing")
    if drift_failed:
        blocked_by.extend([c.name for c in drift_checks if c.status == "fail"])
    if trend_failed:
        blocked_by.append("trend_failed")
    if trend_insufficient and policy.require_trend_evidence:
        blocked_by.append("trend_evidence_insufficient")
    if blocked_by:
        readiness = "no_go"
    elif trend_insufficient and policy.allow_conditional_release:
        readiness = "conditional_go"
    else:
        readiness = "go"
    return GovernanceReport(
        readiness=readiness,
        blocked_by=blocked_by,
        quality_checks=quality_checks,
        drift_checks=drift_checks,
        trend_checks=trend_checks,
        policy_flags={
            "fail_on_quality_gates": policy.fail_on_quality_gates,
            "fail_on_drift_fail": policy.fail_on_drift_fail,
            "fail_on_trend_fail": policy.fail_on_trend_fail,
            "require_trend_evidence": policy.require_trend_evidence,
            "allow_conditional_release": policy.allow_conditional_release,
        },
    )


def build_release_gate_report(
    *,
    audit: dict[str, Any],
    drift_report: dict[str, Any],
    trend_report: dict[str, Any],
    allow_conditional_trend_release: bool = True,
) -> dict[str, Any]:
    quality_passed = bool(audit.get("all_quality_gates_passed", False))
    drift_passed = bool(drift_report.get("all_passed", False))
    trend_status = str(trend_report.get("status", "unknown"))
    trend_checks_passed = bool(trend_report.get("all_passed", False)) if trend_status == "ok" else False
    trend_evidence_sufficient = bool(trend_report.get("trend_evidence_sufficient", trend_status == "ok"))
    checks = {
        "quality_gates": quality_passed,
        "drift_checks": drift_passed,
        "trend_checks": trend_checks_passed,
        "trend_evidence": trend_evidence_sufficient,
    }
    score = (40 if quality_passed else 0) + (35 if drift_passed else 0) + (25 if trend_checks_passed and trend_evidence_sufficient else 0)
    if quality_passed and drift_passed and not trend_evidence_sufficient and allow_conditional_trend_release:
        readiness = "conditional_go"
        blocked_by = ["trend_evidence"]
    elif quality_passed and drift_passed and trend_checks_passed and trend_evidence_sufficient:
        readiness = "go"
        blocked_by = []
    else:
        readiness = "no_go"
        blocked_by = [k for k, v in checks.items() if not v]
    return {
        "readiness": readiness,
        "score": score,
        "max_score": 100,
        "checks": checks,
        "trend_status": trend_status,
        "blocked_by": blocked_by,
        "allow_conditional_trend_release": allow_conditional_trend_release,
        "policy_version": "release_gate_v2",
    }
