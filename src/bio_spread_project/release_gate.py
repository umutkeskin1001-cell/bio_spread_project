"""Release-gate evaluation for run artifacts."""

from __future__ import annotations

from typing import Any


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

    check_results = {
        "quality_gates": quality_passed,
        "drift_checks": drift_passed,
        "trend_checks": trend_checks_passed,
        "trend_evidence": trend_evidence_sufficient,
    }
    score = (
        (40 if quality_passed else 0)
        + (35 if drift_passed else 0)
        + (25 if trend_checks_passed and trend_evidence_sufficient else 0)
    )
    if quality_passed and drift_passed and not trend_evidence_sufficient and allow_conditional_trend_release:
        readiness = "conditional_go"
        blocked_by = ["trend_evidence"]
    elif quality_passed and drift_passed and trend_checks_passed and trend_evidence_sufficient:
        readiness = "go"
        blocked_by = []
    else:
        readiness = "no_go"
        blocked_by = [name for name, passed in check_results.items() if not passed]
    return {
        "readiness": readiness,
        "score": score,
        "max_score": 100,
        "checks": check_results,
        "trend_status": trend_status,
        "blocked_by": blocked_by,
        "allow_conditional_trend_release": allow_conditional_trend_release,
        "policy_version": "release_gate_v2",
    }
