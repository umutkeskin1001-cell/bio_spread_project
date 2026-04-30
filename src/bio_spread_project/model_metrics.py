"""Helpers for reporting evaluated versus unavailable model metrics."""

from __future__ import annotations

from typing import Any


def validation_summary(metrics: dict[str, Any]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in (
        "roc_auc",
        "average_precision",
        "group_oof_roc_auc",
        "temporal_holdout_roc_auc",
        "external_holdout_roc_auc",
        "bootstrap_roc_auc_ci_low",
        "bootstrap_roc_auc_ci_high",
        "max_single_feature_auc",
        "suspicious_feature_count",
    ):
        if key in metrics:
            summary[key] = float(metrics[key])
    return summary


def validation_tracks(metrics: dict[str, Any]) -> dict[str, dict[str, Any]]:
    def scalar(name: str) -> float | None:
        return float(metrics[name]) if name in metrics else None

    def track(status: str, **values: float | None) -> dict[str, Any]:
        payload: dict[str, Any] = {"status": status}
        for key, value in values.items():
            if value is not None:
                payload[key] = value
        return payload

    tracks = {
        "oof": track(
            "evaluated" if "roc_auc" in metrics else "not_evaluated",
            roc_auc=scalar("roc_auc"),
            average_precision=scalar("average_precision"),
            expected_calibration_error=scalar("expected_calibration_error"),
            brier_score=scalar("brier_score"),
            n_backbones=scalar("n_backbones"),
            n_positive=scalar("n_positive"),
            prevalence=scalar("prevalence"),
            bootstrap_roc_auc_ci_low=scalar("bootstrap_roc_auc_ci_low"),
            bootstrap_roc_auc_ci_high=scalar("bootstrap_roc_auc_ci_high"),
            bootstrap_average_precision_ci_low=scalar("bootstrap_average_precision_ci_low"),
            bootstrap_average_precision_ci_high=scalar("bootstrap_average_precision_ci_high"),
        ),
        "group_oof": track(
            "evaluated" if "group_oof_roc_auc" in metrics else "not_evaluated",
            roc_auc=scalar("group_oof_roc_auc"),
            average_precision=scalar("group_oof_average_precision"),
            expected_calibration_error=scalar("group_expected_calibration_error"),
        ),
        "temporal_holdout": track(
            "evaluated" if "temporal_holdout_roc_auc" in metrics else "not_evaluated",
            roc_auc=scalar("temporal_holdout_roc_auc"),
            average_precision=scalar("temporal_holdout_average_precision"),
            expected_calibration_error=scalar("temporal_holdout_expected_calibration_error"),
            n_backbones=scalar("temporal_holdout_n_backbones"),
        ),
        "external_holdout": track(
            "evaluated" if "external_holdout_roc_auc" in metrics else "not_evaluated",
            roc_auc=scalar("external_holdout_roc_auc"),
            average_precision=scalar("external_holdout_average_precision"),
            n_backbones=scalar("external_holdout_n_backbones"),
            prevalence=scalar("external_holdout_prevalence"),
            bootstrap_roc_auc_ci_low=scalar("external_holdout_bootstrap_roc_auc_ci_low"),
            bootstrap_roc_auc_ci_high=scalar("external_holdout_bootstrap_roc_auc_ci_high"),
        ),
        "adversarial_leakage": track(
            "evaluated" if "max_single_feature_auc" in metrics else "not_evaluated",
            max_single_feature_auc=scalar("max_single_feature_auc"),
            suspicious_feature_count=scalar("suspicious_feature_count"),
        ),
    }
    return tracks
