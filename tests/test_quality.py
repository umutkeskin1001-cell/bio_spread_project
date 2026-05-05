from bio_spread_project.governance import QualityThresholds
from bio_spread_project.governance import evaluate_quality_gates as evaluate_quality_checks


def test_geo_quality_gates_fail_when_group_and_temporal_metrics_are_missing():
    gates = {
        c.name: c.passed
        for c in evaluate_quality_checks(
        metrics={
            "validation_mode": "cross_validated",
            "roc_auc": 0.9,
            "average_precision": 0.8,
            "prevalence": 0.3,
            "expected_calibration_error": 0.02,
            "bootstrap_roc_auc_ci_low": 0.8,
            "bootstrap_average_precision_ci_low": 0.5,
            "max_single_feature_auc": 0.7,
            "suspicious_feature_count": 0.0,
        },
        input_mode="geo_reliability_feature_surface",
        leakage_audit_passed=True,
        thresholds=QualityThresholds(external_holdout_required=True),
    )}
    assert gates["group_auc_at_least_target"] is False
    assert gates["temporal_holdout_auc_at_least_target"] is False


def test_geo_quality_does_not_backfill_temporal_from_spatial_group_oof():
    gates = {
        c.name: c.passed
        for c in evaluate_quality_checks(
        metrics={
            "validation_mode": "spatial_group_cv_stacked",
            "roc_auc": 0.99,
            "average_precision": 0.90,
            "prevalence": 0.10,
            "expected_calibration_error": 0.01,
            "bootstrap_roc_auc_ci_low": 0.95,
            "bootstrap_average_precision_ci_low": 0.80,
            "group_oof_roc_auc": 0.95,
            "max_single_feature_auc": 0.50,
            "suspicious_feature_count": 0.0,
        },
        input_mode="geo_reliability_feature_surface",
        leakage_audit_passed=True,
        thresholds=QualityThresholds(external_holdout_required=True),
    )}

    assert gates["group_auc_at_least_target"] is True
    assert gates["temporal_holdout_auc_at_least_target"] is False
    assert gates["external_holdout_auc_at_least_target"] is False


def test_geo_quality_requires_external_holdout_by_default():
    gates = {
        c.name: c.passed
        for c in evaluate_quality_checks(
        metrics={
            "validation_mode": "spatial_group_cv_stacked",
            "roc_auc": 0.9,
            "average_precision": 0.8,
            "prevalence": 0.3,
            "expected_calibration_error": 0.02,
            "bootstrap_roc_auc_ci_low": 0.8,
            "bootstrap_average_precision_ci_low": 0.5,
            "group_oof_roc_auc": 0.86,
            "temporal_holdout_roc_auc": 0.81,
            "max_single_feature_auc": 0.7,
            "suspicious_feature_count": 0.0,
        },
        input_mode="geo_reliability_feature_surface",
        leakage_audit_passed=True,
        thresholds=QualityThresholds(),
    )}

    assert gates["external_holdout_auc_at_least_target"] is False


def test_geo_quality_fails_temporal_consistency_when_flagged() -> None:
    gates = {
        c.name: c.passed
        for c in evaluate_quality_checks(
            metrics={
                "validation_mode": "spatial_group_cv_stacked",
                "roc_auc": 0.9,
                "average_precision": 0.8,
                "prevalence": 0.3,
                "expected_calibration_error": 0.02,
                "bootstrap_roc_auc_ci_low": 0.8,
                "bootstrap_average_precision_ci_low": 0.5,
                "group_oof_roc_auc": 0.86,
                "temporal_holdout_roc_auc": 0.81,
                "temporal_consistency_status": "fail",
                "max_single_feature_auc": 0.7,
                "suspicious_feature_count": 0.0,
            },
            input_mode="geo_reliability_feature_surface",
            leakage_audit_passed=True,
            thresholds=QualityThresholds(external_holdout_required=False, temporal_consistency_required=True),
        )
    }
    assert gates["temporal_consistency_passed"] is False


def test_geo_quality_fails_when_feature_lineage_unknown_count_exceeds_threshold() -> None:
    gates = {
        c.name: c.passed
        for c in evaluate_quality_checks(
            metrics={
                "validation_mode": "spatial_group_cv_stacked",
                "roc_auc": 0.9,
                "average_precision": 0.8,
                "prevalence": 0.3,
                "expected_calibration_error": 0.02,
                "bootstrap_roc_auc_ci_low": 0.8,
                "bootstrap_average_precision_ci_low": 0.5,
                "group_oof_roc_auc": 0.86,
                "temporal_holdout_roc_auc": 0.81,
                "feature_lineage_status": "pass",
                "feature_lineage_unknown_count": 2.0,
                "max_single_feature_auc": 0.7,
                "suspicious_feature_count": 0.0,
            },
            input_mode="geo_reliability_feature_surface",
            leakage_audit_passed=True,
            thresholds=QualityThresholds(
                external_holdout_required=False,
                temporal_consistency_required=False,
                feature_lineage_required=True,
                max_unknown_lineage_count=0,
            ),
        )
    }
    assert gates["feature_lineage_passed"] is False


def test_geo_quality_fails_when_disabled_feature_leak_count_nonzero() -> None:
    gates = {
        c.name: c.passed
        for c in evaluate_quality_checks(
            metrics={
                "validation_mode": "spatial_group_cv_stacked",
                "roc_auc": 0.9,
                "average_precision": 0.8,
                "prevalence": 0.3,
                "expected_calibration_error": 0.02,
                "max_calibration_bin_gap": 0.05,
                "bootstrap_roc_auc_ci_low": 0.8,
                "bootstrap_average_precision_ci_low": 0.5,
                "group_oof_roc_auc": 0.86,
                "temporal_holdout_roc_auc": 0.81,
                "feature_lineage_status": "pass",
                "feature_lineage_unknown_count": 0.0,
                "disabled_feature_leak_count": 1.0,
                "max_single_feature_auc": 0.7,
                "suspicious_feature_count": 0.0,
            },
            input_mode="geo_reliability_feature_surface",
            leakage_audit_passed=True,
            thresholds=QualityThresholds(
                external_holdout_required=False,
                temporal_consistency_required=False,
                feature_lineage_required=True,
                max_unknown_lineage_count=0,
                max_disabled_feature_leak_count=0,
            ),
        )
    }
    assert gates["disabled_feature_leak_passed"] is False
