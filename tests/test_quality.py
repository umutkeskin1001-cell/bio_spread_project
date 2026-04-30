from bio_spread_project.quality import QualityThresholds, evaluate_quality_gates


def test_geo_quality_gates_fail_when_group_and_temporal_metrics_are_missing():
    gates = evaluate_quality_gates(
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
    )
    assert gates["group_auc_at_least_target"] is False
    assert gates["temporal_holdout_auc_at_least_target"] is False


def test_geo_quality_does_not_backfill_temporal_from_spatial_group_oof():
    gates = evaluate_quality_gates(
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
    )

    assert gates["group_auc_at_least_target"] is True
    assert gates["temporal_holdout_auc_at_least_target"] is False
    assert gates["external_holdout_auc_at_least_target"] is False
