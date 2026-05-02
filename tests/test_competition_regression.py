from pathlib import Path

from bio_spread_project.orchestrator import run_pipeline

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GEO_SPREAD_FEATURES = PROJECT_ROOT / "data" / "project_inputs" / "geo_spread" / "inputs" / "backbone_scored.tsv"


def test_packaged_competition_run_stays_above_metric_floor(tmp_path):
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=GEO_SPREAD_FEATURES,
        output_dir=tmp_path / "competition_regression",
    )

    assert result.metrics["roc_auc"] >= 0.82
    assert result.metrics["average_precision"] >= 0.74
    assert result.metrics["bootstrap_average_precision_ci_low"] > result.metrics["prevalence"]
    assert result.metrics["bootstrap_average_precision_ci_low"] < result.metrics["bootstrap_average_precision_ci_high"]
    assert result.metrics["max_single_feature_auc"] < 0.95
    assert result.metrics["suspicious_feature_count"] == 0.0
