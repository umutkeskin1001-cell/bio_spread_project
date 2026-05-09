import json

import pytest

from bio_spread_project.governance import TrendThresholds, evaluate_model_registry_trend, load_trend_thresholds


def test_trend_evaluation_exposes_required_history_when_insufficient():
    report = evaluate_model_registry_trend(
        entries=[{"roc_auc": 0.85, "average_precision": 0.80, "all_quality_gates_passed": True} for _ in range(8)],
        window_size=5,
        thresholds=TrendThresholds(required_entries_for_go=20),
    )
    assert report["status"] == "insufficient_data"
    assert report["all_passed"] is False
    assert report["trend_evidence_sufficient"] is False
    assert report["required_entries"] == 20


def test_invalid_required_entries_for_go_raises_validation_error(tmp_path):
    invalid = tmp_path / "invalid_trend.json"
    invalid.write_text(json.dumps({"required_entries_for_go": 0}), encoding="utf-8")
    with pytest.raises(ValueError, match="required_entries_for_go"):
        load_trend_thresholds(invalid)


def test_fractional_required_entries_for_go_raises_validation_error(tmp_path):
    invalid = tmp_path / "invalid_trend_fractional.json"
    invalid.write_text(json.dumps({"required_entries_for_go": 2.5}), encoding="utf-8")
    with pytest.raises(ValueError, match="required_entries_for_go"):
        load_trend_thresholds(invalid)
