from __future__ import annotations

from bio_spread_project.validation_protocol_v2 import (
    build_rolling_temporal_windows,
    evaluate_temporal_consistency,
)


def test_build_rolling_temporal_windows_basic() -> None:
    years = [2016, 2017, 2018, 2019, 2020, 2021]
    windows = build_rolling_temporal_windows(years, min_train_years=3, gap_years=1, test_span_years=1)
    assert windows
    assert windows[0].train_end_year == 2018
    assert windows[0].test_start_year == 2020
    assert windows[0].test_end_year == 2020


def test_build_rolling_temporal_windows_empty_for_short_history() -> None:
    years = [2020, 2021]
    windows = build_rolling_temporal_windows(years, min_train_years=3, gap_years=1, test_span_years=1)
    assert windows == []


def test_evaluate_temporal_consistency_flags_too_good() -> None:
    result = evaluate_temporal_consistency(oof_roc_auc=0.90, window_rocs=[0.94, 0.95], max_positive_delta=0.03)
    assert result["status"] == "fail"
    assert "temporal_too_good_vs_oof" in result["flags"]


def test_evaluate_temporal_consistency_passes_stable_windows() -> None:
    result = evaluate_temporal_consistency(oof_roc_auc=0.90, window_rocs=[0.89, 0.90, 0.91], max_positive_delta=0.03, max_range=0.12)
    assert result["status"] == "pass"
    assert result["flags"] == []
