from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class TemporalWindow:
    train_end_year: int
    test_start_year: int
    test_end_year: int


def build_rolling_temporal_windows(
    years: list[int],
    *,
    min_train_years: int = 3,
    gap_years: int = 1,
    test_span_years: int = 1,
) -> list[TemporalWindow]:
    if min_train_years < 1:
        raise ValueError("min_train_years must be >= 1")
    if gap_years < 0:
        raise ValueError("gap_years must be >= 0")
    if test_span_years < 1:
        raise ValueError("test_span_years must be >= 1")

    uniq = sorted(set(int(y) for y in years))
    if len(uniq) < (min_train_years + gap_years + test_span_years):
        return []

    windows: list[TemporalWindow] = []
    min_year = uniq[0]
    max_year = uniq[-1]
    for train_end in range(min_year + min_train_years - 1, max_year + 1):
        test_start = train_end + gap_years + 1
        test_end = test_start + test_span_years - 1
        if test_end > max_year:
            continue
        windows.append(
            TemporalWindow(
                train_end_year=train_end,
                test_start_year=test_start,
                test_end_year=test_end,
            )
        )
    return windows


def evaluate_temporal_consistency(
    *,
    oof_roc_auc: float,
    window_rocs: list[float],
    max_positive_delta: float = 0.03,
    max_peak_positive_delta: float = 0.06,
    max_range: float = 0.12,
) -> dict[str, Any]:
    if not window_rocs:
        return {
            "status": "not_evaluated",
            "reason": "no_temporal_windows",
            "window_count": 0,
        }

    arr = np.array(window_rocs, dtype=np.float64)
    median = float(np.median(arr))
    minimum = float(np.min(arr))
    maximum = float(np.max(arr))
    positive_delta = maximum - float(oof_roc_auc)
    median_delta = median - float(oof_roc_auc)
    spread = maximum - minimum
    flags: list[str] = []
    if median_delta > max_positive_delta:
        flags.append("temporal_too_good_vs_oof")
    if positive_delta > max_peak_positive_delta:
        flags.append("temporal_peak_too_good_vs_oof")
    if spread > max_range:
        flags.append("temporal_instability")
    return {
        "status": "pass" if not flags else "fail",
        "reason": "" if not flags else ", ".join(flags),
        "window_count": int(arr.size),
        "median_roc_auc": median,
        "min_roc_auc": minimum,
        "max_roc_auc": maximum,
        "roc_auc_spread": spread,
        "median_delta_vs_oof": float(median_delta),
        "max_positive_delta_vs_oof": float(positive_delta),
        "flags": flags,
    }
