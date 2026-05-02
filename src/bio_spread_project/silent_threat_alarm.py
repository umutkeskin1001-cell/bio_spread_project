from __future__ import annotations

import numpy as np
import polars as pl

from bio_spread_project.conformal import compute_alarm_score


def compute_alarm_scores(predictions: pl.DataFrame) -> pl.DataFrame:
    if predictions.is_empty() or "risk_probability" not in predictions.columns:
        return predictions
    lower = predictions["lower"].cast(pl.Float64).to_numpy() if "lower" in predictions.columns else np.zeros(predictions.height)
    upper = predictions["upper"].cast(pl.Float64).to_numpy() if "upper" in predictions.columns else np.ones(predictions.height)
    known = predictions["knownness_score"].cast(pl.Float64).to_numpy() if "knownness_score" in predictions.columns else np.full(predictions.height, 0.5)
    interval_width = upper - lower
    alarm = compute_alarm_score(predictions["risk_probability"].cast(pl.Float64).to_numpy(), interval_width, known)
    return predictions.with_columns(pl.Series("alarm_score", alarm))
