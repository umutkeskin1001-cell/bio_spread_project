from __future__ import annotations

import polars as pl


def project_outbreaks_prevented(predictions: pl.DataFrame, slope: float = 2.0, intercept: float = 0.0) -> pl.DataFrame:
    if "risk_probability" not in predictions.columns:
        return predictions
    return predictions.with_columns((pl.col("risk_probability").cast(pl.Float64) * slope + intercept).alias("expected_outbreaks_prevented"))
