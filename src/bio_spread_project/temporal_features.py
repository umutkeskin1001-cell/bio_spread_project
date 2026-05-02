from __future__ import annotations

import numpy as np
import polars as pl


def _slope_from_series(years: list[int], values: list[float]) -> float:
    if len(years) < 2:
        return 0.0
    x = np.asarray(years, dtype=np.float64)
    y = np.asarray(values, dtype=np.float64)
    x_centered = x - x.mean()
    denom = float(np.sum(x_centered * x_centered))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(x_centered * (y - y.mean())) / denom)


def build_temporal_trend_features(records: pl.DataFrame, *, split_year: int) -> pl.DataFrame:
    if records.is_empty() or "backbone_id" not in records.columns:
        return pl.DataFrame(
            {
                "backbone_id": [],
                "country_slope_train": [],
                "host_breadth_slope_train": [],
                "mobility_shift_slope_train": [],
                "recent_expansion_flag": [],
            }
        )

    required = {"backbone_id", "year", "country", "host_genus", "mobility_score"}
    if not required.issubset(set(records.columns)):
        return pl.DataFrame(
            {
                "backbone_id": records["backbone_id"].unique() if "backbone_id" in records.columns else [],
                "country_slope_train": [],
                "host_breadth_slope_train": [],
                "mobility_shift_slope_train": [],
                "recent_expansion_flag": [],
            }
        )

    pre = records.filter(pl.col("year") <= split_year)
    if pre.is_empty():
        return pl.DataFrame(
            {
                "backbone_id": records["backbone_id"].unique(),
                "country_slope_train": [0.0] * records["backbone_id"].n_unique(),
                "host_breadth_slope_train": [0.0] * records["backbone_id"].n_unique(),
                "mobility_shift_slope_train": [0.0] * records["backbone_id"].n_unique(),
                "recent_expansion_flag": [0] * records["backbone_id"].n_unique(),
            }
        )

    out_rows: list[dict[str, float | int | str]] = []
    for bb in pre["backbone_id"].unique().to_list():
        bb_df = pre.filter(pl.col("backbone_id") == bb)
        year_min = int(bb_df["year"].min())
        years = list(range(year_min, split_year + 1))
        cum_countries: list[float] = []
        cum_hosts: list[float] = []
        mean_mob: list[float] = []
        seen_countries: set[str] = set()
        seen_hosts: set[str] = set()
        country_counts_by_year: dict[int, int] = {}

        for yr in years:
            year_slice = bb_df.filter(pl.col("year") == yr)
            countries = set(year_slice["country"].drop_nulls().cast(pl.Utf8).to_list())
            hosts = set(year_slice["host_genus"].drop_nulls().cast(pl.Utf8).to_list())
            seen_countries |= countries
            seen_hosts |= hosts
            cum_countries.append(float(len(seen_countries)))
            cum_hosts.append(float(len(seen_hosts)))
            if year_slice.height > 0:
                mean_mob.append(float(year_slice["mobility_score"].cast(pl.Float64).mean()))
            else:
                mean_mob.append(mean_mob[-1] if mean_mob else 0.0)
            country_counts_by_year[yr] = len(countries)

        recent_years = [split_year - 1, split_year]
        recent_expansion = int(any(country_counts_by_year.get(yr, 0) > 0 for yr in recent_years))

        out_rows.append(
            {
                "backbone_id": str(bb),
                "country_slope_train": _slope_from_series(years, cum_countries),
                "host_breadth_slope_train": _slope_from_series(years, cum_hosts),
                "mobility_shift_slope_train": _slope_from_series(years, mean_mob),
                "recent_expansion_flag": recent_expansion,
            }
        )

    return pl.DataFrame(out_rows)
