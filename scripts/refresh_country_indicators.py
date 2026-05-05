from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import fmean
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import polars as pl


WB_BASE = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
OWID_ANTIBIOTIC_CSV = "https://ourworldindata.org/grapher/antibiotic-consumption-rate.csv"


def _http_json(url: str) -> Any:
    req = Request(url, headers={"User-Agent": "bio-spread-project/1.0"})
    with urlopen(req, timeout=30) as response:  # nosec B310 - trusted HTTPS endpoints
        return json.loads(response.read().decode("utf-8"))


def _http_bytes(url: str) -> bytes:
    req = Request(url, headers={"User-Agent": "bio-spread-project/1.0"})
    with urlopen(req, timeout=30) as response:  # nosec B310 - trusted HTTPS endpoints
        return response.read()


def _world_bank_indicator_mean(country_iso2: str, indicator: str, start_year: int, end_year: int) -> float | None:
    params = urlencode(
        {
            "format": "json",
            "per_page": 200,
            "date": f"{start_year}:{end_year}",
        }
    )
    payload = _http_json(f"{WB_BASE.format(country=country_iso2, indicator=indicator)}?{params}")
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[1], list):
        return None
    values = [entry.get("value") for entry in payload[1] if isinstance(entry, dict)]
    numeric = [float(v) for v in values if v is not None]
    if not numeric:
        return None
    return float(fmean(numeric))


def _load_antibiotic_consumption() -> pl.DataFrame:
    # OWID country column usually uses ISO3 under "Code", metric column can vary;
    # we resolve by known prefix.
    csv_bytes = _http_bytes(OWID_ANTIBIOTIC_CSV)
    tmp = Path("/tmp/owid_antibiotic.csv")
    tmp.write_bytes(csv_bytes)
    df = pl.read_csv(tmp)
    metric_candidates = [c for c in df.columns if "defined daily doses" in c.lower()]
    if not metric_candidates:
        raise ValueError("OWID antibiotic dataset does not include expected metric column")
    metric = metric_candidates[0]
    return (
        df.select(["Code", "Year", metric])
        .rename({"Code": "iso3", "Year": "year", metric: "antibiotic"})
        .with_columns(
            [
                pl.col("iso3").cast(pl.Utf8),
                pl.col("year").cast(pl.Int64),
                pl.col("antibiotic").cast(pl.Float64),
            ]
        )
    )


def _iso2_to_iso3_map() -> dict[str, str]:
    # minimal set; extend as needed
    return {
        "AR": "ARG",
        "BR": "BRA",
        "CL": "CHL",
        "DE": "DEU",
        "FR": "FRA",
        "IT": "ITA",
        "NL": "NLD",
        "PE": "PER",
        "PK": "PAK",
        "TR": "TUR",
        "US": "USA",
    }


def refresh_country_indicators(
    *,
    records_path: Path,
    output_path: Path,
    start_year: int = 2015,
    end_year: int = 2020,
) -> None:
    records = pl.read_csv(records_path)
    countries = sorted(records["country"].drop_nulls().cast(pl.Utf8).unique().to_list())
    if not countries:
        raise ValueError("No countries found in records file")

    antibiotics = _load_antibiotic_consumption()
    iso_map = _iso2_to_iso3_map()
    rows: list[dict[str, float | str]] = []
    for iso2 in countries:
        iso3 = iso_map.get(iso2)
        ab_value: float | None = None
        if iso3 is not None:
            ab_slice = antibiotics.filter((pl.col("iso3") == iso3) & (pl.col("year") >= start_year) & (pl.col("year") <= end_year))
            if ab_slice.height > 0:
                ab_value = float(ab_slice["antibiotic"].drop_nulls().mean())

        health_exp = _world_bank_indicator_mean(iso2, "SH.XPD.CHEX.PC.CD", start_year, end_year)
        tourists = _world_bank_indicator_mean(iso2, "ST.INT.ARVL", start_year, end_year)
        rows.append(
            {
                "country": iso2,
                "avg_antibiotic_consumption_2015_2020": float(ab_value or 0.0),
                "health_exp_pc": float(health_exp or 0.0),
                "tourists_per_year": float(tourists or 0.0),
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "country",
                "avg_antibiotic_consumption_2015_2020",
                "health_exp_pc",
                "tourists_per_year",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    refresh_country_indicators(
        records_path=Path("data/sample_plasmid_records.csv"),
        output_path=Path("data/external/country_indicators_train.csv"),
        start_year=2015,
        end_year=2020,
    )
