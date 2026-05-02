from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Sequence

import polars as pl

CLINICAL_TERMS = {"clinical", "hospital", "patient", "human"}

GENUS_TO_ORDER: dict[str, str] = {
    "Escherichia": "Enterobacterales",
    "Klebsiella": "Enterobacterales",
    "Salmonella": "Enterobacterales",
    "Enterobacter": "Enterobacterales",
    "Pseudomonas": "Pseudomonadales",
    "Acinetobacter": "Moraxellales",
    "Staphylococcus": "Bacillales",
    "Enterococcus": "Lactobacillales",
    "Vibrio": "Vibrionales",
    "Campylobacter": "Campylobacterales",
}


@dataclass(frozen=True)
class BackboneFeatureRow:
    backbone_id: str
    n_records_pre: int
    n_countries_pre: int
    n_hosts_pre: int
    host_diversity_pre: int
    mean_amr_gene_count_pre: float
    mean_mobility_score_pre: float
    clinical_fraction_pre: float
    n_new_countries_future: int
    label_geo_spread: int
    knownness_score: float


@dataclass(frozen=True)
class FeatureConfig:
    split_year: int = 2020
    horizon_years: int = 3
    min_new_countries_for_spread: int = 2


def feature_rows_to_frame(rows: list[BackboneFeatureRow]) -> pl.DataFrame:
    return pl.DataFrame([asdict(row) for row in rows]) if rows else pl.DataFrame()


def build_surveillance_intensity(pre_df_records: pl.DataFrame, all_records: pl.LazyFrame, split_year: int) -> pl.DataFrame:
    country_agg = all_records.filter(pl.col("year") <= split_year).group_by("country").len().collect()
    
    unique_b_c = pre_df_records.select(["backbone_id", "country"]).drop_nulls().unique()
    return (
        unique_b_c
        .join(country_agg, on="country", how="left")
        .group_by("backbone_id")
        .agg(pl.col("len").mean().fill_null(0.0).alias("surv_intensity"))
    )

def build_host_sampling_entropy(pre_df_records: pl.DataFrame) -> pl.DataFrame:
    if "host_genus" not in pre_df_records.columns or pre_df_records.is_empty():
        unique_backbones = pre_df_records["backbone_id"].unique() if "backbone_id" in pre_df_records.columns else []
        return pl.DataFrame({"backbone_id": unique_backbones, "host_sampling_shannon": [0.0] * len(unique_backbones)})

    host_counts = (
        pre_df_records
        .group_by(["backbone_id", "host_genus"])
        .len()
        .with_columns(
            (pl.col("len") / pl.col("len").sum().over("backbone_id")).alias("p")
        )
        .with_columns((-pl.col("p") * pl.col("p").log()).alias("entropy_term"))
        .group_by("backbone_id")
        .agg(pl.sum("entropy_term").fill_null(0.0).alias("host_sampling_shannon"))
    )
    return host_counts


def build_backbone_features(
    records: pl.DataFrame | Sequence[object],
    *,
    split_year: int,
    horizon_years: int,
) -> list[BackboneFeatureRow]:
    """Build one feature row per backbone using only records up to split_year."""

    if not isinstance(records, pl.DataFrame):
        records = pl.DataFrame([asdict(record) if is_dataclass(record) and not isinstance(record, type) else record for record in records])

    # Taxonomy mapping
    taxonomy_df = pl.DataFrame({
        "host_genus": list(GENUS_TO_ORDER.keys()),
        "host_order": list(GENUS_TO_ORDER.values()),
    })

    df = records.join(taxonomy_df, on="host_genus", how="left").with_columns(
        pl.col("host_order").fill_null(pl.concat_str([pl.lit("Unknown_"), pl.col("host_genus")]))
    )

    # Pre-split period features
    pre_df = df.filter(pl.col("year") <= split_year).group_by("backbone_id").agg([
        pl.len().alias("n_records_pre"),
        pl.col("country").n_unique().alias("n_countries_pre"),
        pl.col("host_genus").n_unique().alias("n_hosts_pre"),
        pl.col("host_order").n_unique().alias("host_diversity_pre"),
        pl.col("amr_gene_count").mean().alias("mean_amr_gene_count_pre"),
        pl.col("mobility_score").mean().alias("mean_mobility_score_pre"),
        pl.col("clinical_context").str.to_lowercase().is_in(CLINICAL_TERMS).mean().alias("clinical_fraction_pre"),
        pl.col("country").unique().alias("pre_countries"),
    ])

    # Future period (label generation)
    future_df = df.filter((pl.col("year") > split_year) & (pl.col("year") <= split_year + horizon_years)).group_by("backbone_id").agg([
        pl.col("country").unique().alias("future_countries"),
    ])

    # Join and calculate final features
    features = pre_df.join(future_df, on="backbone_id", how="left").with_columns([
        pl.col("future_countries").fill_null([]),
    ]).with_columns([
        pl.col("future_countries").list.set_difference(pl.col("pre_countries")).list.len().alias("n_new_countries_future")
    ]).with_columns([
        pl.when(pl.col("n_new_countries_future") >= 2).then(1).otherwise(0).alias("label_geo_spread"),
        ((pl.col("n_records_pre").clip(upper_bound=8) / 8.0 +
          pl.col("n_countries_pre").clip(upper_bound=5) / 5.0) / 2.0).clip(0.0, 1.0).alias("knownness_score")
    ])

    out = features.drop(["pre_countries", "future_countries"]).sort("backbone_id")
    rows: list[BackboneFeatureRow] = []
    for row in out.to_dicts():
        rows.append(
            BackboneFeatureRow(
                backbone_id=str(row["backbone_id"]),
                n_records_pre=int(row["n_records_pre"]),
                n_countries_pre=int(row["n_countries_pre"]),
                n_hosts_pre=int(row["n_hosts_pre"]),
                host_diversity_pre=int(row["host_diversity_pre"]),
                mean_amr_gene_count_pre=float(row["mean_amr_gene_count_pre"]),
                mean_mobility_score_pre=float(row["mean_mobility_score_pre"]),
                clinical_fraction_pre=float(row["clinical_fraction_pre"]),
                n_new_countries_future=int(row["n_new_countries_future"]),
                label_geo_spread=int(row["label_geo_spread"]),
                knownness_score=float(row["knownness_score"]),
            )
        )
    return rows


def build_backbone_features_lazy(
    records: pl.LazyFrame,
    *,
    config: FeatureConfig,
) -> pl.LazyFrame:
    taxonomy_df = pl.DataFrame(
        {
            "host_genus": list(GENUS_TO_ORDER.keys()),
            "host_order": list(GENUS_TO_ORDER.values()),
        }
    ).lazy()

    df = records.join(taxonomy_df, on="host_genus", how="left").with_columns(
        pl.col("host_order").fill_null(pl.concat_str([pl.lit("Unknown_"), pl.col("host_genus")]))
    )

    pre_df = df.filter(pl.col("year") <= config.split_year).group_by("backbone_id").agg(
        [
            pl.len().alias("n_records_pre"),
            pl.col("country").n_unique().alias("n_countries_pre"),
            pl.col("host_genus").n_unique().alias("n_hosts_pre"),
            pl.col("host_order").n_unique().alias("host_diversity_pre"),
            pl.col("amr_gene_count").mean().alias("mean_amr_gene_count_pre"),
            pl.col("mobility_score").mean().alias("mean_mobility_score_pre"),
            pl.col("clinical_context").str.to_lowercase().is_in(CLINICAL_TERMS).mean().alias("clinical_fraction_pre"),
            pl.col("country").unique().alias("pre_countries"),
        ]
    )
    future_df = df.filter((pl.col("year") > config.split_year) & (pl.col("year") <= config.split_year + config.horizon_years)).group_by(
        "backbone_id"
    ).agg([pl.col("country").unique().alias("future_countries")])
    return (
        pre_df.join(future_df, on="backbone_id", how="left")
        .with_columns(pl.col("future_countries").fill_null([]))
        .with_columns(
            pl.col("future_countries").list.set_difference(pl.col("pre_countries")).list.len().alias("n_new_countries_future")
        )
        .with_columns(
            [
                pl.when(pl.col("n_new_countries_future") >= config.min_new_countries_for_spread)
                .then(1)
                .otherwise(0)
                .alias("label_geo_spread"),
                (
                    (
                        pl.col("n_records_pre").clip(upper_bound=8) / 8.0
                        + pl.col("n_countries_pre").clip(upper_bound=5) / 5.0
                    )
                    / 2.0
                )
                .clip(0.0, 1.0)
                .alias("knownness_score"),
            ]
        )
        .drop(["pre_countries", "future_countries"])
        .sort("backbone_id")
    )
