from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import polars as pl

CLINICAL_TERMS = {"clinical", "hospital", "patient", "human"}
AGRICULTURAL_TERMS = {"livestock", "farm", "animal", "cattle", "pig", "poultry", "swine", "bovine", "ovine"}
ENVIRONMENTAL_TERMS = {"environmental", "soil", "water", "wastewater", "marine", "aquatic", "plant"}

# Major airport hub countries used for gravity-index heuristic
HUB_COUNTRIES = {
    "USA", "China", "United Kingdom", "Germany", "France",
    "Netherlands", "Japan", "Singapore", "UAE", "Canada",
    "Australia", "India", "Brazil", "South Korea", "Turkey",
}

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

# Order-level phylogenetic distance proxy (number of super-order hops)
ORDER_DISTANCE: dict[tuple[str, str], float] = {
    ("Enterobacterales", "Pseudomonadales"): 2.0,
    ("Enterobacterales", "Moraxellales"): 2.0,
    ("Enterobacterales", "Bacillales"): 3.0,
    ("Enterobacterales", "Lactobacillales"): 3.0,
    ("Enterobacterales", "Vibrionales"): 2.5,
    ("Enterobacterales", "Campylobacterales"): 3.0,
    ("Pseudomonadales", "Moraxellales"): 2.0,
    ("Pseudomonadales", "Bacillales"): 3.0,
    ("Pseudomonadales", "Lactobacillales"): 3.0,
    ("Bacillales", "Lactobacillales"): 1.5,
}


def _load_high_priority_genes(config_path: str | Path | None = None) -> set[str]:
    """Load WHO 2024 high-priority resistance gene symbols from project config."""
    if config_path is None:
        config_path = Path(__file__).resolve().parents[3] / "project_config" / "high_priority_genes.json"
    try:
        with open(config_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        genes: set[str] = set()
        for category in data.get("high_priority_genes", {}).values():
            genes.update(category.get("genes", []))
        return genes
    except Exception as exc:
        warnings.warn(f"Could not load high_priority_genes.json: {exc}. Falling back to core set.")
        return {"blaNDM", "blaKPC", "mcr-1", "vanA", "blaCTX-M"}


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
    # --- BioSpread v4.0 Infinite Architect additions ---
    one_health_niche_jump: float = 0.0
    phylogenetic_host_jump_distance: float = 0.0
    mobilization_synergy: float = 0.0
    mash_neighbor_distance: float = 0.0
    gravity_index: float = 0.0
    high_priority_gene_count: int = 0
    has_inc_f: int = 0
    has_inc_i: int = 0


@dataclass(frozen=True)
class FeatureConfig:
    split_year: int = 2020
    horizon_years: int = 3
    min_new_countries_for_spread: int = 2


def feature_rows_to_frame(rows: list[BackboneFeatureRow]) -> pl.DataFrame:
    return pl.DataFrame([asdict(row) for row in rows]) if rows else pl.DataFrame()


def _order_distance(order_a: str, order_b: str) -> float:
    if order_a == order_b:
        return 0.0
    key = tuple(sorted([order_a, order_b]))
    return ORDER_DISTANCE.get(key, 3.5)


def _as_list(value: Any) -> list[Any]:
    """Normalize Polars list/series/null to a plain Python list."""
    if value is None:
        return []
    if hasattr(value, "to_list"):
        return value.to_list()  # type: ignore[union-attr]
    return list(value)


def _compute_niche_jump_score(clinical_contexts: Any) -> float:
    """Score 0-3 based on how many One Health niches are occupied."""
    items = _as_list(clinical_contexts)
    text = " ".join(str(c).lower() for c in items)
    has_clinical = any(term in text for term in CLINICAL_TERMS)
    has_agri = any(term in text for term in AGRICULTURAL_TERMS)
    has_env = any(term in text for term in ENVIRONMENTAL_TERMS)
    return float(has_clinical + has_agri + has_env)


def _compute_phylo_jump_distance(host_genera: Any) -> float:
    """Average pairwise phylogenetic distance between observed host genera."""
    genera = _as_list(host_genera)
    orders = [GENUS_TO_ORDER.get(g, f"Unknown_{g}") for g in genera]
    unique_orders = list(set(orders))
    if len(unique_orders) <= 1:
        return 0.0
    distances = []
    for i in range(len(unique_orders)):
        for j in range(i + 1, len(unique_orders)):
            distances.append(_order_distance(unique_orders[i], unique_orders[j]))
    return float(np.mean(distances)) if distances else 0.0


def _compute_gravity_index(countries: Any) -> float:
    """Higher for plasmids observed in major air-travel hubs (global connectivity proxy)."""
    items = _as_list(countries)
    return float(sum(1.0 for c in items if str(c).strip() in HUB_COUNTRIES)) / max(len(items), 1)


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
        unique_backbones: pl.Series | list[Any] = pre_df_records["backbone_id"].unique() if "backbone_id" in pre_df_records.columns else []
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

    df = records.join(taxonomy_df, on="host_genus", how="left", coalesce=True).with_columns(
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
        pl.col("host_genus").unique().alias("pre_host_genera"),
        pl.col("clinical_context").unique().alias("pre_clinical_contexts"),
    ])

    # Future period (label generation)
    future_df = df.filter((pl.col("year") > split_year) & (pl.col("year") <= split_year + horizon_years)).group_by("backbone_id").agg([
        pl.col("country").unique().alias("future_countries"),
    ])

    # Join and calculate final features
    features = pre_df.join(future_df, on="backbone_id", how="left", coalesce=True).with_columns([
        pl.col("future_countries").fill_null([]),
    ]).with_columns([
        pl.col("future_countries").list.set_difference(pl.col("pre_countries")).list.len().alias("n_new_countries_future")
    ]).with_columns([
        pl.when(pl.col("n_new_countries_future") >= 2).then(1).otherwise(0).alias("label_geo_spread"),
        ((pl.col("n_records_pre").clip(upper_bound=8) / 8.0 +
          pl.col("n_countries_pre").clip(upper_bound=5) / 5.0) / 2.0).clip(0.0, 1.0).alias("knownness_score"),
    ]).with_columns([
        # --- BioSpread v4.0 biological intelligence ---
        pl.col("pre_clinical_contexts").map_elements(
            lambda x: _compute_niche_jump_score(x), return_dtype=pl.Float64
        ).alias("one_health_niche_jump"),
        pl.col("pre_host_genera").map_elements(
            lambda x: _compute_phylo_jump_distance(x), return_dtype=pl.Float64
        ).alias("phylogenetic_host_jump_distance"),
        pl.col("pre_countries").map_elements(
            lambda x: _compute_gravity_index(x), return_dtype=pl.Float64
        ).alias("gravity_index"),
    ]).with_columns([
        (pl.col("phylogenetic_host_jump_distance") * pl.col("one_health_niche_jump") * pl.col("mean_mobility_score_pre")).alias("mobilization_synergy"),
    ])

    # Mash distance: gracefully degrade if column missing
    if "mash_neighbor_distance" in df.columns:
        mash_df = df.filter(pl.col("year") <= split_year).group_by("backbone_id").agg(
            pl.col("mash_neighbor_distance").mean().alias("mash_neighbor_distance")
        )
        features = features.join(mash_df, on="backbone_id", how="left", coalesce=True).with_columns(
            pl.col("mash_neighbor_distance").fill_null(0.5)
        )
    else:
        features = features.with_columns(pl.lit(0.5).alias("mash_neighbor_distance"))
        warnings.warn("`mash_neighbor_distance` not found in input; defaulting to 0.5.")

    out = features.drop(["pre_countries", "future_countries", "pre_host_genera", "pre_clinical_contexts"]).sort("backbone_id")
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
                one_health_niche_jump=float(row.get("one_health_niche_jump", 0.0)),
                phylogenetic_host_jump_distance=float(row.get("phylogenetic_host_jump_distance", 0.0)),
                mobilization_synergy=float(row.get("mobilization_synergy", 0.0)),
                mash_neighbor_distance=float(row.get("mash_neighbor_distance", 0.5)),
                gravity_index=float(row.get("gravity_index", 0.0)),
                high_priority_gene_count=int(row.get("high_priority_gene_count", 0)),
                has_inc_f=int(row.get("has_inc_f", 0)),
                has_inc_i=int(row.get("has_inc_i", 0)),
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

    df = records.join(taxonomy_df, on="host_genus", how="left", coalesce=True).with_columns(
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
            pl.col("host_genus").unique().alias("pre_host_genera"),
            pl.col("clinical_context").unique().alias("pre_clinical_contexts"),
        ]
    )
    future_df = df.filter((pl.col("year") > config.split_year) & (pl.col("year") <= config.split_year + config.horizon_years)).group_by(
        "backbone_id"
    ).agg([pl.col("country").unique().alias("future_countries")])
    return (
        pre_df.join(future_df, on="backbone_id", how="left", coalesce=True)
        .with_columns(
            [
                pl.col("future_countries").fill_null(pl.lit([], dtype=pl.List(pl.String))),
                pl.col("pre_countries").fill_null(pl.lit([], dtype=pl.List(pl.String))),
                pl.col("pre_host_genera").fill_null(pl.lit([], dtype=pl.List(pl.String))),
                pl.col("pre_clinical_contexts").fill_null(pl.lit([], dtype=pl.List(pl.String))),
            ]
        )
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
    )


def enrich_features_with_amr_flags(
    features_df: pl.DataFrame,
    amr_df: pl.DataFrame,
    config_path: str | Path | None = None,
) -> pl.DataFrame:
    """Add boolean flags for high-priority genes (WHO 2024) and IncF/IncI replicon types.

    Parameters
    ----------
    features_df : pl.DataFrame
        Backbone feature surface with at least `backbone_id`.
    amr_df : pl.DataFrame
        AMR hits table containing `backbone_id` and `gene_symbol` columns.
    config_path : optional
        Path to ``high_priority_genes.json``.
    """
    genes = _load_high_priority_genes(config_path)
    if genes and {"backbone_id", "gene_symbol"}.issubset(set(amr_df.columns)):
        gene_counts = (
            amr_df.filter(pl.col("gene_symbol").is_in(list(genes)))
            .group_by("backbone_id")
            .len()
            .rename({"len": "high_priority_gene_count"})
        )
        features_df = features_df.join(gene_counts, on="backbone_id", how="left", coalesce=True).with_columns(
            pl.col("high_priority_gene_count").fill_null(0).cast(pl.Int64)
        )
    else:
        features_df = features_df.with_columns(pl.lit(0).alias("high_priority_gene_count"))

    # Replicon type flags (IncF, IncI) from raw records if available
    # NOTE: callers should join replicon data beforehand or pass it in `amr_df` with a `replicon_types` column.
    if "replicon_types" in amr_df.columns:
        rep_flags = (
            amr_df.select(["backbone_id", "replicon_types"])
            .with_columns([
                pl.col("replicon_types").str.contains("IncF").cast(pl.Int64).alias("has_inc_f"),
                pl.col("replicon_types").str.contains("IncI").cast(pl.Int64).alias("has_inc_i"),
            ])
            .group_by("backbone_id")
            .agg([
                pl.col("has_inc_f").max().alias("has_inc_f"),
                pl.col("has_inc_i").max().alias("has_inc_i"),
            ])
        )
        features_df = features_df.join(rep_flags, on="backbone_id", how="left", coalesce=True).with_columns([
            pl.col("has_inc_f").fill_null(0).cast(pl.Int64),
            pl.col("has_inc_i").fill_null(0).cast(pl.Int64),
        ])
    else:
        features_df = features_df.with_columns([
            pl.lit(0).alias("has_inc_f"),
            pl.lit(0).alias("has_inc_i"),
        ])
    return features_df
