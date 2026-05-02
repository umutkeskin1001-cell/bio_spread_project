from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from bio_spread_project.data import read_table
from bio_spread_project.embeddings import EmbeddingStore
from bio_spread_project.grps import compute_grps

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EnrichmentFlags:
    enable_intrinsic_plasmid_features: bool = True
    enable_host_traits: bool = True
    enable_country_indicators: bool = True
    enable_temporal_trends: bool = True
    enable_amr_diversity: bool = True
    enable_phylogenetic_proximity: bool = False
    enable_phylo_spatial_embedding: bool = False
    enable_rank_focal_loss: bool = False
    enable_soft_country_debiasing: bool = False
    enable_grps: bool = True
    enable_gated_fusion: bool = True
    use_reliability_propensity: bool = True
    enable_conformal: bool = True
    enable_synergy_interactions: bool = True
    enable_phylo_propagation: bool = True
    enable_evidential_meta: bool = True


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_enrichment_flags(config_path: Path) -> EnrichmentFlags:
    if not config_path.exists():
        return EnrichmentFlags()
    values: dict[str, bool] = {}
    for line in config_path.read_text(encoding="utf-8").splitlines():
        clean = line.strip()
        if not clean or clean.startswith("#") or ":" not in clean:
            continue
        key, raw = clean.split(":", 1)
        values[key.strip()] = _parse_bool(raw)
    return EnrichmentFlags(
        enable_intrinsic_plasmid_features=values.get("enable_intrinsic_plasmid_features", True),
        enable_host_traits=values.get("enable_host_traits", True),
        enable_country_indicators=values.get("enable_country_indicators", True),
        enable_temporal_trends=values.get("enable_temporal_trends", True),
        enable_amr_diversity=values.get("enable_amr_diversity", True),
        enable_phylogenetic_proximity=values.get("enable_phylogenetic_proximity", False),
        enable_phylo_spatial_embedding=values.get("enable_phylo_spatial_embedding", False),
        enable_rank_focal_loss=values.get("enable_rank_focal_loss", False),
        enable_soft_country_debiasing=values.get("enable_soft_country_debiasing", False),
        enable_grps=values.get("enable_grps", True),
        enable_gated_fusion=values.get("enable_gated_fusion", True),
        use_reliability_propensity=values.get("use_reliability_propensity", True),
        enable_conformal=values.get("enable_conformal", True),
        enable_synergy_interactions=values.get("enable_synergy_interactions", True),
        enable_phylo_propagation=values.get("enable_phylo_propagation", True),
        enable_evidential_meta=values.get("enable_evidential_meta", True),
    )


def add_grps_to_features(
    features: pl.DataFrame,
    embedding_store: EmbeddingStore,
    *,
    knownness_threshold: float = 0.3,
) -> pl.DataFrame:
    if features.is_empty() or "knownness_score" not in features.columns:
        return features
    ids = (
        features.filter(pl.col("knownness_score").fill_null(0.5) < knownness_threshold)["backbone_id"]
        .cast(pl.Utf8)
        .to_list()
    )
    if not ids:
        return features.with_columns(pl.lit(0.0).alias("grps")) if "grps" not in features.columns else features

    emb = embedding_store.get(features["backbone_id"].cast(pl.Utf8).to_list(), kind="esm2")
    if not emb.is_empty():
        emb_cols_all = [c for c in emb.columns if c != "backbone_id"]
        emb = emb.group_by("backbone_id").agg([pl.col(c).cast(pl.Float64).mean().alias(c) for c in emb_cols_all])
    emb_cols = [c for c in emb.columns if c.startswith("esm2_embed_")]
    if emb.is_empty() or not emb_cols:
        return features.with_columns(pl.lit(0.0).alias("grps")) if "grps" not in features.columns else features

    labels = features.select(["backbone_id", "label_geo_spread"])
    grps_df = compute_grps(ids, emb, labels, emb_cols)
    out = features.join(grps_df, on="backbone_id", how="left", coalesce=True)
    return out.with_columns(pl.col("grps").fill_null(0.0))


def augment_intrinsic_plasmid_features(features: pl.DataFrame, external_dir: Path) -> tuple[pl.DataFrame, bool]:
    path = external_dir / "plasmid_intrinsic_props.tsv"
    if not path.exists():
        return features, False
    lookup = read_table(path)
    if lookup.is_empty() or "backbone_id" not in lookup.columns:
        return features, False
    merged = features.join(lookup, on="backbone_id", how="left", coalesce=True)
    if "inc_group" in merged.columns:
        merged = merged.with_columns(pl.col("inc_group").cast(pl.Utf8).fill_null("UNK").hash(seed=17).cast(pl.Float64).alias("inc_group_code"))
    if "mob_typer" in merged.columns:
        merged = merged.with_columns(pl.col("mob_typer").cast(pl.Utf8).fill_null("UNK").hash(seed=23).cast(pl.Float64).alias("mob_typer_code"))
    numeric_dtypes = {pl.Int8, pl.Int16, pl.Int32, pl.Int64, pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64, pl.Float32, pl.Float64}
    numeric_cols = [c for c in lookup.columns if c != "backbone_id" and lookup[c].dtype in numeric_dtypes]
    cat_cols = [c for c in lookup.columns if c != "backbone_id" and c not in numeric_cols]
    exprs = [pl.col(c).fill_null(0.0).fill_nan(0.0) for c in numeric_cols]
    exprs.extend(pl.col(c).fill_null("UNK") for c in cat_cols)
    return merged.with_columns(exprs), True


def augment_host_trait_features(features: pl.DataFrame, records: pl.DataFrame, external_dir: Path, *, split_year: int) -> tuple[pl.DataFrame, bool]:
    path = external_dir / "host_traits.tsv"
    if not path.exists() or records.is_empty():
        return features, False
    traits = read_table(path)
    needed = {"genus", "is_pathogen", "environment_primary", "gram_stain", "metabolism"}
    if traits.is_empty() or not needed.issubset(set(traits.columns)):
        return features, False

    pre = records.filter(pl.col("year") <= split_year)
    if pre.is_empty():
        return features, False
    with_traits = pre.join(traits, left_on="host_genus", right_on="genus", how="left", coalesce=True)

    rows: list[dict[str, float | str]] = []
    for bb in with_traits["backbone_id"].unique().to_list():
        bb_df = with_traits.filter(pl.col("backbone_id") == bb)
        n = max(1, bb_df.height)
        pathogen_vals = bb_df["is_pathogen"].cast(pl.Utf8).fill_null("").str.to_lowercase().to_list()
        frac_pathogenic = float(sum(v in {"1", "true", "yes", "y"} for v in pathogen_vals)) / float(n)
        env_div = float(bb_df["environment_primary"].drop_nulls().n_unique()) if "environment_primary" in bb_df.columns else 0.0
        metab_div = float(bb_df["metabolism"].drop_nulls().n_unique()) if "metabolism" in bb_df.columns else 0.0
        gram_entropy = 0.0
        if "gram_stain" in bb_df.columns:
            gram_counts = bb_df.group_by("gram_stain").len()
            total = max(1.0, float(gram_counts["len"].sum()))
            for c in gram_counts["len"].to_list():
                p = float(c) / total
                if p > 0.0:
                    gram_entropy += -p * math.log(p)

        rows.append(
            {
                "backbone_id": str(bb),
                "frac_pathogenic_hosts": frac_pathogenic,
                "env_diversity": env_div,
                "gram_stain_entropy": gram_entropy,
                "metabolic_diversity": metab_div,
            }
        )

    agg = pl.DataFrame(rows)
    return features.join(agg, on="backbone_id", how="left", coalesce=True).with_columns(
        [
            pl.col("frac_pathogenic_hosts").fill_null(0.0),
            pl.col("env_diversity").fill_null(0.0),
            pl.col("gram_stain_entropy").fill_null(0.0),
            pl.col("metabolic_diversity").fill_null(0.0),
        ]
    ), True


def augment_country_indicator_features(features: pl.DataFrame, records: pl.DataFrame, external_dir: Path, *, split_year: int) -> tuple[pl.DataFrame, bool]:
    path = external_dir / "country_indicators_train.csv"
    if not path.exists() or records.is_empty():
        return features, False
    indicators = read_table(path)
    req = {"country", "avg_antibiotic_consumption_2015_2020", "health_exp_pc", "tourists_per_year"}
    if indicators.is_empty() or not req.issubset(set(indicators.columns)):
        return features, False

    pre = records.filter(pl.col("year") <= split_year).select(["backbone_id", "country"]).drop_nulls().unique()
    if pre.is_empty():
        return features, False
    joined = pre.join(indicators, on="country", how="left", coalesce=True)
    agg = joined.group_by("backbone_id").agg(
        [
            pl.col("avg_antibiotic_consumption_2015_2020").cast(pl.Float64).mean().fill_null(0.0).alias("mean_antibiotic_pressure"),
            pl.col("health_exp_pc").cast(pl.Float64).max().fill_null(0.0).alias("max_health_exp"),
            pl.col("tourists_per_year").cast(pl.Float64).std().fill_null(0.0).alias("tourist_entropy"),
        ]
    )
    return features.join(agg, on="backbone_id", how="left", coalesce=True).with_columns(
        [
            pl.col("mean_antibiotic_pressure").fill_null(0.0),
            pl.col("max_health_exp").fill_null(0.0),
            pl.col("tourist_entropy").fill_null(0.0),
        ]
    ), True


def augment_amr_diversity_features(features: pl.DataFrame, records: pl.DataFrame, amr_path: Path | None) -> tuple[pl.DataFrame, bool]:
    if amr_path is None or not amr_path.exists() or records.is_empty() or "sequence_accession" not in records.columns:
        return features, False
    amr = read_table(amr_path)
    if amr.is_empty() or "NUCCORE_ACC" not in amr.columns:
        return features, False
    symbol_col = "gene_symbol" if "gene_symbol" in amr.columns else ("gene_name" if "gene_name" in amr.columns else None)
    if symbol_col is None:
        return features, False

    amr = amr.with_columns(pl.col(symbol_col).cast(pl.Utf8).fill_null("").alias("gene"))
    cls = (
        pl.when(pl.col("gene").str.contains("kpc|ndm|vim|imp|oxa", literal=False)).then(pl.lit("carbapenemase"))
        .when(pl.col("gene").str.contains("ctx-m|tem|shv", literal=False)).then(pl.lit("esbl"))
        .when(pl.col("gene").str.contains("mcr", literal=False)).then(pl.lit("colistin"))
        .otherwise(pl.lit("other"))
        .alias("amr_class")
    )
    amr = amr.with_columns(cls)

    map_df = records.select(["backbone_id", "sequence_accession"]).drop_nulls().unique()
    joined = map_df.join(amr, left_on="sequence_accession", right_on="NUCCORE_ACC", how="inner")
    if joined.is_empty():
        return features, False

    counts = joined.group_by(["backbone_id", "amr_class"]).len()
    totals = counts.group_by("backbone_id").agg(pl.col("len").sum().alias("total"))
    probs = counts.join(totals, on="backbone_id", how="left").with_columns((pl.col("len") / pl.col("total").clip(1, None)).alias("p"))
    shannon = probs.with_columns((-pl.col("p") * pl.col("p").log()).alias("h")).group_by("backbone_id").agg(pl.col("h").sum().alias("amr_class_shannon"))

    pivot = counts.pivot(on="amr_class", index="backbone_id", values="len", aggregate_function="sum").rename(
        {
            "carbapenemase": "carbapenemase_count",
            "esbl": "esbl_count",
            "colistin": "colistin_resistance_count",
            "other": "other_amr_count",
        }
    )
    out = pivot.join(shannon, on="backbone_id", how="left", coalesce=True)
    for col in ["carbapenemase_count", "esbl_count", "colistin_resistance_count", "other_amr_count", "amr_class_shannon"]:
        if col not in out.columns:
            out = out.with_columns(pl.lit(0.0).alias(col))
    out = out.with_columns([pl.col(c).fill_null(0.0).cast(pl.Float64) for c in out.columns if c != "backbone_id"])
    return features.join(out, on="backbone_id", how="left", coalesce=True).with_columns(
        [pl.col(c).fill_null(0.0) for c in out.columns if c != "backbone_id"]
    ), True


def augment_phylogenetic_proximity(features: pl.DataFrame, records: pl.DataFrame, external_dir: Path, *, split_year: int) -> tuple[pl.DataFrame, bool]:
    path = external_dir / "mash_distances.csv"
    if not path.exists() or records.is_empty():
        return features, False
    dists = read_table(path)
    req = {"backbone_id_a", "backbone_id_b", "distance"}
    if dists.is_empty() or not req.issubset(set(dists.columns)):
        return features, False

    pre = records.filter(pl.col("year") <= split_year)
    top5 = pre.group_by("backbone_id").agg(pl.col("country").n_unique().alias("n_countries_pre")).sort("n_countries_pre", descending=True).head(5)
    if top5.is_empty():
        return features, False
    travellers = set(top5["backbone_id"].to_list())

    min_dist_rows: list[dict[str, float | str]] = []
    for bb in features["backbone_id"].to_list():
        cand = dists.filter(
            ((pl.col("backbone_id_a") == bb) & (pl.col("backbone_id_b").is_in(list(travellers))))
            | ((pl.col("backbone_id_b") == bb) & (pl.col("backbone_id_a").is_in(list(travellers))))
        )
        min_dist = float(cand["distance"].cast(pl.Float64).min()) if cand.height > 0 else 1.0
        min_dist_rows.append({"backbone_id": str(bb), "min_dist_to_top5_traveller": min_dist})

    out = pl.DataFrame(min_dist_rows)
    return features.join(out, on="backbone_id", how="left", coalesce=True).with_columns(
        pl.col("min_dist_to_top5_traveller").fill_null(1.0)
    ), True


def drop_all_missing_columns(df: pl.DataFrame) -> tuple[pl.DataFrame, list[str]]:
    drop_cols: list[str] = []
    for col in df.columns:
        if col in {"backbone_id", "label_geo_spread", "n_new_countries_future", "knownness_score", "region"}:
            continue
        if df[col].null_count() == df.height:
            drop_cols.append(col)
    if drop_cols:
        logger.warning("Dropping all-missing enrichment columns: %s", ", ".join(sorted(drop_cols)))
        return df.drop(drop_cols), drop_cols
    return df, []
