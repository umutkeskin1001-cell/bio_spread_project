from __future__ import annotations

from pathlib import Path

import polars as pl


def read_table(path: str | Path, *, schema_overrides: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(source)
    if suffix in {".tsv", ".tab"}:
        return pl.read_csv(source, separator="\t", schema_overrides=schema_overrides)
    if suffix == ".csv":
        return pl.read_csv(source, separator=",", schema_overrides=schema_overrides)
    raise ValueError(f"Unsupported table format: {source}")


def scan_table(path: str | Path, *, schema_overrides: dict[str, pl.DataType] | None = None) -> pl.LazyFrame:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pl.scan_parquet(source)
    if suffix in {".tsv", ".tab"}:
        return pl.scan_csv(source, separator="\t", schema_overrides=schema_overrides)
    if suffix == ".csv":
        return pl.scan_csv(source, separator=",", schema_overrides=schema_overrides)
    raise ValueError(f"Unsupported table format: {source}")


def load_observations(path: str | Path) -> pl.LazyFrame:
    df = scan_table(path)
    schema = df.schema
    required = {
        "backbone_id",
        "year",
        "country",
        "host_genus",
        "clinical_context",
        "amr_gene_count",
        "mobility_score",
    }
    missing = sorted(required - set(schema.names()))
    if missing:
        raise ValueError(f"Input records missing required columns: {', '.join(missing)}")
        
    return df.select(
        [
            pl.col("backbone_id").cast(pl.String),
            pl.col("year").cast(pl.Int64),
            pl.col("country").cast(pl.String),
            pl.col("host_genus").cast(pl.String),
            pl.col("clinical_context").cast(pl.String),
            pl.col("amr_gene_count").cast(pl.Float64),
            pl.col("mobility_score").cast(pl.Float64),
        ]
    ).filter(
        (pl.col("backbone_id").is_not_null()) &
        (pl.col("year").is_between(1900, 2100)) &
        (pl.col("amr_gene_count") >= 0.0) &
        (pl.col("mobility_score").is_between(0.0, 1.0))
    )


def load_amr_weights(path: str | Path) -> pl.LazyFrame:
    amr = scan_table(path)
    schema = amr.schema
    if "NUCCORE_ACC" not in schema.names():
        raise ValueError("AMR table missing NUCCORE_ACC column")

    gene_col = "gene_symbol" if "gene_symbol" in schema.names() else "gene_name"
    if gene_col not in schema.names():
        return amr.select(pl.col("NUCCORE_ACC")).group_by("NUCCORE_ACC").agg(pl.lit(0.0).alias("amr_gene_count"))

    weighted = amr.with_columns(pl.col(gene_col).cast(pl.String).fill_null("").alias("gene_id"))
    weight_expr = pl.lit(1.0)
    for pattern, weight in {
        "blaNDM": 5.0,
        "blaKPC": 5.0,
        "blaOXA-48": 5.0,
        "blaVIM": 5.0,
        "blaIMP": 5.0,
        "mcr-1": 5.0,
        "blaCTX-M": 3.0,
        "vanA": 3.0,
        "vanB": 3.0,
    }.items():
        weight_expr = pl.when(pl.col("gene_id").str.contains(pattern)).then(pl.lit(weight)).otherwise(weight_expr)

    return weighted.group_by("NUCCORE_ACC").agg(weight_expr.sum().alias("amr_gene_count"))

