from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import polars as pl

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PlasmidRecord:
    backbone_id: str
    year: int | None
    country: str
    host_genus: str
    clinical_context: str
    amr_gene_count: float
    mobility_score: float

    def __post_init__(self) -> None:
        if not self.backbone_id:
            raise ValueError("backbone_id must be non-empty")
        if self.year is not None and self.year < 0:
            raise ValueError(f"year outside accepted range: {self.year}")
        if not math.isfinite(self.amr_gene_count) or self.amr_gene_count < 0.0:
            raise ValueError("amr_gene_count must be finite and non-negative")
        if not math.isfinite(self.mobility_score) or not 0.0 <= self.mobility_score <= 1.0:
            raise ValueError("mobility_score must be finite in [0, 1]")


def read_table(path: str | Path, *, schema_overrides: dict[str, pl.DataType] | None = None) -> pl.DataFrame:
    """Read CSV, TSV, or Parquet tables with extension-based dispatch."""
    input_path = Path(path)
    suffix = input_path.suffix.lower()
    if suffix == ".parquet":
        return pl.read_parquet(input_path)
    if suffix in {".tsv", ".tab"}:
        return pl.read_csv(input_path, separator="\t", schema_overrides=schema_overrides)
    if suffix == ".csv":
        return pl.read_csv(input_path, separator=",", schema_overrides=schema_overrides)
    raise ValueError(f"Unsupported table format: {input_path}")


def load_records(path: str | Path) -> list[PlasmidRecord]:
    """Load plasmid observation records using Polars."""
    df = read_table(path)
    required = {"backbone_id", "year", "country", "host_genus", "clinical_context", "amr_gene_count", "mobility_score"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {', '.join(missing)}")

    non_id_columns = [column for column in required if column != "backbone_id"]
    blank_id = pl.col("backbone_id").is_null() | (pl.col("backbone_id").cast(pl.String).str.strip_chars() == "")
    has_other_data = pl.any_horizontal([pl.col(column).is_not_null() for column in non_id_columns])

    if df.filter(blank_id & has_other_data).height > 0:
        raise ValueError("backbone_id must be present for every non-empty record")

    df = df.filter(~blank_id).with_columns(
        [
            pl.col("year").fill_null(0).cast(pl.Int64),
            pl.col("amr_gene_count").fill_null(0.0).cast(pl.Float64),
            pl.col("mobility_score").fill_null(0.0).cast(pl.Float64),
        ]
    )

    # Filter years
    df = df.filter((pl.col("year") <= 2100) & ((pl.col("year") >= 1900) | (pl.col("year") == 0)))

    cleaned = df.with_columns([
        pl.col("backbone_id").cast(pl.String),
        pl.col("country").fill_null("unknown").cast(pl.String),
        pl.col("host_genus").fill_null("unknown").cast(pl.String),
        pl.col("clinical_context").fill_null("unknown").cast(pl.String),
        pl.col("amr_gene_count").clip(lower_bound=0.0),
        pl.col("mobility_score").clip(lower_bound=0.0, upper_bound=1.0),
    ])
    return [
        PlasmidRecord(
            backbone_id=str(row["backbone_id"]),
            year=int(row["year"]) if row["year"] is not None else None,
            country=str(row["country"]),
            host_genus=str(row["host_genus"]),
            clinical_context=str(row["clinical_context"]),
            amr_gene_count=float(row["amr_gene_count"]),
            mobility_score=float(row["mobility_score"]),
        )
        for row in cleaned.to_dicts()
    ]


def _derive_mobility(df: pl.DataFrame) -> pl.Expr:
    """Vectorized mobility score derivation."""
    if "predicted_mobility" in df.columns:
        predicted_expr = pl.col("predicted_mobility").fill_null("").cast(pl.String).str.to_lowercase()
    else:
        predicted_expr = pl.lit("")

    score = pl.when(predicted_expr.str.contains("conjug")).then(1.0) \
              .when(predicted_expr.str.contains("mobil")).then(0.7) \
              .otherwise(0.0)

    def is_true(col: str) -> pl.Expr:
        if col not in df.columns:
            return pl.lit(False)
        c = pl.col(col)
        if df.get_column(col).dtype == pl.Boolean:
            return c
        return c.cast(pl.String).str.to_lowercase().is_in(["1", "true", "yes", "y", "t"])

    support = pl.lit(0.0)
    support += pl.when(is_true("has_mpf")).then(0.35).otherwise(0.0)
    support += pl.when(is_true("has_relaxase")).then(0.25).otherwise(0.0)
    support += pl.when(is_true("has_orit")).then(0.20).otherwise(0.0)
    support += pl.when(is_true("is_mobilizable")).then(0.20).otherwise(0.0)

    return pl.max_horizontal(score, support).clip(0.0, 1.0)


def _derive_clinical_context(df: pl.DataFrame) -> pl.Expr:
    """Vectorized clinical context derivation."""
    cols = ["BIOSAMPLE_pathogenicity", "BIOSAMPLE_package", "ECOSYSTEM_tags", "DISEASE_tags", "record_origin"]
    present_cols = [c for c in cols if c in df.columns]
    combined = pl.concat_str([pl.col(c).fill_null("") for c in present_cols], separator=" ").str.to_lowercase()

    return pl.when(combined.str.contains("clinical|hospital|patient|pathogen|disease|human")).then(pl.lit("clinical")) \
             .when(combined.str.contains("animal|host-associated|host associated")).then(pl.lit("host_associated")) \
             .when(combined.str.contains("food")).then(pl.lit("food")) \
             .when(combined.str.contains("soil|water|environment")).then(pl.lit("environment")) \
             .otherwise(pl.lit("unknown"))


PRIORITY_AMR_WEIGHTS: dict[str, float] = {
    "blaNDM": 5.0, "blaKPC": 5.0, "blaOXA-48": 5.0, "blaVIM": 5.0, "blaIMP": 5.0,
    "mcr-1": 5.0, "blaCTX-M": 3.0, "vanA": 3.0, "vanB": 3.0,
}


def _calculate_weighted_amr(amr_df: pl.DataFrame) -> pl.DataFrame:
    """Calculate priority-weighted AMR score per accession."""
    if "gene_symbol" not in amr_df.columns and "gene_name" not in amr_df.columns:
        return amr_df.group_by("NUCCORE_ACC").agg(pl.lit(0.0).alias("amr_gene_count"))

    amr_df = amr_df.with_columns(
        pl.coalesce([pl.col(c) for c in ["gene_symbol", "gene_name"] if c in amr_df.columns]).alias("gene_id")
    )

    weight_expr = pl.lit(1.0)
    for pattern, weight in PRIORITY_AMR_WEIGHTS.items():
        weight_expr = pl.when(pl.col("gene_id").str.contains(pattern)).then(weight).otherwise(weight_expr)

    return amr_df.group_by("NUCCORE_ACC").agg(weight_expr.unique().sum().alias("amr_gene_count"))


def load_backbone_records(
    path: str | Path,
    *,
    amr_path: str | Path | None = None,
    limit: int | None = None,
) -> list[PlasmidRecord]:
    """Load raw records and derive model observations using Polars."""
    df = read_table(path)
    if limit:
        df = df.head(limit)

    amr_gene_count = pl.lit(0.0)
    if amr_path and Path(amr_path).exists():
        amr_df = read_table(amr_path)
        amr_counts = _calculate_weighted_amr(amr_df)
        df = df.join(amr_counts, left_on="sequence_accession", right_on="NUCCORE_ACC", how="left")
        amr_gene_count = pl.col("amr_gene_count").fill_null(0.0).cast(pl.Float64)

    backbone_id = pl.coalesce([pl.col(c) for c in ["backbone_id", "canonical_id", "sequence_accession"] if c in df.columns])
    year = pl.col("resolved_year").fill_null(0).cast(pl.Int64)

    out = df.with_columns([
        backbone_id.alias("backbone_id"),
        year.alias("year"),
        amr_gene_count.alias("amr_gene_count"),
        _derive_mobility(df).alias("mobility_score"),
        _derive_clinical_context(df).alias("clinical_context"),
        pl.col("country").fill_null("unknown").alias("country"),
        pl.col("genus").fill_null("unknown").alias("host_genus"),
    ]).filter(pl.col("year") > 0)
    return [
        PlasmidRecord(
            backbone_id=str(row["backbone_id"]),
            year=int(row["year"]) if row["year"] is not None else None,
            country=str(row["country"]),
            host_genus=str(row["host_genus"]),
            clinical_context=str(row["clinical_context"]),
            amr_gene_count=float(row["amr_gene_count"]),
            mobility_score=float(row["mobility_score"]),
        )
        for row in out.to_dicts()
    ]
