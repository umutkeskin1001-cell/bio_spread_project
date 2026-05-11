import polars as pl
from dataclasses import dataclass
from typing import Optional

# Canonical Schema for Plasmid Records
PLASMID_SCHEMA = {
    "backbone_id": pl.Utf8,
    "sequence_accession": pl.Utf8,
    "year": pl.Int32,
    "country": pl.Utf8,
    "host_genus": pl.Utf8,
    "clinical_context": pl.Utf8,
    "mobility_score": pl.Float64,
    "T_eff_norm": pl.Float64,
    "spread_label": pl.Int32
}

@dataclass(frozen=True)
class DataContract:
    """Strict data container for validated project inputs."""
    records: pl.DataFrame
    genetic_map: dict[str, list[str]]
    vocab: dict[str, int]
