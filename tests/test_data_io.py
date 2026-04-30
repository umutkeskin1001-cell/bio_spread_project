from dataclasses import asdict

import polars as pl

from bio_spread_project.data import PlasmidRecord, load_records


def test_load_records_accepts_parquet_input(tmp_path):
    rows = [
        PlasmidRecord(
            backbone_id="bb1",
            year=2020,
            country="TR",
            host_genus="Klebsiella",
            clinical_context="clinical",
            amr_gene_count=2.0,
            mobility_score=0.7,
        )
    ]
    path = tmp_path / "records.parquet"
    pl.DataFrame([asdict(row) for row in rows]).write_parquet(path)

    loaded = load_records(path)

    assert loaded == rows
