from dataclasses import asdict

import polars as pl
import pytest

from bio_spread_project.data import PlasmidRecord, load_backbone_records_frame, load_records


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


def test_load_records_rejects_invalid_values_instead_of_rewriting(tmp_path):
    path = tmp_path / "bad_records.csv"
    path.write_text(
        "\n".join(
            [
                "backbone_id,year,country,host_genus,clinical_context,amr_gene_count,mobility_score",
                "bb_bad,1899,TR,Klebsiella,clinical,-1,1.5",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid"):
        load_records(path)


def test_load_backbone_records_frame_reports_missing_raw_columns(tmp_path):
    path = tmp_path / "raw.tsv"
    path.write_text(
        "\n".join(
            [
                "sequence_accession\tresolved_year\tgenus",
                "acc1\t2020\tKlebsiella",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="missing required columns: country"):
        load_backbone_records_frame(path)


def test_load_backbone_records_frame_accepts_minimal_raw_table(tmp_path):
    path = tmp_path / "raw.tsv"
    path.write_text(
        "\n".join(
            [
                "sequence_accession\tresolved_year\tcountry\tgenus",
                "acc1\t2020\tTR\tKlebsiella",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out = load_backbone_records_frame(path)

    assert out.select("backbone_id", "year", "country", "host_genus", "clinical_context", "mobility_score").to_dicts() == [
        {
            "backbone_id": "acc1",
            "year": 2020,
            "country": "TR",
            "host_genus": "Klebsiella",
            "clinical_context": "unknown",
            "mobility_score": 0.0,
        }
    ]
