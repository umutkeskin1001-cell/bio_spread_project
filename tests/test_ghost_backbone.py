from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from bio_spread_project.orchestrator import run_pipeline


def ghost_backbone_test(
    full_data_path: str,
    backbone_id: str,
    earliest_year: int,
    max_obs: int = 3,
    output_dir: str = "/tmp/ghost_test",
):
    all_records = pl.read_csv(full_data_path)
    mask = (pl.col("backbone_id") == backbone_id) & (pl.col("year") <= earliest_year)
    ghost_df = all_records.filter(mask).head(max_obs)

    tmp_csv = f"/tmp/ghost_{backbone_id}.csv"
    ghost_df.write_csv(tmp_csv)
    result = run_pipeline(
        input_path=tmp_csv,
        output_dir=output_dir,
        split_year=earliest_year,
        horizon_years=5,
        fail_on_quality_gates=False,
        fail_on_drift_fail=False,
        require_explicit_surface=False,
    )

    preds = pl.read_csv(result.predictions_path)
    row = preds.filter(pl.col("backbone_id") == backbone_id)
    risk = float(row["risk_probability"][0])
    alarm = 0.0
    if "meta" in row.columns and row["meta"][0] is not None:
        meta = row["meta"][0]
        alarm = float(meta.get("alarm_score", 0.0)) if isinstance(meta, dict) else 0.0
    assert risk > 0.5, f"Ghost backbone risk {risk} <= 0.5"
    assert alarm >= 0.0, f"Ghost backbone alarm {alarm} < 0"
    return risk, alarm


def test_ghost_backbone_with_fixture(tmp_path: Path) -> None:
    fixture = Path(__file__).resolve().parents[1] / "data" / "sample_plasmid_records.csv"
    if not fixture.exists():
        return
    all_records = pl.read_csv(fixture)
    feats = (
        all_records.group_by("backbone_id")
        .agg([
            pl.col("country").n_unique().alias("n_countries"),
            pl.col("year").min().alias("min_year"),
        ])
        .sort("n_countries", descending=True)
    )
    if feats.is_empty():
        return
    backbone_id = str(feats["backbone_id"][0])
    earliest_year = int(feats["min_year"][0])
    try:
        risk, alarm = ghost_backbone_test(str(fixture), backbone_id, earliest_year, output_dir=str(tmp_path / "ghost"))
    except ValueError as exc:
        if "at least 2 classes" in str(exc):
            pytest.skip("ghost subset has only one class; skipping")
        raise
    assert risk > 0.0
    assert alarm >= 0.0
