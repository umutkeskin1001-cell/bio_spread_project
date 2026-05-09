from __future__ import annotations

from pathlib import Path

import polars as pl

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
    if all_records.is_empty():
        return

    candidate: tuple[str, int] | None = None
    backbones = all_records["backbone_id"].unique().to_list()
    for bb in backbones:
        bb_df = all_records.filter(pl.col("backbone_id") == bb).sort("year")
        years = bb_df["year"].unique().sort().to_list()
        for y in years:
            subset = bb_df.filter(pl.col("year") <= y)
            if subset.height < 2:
                continue
            if "label_geo_spread" in subset.columns:
                labels = subset["label_geo_spread"].fill_null(0).cast(pl.Int64)
            elif "n_new_countries_future" in subset.columns:
                labels = subset["n_new_countries_future"].fill_null(0).cast(pl.Int64).gt(0).cast(pl.Int64)
            else:
                continue
            if labels.n_unique() >= 2:
                candidate = (str(bb), int(y))
                break
        if candidate is not None:
            break

    if candidate is None:
        return

    backbone_id, earliest_year = candidate
    risk, alarm = ghost_backbone_test(str(fixture), backbone_id, earliest_year, output_dir=str(tmp_path / "ghost"))
    assert risk > 0.0
    assert alarm >= 0.0
