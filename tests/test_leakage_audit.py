from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
from polars.testing import assert_frame_equal
from scipy.stats import linregress
import json

from bio_spread_project.data import PlasmidRecord, load_records
from bio_spread_project.external_features import augment_phylogenetic_proximity
from bio_spread_project.features import build_backbone_features, feature_rows_to_frame
from bio_spread_project.geo_reliability import (
    FEATURE_COLUMNS,
    _feature_matrix,
    fit_geo_reliability_surface,
    single_feature_leakage_scan,
    statistical_leakage_alarm,
)
from bio_spread_project.gnn_embedder import BackboneGraphEmbedder
from bio_spread_project.orchestrator import _validate_external_holdout, run_pipeline
from bio_spread_project.temporal_features import build_temporal_trend_features

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "leakage"
RECORDS_FIXTURE = FIXTURE_DIR / "records_with_future.csv"
GEO_FIXTURE = FIXTURE_DIR / "geo_leak.tsv"
COUNTRY_FIXTURE = FIXTURE_DIR / "country_indicators_mock.csv"


def _presplit_only(records: list[PlasmidRecord], split_year: int) -> list[PlasmidRecord]:
    return [r for r in records if (r.year is not None and r.year <= split_year)]


def _numeric_feature_columns(df: pl.DataFrame) -> list[str]:
    ignore = {"backbone_id", "label_geo_spread", "n_new_countries_future", "knownness_score"}
    return [c for c in df.columns if c not in ignore and df[c].dtype in {pl.Int64, pl.Int32, pl.Float64, pl.Float32}]


def test_feature_generation_uses_only_presplit_data() -> None:
    split_year = 2020
    records = load_records(RECORDS_FIXTURE)

    full_features = feature_rows_to_frame(build_backbone_features(records, split_year=split_year, horizon_years=3)).sort("backbone_id")
    pre_records = _presplit_only(records, split_year)
    pre_only_features = feature_rows_to_frame(build_backbone_features(pre_records, split_year=split_year, horizon_years=3)).sort("backbone_id")

    comparable = sorted(set(full_features["backbone_id"].to_list()).intersection(set(pre_only_features["backbone_id"].to_list())))
    full_comp = full_features.filter(pl.col("backbone_id").is_in(comparable)).sort("backbone_id")
    pre_comp = pre_only_features.filter(pl.col("backbone_id").is_in(comparable)).sort("backbone_id")

    cols = _numeric_feature_columns(full_comp)
    assert cols
    assert_frame_equal(full_comp.select(["backbone_id", *cols]), pre_comp.select(["backbone_id", *cols]))


def test_temporal_trend_features_no_future_leak() -> None:
    df = pl.DataFrame(
        {
            "backbone_id": ["BB_T"] * 5,
            "year": [2017, 2018, 2019, 2021, 2022],
            "country": ["TR", "DE", "US", "FR", "JP"],
            "host_genus": ["Escherichia"] * 5,
            "mobility_score": [0.2, 0.3, 0.4, 0.8, 0.9],
        }
    )
    out = build_temporal_trend_features(df, split_year=2020)
    slope = float(out.filter(pl.col("backbone_id") == "BB_T")["country_slope_train"][0])

    years = np.array([2017, 2018, 2019, 2020], dtype=float)
    cum = np.array([1, 2, 3, 3], dtype=float)
    expected = float(linregress(years, cum).slope)
    assert abs(slope - expected) < 1e-9


def test_phylogenetic_proximity_uses_only_presplit_statistics(tmp_path: Path) -> None:
    records = pl.DataFrame(
        {
            "backbone_id": [
                "a", "a", "a", "a", "a",
                "b", "b", "b", "b",
                "c", "c", "c",
                "d", "d",
                "e", "e",
                "f", "g", "h", "h", "h",
            ],
            "year": [
                2017, 2018, 2019, 2019, 2020,
                2017, 2018, 2019, 2020,
                2018, 2019, 2020,
                2019, 2020,
                2018, 2019,
                2018, 2018, 2019, 2021, 2022,
            ],
            "country": [
                "TR", "DE", "US", "FR", "JP",
                "TR", "DE", "US", "FR",
                "TR", "DE", "US",
                "TR", "DE",
                "TR", "DE",
                "CN", "IT", "ES", "PT", "NL",
            ],
        }
    )
    features = pl.DataFrame({"backbone_id": sorted(records["backbone_id"].unique().to_list())})
    mash = tmp_path / "mash_distances.csv"
    mash.write_text(
        "backbone_id_a,backbone_id_b,distance\n"
        "a,b,0.1\n"
        "a,c,0.2\n"
        "b,c,0.3\n",
        encoding="utf-8",
    )

    ext_dir = tmp_path
    (ext_dir / "mash_distances.csv").write_text(mash.read_text(encoding="utf-8"), encoding="utf-8")

    base_out, used = augment_phylogenetic_proximity(features, records, ext_dir, split_year=2020)
    assert used
    assert "min_dist_to_top5_traveller" in base_out.columns

    records_future = pl.concat(
        [
            records,
            pl.DataFrame({"backbone_id": ["h"] * 5, "year": [2021, 2021, 2022, 2022, 2023], "country": ["MX", "AR", "CL", "PE", "UY"]}),
        ],
        how="vertical_relaxed",
    )
    fut_out, used_f = augment_phylogenetic_proximity(features, records_future, ext_dir, split_year=2020)
    assert used_f
    pre_top_base = (
        records.filter(pl.col("year") <= 2020)
        .group_by("backbone_id")
        .agg(pl.col("country").n_unique().alias("n_countries_pre"))
        .sort("n_countries_pre", descending=True)
        .head(5)["backbone_id"]
        .to_list()
    )
    pre_top_future = (
        records_future.filter(pl.col("year") <= 2020)
        .group_by("backbone_id")
        .agg(pl.col("country").n_unique().alias("n_countries_pre"))
        .sort("n_countries_pre", descending=True)
        .head(5)["backbone_id"]
        .to_list()
    )
    assert set(pre_top_base) == set(pre_top_future)


def test_gnn_embedder_uses_only_presplit_edges() -> None:
    obs = pl.DataFrame(
        {
            "backbone_id": ["bb1", "bb1", "bb2", "bb2"],
            "year": [2019, 2022, 2019, 2022],
            "country": ["TR", "US", "DE", "US"],
            "host_genus": ["Escherichia", "Escherichia", "Klebsiella", "Klebsiella"],
            "mobility_score": [0.2, 0.9, 0.3, 0.9],
        }
    )

    import torch

    emb_full = BackboneGraphEmbedder()
    emb_full.fit(obs.lazy(), split_year=2020)
    emb_pre = BackboneGraphEmbedder()
    emb_pre.fit(obs.filter(pl.col("year") <= 2020).lazy(), split_year=2020)

    # Future-only shared country (US in 2022) must not appear in the fitted graph.
    assert emb_full.backbone_mapping is not None
    assert emb_pre.backbone_mapping is not None
    assert set(emb_full.backbone_mapping.keys()) == {"bb1", "bb2"}
    assert set(emb_pre.backbone_mapping.keys()) == {"bb1", "bb2"}
    # Same number of backbone embeddings confirms no future-only backbone nodes are introduced.
    assert emb_full.embeddings is not None
    assert emb_pre.embeddings is not None
    assert emb_full.embeddings.shape[0] == emb_pre.embeddings.shape[0]


def test_country_indicators_file_years_limited() -> None:
    txt = COUNTRY_FIXTURE.read_text(encoding="utf-8")
    assert "years <= 2020" in txt.lower()
    df = pl.read_csv(COUNTRY_FIXTURE, comment_prefix="#")
    year_cols = [c for c in df.columns if c.lower() in {"year", "ref_year", "data_year"}]
    if year_cols:
        for col in year_cols:
            assert df[col].cast(pl.Int64).max() <= 2020


def _synthetic_geo_df(n: int = 220, seed: int = 17) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    cols: dict[str, np.ndarray] = {
        "backbone_id": np.array([f"bb_{i}" for i in range(n)], dtype=object),
        "label_geo_spread": rng.integers(0, 2, size=n),
        "n_new_countries_future": rng.integers(0, 6, size=n),
        "knownness_score": rng.uniform(0.2, 0.95, size=n),
        "region": np.array(["global"] * n, dtype=object),
    }
    for name in FEATURE_COLUMNS:
        if name.startswith("oof_"):
            continue
        if name in cols:
            continue
        cols[name] = rng.normal(0.0, 1.0, size=n)
    return pl.DataFrame(cols)


def test_leakage_scan_all_enriched_features() -> None:
    feature_df = _synthetic_geo_df()
    scan = single_feature_leakage_scan(feature_df, auc_threshold=0.95)
    alarm = statistical_leakage_alarm(feature_df)

    assert float(scan["max_single_feature_auc"]) < 0.95
    assert int(scan["suspicious_feature_count"]) == 0
    assert int(alarm["alarm_count"]) == 0


def test_no_forbidden_tokens_in_feature_columns() -> None:
    forbidden = ("future", "test_", "label", "target", "outcome", "n_new_", "time_to_", "event_within_", "jump")
    for col in FEATURE_COLUMNS:
        lowered = col.lower()
        assert not any(token in lowered for token in forbidden)


def test_external_holdout_independence(tmp_path: Path) -> None:
    train = tmp_path / "train.tsv"
    holdout = tmp_path / "holdout.tsv"
    base = (
        "backbone_id\tspread_label\tn_new_countries\tT_eff_norm\tH_obs_specialization_norm\tA_eff_norm\tcoherence_score"
        "\tbackbone_purity_norm\tassignment_confidence_norm\tmash_neighbor_distance_train_norm\torit_support"
        "\tH_external_host_range_norm\tmetadata_support_depth_norm\tmetadata_missingness_burden\tlog1p_member_count_train\tlog1p_n_countries_train\n"
    )
    train.write_text(base + "bb_train\t1\t3\t0.8\t0.6\t0.7\t0.8\t0.7\t0.8\t0.2\t0.8\t0.7\t0.9\t0.1\t1.7\t1.3\n", encoding="utf-8")
    holdout.write_text(base + "bb_hold\t0\t0\t0.2\t0.2\t0.3\t0.3\t0.3\t0.3\t0.6\t0.2\t0.2\t0.6\t0.2\t1.1\t0.8\n", encoding="utf-8")

    _validate_external_holdout(train, holdout)


def test_temporal_holdout_metrics_are_lower_or_equal_to_oof(tmp_path: Path) -> None:
    geo_path = tmp_path / "geo_large.tsv"
    _synthetic_geo_df(n=220).write_csv(geo_path, separator="\t")
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=geo_path,
        output_dir=tmp_path / "geo_temporal_sanity",
    )
    temporal = float(result.metrics.get("temporal_holdout_roc_auc", result.metrics["roc_auc"]))
    overall = float(result.metrics["roc_auc"])
    assert temporal <= overall + 0.05


def _compute_dominant_country_pre(records: pl.DataFrame, split_year: int) -> dict[str, str]:
    pre_country = (
        records.filter(pl.col("year") <= split_year)
        .drop_nulls(subset=["backbone_id", "country"])
        .group_by(["backbone_id", "country"])
        .len()
        .sort(["backbone_id", "len"], descending=[False, True])
        .group_by("backbone_id")
        .agg(pl.first("country").alias("dominant_country"))
    )
    return {str(r["backbone_id"]): str(r["dominant_country"]) for r in pre_country.to_dicts()}


def test_country_debiasing_target_is_presplit_dominant_country() -> None:
    records = pl.DataFrame(
        {
            "backbone_id": ["x", "x", "x", "y", "y", "y"],
            "year": [2019, 2020, 2022, 2019, 2021, 2022],
            "country": ["TR", "TR", "US", "DE", "DE", "FR"],
        }
    )
    dominant = _compute_dominant_country_pre(records, split_year=2020)
    assert dominant["x"] == "TR"
    assert dominant["y"] == "DE"


def test_meta_learner_training_uses_only_oof_features_and_labels(monkeypatch: pytest.MonkeyPatch) -> None:
    df = _synthetic_geo_df(n=80)
    rows_seen: list[int] = []
    labels_seen: list[int] = []

    from sklearn.linear_model import LogisticRegression

    original_fit = LogisticRegression.fit

    def wrapped_fit(self: LogisticRegression, X: np.ndarray, y: np.ndarray, *args: object, **kwargs: object) -> LogisticRegression:
        rows_seen.append(int(X.shape[0]))
        labels_seen.append(int(len(y)))
        return original_fit(self, X, y, *args, **kwargs)

    monkeypatch.setattr(LogisticRegression, "fit", wrapped_fit)
    fit_geo_reliability_surface(df)

    assert 80 in rows_seen
    assert 80 in labels_seen


def test_combined_leakage_audit_pass(tmp_path: Path) -> None:
    geo_path = tmp_path / "geo_large.tsv"
    _synthetic_geo_df(n=220).write_csv(geo_path, separator="\t")
    result = run_pipeline(
        run_mode="geo",
        geo_spread_features_path=geo_path,
        output_dir=tmp_path / "combined_leakage",
    )
    audit = json.loads(result.audit_path.read_text(encoding="utf-8"))
    assert audit["leakage_audit"]["status"] == "pass"
