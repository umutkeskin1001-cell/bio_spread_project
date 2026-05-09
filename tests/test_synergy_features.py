import polars as pl

from bio_spread_project.synergy_features import build_synergy_features


def test_build_synergy_features():
    df = pl.DataFrame({
        "T_eff_norm": [1.0, 2.0],
        "H_obs_specialization_norm": [3.0, 4.0],
        "other": [5.0, 6.0]
    })
    pairs = [("T_eff_norm", "H_obs_specialization_norm"), ("T_eff_norm", "missing")]
    out = build_synergy_features(df, pairs)

    assert "synergy_T_eff_norm__H_obs_specialization_norm" in out.columns
    assert out["synergy_T_eff_norm__H_obs_specialization_norm"].to_list() == [3.0, 8.0]
    assert "synergy_T_eff_norm__missing" not in out.columns
