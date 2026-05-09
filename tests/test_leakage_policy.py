from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from bio_spread_project.leakage_policy import detect_forbidden_columns, validate_feature_surface


def test_leakage_policy_blocks_forbidden_token_columns() -> None:
    df = pl.DataFrame(
        {
            "backbone_id": ["a", "b"],
            "label_geo_spread": [0, 1],
            "my_future_proxy": [0.1, 0.9],
        }
    )
    with pytest.raises(ValueError, match="forbidden feature tokens"):
        validate_feature_surface(df)


def test_leakage_policy_blocks_documented_outcome_aliases() -> None:
    blocked = detect_forbidden_columns(
        [
            "label_geo_spread",
            "n_new_countries_future",
            "spread_label",
            "n_new_countries",
            "macro_region_jump_label",
            "test_fold_probability",
        ]
    )

    assert blocked == [
        "macro_region_jump_label",
        "n_new_countries",
        "spread_label",
        "test_fold_probability",
    ]


def test_leakage_policy_passes_clean_surface() -> None:
    df = pl.DataFrame(
        {
            "backbone_id": ["a", "b"],
            "label_geo_spread": [0, 1],
            "T_eff_norm": [0.1, 0.9],
        }
    )
    report = validate_feature_surface(
        df,
        lineage_path=Path("project_config/config/feature_lineage.json"),
        strict_lineage=False,
    )
    assert report["status"] == "pass"


def test_leakage_policy_strict_lineage_blocks_unknown_features() -> None:
    df = pl.DataFrame(
        {
            "backbone_id": ["a", "b"],
            "label_geo_spread": [0, 1],
            "unknown_feature_x": [0.1, 0.2],
        }
    )
    with pytest.raises(ValueError, match="missing lineage"):
        validate_feature_surface(
            df,
            lineage_path=Path("project_config/config/feature_lineage.json"),
            strict_lineage=True,
        )
