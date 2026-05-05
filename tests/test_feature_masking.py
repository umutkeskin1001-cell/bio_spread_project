from __future__ import annotations

import polars as pl

from bio_spread_project.external_features import EnrichmentFlags, apply_disabled_feature_mask


def test_apply_disabled_feature_mask_drops_disabled_module_columns() -> None:
    df = pl.DataFrame(
        {
            "backbone_id": ["a"],
            "label_geo_spread": [1],
            "synergy_x": [0.1],
            "psge_0": [0.2],
            "fastrp_0": [0.3],
            "grps": [0.4],
            "phylo_prop_risk": [0.5],
        }
    )
    flags = EnrichmentFlags(
        enable_synergy_interactions=False,
        enable_phylo_spatial_embedding=False,
        enable_graph_contagion=False,
        enable_grps=False,
        enable_phylo_propagation=False,
    )
    out, dropped = apply_disabled_feature_mask(df, flags)
    assert set(dropped) == {"synergy_x", "psge_0", "fastrp_0", "grps", "phylo_prop_risk"}
    assert "synergy_x" not in out.columns
