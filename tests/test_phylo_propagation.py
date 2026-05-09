
import polars as pl

from bio_spread_project.phylo_propagation import build_phylo_propagation


def test_build_phylo_propagation(tmp_path):
    mash_path = tmp_path / "mash.tsv"
    mash_path.write_text("backbone_id_1\tbackbone_id_2\tmash_distance\nBB1\tBB2\t0.1\nBB2\tBB3\t0.05\n")

    features = pl.DataFrame({
        "backbone_id": ["BB1", "BB2", "BB3"],
        "label_geo_spread": [1, 0, None]
    })

    out = build_phylo_propagation(features, mash_path, split_year=2020)

    assert "phylo_prop_risk" in out.columns
    assert out.height == 3
    # Check that risks are between 0 and 1
    assert out["phylo_prop_risk"].min() >= 0.0
    assert out["phylo_prop_risk"].max() <= 1.0
