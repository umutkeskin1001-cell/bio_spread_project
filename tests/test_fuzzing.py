"""Property-based tests for BioSpread data logic."""

from hypothesis import given
from hypothesis import strategies as st

from bio_spread_project.data import PlasmidRecord


@given(
    backbone_id=st.text(min_size=1),
    year=st.integers(min_value=0, max_value=3000),
    country=st.text(),
    amr_gene_count=st.floats(min_value=0, max_value=1000, allow_nan=False),
    mobility_score=st.floats(min_value=0, max_value=1, allow_nan=False)
)
def test_plasmid_record_invariant(backbone_id, year, country, amr_gene_count, mobility_score):
    """Ensure PlasmidRecord can handle various inputs without crashing."""
    record = PlasmidRecord(
        backbone_id=backbone_id,
        year=year,
        country=country,
        host_genus="unknown",
        clinical_context="unknown",
        amr_gene_count=amr_gene_count,
        mobility_score=mobility_score
    )
    assert record.backbone_id == backbone_id
    assert 0 <= record.mobility_score <= 1
