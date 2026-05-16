"""Edge case tests for taxonomy, sampler, and split."""

import polars as pl
import pytest

from bio_spread.data.dataset import SequenceBatchSampler
from bio_spread.data.snapshot import build_taxonomy_vocab, disjoint_backbone_split


def test_taxonomy_vocab_unknown_collision():
    """build_taxonomy_vocab should handle data containing literal 'UNKNOWN'."""
    df = pl.DataFrame(
        {
            "phylum": ["UNKNOWN", "Proteobacteria"],
            "class": ["Gammaproteobacteria"] * 2,
            "order": ["Enterobacterales"] * 2,
            "family": ["Enterobacteriaceae"] * 2,
            "genus": ["Escherichia"] * 2,
        }
    )
    # The column names need to match the internal taxonomy column convention
    df = df.rename(
        {
            "phylum": "TAXONOMY_phylum",
            "class": "TAXONOMY_class",
            "order": "TAXONOMY_order",
            "family": "TAXONOMY_family",
        }
    )
    # Should raise because "UNKNOWN" conflicts with sentinel
    with pytest.raises(ValueError, match="UNKNOWN"):
        build_taxonomy_vocab(df)


def test_sampler_empty_dataset():
    sampler = SequenceBatchSampler(n_samples=0, batch_size=4)
    assert len(sampler) == 0
    assert list(sampler) == []


def test_sampler_single_sample():
    sampler = SequenceBatchSampler(n_samples=1, batch_size=4)
    assert len(sampler) == 1
    batches = list(sampler)
    assert len(batches) == 1
    assert len(batches[0]) == 1


def test_disjoint_split_minimum_backbones():
    raw = pl.DataFrame(
        {
            "backbone_id": ["A", "B"],
            "year": [2018, 2021],
            "country": ["US", "UK"],
        }
    )
    with pytest.raises(ValueError, match=">= 4"):
        disjoint_backbone_split(raw, split_year=2020)
