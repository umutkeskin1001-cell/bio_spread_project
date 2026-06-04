"""Additional feature tests for coverage."""

import torch

from dna_sentinel.features import (
    CanonicalKmerConfig,
    CanonicalKmerExtractor,
    _canonical_vocab,
    _canonical_map,
    _vocab_offsets,
    _resolve_max_windows,
    preprocess_consistency_features,
)
from dna_sentinel.utils import LabeledSequence


def test_canonical_vocab():
    v4 = _canonical_vocab(4)
    assert v4 > 0
    assert isinstance(v4, int)


def test_canonical_map_size():
    cmap = _canonical_map(4)
    assert cmap.shape[0] == 256


def test_canonical_map_rc():
    cmap = _canonical_map(4)
    idx_a = 0  # AAAA
    idx_t = 255  # TTTT
    assert cmap[idx_a] == cmap[idx_t]  # reverse complements should map together


def test_vocab_offsets():
    offsets = _vocab_offsets(4, 6)
    assert 4 in offsets
    assert 5 in offsets
    assert 6 in offsets
    assert offsets[5] > offsets[4]


def test_resolve_max_windows_tuple():
    assert _resolve_max_windows((16, 8, 4)) == (16, 8, 4)


def test_resolve_max_windows_int():
    result = _resolve_max_windows(56)
    assert sum(result) == 56
    assert len(result) == 3


def test_resolve_max_windows_small():
    result = _resolve_max_windows(3)
    assert sum(result) == 3
    assert len(result) == 3


def test_extractor_creation():
    config = CanonicalKmerConfig(ngram_min=4, ngram_max=5, max_windows=(4, 2, 1))
    ex = CanonicalKmerExtractor(config)
    assert ex.n_features > 0
    assert ex.config.ngram_min == 4
    assert ex.config.ngram_max == 5


def test_extract_simple_dna():
    config = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=10)
    ex = CanonicalKmerExtractor(config)
    feat, struct, mask, scale = ex.extract("ATGCGT" * 50)
    assert feat.shape[0] == sum(config.max_windows)
    assert mask.sum().item() > 0
    assert scale.shape[0] == sum(config.max_windows)


def test_extract_with_n():
    config = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=10)
    ex = CanonicalKmerExtractor(config)
    feat, struct, mask, scale = ex.extract("ATNGCGT" * 50)
    assert feat.shape[0] == sum(config.max_windows)


def test_extract_short_dna():
    config = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=10)
    ex = CanonicalKmerExtractor(config)
    feat, struct, mask, scale = ex.extract("AT")
    assert mask.sum().item() > 0


def test_extract_rc_consensus():
    config = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=10, rc_consensus=True)
    ex = CanonicalKmerExtractor(config)
    feat, struct, mask, scale = ex.extract("ATGCGT" * 50)
    assert feat.shape[0] == sum(config.max_windows)


def test_consistency_features(tmp_path):
    records = [LabeledSequence("s1", "ATGCGT" * 100, 0, 0, 0)]
    config = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=10)
    out = tmp_path / "consistency.pt"
    preprocess_consistency_features(records, config, out, num_workers=1)
    assert out.exists()
    data = torch.load(out, weights_only=True)
    assert "features" in data


def test_motif_counts():
    config = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=49)
    ex = CanonicalKmerExtractor(config)
    seq = torch.tensor([[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]])
    motifs = ex._motif_counts(seq)
    assert motifs.shape[1] == ex._n_motifs


def test_extractor_empty_dna():
    import pytest
    from dna_sentinel.utils import ValidationError
    config = CanonicalKmerConfig(max_windows=(2, 1, 1))
    ex = CanonicalKmerExtractor(config)
    with pytest.raises(ValidationError):
        ex.extract("")
