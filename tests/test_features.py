import torch

from dna_sentinel.features import (
    CanonicalKmerConfig,
    CanonicalKmerExtractor,
    _canonical_map,
    _canonical_vocab,
    _resolve_max_windows,
    _vocab_offsets,
    preprocess_consistency_features,
)
from dna_sentinel.utils import LabeledSequence, ValidationError, revcomp


def test_canonical_map_reduces_vocabulary():
    assert _canonical_vocab(4) == 136 and _canonical_vocab(5) == 512 and _canonical_vocab(6) == 2080
    cm4 = _canonical_map(4)
    assert cm4.shape[0] == 256 and len(set(cm4.tolist())) == 136


def test_vocab_offsets_sum():
    off = _vocab_offsets(4, 6)
    assert off[4] == 0 and off[5] == 136 and off[6] == 648
    assert off[6] + 2080 == 2728


def test_resolve_max_windows_preserves_default_scale_ratio():
    assert _resolve_max_windows(56) == (32, 16, 8)


def test_resolve_max_windows_int():
    r = _resolve_max_windows(28)
    assert sum(r) == 28 and len(r) == 3


def test_canonical_rc_symmetry():
    ext = CanonicalKmerExtractor(CanonicalKmerConfig(rc_consensus=True))
    seq = "ATGCGT" * 1000
    f1, _, m1, _ = ext.extract(seq)
    f2, _, m2, _ = ext.extract(revcomp(seq))
    assert torch.allclose(f1[m1].sum(dim=0), f2[m2].sum(dim=0), atol=1e-3)


def test_extract_shapes():
    feat, _, mask, sids = CanonicalKmerExtractor(CanonicalKmerConfig()).extract("ATGCGT" * 200)
    total = sum(CanonicalKmerConfig().max_windows)
    assert feat.shape == (total, 2728) and mask.shape == (total,) and sids.shape == (total,)


def test_extractor_uses_configured_stride():
    config = CanonicalKmerConfig(window_sizes=(8,), strides=(4,), max_windows=(10,), rc_consensus=False)
    _, _, mask, _ = CanonicalKmerExtractor(config).extract("ACGT" * 5)
    assert int(mask.sum().item()) == 4


def test_includes_unaligned_tail_window():
    config = CanonicalKmerConfig(window_sizes=(8,), strides=(5,), max_windows=(10,), rc_consensus=False)
    _, _, mask, _ = CanonicalKmerExtractor(config).extract("ACGT" * 5 + "AC")
    assert int(mask.sum().item()) == 4


def test_kmer_counts_ignore_padding():
    ext = CanonicalKmerExtractor(CanonicalKmerConfig(window_sizes=(8,), strides=(8,), max_windows=(1,), rc_consensus=False))
    feat, _, mask, _ = ext.extract("CCCC")
    assert mask.tolist() == [True]
    assert feat[0].topk(5).values[0] > feat[0].topk(5).values[-1] * 10


def test_preprocess_consistency_features_saves_expected_keys(tmp_path):
    records = [LabeledSequence("a", "ATGCGT" * 200, 1, 0, 1), LabeledSequence("b", "CGTATG" * 180, 0, 1, 0)]
    out = tmp_path / "consistency.pt"
    cfg = CanonicalKmerConfig(window_sizes=(32,), strides=(16,), max_windows=(4,))
    preprocess_consistency_features(records, cfg, out, num_workers=1)
    saved = torch.load(out, weights_only=True)
    core_keys = {"features", "struct_features", "masks", "scale_ids"}
    assert core_keys.issubset(set(saved))
    assert saved["_schema_version"] is not None
    assert saved["_n_structural_features"] > 0


def test_extract_single_nucleotide():
    config = CanonicalKmerConfig(window_sizes=(8,), strides=(8,), max_windows=(1,), rc_consensus=False)
    feat, _, mask, _ = CanonicalKmerExtractor(config).extract("A")
    assert mask.tolist() == [True]
    assert torch.isfinite(feat).all()


def test_extract_all_n():
    config = CanonicalKmerConfig(window_sizes=(8,), strides=(8,), max_windows=(1,), rc_consensus=False)
    feat, _, mask, _ = CanonicalKmerExtractor(config).extract("NNNNNNNN")
    assert mask.tolist() == [True]
    assert torch.isfinite(feat).all()


def test_extract_empty_raises():
    import pytest
    with pytest.raises(ValidationError):
        CanonicalKmerExtractor().extract("")
