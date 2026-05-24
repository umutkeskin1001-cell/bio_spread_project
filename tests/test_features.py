import torch

from dna_sentinel.features import (
    CanonicalKmerConfig,
    CanonicalKmerExtractor,
    _canonical_map,
    _canonical_vocab,
    _vocab_offsets,
)
from dna_sentinel.utils import revcomp


def test_canonical_map_reduces_vocabulary():
    assert _canonical_vocab(4) == 136
    assert _canonical_vocab(5) == 512
    assert _canonical_vocab(6) == 2080
    cm4 = _canonical_map(4)
    assert cm4.shape[0] == 256 and len(set(cm4.tolist())) == 136
    cm5 = _canonical_map(5)
    assert cm5.shape[0] == 1024 and len(set(cm5.tolist())) == 512
    cm6 = _canonical_map(6)
    assert cm6.shape[0] == 4096 and len(set(cm6.tolist())) == 2080


def test_vocab_offsets_sum():
    off = _vocab_offsets(4, 6)
    assert off[4] == 0 and off[5] == 136 and off[6] == 648
    assert off[6] + 2080 == 2728


def test_canonical_rc_symmetry():
    ext = CanonicalKmerExtractor(CanonicalKmerConfig(rc_consensus=True))
    seq = "ATGCGT" * 1000
    f1, s1, m1, _ = ext.extract(seq)
    f2, s2, m2, _ = ext.extract(revcomp(seq))
    assert torch.allclose(f1[m1].sum(dim=0), f2[m2].sum(dim=0), atol=1e-3)
    assert torch.allclose(s1[m1].sum(dim=0), s2[m2].sum(dim=0), atol=1e-3)


def test_extract_shapes():
    ext = CanonicalKmerExtractor(CanonicalKmerConfig())
    feat, struct_feat, mask, sids = ext.extract("ATGCGT" * 200)
    total = sum(CanonicalKmerConfig().max_windows)
    assert feat.shape == (total, ext.n_features)
    assert mask.shape == (total,)
    assert sids.shape == (total,)
    assert mask.any()


def test_extractor_uses_configured_stride():
    cfg = CanonicalKmerConfig(window_sizes=(8,), strides=(4,), max_windows=(10,), rc_consensus=False)
    ext = CanonicalKmerExtractor(cfg)
    _, _, mask, _ = ext.extract("ACGT" * 5)
    assert int(mask.sum().item()) == 4


def test_includes_unaligned_tail_window():
    cfg = CanonicalKmerConfig(window_sizes=(8,), strides=(5,), max_windows=(10,), rc_consensus=False)
    ext = CanonicalKmerExtractor(cfg)
    _, _, mask, _ = ext.extract("ACGT" * 5 + "AC")
    assert int(mask.sum().item()) == 4


def test_kmer_counts_ignore_padding():
    cfg = CanonicalKmerConfig(window_sizes=(8,), strides=(8,), max_windows=(1,), rc_consensus=False)
    ext = CanonicalKmerExtractor(cfg)
    feat, _, mask, _ = ext.extract("CCCC")
    assert mask.tolist() == [True]
    top_k = feat[0].topk(5).values
    assert top_k[0] > top_k[-1] * 10  # top feature dominates
