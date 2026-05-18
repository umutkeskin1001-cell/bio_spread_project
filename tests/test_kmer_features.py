import torch
from dna_sentinel.kmer_features import MultiScaleKmerConfig, MultiScaleKmerExtractor

def test_extract_shapes():
    ext = MultiScaleKmerExtractor(MultiScaleKmerConfig(n_features=256))
    feat, mask, sids = ext.extract("ATGCGT" * 200)
    total = sum(MultiScaleKmerConfig.max_windows)
    assert feat.shape == (total, 256)
    assert mask.shape == (total,)
    assert sids.shape == (total,)
    assert mask.any()

def test_rc_consensus_makes_features_symmetric():
    from dna_sentinel.fasta import revcomp
    ext = MultiScaleKmerExtractor(MultiScaleKmerConfig(n_features=256, rc_consensus=True))
    seq = "ATGCGT" * 50
    f1, m1, _ = ext.extract(seq)
    f2, m2, _ = ext.extract(revcomp(seq))
    assert torch.allclose(f1[m1], f2[m2], atol=1e-4)
