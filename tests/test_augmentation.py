import torch

from dna_sentinel.utils import WindowDropout


def test_window_dropout_keeps_at_least_one_window():
    wd = WindowDropout(0.5)
    feat = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, dtype=torch.bool)
    _, new_mask = wd(feat, mask, training=True)
    assert new_mask.any(dim=1).all()


def test_window_dropout_noop_when_not_training():
    wd = WindowDropout(0.9)
    feat = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, dtype=torch.bool)
    f2, m2 = wd(feat, mask, training=False)
    assert torch.equal(f2, feat)
    assert torch.equal(m2, mask)
