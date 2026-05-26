import torch

from dna_sentinel.utils import WindowDropout


def test_window_dropout_keeps_at_least_one_window():
    _, new_mask = WindowDropout(0.5)(torch.randn(4, 10, 64), torch.ones(4, 10, dtype=torch.bool), training=True)
    assert new_mask.any(dim=1).all()


def test_window_dropout_noop_when_not_training():
    wd = WindowDropout(0.9)
    feat, mask = torch.randn(4, 10, 64), torch.ones(4, 10, dtype=torch.bool)
    f2, m2 = wd(feat, mask, training=False)
    assert torch.equal(f2, feat) and torch.equal(m2, mask)


def test_window_dropout_all_true_when_drop_rate_0():
    _, m = WindowDropout(0.0)(torch.randn(4, 10, 64), torch.ones(4, 10, dtype=torch.bool), training=True)
    assert m.all()
