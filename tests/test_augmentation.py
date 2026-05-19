import torch

from dna_sentinel.utils import LabeledSequence, WindowDropout, rc_augment


def test_rc_augment_doubles_count():
    records = [LabeledSequence("s1", "ACGT" * 10, 1, 0, 0)]
    assert len(rc_augment(records)) == 2

def test_window_dropout_preserves_first_window():
    wd = WindowDropout(0.5)
    feat = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, dtype=torch.bool)
    _, new_mask = wd(feat, mask, training=True)
    assert new_mask[:, 0].all()

def test_window_dropout_noop_when_not_training():
    wd = WindowDropout(0.9)
    feat = torch.randn(4, 10, 64)
    mask = torch.ones(4, 10, dtype=torch.bool)
    f2, m2 = wd(feat, mask, training=False)
    assert torch.equal(f2, feat)
    assert torch.equal(m2, mask)
