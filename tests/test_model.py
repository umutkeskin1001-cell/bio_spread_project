import torch

from dna_sentinel.model import DnaSentinel, DnaSentinelConfig
from dna_sentinel.tokenizer import DnaTokenizer


def test_model_forward_shapes_are_task_specific():
    model = DnaSentinel(DnaSentinelConfig(channels=24, layers=2, max_windows=3, window_size=32))
    tokens = torch.randint(0, 5, (4, 3, 32))
    mask = torch.ones(4, 3, 32)

    out = model(tokens, mask)

    assert out.mobility_logits.shape == (4, 3)
    assert out.amr_logits.shape == (4,)
    assert out.expansion_logits.shape == (4,)
    assert out.window_scores.shape == (4, 3)
    assert out.evidence_weights.shape == (4, 3)
    assert torch.allclose(out.evidence_weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_model_predictions_are_stable_under_reverse_complement_after_training_mode_off():
    tok = DnaTokenizer()
    cfg = DnaSentinelConfig(channels=16, layers=2, max_windows=2, window_size=32, rc_consensus=True)
    model = DnaSentinel(cfg).eval()
    seq = "AACCGGTTAACCGGTT"
    rc = "AACCGGTTAACCGGTT"[::-1].translate(str.maketrans("ACGT", "TGCA"))
    x1, m1 = tok.batch_windows([seq], cfg.window_size, cfg.stride, cfg.max_windows)
    x2, m2 = tok.batch_windows([rc], cfg.window_size, cfg.stride, cfg.max_windows)

    with torch.no_grad():
        p1 = model(x1, m1).amr_logits
        p2 = model(x2, m2).amr_logits

    assert torch.allclose(p1, p2, atol=1e-4)
