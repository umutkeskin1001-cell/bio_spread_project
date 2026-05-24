import torch

from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.train import (
    _balanced_sample_weights,
    _build_optimizer,
    _epoch_indices,
    _selection_score,
)


def test_optimizer_uses_backbone_and_head_lr():
    model = Cassiopeia(CassiopeiaConfig(n_canonical_features=128, hidden_dim=16))
    opt = _build_optimizer(model, {"lr": 1e-3, "backbone_lr": 1e-4, "head_lr": 2e-3, "weight_decay": 0.05})
    groups = opt.param_groups
    assert len(groups) >= 2
    assert groups[0]["lr"] == 1e-4
    assert groups[-1]["lr"] == 2e-3


def test_selection_score_equal_weights_by_default():
    metrics = {"mobility_balanced_accuracy": 0.6, "amr_auroc": 0.9, "expansion_auroc": 0.75}
    assert _selection_score(metrics, {"score_mode": "equal"}) == (0.6 + 0.9 + 0.75) / 3


def test_selection_score_legacy_mode_available():
    metrics = {"mobility_balanced_accuracy": 0.6, "amr_auroc": 0.9, "expansion_auroc": 0.75}
    assert abs(_selection_score(metrics, {"score_mode": "legacy"}) - (0.9 + 0.75 + 2.0 * 0.6)) < 1e-6


def test_balanced_sample_weights_upweight_rare_labels():
    data = {
        "mobility": torch.tensor([0, 0, 0, 1, 2]),
        "amr": torch.tensor([0, 0, 0, 0, 1], dtype=torch.float),
        "expansion": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float),
    }
    w = _balanced_sample_weights(data)
    assert w.shape == (5,)
    assert torch.isfinite(w).all()
    assert w[4] > w[0]
    assert abs(float(w.mean()) - 1.0) < 1e-6


def test_epoch_indices_respects_balanced_sampling_length():
    data = {
        "mobility": torch.tensor([0, 0, 0, 1, 2]),
        "amr": torch.tensor([0, 0, 0, 0, 1], dtype=torch.float),
        "expansion": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float),
    }
    idx = _epoch_indices(5, data, {"balanced_sampling": True}, torch.Generator().manual_seed(123))
    assert idx.shape == (5,)
    assert idx.min() >= 0
    assert idx.max() < 5
