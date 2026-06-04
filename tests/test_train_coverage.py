"""Additional training tests for coverage."""

import torch
import pytest

from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.train import (
    _inverse_label_frequency,
    _balanced_sample_weights,
    _build_optimizer,
    _build_scheduler,
    _selection_score,
    _device,
    _fit_temperature,
    _fit_temperature_class,
)


def test_inverse_label_frequency():
    labels = torch.tensor([0, 0, 0, 1, 1, 2])
    w = _inverse_label_frequency(labels)
    assert w.shape == labels.shape
    assert w[0] < w[3]  # class 0 has 3 samples (higher count -> lower weight)
    assert w[5] > w[0]  # class 2 has 1 sample (lower count -> higher weight)


def test_inverse_label_frequency_single_class():
    labels = torch.tensor([0, 0, 0])
    w = _inverse_label_frequency(labels)
    assert torch.isfinite(w).all()


def test_balanced_sample_weights():
    data = {
        "mobility": torch.tensor([0, 1, 2, 0, 1]),
        "amr": torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0]),
        "expansion": torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0]),
    }
    w = _balanced_sample_weights(data)
    assert w.shape == (5,)
    assert w.sum() > 0


def test_build_optimizer():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    config = {"lr": 1e-3, "backbone_lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0.1}
    opt = _build_optimizer(model, config)
    assert isinstance(opt, torch.optim.AdamW)
    assert len(opt.param_groups) == 2  # backbone + heads


def test_build_optimizer_default_lr():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    config = {"lr": 1e-3}
    opt = _build_optimizer(model, config)
    assert opt.param_groups[0]["lr"] == 1e-3


def test_build_scheduler_warmup():
    config = {"epochs": 10, "warmup_epochs": 3, "lr": 1e-3}
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    opt = _build_optimizer(model, {"lr": 1e-3})
    sched = _build_scheduler(opt, config)
    assert sched is not None
    # Should be SequentialLR
    assert hasattr(sched, "_schedulers")


def test_build_scheduler_no_warmup():
    config = {"epochs": 10, "lr": 1e-3}
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    opt = _build_optimizer(model, {"lr": 1e-3})
    sched = _build_scheduler(opt, config)
    from torch.optim.lr_scheduler import CosineAnnealingLR
    assert isinstance(sched, CosineAnnealingLR)


def test_selection_score_equal_default():
    metrics = {"mobility_balanced_accuracy": 0.8, "amr_auroc": 0.9, "expansion_auroc": 0.7}
    score = _selection_score(metrics, {"score_mode": "equal"})
    expected = (0.8 + 0.9 + 0.7) / 3.0
    assert abs(score - expected) < 1e-6


def test_selection_score_legacy():
    metrics = {"mobility_balanced_accuracy": 0.8, "amr_auroc": 0.9, "expansion_auroc": 0.7}
    score = _selection_score(metrics, {"score_mode": "legacy"})
    expected = 0.9 + 0.7 + 2.0 * 0.8
    assert abs(score - expected) < 1e-6


def test_selection_score_unknown_mode():
    import pytest
    with pytest.raises(ValueError):
        _selection_score({}, {"score_mode": "invalid"})


def test_device_cpu():
    d = _device(force_cpu=True)
    assert d == "cpu"


def test_device_no_force():
    d = _device(force_cpu=False)
    assert isinstance(d, str)


def test_fit_temperature_binary():
    logits = torch.tensor([2.0, -1.0, 0.5, -2.0])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0])
    t, b = _fit_temperature(logits, targets, torch.nn.functional.binary_cross_entropy_with_logits, "cpu")
    assert t >= 0.1
    assert isinstance(b, float)


def test_fit_temperature_single_class():
    logits = torch.tensor([1.0, 1.0, 1.0])
    targets = torch.tensor([1.0, 1.0, 1.0])
    t, b = _fit_temperature(logits, targets, torch.nn.functional.binary_cross_entropy_with_logits, "cpu")
    assert t == 1.0
    assert b == 0.0


def test_fit_temperature_class():
    logits = torch.tensor([[2.0, -1.0], [0.5, -2.0], [-1.0, 2.0]])
    targets = torch.tensor([0, 0, 1])
    t = _fit_temperature_class(logits, targets, "cpu")
    assert t >= 0.1


def test_fit_temperature_class_single():
    logits = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    targets = torch.tensor([0, 0, 0])
    t = _fit_temperature_class(logits, targets, "cpu")
    assert t == 1.0
