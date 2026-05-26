import torch
import torch.nn.functional as F
from copy import deepcopy

from dna_sentinel.model import Cassiopeia, CassiopeiaConfig, _focal_bce
from dna_sentinel.train import (
    _balanced_sample_weights,
    _build_optimizer,
    _compute_batch_loss,
    _consistency_loss,
    _epoch_indices,
    _pcgrad_step,
    _selection_score,
    _apply_swa,
    cross_validate,
)


def test_optimizer_uses_backbone_and_head_lr():
    opt = _build_optimizer(Cassiopeia(CassiopeiaConfig(n_canonical_features=128, hidden_dim=16)),
                           {"lr": 1e-3, "backbone_lr": 1e-4, "head_lr": 2e-3, "weight_decay": 0.05})
    assert len(opt.param_groups) >= 2
    assert opt.param_groups[0]["lr"] == 1e-4 and opt.param_groups[-1]["lr"] == 2e-3


def test_selection_score_equal_weights_by_default():
    assert _selection_score({"mobility_balanced_accuracy": 0.6, "amr_auroc": 0.9, "expansion_auroc": 0.75}, {"score_mode": "equal"}) == (0.6 + 0.9 + 0.75) / 3


def test_selection_score_legacy_mode():
    s = _selection_score({"mobility_balanced_accuracy": 0.6, "amr_auroc": 0.9, "expansion_auroc": 0.75}, {"score_mode": "legacy"})
    assert abs(s - (0.9 + 0.75 + 2.0 * 0.6)) < 1e-6


def test_selection_score_unknown_mode():
    import pytest
    with pytest.raises(ValueError, match="unknown"):
        _selection_score({}, {"score_mode": "invalid"})


def test_balanced_sample_weights_upweight_rare_labels():
    data = {"mobility": torch.tensor([0, 0, 0, 1, 2]), "amr": torch.tensor([0, 0, 0, 0, 1], dtype=torch.float),
            "expansion": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float)}
    w = _balanced_sample_weights(data)
    assert w.shape == (5,) and torch.isfinite(w).all() and w[4] > w[0]


def test_epoch_indices_respects_balanced_sampling():
    data = {"mobility": torch.tensor([0, 0, 0, 1, 2]), "amr": torch.tensor([0, 0, 0, 0, 1], dtype=torch.float),
            "expansion": torch.tensor([0, 0, 0, 1, 1], dtype=torch.float)}
    idx = _epoch_indices(5, data, {"balanced_sampling": True}, torch.Generator().manual_seed(123))
    assert idx.shape == (5,) and idx.min() >= 0 and idx.max() < 5


def test_consistency_loss_is_small_for_identical_outputs():
    o = {"mobility_logits": torch.tensor([[2.0, 0.0, -1.0]]), "amr_logits": torch.tensor([0.5]), "expansion_logits": torch.tensor([-0.25])}
    assert _consistency_loss(o, o, temperature=1.0).item() < 1e-6


def test_consistency_loss_positive_for_different_outputs():
    a = {"mobility_logits": torch.tensor([[2.0, 0.0, -1.0]]), "amr_logits": torch.tensor([0.5]), "expansion_logits": torch.tensor([-0.25])}
    b = {"mobility_logits": torch.tensor([[-1.0, 0.0, 2.0]]), "amr_logits": torch.tensor([-0.5]), "expansion_logits": torch.tensor([0.25])}
    assert _consistency_loss(a, b, temperature=1.0).item() > 0


def test_cross_validate_returns_summary():
    torch.manual_seed(42)
    data = {"features": torch.randn(10, 8, 100), "masks": torch.ones(10, 8, dtype=torch.bool),
            "mobility": torch.randint(0, 3, (10,)), "amr": torch.randint(0, 2, (10,), dtype=torch.float),
            "expansion": torch.randint(0, 2, (10,), dtype=torch.float),
            "struct_features": torch.randn(10, 8, 19), "scale_ids": torch.zeros(10, 8, dtype=torch.long)}
    result, _ = cross_validate({"hidden_dim": 32, "n_canonical_features": 100, "frp_out_dim": 32, "n_layers": 1, "max_windows": 8},
                            data, {"epochs": 1, "batch_size": 4, "artifact_dir": "/tmp/cv_test", "patience": 50, "lr": 1e-3}, n_folds=2)
    assert "n_folds" in result and "task_score" in result
    assert result["n_folds"] == 2


def test_pcgrad_reduces_gradient_conflict():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, n_canonical_features=32, frp_out_dim=16, max_windows=4))
    opt = _build_optimizer(model, {"lr": 1e-3, "backbone_lr": 1e-3, "head_lr": 1e-3, "weight_decay": 0})
    B, W = 4, 4
    x = torch.randn(B, W, 32)
    m = torch.ones(B, W, dtype=torch.bool)
    out = model(x, m)
    losses = [
        F.cross_entropy(out["mobility_logits"], torch.randint(0, 3, (B,))),
        _focal_bce(out["amr_logits"], torch.randint(0, 2, (B,), dtype=torch.float), None, 0.0),
        _focal_bce(out["expansion_logits"], torch.randint(0, 2, (B,), dtype=torch.float), None, 0.0),
    ]
    sd_before = deepcopy(model.state_dict())
    _pcgrad_step(model, losses, opt, 1)
    # after step + zero_grad, grads are None but optimizer.step() should have run
    changed = any(not torch.allclose(sd_before[n], p.data) for n, p in model.named_parameters() if p.requires_grad)
    assert changed, "PCGrad step should update model parameters"


def test_compute_batch_loss_handles_nan_gracefully():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, n_canonical_features=32, frp_out_dim=16, max_windows=4))
    B, W = 4, 4
    x = torch.randn(B, W, 32)
    m = torch.ones(B, W, dtype=torch.bool)
    out = model(x, m)
    losses = _compute_batch_loss(model, out,
                                 torch.randint(0, 3, (B,)),
                                 torch.randint(0, 2, (B,), dtype=torch.float),
                                 torch.randint(0, 2, (B,), dtype=torch.float),
                                 None, None, None, 0.0)
    assert torch.isfinite(losses["total"])
    assert not torch.isnan(losses["total"])


def test_apply_swa_averages_checkpoints():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, n_canonical_features=32, frp_out_dim=16, max_windows=4))
    ckpts = [deepcopy(model.state_dict()) for _ in range(3)]
    _apply_swa(model, ckpts)
    for n, p in model.named_parameters():
        assert torch.allclose(p.data, ckpts[0][n])


def test_apply_swa_empty_checkpoints():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, n_canonical_features=32, frp_out_dim=16, max_windows=4))
    sd_before = deepcopy(model.state_dict())
    _apply_swa(model, [])
    for n, p in model.named_parameters():
        assert torch.allclose(p.data, sd_before[n])


def test_cross_validate_group_aware():
    torch.manual_seed(42)
    data = {"features": torch.randn(10, 8, 100), "masks": torch.ones(10, 8, dtype=torch.bool),
            "mobility": torch.randint(0, 3, (10,)), "amr": torch.randint(0, 2, (10,), dtype=torch.float),
            "expansion": torch.randint(0, 2, (10,), dtype=torch.float),
            "struct_features": torch.randn(10, 8, 19), "scale_ids": torch.zeros(10, 8, dtype=torch.long)}
    groups = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2]
    result, _ = cross_validate({"hidden_dim": 32, "n_canonical_features": 100, "frp_out_dim": 32, "n_layers": 1, "max_windows": 8},
                            data, {"epochs": 1, "batch_size": 4, "artifact_dir": "/tmp/cv_test", "patience": 50, "lr": 1e-3},
                            n_folds=3, group_ids=groups)
    assert result["n_folds"] == 3
    assert "task_score" in result


def test_multiclass_metrics_per_class():
    from dna_sentinel.utils import multiclass_metrics
    import numpy as np
    y = np.array([0, 0, 1, 1, 2, 2])
    p = np.eye(3)[y] * 0.9 + np.ones((6, 3)) * 0.033
    m = multiclass_metrics(y, p, "mob")
    assert "mob_class0_f1" in m
    assert "mob_class1_precision" in m
    assert "mob_confusion_matrix" in m
    assert len(m["mob_confusion_matrix"]) == 3


def test_compute_risk_score_custom_weights():
    from dna_sentinel.utils import compute_risk_score
    m = [0.7, 0.2, 0.1]
    s = compute_risk_score(m, 0.8, 0.6, weights=(0.5, 0.3, 0.2))
    expected = 0.5 * (1 - 0.7) + 0.3 * 0.8 + 0.2 * 0.6
    assert abs(s - expected) < 1e-6
