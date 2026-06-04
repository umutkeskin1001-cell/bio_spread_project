"""Tests for training calibration and edge functions."""

import torch
import numpy as np

from dna_sentinel.train import (
    fit_calibration,
    _device,
    _focal_bce,
    _selection_score,
)
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.utils import set_seed, WindowDropout


def test_fit_calibration():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56)
    model = Cassiopeia(cfg)
    model.eval()
    val_data = {
        "features": torch.randn(8, 56, 2728),
        "masks": torch.ones(8, 56, dtype=torch.bool),
        "mobility": torch.randint(0, 3, (8,)),
        "amr": torch.randint(0, 2, (8,)).float(),
        "expansion": torch.randint(0, 2, (8,)).float(),
    }
    result = fit_calibration(model, val_data, "cpu")
    assert "cal_mob_probs" in result
    assert "cal_amr_probs" in result
    assert "cal_exp_probs" in result


def test_fit_calibration_expansion_multiclass():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, expansion_classes=3)
    model = Cassiopeia(cfg)
    model.eval()
    val_data = {
        "features": torch.randn(6, 56, 2728),
        "masks": torch.ones(6, 56, dtype=torch.bool),
        "mobility": torch.randint(0, 3, (6,)),
        "amr": torch.randint(0, 2, (6,)).float(),
        "expansion": torch.randint(0, 3, (6,)),
    }
    result = fit_calibration(model, val_data, "cpu")
    assert "cal_mob_probs" in result


def test_fit_calibration_with_struct():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, n_structural_features=5)
    model = Cassiopeia(cfg)
    model.eval()
    val_data = {
        "features": torch.randn(4, 56, 2728),
        "masks": torch.ones(4, 56, dtype=torch.bool),
        "struct_features": torch.randn(4, 56, 5),
        "scale_ids": torch.zeros(4, 56, dtype=torch.long),
        "mobility": torch.randint(0, 3, (4,)),
        "amr": torch.randint(0, 2, (4,)).float(),
        "expansion": torch.randint(0, 2, (4,)).float(),
    }
    result = fit_calibration(model, val_data, "cpu")
    assert "cal_mob_probs" in result


def test_device_force_cpu():
    d = _device(force_cpu=True)
    assert d == "cpu"


def test_selection_score_equal():
    metrics = {"mobility_balanced_accuracy": 0.8, "amr_auroc": 0.9, "expansion_auroc": 0.7}
    score = _selection_score(metrics, {"score_mode": "equal"})
    expected = (0.8 + 0.9 + 0.7) / 3.0
    assert abs(score - expected) < 1e-6


def test_selection_score_with_penalties():
    metrics = {"mobility_balanced_accuracy": 0.8, "amr_auroc": 0.9,
               "expansion_auroc": 0.7, "ece_penalty": 0.05, "drift_penalty": 0.02}
    score = _selection_score(metrics, {"score_mode": "equal"})
    expected = (0.8 + 0.9 + 0.7) / 3.0 - 0.05 - 0.02
    assert abs(score - expected) < 1e-6


def test_focal_bce_edge_cases():
    loss = _focal_bce(torch.tensor([100.0, -100.0]), torch.tensor([1.0, 0.0]), None, 5.0)
    assert torch.isfinite(loss)


def test_window_dropout_with_padded():
    wd = WindowDropout(drop_rate=0.5)
    feat = torch.randn(2, 10, 5)
    mask = torch.zeros(2, 10, dtype=torch.bool)
    mask[:, :5] = True
    f, m = wd(feat, mask, training=True)
    assert (m[:, 5:] == False).all()  # noqa: E712 — padded remains False
    assert m[:, :5].any()  # at least one active window remains


def test_set_seed_deterministic():
    set_seed(42, deterministic=True)


def test_window_dropout_does_not_drop_all():
    wd = WindowDropout(drop_rate=0.99)
    feat = torch.randn(3, 5, 5)
    mask = torch.ones(3, 5, dtype=torch.bool)
    f, m = wd(feat, mask, training=True)
    assert m.any(dim=1).all()


# ── binary_metrics edge cases ─────────────────────────────────────

def test_binary_metrics_single_class():
    from dna_sentinel.utils import binary_metrics
    m = binary_metrics(np.array([0, 0, 0]), np.array([0.1, 0.2, 0.3]), "test")
    assert m["test_auroc"] == 0.5


def test_binary_metrics_normal():
    from dna_sentinel.utils import binary_metrics
    m = binary_metrics(np.array([0, 1, 0, 1]), np.array([0.1, 0.9, 0.2, 0.8]), "test")
    assert m["test_auroc"] > 0.5


def test_multiclass_metrics():
    from dna_sentinel.utils import multiclass_metrics
    y = np.array([0, 1, 2, 0])
    p = np.array([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8], [0.8, 0.1, 0.1]])
    m = multiclass_metrics(y, p, "mob")
    assert "mob_balanced_accuracy" in m
    assert "mob_confusion_matrix" in m


# ── Compression edge cases ───────────────────────────────────────

def test_compress_checkpoint_fp16(tmp_path):
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig, compress_checkpoint
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    src = tmp_path / "src.pt"
    model.save(src)
    dst = tmp_path / "dst.pt"
    result = compress_checkpoint(src, dst, fmt="fp16")
    assert result["format"] == "fp16"
    assert result["dst_bytes"] < result["src_bytes"]


def test_compress_checkpoint_int8(tmp_path):
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig, compress_checkpoint, load_compressed
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    src = tmp_path / "src.pt"
    model.save(src)
    dst = tmp_path / "dst_int8.pt"
    result = compress_checkpoint(src, dst, fmt="int8")
    assert result["format"] == "int8"
    # Test loading
    loaded = load_compressed(dst, "cpu")
    assert loaded.config.hidden_dim == 16


def test_model_save_load_roundtrip(tmp_path):
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    x = torch.randn(2, 56, 2728)
    m = torch.ones(2, 56, dtype=torch.bool)
    out_orig = model(x, m)
    path = tmp_path / "m.pt"
    model.save(path)
    loaded = Cassiopeia.load(path, "cpu")
    loaded.eval()
    out_loaded = loaded(x, m)
    assert torch.allclose(out_orig["mobility_logits"], out_loaded["mobility_logits"], atol=1e-5)


# ── API: test with env variables ─────────────────────────────────

def test_api_version():
    from dna_sentinel.api import _VERSION
    assert _VERSION is not None
