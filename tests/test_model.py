"""Unit tests for Cassiopeia model and data pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from dna_sentinel.model import (
    Cassiopeia,
    CassiopeiaConfig,
    GLUMixer,
    _focal_bce,
    _make_frp,
)

# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestCassiopeiaConfig:
    def test_default_small(self):
        cfg = CassiopeiaConfig()
        assert cfg.variant == "small"
        assert cfg.hidden_dim == 128
        assert cfg.expansion_classes == 1
        assert cfg.amr_classes == 1

    def test_from_yaml(self):
        yaml_dict = {"model": {"variant": "large", "hidden_dim": 512, "n_heads": 12,
                                "n_layers_transformer": 8, "expansion_classes": 50, "amr_classes": 12}}
        cfg = CassiopeiaConfig.from_yaml(yaml_dict)
        assert cfg.variant == "large"
        assert cfg.hidden_dim == 512
        assert cfg.expansion_classes == 50
        assert cfg.amr_classes == 12

    def test_to_dict(self):
        cfg = CassiopeiaConfig()
        d = cfg.to_dict()
        assert isinstance(d, dict)
        assert d["variant"] == "small"

    def test_extra_keys_ignored(self):
        yaml_dict = {"model": {"hidden_dim": 256, "nonexistent_key": 999}}
        cfg = CassiopeiaConfig.from_yaml(yaml_dict)
        assert cfg.hidden_dim == 256
        assert not hasattr(cfg, "nonexistent_key")


# ---------------------------------------------------------------------------
# FRP matrix tests
# ---------------------------------------------------------------------------

class TestFRPMatrix:
    def test_shape(self):
        mat = _make_frp(100, 256)
        assert mat.shape == (100, 256)

    def test_values(self):
        mat = _make_frp(50, 50)
        assert torch.all((mat == 1) | (mat == -1) | (mat == 0))
        assert (mat == 0).float().mean() > 0.5  # mostly zeros (2/3 chance)

    def test_deterministic(self):
        m1 = _make_frp(100, 100, seed=42)
        m2 = _make_frp(100, 100, seed=42)
        assert torch.allclose(m1, m2)


# ---------------------------------------------------------------------------
# Loss tests
# ---------------------------------------------------------------------------

class TestLosses:
    def test_focal_bce_basic(self):
        logits = torch.tensor([2.0, -2.0, 1.0, -1.0])
        target = torch.tensor([1.0, 0.0, 1.0, 0.0])
        loss = _focal_bce(logits, target, pw=None, gamma=0.0)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_focal_bce_with_gamma(self):
        logits = torch.tensor([2.0, -2.0])
        target = torch.tensor([1.0, 0.0])
        loss_g0 = _focal_bce(logits, target, pw=None, gamma=0.0)
        loss_g2 = _focal_bce(logits, target, pw=None, gamma=2.0)
        assert loss_g2.item() < loss_g0.item()  # focal reduces loss for easy examples

    def test_focal_bce_pos_weight(self):
        logits = torch.tensor([0.0, 0.0])
        target = torch.tensor([1.0, 0.0])
        loss_now = _focal_bce(logits, target, pw=None, gamma=0.0)
        loss_pw = _focal_bce(logits, target, pw=torch.tensor([3.0]), gamma=0.0)
        assert loss_pw.item() > loss_now.item()




# ---------------------------------------------------------------------------
# Block tests
# ---------------------------------------------------------------------------

class TestBlocks:
    def test_glu_mixer(self):
        block = GLUMixer(28, 128)
        x = torch.randn(4, 28, 128)
        out = block(x)
        assert out.shape == (4, 28, 128)


# ---------------------------------------------------------------------------
# Model tests
# ---------------------------------------------------------------------------

class TestCassiopeiaSmall:
    @pytest.fixture
    def model(self):
        return Cassiopeia(CassiopeiaConfig(variant="small", hidden_dim=64,
                                             n_canonical_features=100, frp_out_dim=64,
                                             n_layers=1, max_windows=12,
                                             expansion_classes=1, amr_classes=1))

    @pytest.fixture
    def batch(self):
        B, W = 4, 12
        nf = 100
        return {
            "features": torch.randn(B, W, nf),
            "masks": torch.ones(B, W, dtype=torch.bool),
            "mobility": torch.randint(0, 3, (B,)),
            "amr": torch.randint(0, 2, (B,), dtype=torch.float),
            "expansion": torch.randint(0, 2, (B,), dtype=torch.float),
        }

    def test_forward(self, model, batch):
        out = model(batch["features"], batch["masks"])
        assert "mobility_logits" in out
        assert "amr_logits" in out
        assert "expansion_logits" in out
        assert out["mobility_logits"].shape == (4, 3)
        assert out["amr_logits"].shape == (4,)
        assert out["expansion_logits"].shape == (4,)

    def test_compute_loss(self, model, batch):
        out = model(batch["features"], batch["masks"])
        loss = model.compute_loss(
            out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
            batch["mobility"], batch["amr"], batch["expansion"],
        )
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_save_load(self, model, batch):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            model.save(path)
        loaded = Cassiopeia.load(path)
        # Verify inference works on loaded model
        out = loaded(batch["features"], batch["masks"])
        assert out["mobility_logits"].shape == (4, 3)
        assert out["amr_logits"].shape == (4,)
        assert out["expansion_logits"].shape == (4,)
        Path(path).unlink()

    def test_parameter_count(self):
        model = Cassiopeia()
        count = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert count < 500_000

    def test_save_load_roundtrip(self, model, batch):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            model.save(path)
        loaded = Cassiopeia.load(path)
        model.eval()
        with torch.no_grad():
            a = model(batch["features"], batch["masks"])["amr_logits"]
            b = loaded(batch["features"], batch["masks"])["amr_logits"]
        assert torch.allclose(a, b)
        Path(path).unlink()

    def test_backward(self, batch):
        model = Cassiopeia(CassiopeiaConfig(
            variant="small", hidden_dim=64,
            n_canonical_features=100, frp_out_dim=64,
            n_layers=1, max_windows=12,
            expansion_classes=1, amr_classes=1,
        ))
        out = model(batch["features"], batch["masks"])
        loss = model.compute_loss(
            out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
            batch["mobility"], batch["amr"], batch["expansion"],
        )
        loss.backward()
        trainable = [p for n, p in model.named_parameters() if p.requires_grad]
        assert any(p.grad is not None for p in trainable)


class TestCassiopeiaLarge:
    @pytest.fixture
    def model(self):
        return Cassiopeia(CassiopeiaConfig(variant="large", hidden_dim=128,
                                             n_canonical_features=100, frp_out_dim=64,
                                             n_layers=2, max_windows=12,
                                             expansion_classes=10, amr_classes=4))

    @pytest.fixture
    def batch(self):
        B, W = 4, 12
        return {
            "features": torch.randn(B, W, 100),
            "masks": torch.ones(B, W, dtype=torch.bool),
            "mobility": torch.randint(0, 3, (B,)),
            "amr": torch.randint(0, 2, (B, 4), dtype=torch.float),
            "expansion": torch.randint(0, 10, (B,)),
        }

    def test_forward(self, model, batch):
        out = model(batch["features"], batch["masks"])
        assert out["amr_logits"].shape == (4, 4)
        assert out["expansion_logits"].shape == (4, 10)

    def test_loss(self, model, batch):
        out = model(batch["features"], batch["masks"])
        loss = model.compute_loss(
            out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
            batch["mobility"], batch["amr"], batch["expansion"],
        )
        assert loss.item() > 0

    def test_expansion_softmax(self, model, batch):
        out = model(batch["features"], batch["masks"])
        probs = torch.softmax(out["expansion_logits"], dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))


# ---------------------------------------------------------------------------
# Calibration tests
# ---------------------------------------------------------------------------

class TestCalibration:
    def test_fit_calibration(self):
        from dna_sentinel.train import fit_calibration
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64,
                                              max_windows=12, expansion_classes=1, amr_classes=1))
        B, W = 8, 12
        val_data = {
            "features": torch.randn(B, W, 100),
            "masks": torch.ones(B, W, dtype=torch.bool),
            "mobility": torch.randint(0, 3, (B,)),
            "amr": torch.randint(0, 2, (B,), dtype=torch.float),
            "expansion": torch.randint(0, 2, (B,), dtype=torch.float),
        }
        device = "cpu"
        fit_calibration(model, val_data, device)
        assert model.mob_t.item() > 0


# ---------------------------------------------------------------------------
# Utility tests
# ---------------------------------------------------------------------------

class TestUtils:
    def test_window_dropout(self):
        from dna_sentinel.utils import WindowDropout
        wd = WindowDropout(0.5)
        feat = torch.randn(4, 10, 128)
        mask = torch.ones(4, 10, dtype=torch.bool)
        f2, m2 = wd(feat, mask, training=True)
        assert f2.shape == feat.shape
        assert m2.dtype == torch.bool
        assert m2.any(dim=1).all()  # every row has at least one kept

    def test_binary_metrics(self):
        from dna_sentinel.utils import binary_metrics
        y = np.array([1, 0, 1, 0, 1])
        p = np.array([0.9, 0.1, 0.8, 0.2, 0.7])
        m = binary_metrics(y, p, "test")
        assert "test_auroc" in m
        assert "test_brier" in m
        assert "test_ece" in m
        assert m["test_auroc"] > 0.5

    def test_multiclass_metrics(self):
        from dna_sentinel.utils import multiclass_metrics
        y = np.array([0, 1, 2, 0, 1])
        p = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=float)
        m = multiclass_metrics(y, p, "test")
        assert m["test_accuracy"] == 1.0
        assert m["test_balanced_accuracy"] == 1.0


# ---------------------------------------------------------------------------
# Evaluate tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_small(self):
        from dna_sentinel.train import evaluate
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64,
                                              max_windows=12, expansion_classes=1, amr_classes=1))
        B, W = 8, 12
        data = {
            "features": torch.randn(B, W, 100),
            "masks": torch.ones(B, W, dtype=torch.bool),
            "mobility": torch.randint(0, 3, (B,)),
            "amr": torch.randint(0, 2, (B,), dtype=torch.float),
            "expansion": torch.randint(0, 2, (B,), dtype=torch.float),
        }
        m = evaluate(model, data)
        assert "mobility_balanced_accuracy" in m
        assert "amr_auroc" in m
        assert "expansion_auroc" in m

    def test_evaluate_large(self):
        from dna_sentinel.train import evaluate
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64,
                                              max_windows=12, expansion_classes=10, amr_classes=4))
        B, W = 8, 12
        data = {
            "features": torch.randn(B, W, 100),
            "masks": torch.ones(B, W, dtype=torch.bool),
            "mobility": torch.randint(0, 3, (B,)),
            "amr": torch.randint(0, 2, (B, 4), dtype=torch.float),
            "expansion": torch.randint(0, 10, (B,)),
        }
        m = evaluate(model, data)
        assert "mobility_balanced_accuracy" in m
        assert "amr_auroc" in m or any(k.startswith("amr_") and k.endswith("_auroc") for k in m)
        assert "expansion_auroc" in m or "expansion_accuracy" in m


# ---------------------------------------------------------------------------
# Inference tests
# ---------------------------------------------------------------------------

class TestInference:
    def test_predict_one(self):
        from dna_sentinel.utils import predict_one
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256,
                                              max_windows=28))
        pred = predict_one(model, "test", "ACGT" * 50)
        assert pred.sequence_id == "test"
        assert len(pred.mobility_probs) == 3
        assert 0 <= pred.amr_probability <= 1
        assert 0 <= pred.expansion_probability <= 1
        assert 0 <= pred.risk_score <= 1

    def test_predict_batch(self):
        from dna_sentinel.utils import predict_batch
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256,
                                              max_windows=28))
        preds = predict_batch(model, [("s1", "ACGT" * 50), ("s2", "TGCA" * 50)])
        assert len(preds) == 2
        assert preds[0].sequence_id == "s1"
        assert preds[1].sequence_id == "s2"

    def test_inference_service(self):
        import tempfile

        from dna_sentinel.utils import InferenceService
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256,
                                              max_windows=28))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model.save(f.name)
            service = InferenceService(f.name)
            result = service.predict("test", "ACGT" * 50)
            assert "risk_score" in result
            assert "amr_probability" in result
            Path(f.name).unlink()
