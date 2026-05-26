"""Unit tests for Cassiopeia model and data pipeline."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from dna_sentinel.model import (
    Cassiopeia,
    CassiopeiaConfig,
    CircularPositionEncoding,
    DropPath,
    GLUMixer,
    WindowMotifConv,
    _focal_bce,
    _make_frp,
)


class TestCassiopeiaConfig:
    def test_default_small(self):
        cfg = CassiopeiaConfig()
        assert cfg.hidden_dim == 128
        assert cfg.expansion_classes == 1
        assert cfg.amr_classes == 1
        assert cfg.learnable_frp is False

    def test_from_yaml(self):
        cfg = CassiopeiaConfig.from_yaml({"model": {"hidden_dim": 512, "expansion_classes": 50, "amr_classes": 12}})
        assert cfg.hidden_dim == 512
        assert cfg.expansion_classes == 50
        assert cfg.amr_classes == 12

    def test_to_dict(self):
        d = CassiopeiaConfig().to_dict()
        assert isinstance(d, dict) and d["hidden_dim"] == 128

    def test_extra_keys_ignored(self):
        cfg = CassiopeiaConfig.from_yaml({"model": {"hidden_dim": 256, "nonexistent_key": 999}})
        assert cfg.hidden_dim == 256

    def test_learnable_frp_default_off(self):
        assert CassiopeiaConfig().learnable_frp is False

    def test_learnable_frp_custom(self):
        cfg = CassiopeiaConfig(learnable_frp=True)
        assert cfg.learnable_frp is True


class TestFRPMatrix:
    def test_shape_and_values(self):
        mat = _make_frp(100, 256)
        assert mat.shape == (100, 256)
        assert torch.all((mat == 1) | (mat == -1) | (mat == 0))
        assert (mat == 0).float().mean() > 0.5

    def test_deterministic(self):
        assert torch.allclose(_make_frp(100, 100, seed=42), _make_frp(100, 100, seed=42))

    def test_different_seeds_differ(self):
        assert not torch.allclose(_make_frp(100, 100, seed=42), _make_frp(100, 100, seed=99))


class TestDropPath:
    def test_identity_when_eval(self):
        dp = DropPath(0.5).eval()
        x = torch.randn(4, 10)
        assert torch.allclose(dp(x), x)

    def test_identity_when_zero_rate(self):
        dp = DropPath(0.0).train()
        x = torch.randn(4, 10)
        assert torch.allclose(dp(x), x)

    def test_output_shape(self):
        dp = DropPath(0.5).train()
        x = torch.randn(4, 10, 128)
        assert dp(x).shape == x.shape


class TestLosses:
    def test_focal_bce_basic(self):
        loss = _focal_bce(torch.tensor([2.0, -2.0]), torch.tensor([1.0, 0.0]), pw=None, gamma=0.0)
        assert loss.item() > 0 and not torch.isnan(loss)

    def test_focal_bce_with_gamma(self):
        logits, target = torch.tensor([2.0, -2.0]), torch.tensor([1.0, 0.0])
        assert _focal_bce(logits, target, pw=None, gamma=2.0) < _focal_bce(logits, target, pw=None, gamma=0.0)

    def test_focal_bce_pos_weight(self):
        loss_n = _focal_bce(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), pw=None, gamma=0.0)
        loss_p = _focal_bce(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 0.0]), pw=torch.tensor([3.0]), gamma=0.0)
        assert loss_p > loss_n


class TestBlocks:
    def test_glu_mixer(self):
        out = GLUMixer(28, 128)(torch.randn(4, 28, 128))
        assert out.shape == (4, 28, 128)

    def test_glu_mixer_with_mask(self):
        x = torch.randn(4, 28, 128)
        mask = torch.ones(4, 28, dtype=torch.bool)
        mask[:, 20:] = False
        out = GLUMixer(28, 128)(x, mask)
        assert out.shape == x.shape
        assert torch.all(out[:, 20:] == 0)

    def test_cppe_shape_and_mask_zeroing(self):
        cppe = CircularPositionEncoding(32)
        x = torch.zeros(2, 6, 32)
        mask = torch.tensor([[True, True, True, False, False, False],
                             [True, True, True, True, True, True]])
        scale_ids = torch.tensor([[0, 0, 1, 1, 2, 2], [0, 0, 1, 1, 2, 2]])
        out = cppe(x, mask, scale_ids)
        assert out.shape == x.shape and torch.all(out[0, 3:] == 0)

    def test_cppe_no_scale_ids(self):
        cppe = CircularPositionEncoding(32)
        out = cppe(torch.zeros(2, 6, 32), torch.ones(2, 6, dtype=torch.bool))
        assert out.shape == (2, 6, 32)

    def test_window_motif_conv(self):
        conv = WindowMotifConv(hidden_dim=32, kernel_size=5, dropout=0.0)
        x = torch.randn(2, 8, 32)
        mask = torch.tensor([[True, True, True, True, False, False, False, False],
                             [True, True, True, True, True, True, True, True]])
        out = conv(x, mask)
        assert out.shape == x.shape and torch.all(out[0, 4:] == 0) and torch.isfinite(out).all()

    def test_window_motif_conv_odd_kernel(self):
        with pytest.raises(ValueError, match="odd"):
            WindowMotifConv(32, kernel_size=4)


class TestCassiopeiaSmall:
    @pytest.fixture
    def model(self):
        return Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64, n_layers=1, max_windows=12, expansion_classes=1, amr_classes=1))

    @pytest.fixture
    def batch(self):
        return {"features": torch.randn(4, 12, 100), "masks": torch.ones(4, 12, dtype=torch.bool),
                "mobility": torch.randint(0, 3, (4,)), "amr": torch.randint(0, 2, (4,), dtype=torch.float),
                "expansion": torch.randint(0, 2, (4,), dtype=torch.float)}

    def test_forward(self, model, batch):
        out = model(batch["features"], batch["masks"])
        assert out["mobility_logits"].shape == (4, 3)
        assert out["amr_logits"].shape == (4,)
        assert out["expansion_logits"].shape == (4,)

    def test_compute_loss(self, model, batch):
        out = model(batch["features"], batch["masks"])
        loss = model.compute_loss(out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
                                   batch["mobility"], batch["amr"], batch["expansion"])
        assert loss["total"].item() > 0 and not torch.isnan(loss["total"])

    def test_save_load(self, model, batch):
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            path = f.name
            model.save(path)
        loaded = Cassiopeia.load(path)
        assert loaded(batch["features"], batch["masks"])["mobility_logits"].shape == (4, 3)
        Path(path).unlink()

    def test_parameter_count(self):
        model = Cassiopeia(CassiopeiaConfig(use_hierarchical=False))
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 500_000

    def test_prime_parameter_count(self):
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=160, frp_out_dim=384, n_layers=3, lora_rank=12, adapter_rank=16,
                                             use_scale_embedding=False, use_cppe=True, use_window_conv=True,
                                             window_conv_kernel=5, use_hierarchical=False))
        assert 700_000 < sum(p.numel() for p in model.parameters() if p.requires_grad) <= 1_100_000

    def test_hierarchical_parameter_count(self):
        model = Cassiopeia(CassiopeiaConfig(use_hierarchical=True, n_scale_layers=2, use_scale_gate=True))
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 400_000 < n <= 600_000, f"got {n}"

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
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64, n_layers=1, max_windows=12))
        out = model(batch["features"], batch["masks"])
        loss = model.compute_loss(out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
                                   batch["mobility"], batch["amr"], batch["expansion"])
        loss["total"].backward()
        assert any(p.grad is not None for n, p in model.named_parameters() if p.requires_grad)

    def test_forward_returns_task_specific_evidence(self, model, batch):
        out = model(batch["features"], batch["masks"])
        assert out["mobility_evidence"].shape == batch["masks"].shape

    def test_forward_with_struct_and_scale(self):
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=80, n_layers=2, max_windows=12,
                                             lora_rank=4, adapter_rank=8, use_scale_embedding=False, use_cppe=True, use_window_conv=True, window_conv_kernel=3))
        out = model(torch.randn(3, 12, 100), torch.ones(3, 12, dtype=torch.bool),
                     struct_features=torch.randn(3, 12, 19),
                     scale_ids=torch.tensor([[0]*4 + [1]*4 + [2]*4]*3))
        assert out["mobility_logits"].shape == (3, 3)

    def test_forward_all_n_mask(self, model):
        mask = torch.zeros(4, 12, dtype=torch.bool)
        out = model(torch.randn(4, 12, 100), mask)
        assert torch.isfinite(out["mobility_logits"]).all()

    def test_learnable_frp_forward(self):
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=32, n_canonical_features=100, frp_out_dim=32, n_layers=1, max_windows=8, learnable_frp=True))
        out = model(torch.randn(2, 8, 100), torch.ones(2, 8, dtype=torch.bool))
        assert out["mobility_logits"].shape == (2, 3)


class TestCassiopeiaLarge:
    @pytest.fixture
    def model(self):
        return Cassiopeia(CassiopeiaConfig(hidden_dim=128, n_canonical_features=100, frp_out_dim=64, n_layers=2, max_windows=12, expansion_classes=10, amr_classes=4))

    @pytest.fixture
    def batch(self):
        return {"features": torch.randn(4, 12, 100), "masks": torch.ones(4, 12, dtype=torch.bool),
                "mobility": torch.randint(0, 3, (4,)), "amr": torch.randint(0, 2, (4, 4), dtype=torch.float),
                "expansion": torch.randint(0, 10, (4,))}

    def test_forward(self, model, batch):
        out = model(batch["features"], batch["masks"])
        assert out["amr_logits"].shape == (4, 4) and out["expansion_logits"].shape == (4, 10)

    def test_loss(self, model, batch):
        out = model(batch["features"], batch["masks"])
        assert model.compute_loss(out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
                                   batch["mobility"], batch["amr"], batch["expansion"])["total"].item() > 0

    def test_expansion_softmax(self, model, batch):
        probs = torch.softmax(model(batch["features"], batch["masks"])["expansion_logits"], dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(4))


class TestCalibration:
    def test_fit_calibration(self):
        from dna_sentinel.train import fit_calibration
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64, max_windows=12, expansion_classes=1, amr_classes=1))
        B, W = 8, 12
        val_data = {"features": torch.randn(B, W, 100), "masks": torch.ones(B, W, dtype=torch.bool),
                    "mobility": torch.randint(0, 3, (B,)), "amr": torch.randint(0, 2, (B,), dtype=torch.float),
                    "expansion": torch.randint(0, 2, (B,), dtype=torch.float)}
        fit_calibration(model, val_data, "cpu")
        assert model.mob_t.item() > 0

    def test_fit_calibration_single_class(self):
        from dna_sentinel.train import fit_calibration
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=32, n_canonical_features=100, frp_out_dim=32, max_windows=8))
        B = 8
        val_data = {"features": torch.randn(B, 8, 100), "masks": torch.ones(B, 8, dtype=torch.bool),
                    "mobility": torch.zeros(B, dtype=torch.long), "amr": torch.zeros(B, dtype=torch.float),
                    "expansion": torch.zeros(B, dtype=torch.float)}
        fit_calibration(model, val_data, "cpu")
        assert model.mob_t.item() > 0


class TestEvaluate:
    def test_evaluate_small(self):
        from dna_sentinel.train import evaluate
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64, max_windows=12))
        m = evaluate(model, {"features": torch.randn(8, 12, 100), "masks": torch.ones(8, 12, dtype=torch.bool),
                             "mobility": torch.randint(0, 3, (8,)), "amr": torch.randint(0, 2, (8,), dtype=torch.float),
                             "expansion": torch.randint(0, 2, (8,), dtype=torch.float)})
        assert "mobility_balanced_accuracy" in m and "amr_auroc" in m and "expansion_auroc" in m

    def test_evaluate_large(self):
        from dna_sentinel.train import evaluate
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64, max_windows=12, expansion_classes=10, amr_classes=4))
        m = evaluate(model, {"features": torch.randn(8, 12, 100), "masks": torch.ones(8, 12, dtype=torch.bool),
                             "mobility": torch.randint(0, 3, (8,)), "amr": torch.randint(0, 2, (8, 4), dtype=torch.float),
                             "expansion": torch.randint(0, 10, (8,))})
        assert "mobility_balanced_accuracy" in m and "amr_auroc" in m

class TestInference:
    def test_predict_one(self):
        from dna_sentinel.utils import predict_one
        pred = predict_one(Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256, max_windows=28)), "test", "ACGT" * 50)
        assert pred.sequence_id == "test" and len(pred.mobility_probs) == 3 and 0 <= pred.amr_probability <= 1

    def test_predict_one_has_task_specific_windows(self):
        from dna_sentinel.utils import predict_one
        pred = predict_one(Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256, max_windows=28)), "test", "ACGT" * 50)
        assert isinstance(pred.top_windows, list) and isinstance(pred.top_mobility_windows, list)

    def test_predict_batch(self):
        from dna_sentinel.utils import predict_batch
        preds = predict_batch(Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256, max_windows=28)),
                               [("s1", "ACGT" * 50), ("s2", "TGCA" * 50)])
        assert len(preds) == 2 and preds[0].sequence_id == "s1"

    def test_predict_batch_empty(self):
        from dna_sentinel.utils import predict_batch
        assert predict_batch(Cassiopeia(CassiopeiaConfig(max_windows=28)), []) == []

    def test_inference_service(self):
        from dna_sentinel.utils import InferenceService
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=2728, frp_out_dim=256, max_windows=28))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            model.save(f.name)
            svc = InferenceService(f.name)
            result = svc.predict("test", "ACGT" * 50)
            assert "risk_score" in result and "amr_probability" in result
            Path(f.name).unlink()
