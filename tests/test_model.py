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
        model = Cassiopeia(CassiopeiaConfig(use_hierarchical=False, n_structural_features=49))
        assert sum(p.numel() for p in model.parameters() if p.requires_grad) < 700_000

    def test_prime_parameter_count(self):
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=160, frp_out_dim=384, n_layers=3, lora_rank=12, adapter_rank=16,
                                             use_cppe=True, use_hierarchical=False, n_structural_features=49))
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 500_000 < n <= 1_300_000, f"got {n}"

    def test_hierarchical_parameter_count(self):
        model = Cassiopeia(CassiopeiaConfig(use_hierarchical=True, n_scale_layers=2, n_structural_features=49))
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 300_000 < n <= 800_000, f"got {n}"

    def test_v9_ring_ssm_parameter_count(self):
        model = Cassiopeia(CassiopeiaConfig(
            n_structural_features=49, hidden_dim=128, frp_out_dim=320, n_layers=3,
            ring_ssm_kernel=7, use_hierarchical=True,
            n_scale_layers=2, use_cppe=True, max_windows=56,
        ))
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        assert 300_000 < n <= 900_000, f"got {n}"

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
        model = Cassiopeia(CassiopeiaConfig(n_structural_features=49, hidden_dim=64, n_canonical_features=100, frp_out_dim=80, n_layers=2, max_windows=12,
                                             lora_rank=4, adapter_rank=8, use_cppe=True))
        out = model(torch.randn(3, 12, 100), torch.ones(3, 12, dtype=torch.bool),
                     struct_features=torch.randn(3, 12, 49),
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
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=64, n_canonical_features=100, frp_out_dim=64, max_windows=12, n_structural_features=49))
        B, W = 8, 12
        val_data = {"features": torch.randn(B, W, 100), "masks": torch.ones(B, W, dtype=torch.bool),
                    "mobility": torch.randint(0, 3, (B,)), "amr": torch.randint(0, 2, (B,), dtype=torch.float),
                    "expansion": torch.randint(0, 2, (B,), dtype=torch.float),
                    "struct_features": torch.randn(B, W, 49)}
        result = fit_calibration(model, val_data, "cpu")
        assert "cal_mob_probs" in result

    def test_fit_calibration_single_class(self):
        from dna_sentinel.train import fit_calibration
        model = Cassiopeia(CassiopeiaConfig(hidden_dim=32, n_canonical_features=100, frp_out_dim=32, max_windows=8, n_structural_features=49))
        B = 8
        val_data = {"features": torch.randn(B, 8, 100), "masks": torch.ones(B, 8, dtype=torch.bool),
                    "mobility": torch.zeros(B, dtype=torch.long), "amr": torch.zeros(B, dtype=torch.float),
                    "expansion": torch.zeros(B, dtype=torch.float),
                    "struct_features": torch.randn(B, 8, 49)}
        result = fit_calibration(model, val_data, "cpu")
        assert "cal_mob_probs" in result


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


class TestV9RingSSM:
    def test_ring_ssm_block_output_shape(self):
        from dna_sentinel.model import RingSSMBlock
        block = RingSSMBlock(64, kernel=7, dropout=0.0)
        x = torch.randn(2, 10, 64)
        mask = torch.ones(2, 10, dtype=torch.bool)
        out = block(x, mask)
        assert out.shape == (2, 10, 64)

    def test_ring_ssm_mask_zeroing(self):
        from dna_sentinel.model import RingSSMBlock
        block = RingSSMBlock(64, kernel=5, dropout=0.0)
        x = torch.randn(2, 10, 64)
        mask = torch.tensor([[True]*6 + [False]*4, [True]*10])
        out = block(x, mask)
        assert torch.all(out[0, 6:] == 0)
        assert torch.isfinite(out).all()

    def test_ring_ssm_backward(self):
        from dna_sentinel.model import RingSSMBlock
        block = RingSSMBlock(32, kernel=5, dropout=0.0)
        x = torch.randn(2, 8, 32, requires_grad=True)
        mask = torch.ones(2, 8, dtype=torch.bool)
        out = block(x, mask)
        out.sum().backward()
        assert x.grad is not None and torch.isfinite(x.grad).all()


class TestArchitecture:
    def test_forward_eval(self):
        cfg = CassiopeiaConfig(
            hidden_dim=64, n_canonical_features=100, frp_out_dim=64,
            n_layers=1, max_windows=8, n_structural_features=49,
        )
        model = Cassiopeia(cfg)
        model.eval()
        out = model(torch.randn(2, 8, 100), torch.ones(2, 8, dtype=torch.bool),
                     struct_features=torch.randn(2, 8, 49))
        assert out["mobility_logits"].shape == (2, 3)

    def test_flat_vs_hierarchical(self):
        flat = Cassiopeia(CassiopeiaConfig(use_hierarchical=False, n_structural_features=49, max_windows=8))
        hier = Cassiopeia(CassiopeiaConfig(use_hierarchical=True, n_structural_features=49, max_windows=8))
        x = torch.randn(2, 8, 2728)
        m = torch.ones(2, 8, dtype=torch.bool)
        s = torch.randn(2, 8, 49)
        flat.eval(); hier.eval()
        of = flat(x, m, struct_features=s)["mobility_logits"]
        oh = hier(x, m, struct_features=s)["mobility_logits"]
        assert of.shape == oh.shape == (2, 3)


class TestConsistencyGradient:
    def test_consistency_loss_has_gradient(self):
        model = Cassiopeia(CassiopeiaConfig(
            hidden_dim=32, n_canonical_features=100, frp_out_dim=32,
            n_layers=1, max_windows=8, consistency_alpha=0.1,
        ))
        model.train()
        feat = torch.randn(4, 8, 100)
        mask = torch.ones(4, 8, dtype=torch.bool)
        out = model(feat, mask)
        mob_target = torch.randint(0, 3, (4,))
        amr_target = torch.rand(4)
        exp_target = torch.rand(4)
        loss = model.compute_loss(
            out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
            mob_target, amr_target, exp_target,
            exp_proxy_logits=out.get("exp_proxy_logits"),
        )
        loss["total"].backward()
        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                assert torch.isfinite(p.grad).any(), f"NaN grad in {n}"


class TestCompression:
    def test_compress_fp16_roundtrip(self):
        from dna_sentinel.model import compress_checkpoint, load_compressed
        model = Cassiopeia(CassiopeiaConfig(
            hidden_dim=32, n_canonical_features=100, frp_out_dim=32,
            n_layers=1, max_windows=8,
        ))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as src:
            model.save(src.name)
            src_path = src.name
        dst_path = src_path + ".fp16"
        info = compress_checkpoint(src_path, dst_path, fmt="fp16")
        loaded = load_compressed(dst_path)
        feat = torch.randn(2, 8, 100)
        mask = torch.ones(2, 8, dtype=torch.bool)
        model.eval()
        loaded.eval()
        with torch.no_grad():
            a = model(feat, mask)["mobility_logits"]
            b = loaded(feat, mask)["mobility_logits"]
        assert torch.allclose(a, b, atol=1e-2)
        assert info["dst_bytes"] < info["src_bytes"]
        Path(src_path).unlink()
        Path(dst_path).unlink()


class TestFeatureSchema:
    def test_schema_version_in_cache(self):
        from dna_sentinel.features import CanonicalKmerConfig, preprocess_all_features
        from dna_sentinel.utils import LabeledSequence
        records = [LabeledSequence("a", "ACGT" * 100, 1, 0, 1)]
        cfg = CanonicalKmerConfig(window_sizes=(32,), strides=(16,), max_windows=(4,))
        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            out = Path(f.name)
        preprocess_all_features(records, cfg, out, num_workers=1)
        saved = torch.load(out, weights_only=True)
        assert saved["_schema_version"] is not None
        assert saved["_n_structural_features"] == cfg.n_structural_features
        assert "_manifest_hash" in saved
        out.unlink()
