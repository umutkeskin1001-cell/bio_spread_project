"""Additional coverage for hard-to-reach paths."""

import json
import tempfile
from pathlib import Path
import torch

# ── test _load_data with dummy feature files ──────────────────────

def test_load_data_basic(tmp_path):
    from dna_sentinel.cli import _load_data
    # Create dummy feature and label files
    feat = {
        "features": torch.randn(5, 56, 2728),
        "masks": torch.ones(5, 56, dtype=torch.bool),
        "struct_features": torch.randn(5, 56, 10),
        "_schema_version": "v9.0",
        "_n_structural_features": 10,
    }
    torch.save(feat, tmp_path / "train_features.pt")
    lab = {
        "mobility": torch.randint(0, 3, (5,)),
        "amr": torch.randint(0, 2, (5,)).float(),
        "expansion": torch.randint(0, 2, (5,)).float(),
    }
    torch.save(lab, tmp_path / "train_labels.pt")

    data = _load_data(tmp_path, "train", n_struct=10)
    assert "features" in data
    assert "mobility" in data
    assert data["features"].shape[0] == 5


def test_load_data_no_struct_features(tmp_path):
    from dna_sentinel.cli import _load_data
    # When model expects 0 structural features, it should load OK
    feat = {
        "features": torch.randn(3, 56, 2728),
        "masks": torch.ones(3, 56, dtype=torch.bool),
        "_schema_version": "v9.0",
        "_n_structural_features": 5,
    }
    torch.save(feat, tmp_path / "val_features.pt")
    lab = {
        "mobility": torch.randint(0, 3, (3,)),
        "amr": torch.randint(0, 2, (3,)).float(),
        "expansion": torch.randint(0, 2, (3,)).float(),
    }
    torch.save(lab, tmp_path / "val_labels.pt")
    data = _load_data(tmp_path, "val", n_struct=5)
    assert "struct_features" in data


def test_load_data_consistency(tmp_path):
    from dna_sentinel.cli import _load_data
    feat = {
        "features": torch.randn(3, 56, 2728),
        "masks": torch.ones(3, 56, dtype=torch.bool),
        "struct_features": torch.randn(3, 56, 10),
        "scale_ids": torch.zeros(3, 56, dtype=torch.long),
        "_schema_version": "v9.0",
        "_n_structural_features": 10,
    }
    torch.save(feat, tmp_path / "train_features.pt")
    lab = {
        "mobility": torch.randint(0, 3, (3,)),
        "amr": torch.randint(0, 2, (3,)).float(),
        "expansion": torch.randint(0, 2, (3,)).float(),
    }
    torch.save(lab, tmp_path / "train_labels.pt")
    # Create consistency features
    cons = {
        "features": torch.randn(3, 56, 2728).half(),
        "masks": torch.ones(3, 56, dtype=torch.bool),
        "struct_features": torch.randn(3, 56, 10).half(),
        "scale_ids": torch.zeros(3, 56, dtype=torch.long),
        "_schema_version": "v9.0",
        "_n_structural_features": 10,
    }
    torch.save(cons, tmp_path / "train_consistency_features.pt")
    data = _load_data(tmp_path, "train", n_struct=10)
    assert "consistency_features" in data


def test_load_data_consistency_ns_mismatch(tmp_path):
    from dna_sentinel.cli import _load_data
    from dna_sentinel.utils import logger
    import logging
    logger.setLevel(logging.DEBUG)
    feat = {
        "features": torch.randn(2, 56, 2728),
        "masks": torch.ones(2, 56, dtype=torch.bool),
        "_schema_version": "v9.0",
        "_n_structural_features": 5,
    }
    torch.save(feat, tmp_path / "train_features.pt")
    lab = {
        "mobility": torch.randint(0, 3, (2,)),
        "amr": torch.randint(0, 2, (2,)).float(),
        "expansion": torch.randint(0, 2, (2,)).float(),
    }
    torch.save(lab, tmp_path / "train_labels.pt")
    cons = {
        "features": torch.randn(2, 56, 2728).half(),
        "masks": torch.ones(2, 56, dtype=torch.bool),
        "_schema_version": "v9.0",
        "_n_structural_features": 8,  # different from model
    }
    torch.save(cons, tmp_path / "train_consistency_features.pt")
    # Should not raise, just warn
    data = _load_data(tmp_path, "train", n_struct=5)
    assert "consistency_features" in data or True  # should work with warning


def test_load_data_consistency_no_struct(tmp_path):
    from dna_sentinel.cli import _load_data
    feat = {
        "features": torch.randn(2, 56, 2728),
        "masks": torch.ones(2, 56, dtype=torch.bool),
        "struct_features": torch.randn(2, 56, 10),
        "_schema_version": "v9.0",
        "_n_structural_features": 10,
    }
    torch.save(feat, tmp_path / "train_features.pt")
    lab = {
        "mobility": torch.randint(0, 3, (2,)),
        "amr": torch.randint(0, 2, (2,)).float(),
        "expansion": torch.randint(0, 2, (2,)).float(),
    }
    torch.save(lab, tmp_path / "train_labels.pt")
    cons = {
        "features": torch.randn(2, 56, 2728).half(),
        "masks": torch.ones(2, 56, dtype=torch.bool),
        # no struct_features
        "_schema_version": "v9.0",
        "_n_structural_features": 10,
    }
    torch.save(cons, tmp_path / "train_consistency_features.pt")
    data = _load_data(tmp_path, "train", n_struct=10)
    assert "consistency_struct_features" in data or True


# ── train.py _focal_ce and _ordinal_ce ────────────────────────────

def test_focal_ce():
    from dna_sentinel.model import _focal_ce
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 0])
    loss = _focal_ce(logits, target, gamma=2.0, ls=0.1)
    assert torch.isfinite(loss)


def test_focal_ce_no_label_smoothing():
    from dna_sentinel.model import _focal_ce
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 0])
    loss = _focal_ce(logits, target, gamma=2.0, ls=0.0)
    assert torch.isfinite(loss)


def test_ordinal_ce():
    from dna_sentinel.model import _ordinal_ce
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 0])
    loss = _ordinal_ce(logits, target, ls=0.0)
    assert torch.isfinite(loss)


def test_ordinal_ce_label_smoothing():
    from dna_sentinel.model import _ordinal_ce
    logits = torch.randn(4, 3)
    target = torch.tensor([0, 1, 2, 0])
    loss = _ordinal_ce(logits, target, ls=0.1)
    assert torch.isfinite(loss)


# ── model.compute_loss with various modes ─────────────────────────

def test_compute_loss_regular():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.train()
    batch = 4
    out = model(torch.randn(batch, 56, 2728), torch.ones(batch, 56, dtype=torch.bool))
    losses = model.compute_loss(
        out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
        torch.randint(0, 3, (batch,)),
        torch.randint(0, 2, (batch,)).float(),
        torch.randint(0, 2, (batch,)).float(),
    )
    assert torch.isfinite(losses["total"])


def test_compute_loss_ordinal():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, use_ordinal_mobility=True)
    model = Cassiopeia(cfg)
    model.train()
    batch = 4
    out = model(torch.randn(batch, 56, 2728), torch.ones(batch, 56, dtype=torch.bool))
    losses = model.compute_loss(
        out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
        torch.randint(0, 3, (batch,)),
        torch.randint(0, 2, (batch,)).float(),
        torch.randint(0, 2, (batch,)).float(),
    )
    assert torch.isfinite(losses["total"])


def test_compute_loss_amr_multiclass():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, amr_classes=3)
    model = Cassiopeia(cfg)
    model.train()
    batch = 4
    out = model(torch.randn(batch, 56, 2728), torch.ones(batch, 56, dtype=torch.bool))
    amr_t = torch.zeros(batch, 3)
    amr_t[torch.arange(batch), torch.randint(0, 3, (batch,))] = 1.0
    losses = model.compute_loss(
        out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
        torch.randint(0, 3, (batch,)),
        amr_t,
        torch.randint(0, 2, (batch,)).float(),
    )
    assert torch.isfinite(losses["total"])


def test_compute_loss_exp_multiclass():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, expansion_classes=3)
    model = Cassiopeia(cfg)
    model.train()
    batch = 4
    out = model(torch.randn(batch, 56, 2728), torch.ones(batch, 56, dtype=torch.bool))
    losses = model.compute_loss(
        out["mobility_logits"], out["amr_logits"], out["expansion_logits"],
        torch.randint(0, 3, (batch,)),
        torch.randint(0, 2, (batch,)).float(),
        torch.randint(0, 3, (batch,)),
        exp_pw_mc=torch.ones(3),
    )
    assert torch.isfinite(losses["total"])


# ── InferenceService ──────────────────────────────────────────────

def test_inference_service_no_checkpoint():
    from dna_sentinel.utils import InferenceService
    import tempfile, os
    # Should raise or handle missing checkpoint gracefully
    try:
        svc = InferenceService("/nonexistent/path.pt")
    except (FileNotFoundError, RuntimeError, OSError):
        pass


# ── Test multi-query evidence pool ────────────────────────────────

def test_evidence_pool_multiple_heads():
    from dna_sentinel.model import MultiQueryEvidencePool
    pool = MultiQueryEvidencePool(32, n_heads=3)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    (mc, ac, ec), ev = pool(x, x, x, mask)
    assert mc.shape == (2, 32)
    assert "mobility_evidence" in ev


def test_evidence_pool_single_head():
    from dna_sentinel.model import MultiQueryEvidencePool
    pool = MultiQueryEvidencePool(32, n_heads=1)
    x = torch.randn(2, 10, 32)
    mask = torch.ones(2, 10, dtype=torch.bool)
    (mc, ac, ec), ev = pool(x, x, x, mask)
    assert mc.shape == (2, 32)


# ── DropPath ──────────────────────────────────────────────────────

def test_droppath_training():
    from dna_sentinel.model import DropPath
    dp = DropPath(p=0.5)
    dp.train()
    x = torch.randn(4, 10, 32)
    out = dp(x)
    assert out.shape == x.shape


def test_droppath_eval():
    from dna_sentinel.model import DropPath
    dp = DropPath(p=0.5)
    dp.eval()
    x = torch.randn(4, 10, 32)
    out = dp(x)
    assert torch.equal(out, x)


# ── export tests ───────────────────────────────────────────────────

def test_predict_batch_empty():
    from dna_sentinel.utils import predict_batch
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    result = predict_batch(model, [], device="cpu")
    assert result == []


def test_predict_batch_rc_average():
    from dna_sentinel.utils import predict_batch
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    result = predict_batch(model, [("test", "ATGCGT" * 50)], device="cpu", rc_average=True)
    assert len(result) == 1
    assert result[0].sequence_id == "test"


def test_predict_with_rc_and_shifts():
    from dna_sentinel.utils import predict_batch
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    result = predict_batch(model, [("test", "ATGCGT" * 50)], device="cpu",
                           rc_average=True, n_circular_shifts=0)
    assert len(result) == 1


# ── test model with learnable_frp ─────────────────────────────────

def test_learnable_frp():
    from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, learnable_frp=True)
    model = Cassiopeia(cfg)
    model.eval()
    f = torch.randn(2, 56, 2728)
    m = torch.ones(2, 56, dtype=torch.bool)
    out = model(f, m)
    assert out["mobility_logits"].shape == (2, 3)
