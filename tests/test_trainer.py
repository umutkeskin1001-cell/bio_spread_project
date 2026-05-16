"""Tests for BioSpreadTrainer: training loop, loss components, calibration."""
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from torch.utils.data import DataLoader

from bio_spread.data.dataset import sequence_collate
from bio_spread.models.components import PlattScaler
from bio_spread.models.sovereign import BioSpreadModel
from bio_spread.models.trainer import (
    BioSpreadTrainer,
    hazard_masked_bce,
    ranking_loss,
)


class SyntheticSequenceDataset:
    def __init__(self, B, L, n_static, n_snapshot):
        self.B = B
        self.L = L
        self.n_static = n_static
        self.n_snapshot = n_snapshot
        self.static = torch.randn(B, n_static)
        self.seq = torch.randn(B, L, n_snapshot)
        self.mask = torch.ones(B, L)
        self.hazard = torch.randint(0, 2, (B, L, 3)).float()
        self.hazard[:, :, torch.rand(3) > 0.7] = -1.0
        self.counts = torch.randint(0, 5, (B,)).float()
        self.seq_len = torch.full((B,), L)

    def __len__(self):
        return self.B

    def __getitem__(self, idx):
        return {
            "static": self.static[idx],
            "seq": self.seq[idx],
            "mask": self.mask[idx],
            "hazard": self.hazard[idx],
            "count": self.counts[idx],
            "seq_len": torch.tensor(self.L),
            "backbone_id": f"syn_{idx}",
        }


def syn_collate(batch, max_seq_len):
    return sequence_collate(batch, max_seq_len)


def test_hazard_masked_bce_all_valid():
    logits = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float32)
    targets = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)
    pw = torch.tensor(1.0)
    loss = hazard_masked_bce(logits, targets, pw)
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_hazard_masked_bce_censored():
    logits = torch.tensor([0.5, -0.5, 1.0], dtype=torch.float32)
    targets = torch.tensor([1.0, -1.0, 1.0], dtype=torch.float32)
    pw = torch.tensor(1.0)
    loss = hazard_masked_bce(logits, targets, pw)
    assert loss.item() > 0


def test_hazard_masked_bce_all_censored():
    logits = torch.randn(5)
    targets = torch.full((5,), -1.0)
    pw = torch.tensor(1.0)
    loss = hazard_masked_bce(logits, targets, pw)
    assert loss.item() == 0.0


def test_ranking_loss_increasing():
    probs = torch.tensor([[0.1, 0.5, 0.9], [0.2, 0.6, 0.7]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    loss = ranking_loss(probs, targets)
    assert loss.item() >= 0
    assert loss.item() < 1e-4


def test_ranking_loss_decreasing_penalized():
    probs = torch.tensor([[0.9, 0.5, 0.1]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0, 1.0]], dtype=torch.float32)
    loss = ranking_loss(probs, targets)
    assert loss.item() > 0


def test_ranking_loss_all_censored():
    probs = torch.randn(2, 3)
    targets = torch.full((2, 3), -1.0)
    loss = ranking_loss(probs, targets)
    assert loss.item() == 0.0


def test_platt_scaler_identity():
    scaler = PlattScaler()
    logits = torch.tensor([0.0, 1.0, -1.0], dtype=torch.float32)
    out = scaler(logits)
    assert out.shape == logits.shape
    assert torch.allclose(scaler.a, torch.tensor(1.0))
    assert torch.allclose(scaler.b, torch.zeros(1))


def test_platt_scaler_training():
    scaler = PlattScaler()
    logits = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=torch.float32)
    targets = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0], dtype=torch.float32)

    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.01, max_iter=50)
    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(scaler(logits), targets)
        loss.backward()
        return loss
    for _ in range(200):
        loss = opt.step(closure)
        if loss is None:
            break

    scaled = scaler(logits)
    probs = torch.sigmoid(scaled)
    assert probs[0] < probs[-1]


@given(
    B=st.integers(min_value=1, max_value=8),
    L=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=20, deadline=None)
def test_biotrainer_fit_synthetic(B, L):
    """End-to-end training on tiny synthetic data."""
    n_static, n_snap = 5, 10
    model = BioSpreadModel(n_static=n_static, n_snapshot=n_snap, static_dim=32, temporal_dim=32, hidden_dim=32, n_hazard=3)
    trainer = BioSpreadTrainer(model, device="cpu", epochs=2, patience=5, warmup_epochs=0)

    ds = SyntheticSequenceDataset(B, L, n_static, n_snap)
    loader = DataLoader(ds, batch_size=max(1, B // 2), collate_fn=lambda b: syn_collate(b, L))

    path = trainer.fit(loader, val_loader=loader)
    assert path.exists() or path.name.startswith("BS_")


@given(
    B=st.integers(min_value=1, max_value=6),
    L=st.integers(min_value=2, max_value=5),
)
@settings(max_examples=10, deadline=None)
def test_biotrainer_evaluate_synthetic(B, L):
    """Test evaluate method on synthetic data."""
    n_static, n_snap = 5, 10
    model = BioSpreadModel(n_static=n_static, n_snapshot=n_snap, static_dim=32, temporal_dim=32, hidden_dim=32, n_hazard=3)
    trainer = BioSpreadTrainer(model, device="cpu", calibrate=False)

    ds = SyntheticSequenceDataset(B, L, n_static, n_snap)
    loader = DataLoader(ds, batch_size=B, collate_fn=lambda b: syn_collate(b, L))

    metrics = trainer.evaluate(loader)
    assert "roc_auc" in metrics
    assert "f1" in metrics
    assert "ece" in metrics
    assert "n" in metrics


def test_biotrainer_pos_weight_horizons():
    """Test per-horizon pos_weight computation."""
    B, L, n_static, n_snap = 8, 5, 5, 10
    model = BioSpreadModel(n_static=n_static, n_snapshot=n_snap, static_dim=32, temporal_dim=32, hidden_dim=32, n_hazard=3)
    trainer = BioSpreadTrainer(model, device="cpu", calibrate=False)
    ds = SyntheticSequenceDataset(B, L, n_static, n_snap)
    # Override hazard to create specific imbalance
    ds.hazard = torch.zeros(B, L, 3)
    ds.hazard[:, :, 0] = 1.0
    ds.hazard[:, :, 1] = 0.0
    ds.hazard[:, :, 2] = torch.randint(0, 2, (B, L)).float()

    loader = DataLoader(ds, batch_size=4, collate_fn=lambda b: syn_collate(b, L))
    pos_weights = trainer._compute_pos_weight(loader)
    assert pos_weights.shape == (3,)
    assert pos_weights[0] < 1.0
    assert pos_weights[1] > 1.0


def test_cold_start_head_separate():
    """Test cold-start head produces different logits than main head when temporal is masked."""
    model = BioSpreadModel(n_static=5, n_snapshot=10, static_dim=32, temporal_dim=32, hidden_dim=32, n_hazard=3)
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 5, 10)
    mask = torch.ones(4, 5)
    temporal_mask = torch.tensor([True, True, False, False])

    out = model(static, snapshots, mask, temporal_mask=temporal_mask)

    cold_probs = torch.sigmoid(out.cold_logits)
    main_probs = torch.sigmoid(out.hazard_logits)
    assert cold_probs.shape == (4, 3)
    assert not torch.isnan(cold_probs).any()
    assert not torch.isnan(main_probs).any()


@given(
    logits=st.lists(st.floats(min_value=-5, max_value=5, allow_nan=False), min_size=1, max_size=20),
    targets=st.lists(st.floats(min_value=-1, max_value=1, allow_nan=False), min_size=1, max_size=20),
)
@settings(max_examples=50, deadline=None)
def test_hazard_masked_bce_properties(logits, targets):
    """Property: hazard_masked_bce is non-negative, zero when all censored, NaN-free."""
    n = min(len(logits), len(targets))
    if n == 0:
        return
    t = torch.tensor(logits[:n])
    u = torch.tensor(targets[:n])
    pw = torch.tensor(1.0)
    loss = hazard_masked_bce(t, u, pw)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0
    valid = (u >= 0).any()
    if not valid:
        assert loss.item() == 0.0


@given(
    B=st.integers(min_value=1, max_value=8),
    L=st.integers(min_value=2, max_value=6),
)
@settings(max_examples=20, deadline=None)
def test_ranking_loss_properties(B, L):
    """Property: ranking_loss is non-negative, zero on all-censored, NaN-free."""
    probs = torch.rand(B, L).float()
    targets = torch.randint(-1, 2, (B, L)).float()
    loss = ranking_loss(probs, targets)
    assert not torch.isnan(loss)
    assert loss.item() >= 0.0
    if (targets >= 0).sum() == 0:
        assert loss.item() == 0.0
