"""Final push to reach 85% coverage."""

import numpy as np
import torch

from dna_sentinel.features import (
    CanonicalKmerConfig,
    CanonicalKmerExtractor,
    _canonical_vocab,
    _resolve_max_windows,
    _vocab_offsets,
    preprocess_all_features,
)
from dna_sentinel.model import Cassiopeia, CassiopeiaConfig
from dna_sentinel.prepare import _DSU, SequenceRecord, _jaccard, _sampled_kmers, cluster_split
from dna_sentinel.train import (
    _compute_batch_loss,
    _consistency_loss,
    _run_model,
)
from dna_sentinel.utils import configure_logging, task_score

# ── train.py ────────────────────────────────────────────────────────

def test_consistency_loss_binary():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, expansion_classes=1))
    model.eval()
    batch = 4
    f = torch.randn(batch, 56, 16)
    m = torch.ones(batch, 56, dtype=torch.bool)
    out_ref = model.forward_from_encoder(f, m)
    out_aug = model.forward_from_encoder(f + 0.01, m)
    loss = _consistency_loss(out_ref, out_aug, temperature=1.0, expansion_classes=1)
    assert loss > 0, f"consistency loss should be positive, got {loss}"
    assert torch.isfinite(loss)


def test_consistency_loss_multiclass():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, expansion_classes=3))
    model.eval()
    batch = 4
    f = torch.randn(batch, 56, 16)
    m = torch.ones(batch, 56, dtype=torch.bool)
    out_ref = model.forward_from_encoder(f, m)
    out_aug = model.forward_from_encoder(f + 0.01, m)
    loss = _consistency_loss(out_ref, out_aug, temperature=1.0, expansion_classes=3)
    assert loss > 0
    assert torch.isfinite(loss)


def test_consistency_loss_identical():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, expansion_classes=1))
    model.eval()
    f = torch.randn(2, 56, 16)
    m = torch.ones(2, 56, dtype=torch.bool)
    out = model.forward_from_encoder(f, m)
    loss = _consistency_loss(out, out, temperature=1.0, expansion_classes=1)
    assert loss < 0.1


def test_run_model():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    data = {
        "features": torch.randn(8, 56, 2728),
        "masks": torch.ones(8, 56, dtype=torch.bool),
        "mobility": torch.randint(0, 3, (8,)),
        "amr": torch.randint(0, 2, (8,)).float(),
        "expansion": torch.randint(0, 2, (8,)).float(),
    }
    mob, amr, exp = _run_model(data, model, "cpu", bs=4)
    assert mob.shape == (8, 3)
    assert amr.shape == (8,)
    assert exp.shape == (8,)


def test_run_model_with_struct():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, n_structural_features=5)
    model = Cassiopeia(cfg)
    model.eval()
    data = {
        "features": torch.randn(4, 56, 2728),
        "masks": torch.ones(4, 56, dtype=torch.bool),
        "struct_features": torch.randn(4, 56, 5),
        "scale_ids": torch.zeros(4, 56, dtype=torch.long),
        "mobility": torch.randint(0, 3, (4,)),
        "amr": torch.randint(0, 2, (4,)).float(),
        "expansion": torch.randint(0, 2, (4,)).float(),
    }
    mob, amr, exp = _run_model(data, model, "cpu", bs=2)
    assert mob.shape == (4, 3)


def test_compute_batch_loss():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    model.eval()
    batch = 4
    out = {
        "mobility_logits": torch.randn(batch, 3),
        "amr_logits": torch.randn(batch),
        "expansion_logits": torch.randn(batch),
    }
    losses = _compute_batch_loss(
        model, out,
        torch.randint(0, 3, (batch,)),
        torch.randint(0, 2, (batch,)).float(),
        torch.randint(0, 2, (batch,)).float(),
        None, None, None, 2.0,
    )
    assert "total" in losses
    assert torch.isfinite(losses["total"])


def test_compute_batch_loss_with_consistency():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, consistency_alpha=0.1)
    model = Cassiopeia(cfg)
    model.train()
    batch = 4
    out = model(torch.randn(batch, 56, 2728), torch.ones(batch, 56, dtype=torch.bool))
    losses = _compute_batch_loss(
        model, out,
        torch.randint(0, 3, (batch,)),
        torch.randint(0, 2, (batch,)).float(),
        torch.randint(0, 2, (batch,)).float(),
        None, None, None, 2.0,
    )
    assert torch.isfinite(losses["total"])


# ── utils.py ────────────────────────────────────────────────────────

def test_configure_logging_default():
    configure_logging()


def test_task_score_with_metrics():
    result = task_score({"mobility_balanced_accuracy": 0.8, "amr_auroc": 0.9, "expansion_auroc": 0.7})
    assert abs(result - 0.8) < 1e-10


def test_task_score_missing_key():
    score = task_score({"mobility_balanced_accuracy": 0.8})
    assert score <= 0.8


# ── features.py ─────────────────────────────────────────────────────

def test_canonical_vocab_values():
    v = _canonical_vocab(4)
    assert v > 0
    v5 = _canonical_vocab(5)
    assert v5 > v


def test_vocab_offsets_monotonic():
    offsets = _vocab_offsets(3, 6)
    for k in range(3, 6):
        assert offsets[k + 1] > offsets[k]


def test_resolve_max_windows_large():
    result = _resolve_max_windows(100)
    assert sum(result) == 100
    assert len(result) == 3


def test_preprocess_empty_records(tmp_path):
    from dna_sentinel.features import preprocess_all_features
    cfg = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=5)
    preprocess_all_features([], cfg, tmp_path / "empty.pt")
    assert not (tmp_path / "empty.pt").exists()


def test_preprocess_all_features(tmp_path):
    from dna_sentinel.utils import LabeledSequence
    records = [LabeledSequence("s1", "ATGCGT" * 50, 0, 0, 0)]
    cfg = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=5)
    preprocess_all_features(records, cfg, tmp_path / "out.pt", num_workers=1)
    assert (tmp_path / "out.pt").exists()
    data = torch.load(tmp_path / "out.pt", weights_only=True)
    assert "features" in data


def test_extractor_rc_consensus():
    cfg = CanonicalKmerConfig(max_windows=(2, 1, 1), n_structural_features=5, rc_consensus=True)
    ex = CanonicalKmerExtractor(cfg)
    feat, struct, mask, scale = ex.extract("ATGCGT" * 100)
    assert mask.sum().item() > 0


def test_extractor_narrow_window():
    cfg = CanonicalKmerConfig(window_sizes=(100,), strides=(50,), max_windows=(2,), n_structural_features=5)
    ex = CanonicalKmerExtractor(cfg)
    feat, struct, mask, scale = ex.extract("ATGCGT" * 25)
    assert mask.sum().item() > 0


# ── prepare.py ──────────────────────────────────────────────────────

def test_sampled_kmers_various_sizes():
    seq = "".join(np.random.choice(["A", "C", "G", "T"], size=10000).tolist())
    res1 = _sampled_kmers(seq, k=21, n=500)
    assert len(res1) <= 500
    assert len(res1) > 0
    res2 = _sampled_kmers(seq, k=21, n=2000)
    assert len(res2) <= 2000


def test_jaccard_edge_cases():
    assert _jaccard({"A"}, {"A"}) == 1.0
    assert _jaccard({"A"}, {"B"}) == 0.0
    assert _jaccard(set(), {"A"}) == 0.0


def test_dsu_large():
    dsu = _DSU(100)
    for i in range(0, 100, 2):
        dsu.union(i, i + 1)
    roots = set(dsu.find(i) for i in range(100))
    assert len(roots) == 50


def test_cluster_split_multiple():
    records = [
        SequenceRecord(f"s{i}", "ATGC" * 250, {"mobility": i % 3, "amr": i % 2, "expansion": 0})
        for i in range(20)
    ]
    result, sketches, clusters = cluster_split(records, seed=42)
    total = sum(len(v) for v in result.values())
    assert total == 20
    assert len(result["train"]) > 0


# ── model.py ────────────────────────────────────────────────────────

def test_model_forward_with_struct():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, n_structural_features=5)
    model = Cassiopeia(cfg)
    model.eval()
    f = torch.randn(2, 56, 2728)
    m = torch.ones(2, 56, dtype=torch.bool)
    s = torch.randn(2, 56, 5)
    sc = torch.zeros(2, 56, dtype=torch.long)
    out = model(f, m, struct_features=s, scale_ids=sc)
    assert out["mobility_logits"].shape == (2, 3)


def test_model_forward_frp():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56)
    model = Cassiopeia(cfg)
    model.eval()
    f = torch.randn(2, 56, 2728)
    m = torch.ones(2, 56, dtype=torch.bool)
    frp = f @ model.encoder.frp
    out = model(f, m, frp_features=frp)
    assert out["mobility_logits"].shape == (2, 3)


def test_calibrated_params():
    model = Cassiopeia(CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56))
    assert hasattr(model, "mob_t")
    assert hasattr(model, "amr_t")
    assert hasattr(model, "exp_t")
    assert model.mob_t.item() == 1.0


def test_model_forward_non_hierarchical():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, use_hierarchical=False)
    model = Cassiopeia(cfg)
    model.eval()
    f = torch.randn(2, 56, 2728)
    m = torch.ones(2, 56, dtype=torch.bool)
    out = model(f, m)
    assert out["mobility_logits"].shape == (2, 3)


def test_model_forward_cppe():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, use_cppe=True)
    model = Cassiopeia(cfg)
    model.eval()
    f = torch.randn(2, 56, 2728)
    m = torch.ones(2, 56, dtype=torch.bool)
    out = model(f, m)
    assert out["mobility_logits"].shape == (2, 3)


def test_expansion_multiclass():
    cfg = CassiopeiaConfig(hidden_dim=16, frp_out_dim=16, max_windows=56, expansion_classes=3)
    model = Cassiopeia(cfg)
    model.eval()
    f = torch.randn(2, 56, 2728)
    m = torch.ones(2, 56, dtype=torch.bool)
    out = model(f, m)
    assert out["expansion_logits"].shape == (2, 3)
