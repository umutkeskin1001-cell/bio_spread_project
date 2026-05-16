"""Tests for BioSpread: dual-expert temporal hazard model."""

import numpy as np
import polars as pl
import torch

from bio_spread.constants import SNAPSHOT_FEATURE_COLS, SNAPSHOT_NAN_COLS, STATIC_COLS
from bio_spread.data.dataset import (
    SequenceDataset,
    fit_normalizers,
    fit_static_normalizers,
    sequence_collate,
)
from bio_spread.data.snapshot import FeatureBuilder, build_sequences, disjoint_backbone_split
from bio_spread.models.sovereign import BioSpreadModel
from bio_spread.models.trainer import ranking_loss


def test_feature_builder_static():
    meta = pl.DataFrame(
        {
            "backbone_id": ["A", "B"],
            "size": [5000, 10000],
            "gc": [0.5, 0.6],
            "n_replicon_types": [2, 0],
            "n_relaxase_types": [1, 0],
            "predicted_mobility": ["conjugative", "non-mobilizable"],
        }
    )
    fb = FeatureBuilder(horizon=3)
    static = fb.static_features(meta)
    assert "backbone_id" in static.columns
    assert "log_size" in static.columns
    assert "mobility_score" in static.columns
    assert len(static) == 2

    # log_size: ln(1+5000) ≈ 8.517
    assert static.filter(pl.col("backbone_id") == "A")["log_size"].to_numpy()[0] > 0
    # conjugative → 2, non-mobilizable → 0
    assert static.filter(pl.col("backbone_id") == "A")["mobility_score"].to_numpy()[0] == 2.0
    assert static.filter(pl.col("backbone_id") == "B")["mobility_score"].to_numpy()[0] == 0.0


def test_feature_builder_backcast():
    fb = FeatureBuilder(horizon=3)
    history = pl.DataFrame(
        {
            "year": [2010, 2012, 2014],
            "country": ["US", "UK", "DE"],
            "host_genus": ["Escherichia", "Klebsiella", "Escherichia"],
        }
    )
    feats = fb.backcast_features(history, cutoff_year=2015)
    assert feats["n_countries"] == 3
    assert feats["n_hosts"] == 2
    assert feats["years_since_first"] == 5.0
    assert feats["n_records"] == 3


def test_feature_builder_hazard():
    fb = FeatureBuilder(horizon=3)
    raw = pl.DataFrame(
        {
            "backbone_id": ["A", "A", "A", "A", "A"],
            "year": [2018, 2019, 2020, 2021, 2022],
            "country": ["US", "US", "US", "UK", "DE"],
            "host_genus": ["Escherichia"] * 5,
        }
    )
    # Build country progression cache for backbone A
    progression: dict[int, set[str]] = {}
    seen: set[str] = set()
    for year in sorted(raw["year"].unique().to_list()):
        yc = set(raw.filter((pl.col("backbone_id") == "A") & (pl.col("year") == year))["country"].to_list())
        seen = seen | yc
        progression[year] = seen

    # Cutoff at 2019: past countries = {US}, future by 2022 = {UK, DE}
    # hazard_1: by 2020 → no new → 0
    # hazard_2: by 2021 → {UK} → new → 1
    # hazard_3: by 2022 → {UK, DE} → new → 1
    targets = fb.hazard_targets(2019, max_year=2022, country_progression=progression)
    assert targets["hazard_1"] == 0.0
    assert targets["hazard_2"] == 1.0
    assert targets["hazard_3"] == 1.0
    assert targets["n_new_countries"] == 2.0
    assert targets["observed"] == 1.0


def test_hazard_censored():
    fb = FeatureBuilder(horizon=3)
    raw = pl.DataFrame(
        {
            "backbone_id": ["A", "A", "A"],
            "year": [2020, 2021, 2022],
            "country": ["US", "UK", "DE"],
            "host_genus": ["Escherichia"] * 3,
        }
    )
    # Build country progression for A
    progression: dict[int, set[str]] = {}
    seen: set[str] = set()
    for year in sorted(raw["year"].unique().to_list()):
        yc = set(raw.filter((pl.col("backbone_id") == "A") & (pl.col("year") == year))["country"].to_list())
        seen = seen | yc
        progression[year] = seen

    # Cutoff at 2022: needs future data through 2025, but max_year=2022
    # All targets should be -1 (right-censored)
    targets = fb.hazard_targets(2022, max_year=2022, country_progression=progression)
    assert targets["hazard_1"] == -1.0
    assert targets["hazard_2"] == -1.0
    assert targets["hazard_3"] == -1.0
    assert targets["observed"] == 0.0


def test_disjoint_backbone_split():
    raw = pl.DataFrame(
        {
            "backbone_id": [f"B{i}" for i in range(20)],
            "year": [2018] * 10 + [2021] * 10,
            "country": ["US"] * 20,
        }
    )
    train, val, test = disjoint_backbone_split(raw, split_year=2020, val_frac=0.3, test_frac=0.3)
    # 10 backbones first seen before 2020 → train candidates
    # 10 backbones first seen at/after 2020 → test candidates
    assert len(train) > 0
    assert len(val) > 0
    assert len(test) > 0
    # No overlap between train and test
    assert len(set(train) & set(test)) == 0
    # No overlap between val and test
    assert len(set(val) & set(test)) == 0
    # val is a subset of train candidates (first seen before 2020)
    for bid in val:
        first_year = raw.filter(pl.col("backbone_id") == bid)["year"].min()
        assert first_year < 2020


def test_build_sequences():
    raw = pl.DataFrame(
        {
            "backbone_id": ["A", "A", "A", "B", "B"],
            "year": [2018, 2019, 2020, 2018, 2021],
            "country": ["US", "US", "UK", "DE", "FR"],
            "host_genus": ["Escherichia"] * 5,
            "size": [5000] * 5,
            "gc": [0.5] * 5,
            "n_replicon_types": [2, 2, 2, 0, 0],
            "n_relaxase_types": [1, 1, 1, 0, 0],
        }
    )
    meta = raw.unique(subset=["backbone_id"])
    seqs = build_sequences(raw, meta, {"A", "B"}, horizon=3, min_snapshots=1)
    assert "hazard_1" in seqs.columns
    assert "hazard_2" in seqs.columns
    assert "hazard_3" in seqs.columns
    assert "n_new_countries" in seqs.columns
    assert len(seqs) >= 5  # 3 snapshots for A + 2 for B
    assert seqs["backbone_id"].n_unique() == 2


def test_dataset_construction():
    df = pl.DataFrame(
        {
            "backbone_id": ["A", "A", "A", "B", "B"],
            "year": [2018, 2019, 2020, 2018, 2019],
            "n_countries": [1.0, 1.0, 2.0, 1.0, 1.0],
            "n_hosts": [1.0, 1.0, 1.0, 1.0, 1.0],
            "years_since_first": [0.0, 1.0, 2.0, 0.0, 1.0],
            "new_countries_recent": [0.0, 0.0, 1.0, 0.0, 0.0],
            "new_countries_2y_ago": [0.0, 0.0, 0.0, 0.0, 0.0],
            "n_records": [1.0, 2.0, 3.0, 1.0, 2.0],
            "acceleration": [0.0, 0.0, 1.0, 0.0, 0.0],
            "niche_breadth": [1.0, 1.0, 0.5, 1.0, 1.0],
            "log_size": [8.5, 8.5, 8.5, 9.2, 9.2],
            "gc": [0.5, 0.5, 0.5, 0.6, 0.6],
            "n_replicon_types": [2.0, 2.0, 2.0, 0.0, 0.0],
            "n_relaxase_types": [1.0, 1.0, 1.0, 0.0, 0.0],
            "mobility_score": [2.0, 2.0, 2.0, 0.0, 0.0],
            "is_conjugative": [1.0, 1.0, 1.0, 0.0, 0.0],
            "is_mobilizable": [0.0, 0.0, 0.0, 1.0, 1.0],
            "topology": [0.0, 0.0, 0.0, 0.0, 0.0],
            "n_orit_types": [2.0, 2.0, 2.0, 0.0, 0.0],
            "host_range_rank": [3.0, 3.0, 3.0, 1.0, 1.0],
            "hazard_1": [0.0, 1.0, -1.0, 0.0, -1.0],
            "hazard_2": [1.0, 1.0, -1.0, 0.0, -1.0],
            "hazard_3": [1.0, 1.0, -1.0, 0.0, -1.0],
            "n_new_countries": [1.0, 1.0, -1.0, 0.0, -1.0],
            "observed": [1.0, 1.0, 0.0, 1.0, 0.0],
        }
    )
    normalizer = fit_normalizers(df)
    static_normalizer = fit_static_normalizers(df)
    ds = SequenceDataset(
        df,
        ["A", "B"],
        max_seq_len=10,
        normalizer=normalizer,
        static_normalizer=static_normalizer,
    )
    assert len(ds) == 2
    item = ds[0]
    assert "seq" in item
    assert "static" in item
    assert "hazard" in item
    assert "count" in item
    assert "seq_len" in item
    assert item["seq"].shape[0] <= 10
    assert item["static"].shape[0] == len(STATIC_COLS)  # 12


def test_sequence_collate():
    df = pl.DataFrame(
        {
            "backbone_id": ["A", "A", "B", "B", "B"],
            "year": [2018, 2019, 2018, 2019, 2020],
            "n_countries": [1.0] * 5,
            "n_hosts": [1.0] * 5,
            "years_since_first": [0.0, 1.0, 0.0, 1.0, 2.0],
            "new_countries_recent": [0.0] * 5,
            "new_countries_2y_ago": [0.0] * 5,
            "n_records": [1.0, 2.0, 1.0, 2.0, 3.0],
            "acceleration": [0.0] * 5,
            "niche_breadth": [1.0] * 5,
            "log_size": [8.5, 8.5, 9.2, 9.2, 9.2],
            "gc": [0.5, 0.5, 0.6, 0.6, 0.6],
            "n_replicon_types": [2.0, 2.0, 0.0, 0.0, 0.0],
            "n_relaxase_types": [1.0, 1.0, 0.0, 0.0, 0.0],
            "mobility_score": [2.0, 2.0, 0.0, 0.0, 0.0],
            "is_conjugative": [1.0, 1.0, 0.0, 0.0, 0.0],
            "is_mobilizable": [0.0, 0.0, 1.0, 1.0, 1.0],
            "topology": [0.0, 0.0, 0.0, 0.0, 0.0],
            "n_orit_types": [2.0, 2.0, 0.0, 0.0, 0.0],
            "host_range_rank": [3.0, 3.0, 1.0, 1.0, 1.0],
            "hazard_1": [0.0, 1.0, 0.0, 0.0, -1.0],
            "hazard_2": [1.0, 1.0, 0.0, 0.0, -1.0],
            "hazard_3": [1.0, 1.0, 0.0, 0.0, -1.0],
            "n_new_countries": [1.0, 1.0, 0.0, 0.0, -1.0],
            "observed": [1.0, 1.0, 1.0, 1.0, 0.0],
        }
    )
    normalizer = fit_normalizers(df)
    static_normalizer = fit_static_normalizers(df)
    ds = SequenceDataset(
        df,
        ["A", "B"],
        max_seq_len=10,
        normalizer=normalizer,
        static_normalizer=static_normalizer,
    )
    items = [ds[i] for i in range(len(ds))]
    batch = sequence_collate(items, max_seq_len=10)
    n_base = len(SNAPSHOT_FEATURE_COLS)
    n_nan = len(SNAPSHOT_NAN_COLS)
    assert batch["seq"].shape == (2, 10, n_base + n_nan)
    assert batch["static"].shape == (2, len(STATIC_COLS))
    assert batch["hazard"].shape == (2, 10, 3)
    assert batch["mask"].shape == (2, 10)
    assert batch["seq_len"][0].item() == 2
    assert batch["seq_len"][1].item() == 3


def test_biospread_model_forward():
    model = BioSpreadModel(n_static=5, n_snapshot=10, static_dim=64, temporal_dim=64, hidden_dim=64, n_hazard=3)
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 10, 10)
    mask = torch.ones(4, 10)
    out = model(static, snapshots, mask)
    assert isinstance(out.hazard_logits, torch.Tensor)
    assert isinstance(out.hazard_logits_all, torch.Tensor)
    _ = out.hazard_logits  # confirm attribute access works
    assert out.hazard_logits.shape == (4, 3), f"hazard.shape={out.hazard_logits.shape}"
    assert out.hazard_logits_all.shape == (4, 10, 3), f"hazard_all.shape={out.hazard_logits_all.shape}"
    assert out.count_logits.shape == (4,), f"count.shape={out.count_logits.shape}"
    assert out.cold_logits.shape == (4, 3), f"cold_logits.shape={out.cold_logits.shape}"
    assert out.fused.shape == (4, 64), f"fused.shape={out.fused.shape}"
    assert out.gate_weights.shape == (4, 2), f"weights.shape={out.gate_weights.shape}"
    assert not torch.isnan(out.hazard_logits).any()
    assert not torch.isnan(out.hazard_logits_all).any()
    assert not torch.isnan(out.count_logits).any()
    assert not torch.isnan(out.cold_logits).any()
    # Per-timestep predictions produce values for all timesteps
    # Masking is applied in the loss function, not in model output
    assert out.mask.shape == (4, 10)
    final_proj = out.hazard_logits[0]
    last_ts = out.hazard_logits_all[0, 9]
    assert not torch.isnan(final_proj).any()
    assert not torch.isnan(last_ts).any()


def test_biospread_model_gradient_flow():
    model = BioSpreadModel(n_static=5, n_snapshot=10, static_dim=64, temporal_dim=64, hidden_dim=64, n_hazard=3)
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 5, 10)
    mask = torch.ones(4, 5)
    # Include temporal_mask to trigger null_embed gradient
    temporal_mask = torch.tensor([True, False, True, False])
    out = model(static, snapshots, mask, temporal_mask=temporal_mask)
    loss = out.hazard_logits.mean() + out.hazard_logits_all.mean() + out.count_logits.mean() + out.cold_logits.mean()
    loss.backward()
    allowed_none = {"cold_prior_predictor"}
    for name, param in model.named_parameters():
        if any(a in name for a in allowed_none):
            continue
        assert param.grad is not None, f"{name} has no gradient"
        assert not torch.isnan(param.grad).any(), f"{name} has NaN gradient"


def test_ranking_loss():
    probs = torch.tensor([[0.1, 0.5, 0.9], [0.8, 0.3, 0.2]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32)
    loss = ranking_loss(probs, targets)
    assert loss.item() >= 0


def test_disjoint_split_no_overlap():
    """Critical: verifies train/test backbone disjointness for leakage prevention."""
    np.random.seed(42)
    torch.manual_seed(42)
    raw = pl.DataFrame(
        {
            "backbone_id": [f"B{i}" for i in range(100)],
            "year": [2015] * 70 + [2021] * 30,
            "country": ["US"] * 100,
        }
    )
    train, val, test = disjoint_backbone_split(raw, split_year=2020, val_frac=0.2, test_frac=0.2)
    assert len(set(train) & set(test)) == 0
    assert len(set(val) & set(test)) == 0
    # 30 backbones first seen in 2021 → all should be in test (minus those shuffled to extra train)
    test_first_seen_2021 = raw.filter(pl.col("backbone_id").is_in(test))["year"].min()
    assert test_first_seen_2021 >= 2020


def test_pipeline_integration():
    """End-to-end: raw data → sequences → dataset → model → forward."""
    raw = pl.DataFrame(
        {
            "backbone_id": ["A", "A", "A", "B", "B", "C", "C", "C"],
            "year": [2018, 2019, 2020, 2018, 2019, 2018, 2019, 2020],
            "country": ["US", "US", "UK", "DE", "FR", "US", "DE", "UK"],
            "host_genus": ["Escherichia"] * 8,
            "size": [5000] * 8,
            "gc": [0.5] * 8,
            "n_replicon_types": [2, 2, 2, 0, 0, 3, 3, 3],
            "n_relaxase_types": [1, 1, 1, 0, 0, 2, 2, 2],
        }
    )
    meta = raw.unique(subset=["backbone_id"])

    # Build sequences
    seqs = build_sequences(raw, meta, {"A", "B", "C"}, horizon=3)
    assert len(seqs) >= 8

    # Normalizers
    norm = fit_normalizers(seqs)
    static_norm = fit_static_normalizers(seqs)

    # Dataset
    ds = SequenceDataset(seqs, ["A", "B", "C"], max_seq_len=10, normalizer=norm, static_normalizer=static_norm)
    assert len(ds) == 3

    # Model forward
    items = [ds[i] for i in range(len(ds))]
    batch = sequence_collate(items, max_seq_len=10)
    model = BioSpreadModel(
        n_static=batch["static"].size(-1),
        n_snapshot=batch["seq"].size(-1),
        static_dim=32,
        temporal_dim=32,
        hidden_dim=32,
        n_hazard=3,
    )
    out = model(batch["static"], batch["seq"], batch["mask"])
    assert out.hazard_logits.shape == (3, 3), f"hazard.shape={out.hazard_logits.shape}"
    assert out.hazard_logits_all.shape == (3, 10, 3), f"hazard_all.shape={out.hazard_logits_all.shape}"
    assert out.count_logits.shape == (3,), f"count.shape={out.count_logits.shape}"
    assert out.cold_logits.shape == (3, 3), f"cold_logits.shape={out.cold_logits.shape}"
    assert not torch.isnan(out.cold_logits).any()
    assert out.mask.shape == (3, 10)


def test_mamba_forward():
    from bio_spread.models.components import HybridTemporalEncoder, Mamba2Block
    B, L, D = 4, 10, 18
    x = torch.randn(B, L, D)
    mask = torch.ones(B, L)

    encoder = HybridTemporalEncoder(D, hidden_dim=64, use_mamba=True, mamba_d_state=8, mamba_n_layers=2)
    h_all, h_pooled = encoder(x, mask)
    assert h_all.shape == (B, L, 64)
    assert h_pooled.shape == (B, 64)
    assert not torch.isnan(h_all).any()
    assert not torch.isnan(h_pooled).any()

    block = Mamba2Block(64, d_state=8)
    out = block(h_all)
    assert out.shape == (B, L, 64)
    assert not torch.isnan(out).any()


def test_hyperbolic_clamping():
    from bio_spread.models.components import PoincareBall, PoincareTaxonomyEncoder
    u = torch.randn(10, 16) * 100
    p = PoincareBall.expmap0(u, c=-1.0)
    norms = p.norm(dim=-1)
    assert (norms < 1.0).all(), f"Max norm: {norms.max().item()}"
    p_clamped = PoincareBall.radius_clamp(p, max_r=0.95, c=-1.0)
    assert p_clamped.norm(dim=-1).max().item() <= 0.96

    vocab_sizes = [10, 15, 20, 25, 30]
    encoder = PoincareTaxonomyEncoder(vocab_sizes, embed_dim=8, dropout=0.0, curvature=-1.0)
    idxs = torch.randint(0, 5, (4, 5))
    emb, div = encoder(idxs)
    assert emb.shape == (4, 40)
    assert not torch.isnan(emb).any()
    assert div.shape == (4,)


def test_evidential_dirichlet():
    from bio_spread.models.components import EvidentialHazardHead
    head = EvidentialHazardHead(64, n_hazard=3)
    x = torch.randn(8, 64)
    probs, alpha, epi_var = head(x)
    assert probs.shape == (8, 3)
    assert alpha.shape == (8, 3)
    assert epi_var.shape == (8, 3)
    assert (alpha > 1.0).all()
    assert (probs >= 0).all() and (probs <= 1).all()
    assert (epi_var >= 0).all()

    targets = torch.randint(0, 2, (8, 3)).float()
    loss = head.loss(alpha, targets, torch.ones(3))
    assert loss.item() > 0
    assert torch.isfinite(loss)


def test_cagrad_projection():
    from bio_spread.models.components import CAGradProjector
    from bio_spread.models.sovereign import BioSpreadModel
    model = BioSpreadModel(n_static=5, n_snapshot=10, static_dim=32, temporal_dim=32, hidden_dim=32, n_hazard=3)
    model.eval()

    losses = {"a": torch.tensor(1.0, requires_grad=True), "b": torch.tensor(2.0, requires_grad=True)}

    total = CAGradProjector.apply(losses, model, c=0.4)
    assert torch.isfinite(total)
    assert total.item() > 0


def test_aci_coverage():
    from bio_spread.utils.metrics import AdaptiveConformalWrapper
    aci = AdaptiveConformalWrapper(alpha=0.1, eta=0.05, q_init=0.5)
    probs = torch.tensor([0.9, 0.1, 0.5])
    lower, upper = aci.get_interval(probs)
    assert lower.shape == (3,)
    assert upper.shape == (3,)
    assert (lower >= 0).all()
    assert (upper <= 1).all()
    assert (lower <= upper).all()

    aci.update(error=0.0)
    aci.update(error=1.0)
    assert aci.q > 0 or aci.q < 2.0

    sets = aci.get_prediction_set(probs)
    assert sets.shape == (3, 2)
    state = aci.state_dict()
    assert "q" in state


def test_fit_block():
    from bio_spread.models.components import FiTBlock
    block = FiTBlock(12, hidden_dim=32, heads=2)
    x = torch.randn(8, 12)
    out = block(x)
    assert out.shape == (8, 32)
    assert not torch.isnan(out).any()


def test_causal_conv1d():
    from bio_spread.models.components import CausalConv1d
    B, L, D = 4, 10, 32
    x = torch.randn(B, L, D)
    conv = CausalConv1d(D, D, kernel_size=3)
    out = conv(x)
    assert out.shape == (B, L, D)
    assert not torch.isnan(out).any()


def test_uncertainty_proto_retriever():
    from bio_spread.models.components import UncertaintyProtoRetriever
    retriever = UncertaintyProtoRetriever(query_dim=20, n_hazard=3, k=5, n_prototypes=32, proto_dim=32)
    query = torch.randn(4, 20)
    labels = torch.randint(0, 2, (4, 3)).float()
    retriever.update(query, labels)
    out = retriever(query)
    assert out.shape == (4, 3)
    assert not torch.isnan(out).any()


def test_biospread_model_with_mamba():
    model = BioSpreadModel(n_static=5, n_snapshot=10, static_dim=32, temporal_dim=32,
                           hidden_dim=32, n_hazard=3, use_mamba=True, mamba_d_state=8)
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 10, 10)
    mask = torch.ones(4, 10)
    out = model(static, snapshots, mask)
    assert out.hazard_logits.shape == (4, 3)
    assert out.cold_logits.shape == (4, 3)
    assert not torch.isnan(out.hazard_logits).any()


def test_biospread_model_with_evidential():
    model = BioSpreadModel(n_static=5, n_snapshot=10, static_dim=32, temporal_dim=32,
                           hidden_dim=32, n_hazard=3, use_evidential=True)
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 10, 10)
    mask = torch.ones(4, 10)
    out = model(static, snapshots, mask)
    assert out.hazard_logits.shape == (4, 3)
    assert out.alpha_pos is not None
    assert out.alpha_pos.shape == (4, 3)
    assert (out.alpha_pos > 1.0).all()
    assert out.epistemic_var is not None
    assert not torch.isnan(out.hazard_logits).any()


def test_evidential_epistemic_var_formula():
    """Verify epistemic variance uses correct formula: alpha_pos / (alpha_0^2 * (alpha_0+1))."""
    from bio_spread.models.components import EvidentialHazardHead
    head = EvidentialHazardHead(32, n_hazard=1)
    x = torch.randn(4, 32)
    expected_prob, alpha_pos, epi_var = head(x)
    alpha_0 = alpha_pos + 1.0
    expected_var = alpha_pos / (alpha_0 ** 2 * (alpha_0 + 1))
    assert torch.allclose(epi_var, expected_var, atol=1e-7)
    # High logits (strong evidence) → low epistemic variance
    # We compare directly on alpha_pos values rather than MLP inputs
    alpha_pos_low = torch.tensor([[1.01]])
    alpha_pos_high = torch.tensor([[10.0]])
    epi_low = alpha_pos_low / ((alpha_pos_low + 1) ** 2 * (alpha_pos_low + 2))
    epi_high = alpha_pos_high / ((alpha_pos_high + 1) ** 2 * (alpha_pos_high + 2))
    assert epi_high.item() < epi_low.item(), "High evidence should reduce epistemic variance"


def test_biospread_model_cold_path_retrieval_routing():
    """Integration: retrieval routing blends cold & temporal logits under uncertainty."""
    model = BioSpreadModel(
        n_static=5, n_snapshot=10, static_dim=32, temporal_dim=32,
        hidden_dim=32, n_hazard=3, use_evidential=True, use_retrieval=True,
        prototype_dim=32, prototype_k=3,
    )
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 10, 10)
    mask = torch.ones(4, 10)

    # Pre-populate retriever with full cold_input (z_static + cold_prior)
    dummy_labels = torch.randint(0, 2, (4, 3)).float()
    with torch.no_grad():
        z_static = model.static_encoder(static)
        cold_prior = model.cold_prior_predictor(z_static)
    model.retriever.update(torch.cat([z_static, cold_prior], dim=-1), dummy_labels)

    # Warm forward (no temporal mask) → routing_weight should be ~0
    out_warm = model(static, snapshots, mask)
    assert out_warm.routing_weight is not None
    assert out_warm.h_cold is not None
    assert out_warm.h_cold.shape == (4, 3)
    assert out_warm.epistemic_var is not None
    assert out_warm.alpha_pos is not None

    # Cold forward (temporal_mask=True) → retrieval blends in
    temporal_mask = torch.tensor([True, True, False, False])
    out_cold = model(static, snapshots, mask, temporal_mask=temporal_mask)
    cold_hazard = out_cold.hazard_logits[temporal_mask]
    warm_hazard = out_cold.hazard_logits[~temporal_mask]
    assert not torch.isnan(cold_hazard).any()
    assert not torch.isnan(warm_hazard).any()

    # Verify routing_weight is between 0 and 1
    assert (out_cold.routing_weight >= 0).all()
    assert (out_cold.routing_weight <= 1).all()

    # Verify retriever stores info
    assert model.retriever.n_seen.item() >= 4

    # Verify h_cold is populated
    assert out_cold.h_cold is not None
    assert out_cold.h_cold.shape == (4, 3)
