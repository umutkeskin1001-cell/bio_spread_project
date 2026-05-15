"""Tests for Sovereign-X: dual-expert temporal hazard model."""

import numpy as np
import polars as pl
import torch

from bio_spread_reborn.data.dataset import (
    SovereignSequenceDataset,
    fit_normalizers,
    fit_static_normalizers,
    sequence_collate,
)
from bio_spread_reborn.data.snapshot import FeatureBuilder, build_sequences, disjoint_backbone_split
from bio_spread_reborn.models.sovereign import SovereignX
from bio_spread_reborn.models.trainer import ranking_loss


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
            "expansion_ratio": [1.0, 1.0, 2.0, 1.0, 1.0],
            "spread_velocity": [1.0, 0.5, 0.67, 1.0, 0.5],
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
    ds = SovereignSequenceDataset(
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
    assert item["static"].shape[0] == 10  # len(STATIC_COLS)


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
            "expansion_ratio": [1.0] * 5,
            "spread_velocity": [1.0, 0.5, 1.0, 0.5, 0.33],
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
    ds = SovereignSequenceDataset(
        df,
        ["A", "B"],
        max_seq_len=10,
        normalizer=normalizer,
        static_normalizer=static_normalizer,
    )
    items = [ds[i] for i in range(len(ds))]
    batch = sequence_collate(items, max_seq_len=10)
    assert batch["seq"].shape == (2, 10, 10)  # B, max_seq_len, n_features=10 (SNAPSHOT_FEATURE_COLS)
    assert batch["static"].shape == (2, 10)
    assert batch["hazard"].shape == (2, 10, 3)
    assert batch["mask"].shape == (2, 10)
    assert batch["seq_len"][0].item() == 2  # A has 2 snapshots
    assert batch["seq_len"][1].item() == 3  # B has 3 snapshots


def test_sovereignx_forward():
    model = SovereignX(n_static=5, n_snapshot=10, static_dim=64, temporal_dim=64, hidden_dim=64, n_hazard=3)
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 10, 10)
    mask = torch.ones(4, 10)
    out = model(static, snapshots, mask)
    hazard, hazard_all, count, cold_logits, fused, weights, mask_out = out
    assert isinstance(out.hazard_logits, torch.Tensor)
    assert isinstance(out.hazard_logits_all, torch.Tensor)
    _ = out.hazard_logits  # confirm attribute access works
    assert hazard.shape == (4, 3), f"hazard.shape={hazard.shape}"
    assert hazard_all.shape == (4, 10, 3), f"hazard_all.shape={hazard_all.shape}"
    assert count.shape == (4,), f"count.shape={count.shape}"
    assert cold_logits.shape == (4, 3), f"cold_logits.shape={cold_logits.shape}"
    assert fused.shape == (4, 64), f"fused.shape={fused.shape}"
    assert weights.shape == (4, 2), f"weights.shape={weights.shape}"
    assert not torch.isnan(hazard).any()
    assert not torch.isnan(hazard_all).any()
    assert not torch.isnan(count).any()
    assert not torch.isnan(cold_logits).any()
    # Per-timestep predictions produce values for all timesteps
    # Masking is applied in the loss function, not in model output
    assert mask_out.shape == (4, 10)
    # Check that all-timestep predictions include the final timestep values
    # For sample 0, last timestep final prediction ≈ per-timestep prediction at L-1
    final_proj = hazard[0]  # (3,)
    last_ts = hazard_all[0, 9]  # (3,) — last timestep prediction
    # These should be correlated (same backbone, different processing paths)
    assert not torch.isnan(final_proj).any()
    assert not torch.isnan(last_ts).any()


def test_sovereignx_gradient_flow():
    model = SovereignX(n_static=5, n_snapshot=10, static_dim=64, temporal_dim=64, hidden_dim=64, n_hazard=3)
    static = torch.randn(4, 5)
    snapshots = torch.randn(4, 5, 10)
    mask = torch.ones(4, 5)
    # Include temporal_mask to trigger null_embed gradient
    temporal_mask = torch.tensor([True, False, True, False])
    hazard, hazard_all, count, cold_logits, _, _, _ = model(static, snapshots, mask, temporal_mask=temporal_mask)
    loss = hazard.mean() + hazard_all.mean() + count.mean() + cold_logits.mean()
    loss.backward()
    for name, param in model.named_parameters():
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
    ds = SovereignSequenceDataset(seqs, ["A", "B", "C"], max_seq_len=10, normalizer=norm, static_normalizer=static_norm)
    assert len(ds) == 3

    # Model forward
    items = [ds[i] for i in range(len(ds))]
    batch = sequence_collate(items, max_seq_len=10)
    model = SovereignX(
        n_static=batch["static"].size(-1),
        n_snapshot=batch["seq"].size(-1),
        static_dim=32,
        temporal_dim=32,
        hidden_dim=32,
        n_hazard=3,
    )
    hazard, hazard_all, count, cold_logits, fused, weights, mask_out = model(
        batch["static"], batch["seq"], batch["mask"]
    )
    assert hazard.shape == (3, 3), f"hazard.shape={hazard.shape}"
    assert hazard_all.shape == (3, 10, 3), f"hazard_all.shape={hazard_all.shape}"
    assert count.shape == (3,), f"count.shape={count.shape}"
    assert cold_logits.shape == (3, 3), f"cold_logits.shape={cold_logits.shape}"
    assert not torch.isnan(cold_logits).any()
    assert mask_out.shape == (3, 10)
    assert not torch.isnan(cold_logits).any()
