import pytest
import torch
import polars as pl
from pathlib import Path
from bio_spread_reborn.data.snapshot import TemporalSnapshotBuilder
from bio_spread_reborn.models.components import TimeGate, GeneEncoder, EvidentialHead
from bio_spread_reborn.models.evidential import SovereignNet

def test_temporal_snapshot_logic():
    # Create synthetic records
    raw_records = pl.DataFrame({
        "backbone_id": ["A", "A", "A", "B", "B"],
        "year": [2018, 2019, 2021, 2018, 2019],
        "country": ["USA", "USA", "UK", "Germany", "Germany"]
    })
    
    # A spreads in 2021 (to UK)
    # B never spreads
    
    builder = TemporalSnapshotBuilder(raw_records, pl.DataFrame(), pl.DataFrame())
    
    # Snapshot for A at 2018: Should look at 2019+ for spread
    # At 2018, past countries = {USA}
    # Future countries = {USA, UK}
    # Spread = True (UK is new)
    assert builder.get_label("A", 2018) == 1
    
    # Snapshot for A at 2019: Should look at 2020+
    # At 2019, past = {USA}
    # Future = {UK}
    assert builder.get_label("A", 2019) == 1
    
    # Snapshot for A at 2021: Should look at 2022+
    # Future = {}
    assert builder.get_label("A", 2021) == 0
    
    # B never spreads
    assert builder.get_label("B", 2018) == 0

def test_time_gate():
    gate = TimeGate(hidden_dim=64, n_freq=12)
    t = torch.tensor([[2018.0], [2021.0]])
    out = gate(t)
    assert out.shape == (2, 64)
    assert torch.all(out >= -1) and torch.all(out <= 1)

def test_gene_encoder():
    vocab_size = 100
    encoder = GeneEncoder(vocab_size=vocab_size, emb_dim=32, num_heads=4, num_layers=2)
    # Batch of 2, seq_len of 10
    x = torch.randint(0, vocab_size, (2, 10))
    x[0, 5:] = 0 # Padding
    
    out = encoder(x)
    assert out.shape == (2, 32)
    assert not torch.isnan(out).any()

def test_sovereign_net_full_pass():
    model = SovereignNet(vocab_size=100, emb_dim=32, hidden_dim=64)
    x = torch.randint(0, 100, (4, 50))
    t = torch.randn(4, 1)
    
    prob, unc, alpha = model(x, t)
    
    assert prob.shape == (4,)
    assert unc.shape == (4,)
    assert alpha.shape == (4, 2)
    assert torch.all(prob >= 0) and torch.all(prob <= 1)
    assert torch.all(unc >= 0)
    assert torch.all(alpha >= 1)
