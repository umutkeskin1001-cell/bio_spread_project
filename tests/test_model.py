import torch
from bio_spread_reborn.models.evidential import SovereignNet

def test_sovereign_net_shapes():
    vocab_size = 10
    model = SovereignNet(vocab_size=vocab_size, emb_dim=16, hidden_dim=32)
    
    batch_size = 4
    seq_len = 5
    x = torch.randint(0, vocab_size, (batch_size, seq_len))
    t = torch.randn(batch_size, 1)
    
    prob, unc, alpha = model(x, t)
    
    assert prob.shape == (batch_size,)
    assert unc.shape == (batch_size,)
    assert alpha.shape == (batch_size, 2)
    assert torch.all(alpha >= 1.0)
    assert torch.all(prob >= 0.0) and torch.all(prob <= 1.0)
    assert torch.all(unc >= 0.0)

def test_uncertainty_calibration():
    # If alpha is very large, uncertainty should be small
    vocab_size = 10
    model = SovereignNet(vocab_size=vocab_size)
    
    x = torch.zeros((1, 5), dtype=torch.long)
    t = torch.zeros((1, 1))
    
    # Mocking evidential layer to give high evidence
    model.evidential.bias.data.fill_(10.0) 
    _, unc_low, _ = model(x, t)
    
    model.evidential.bias.data.fill_(-10.0)
    _, unc_high, _ = model(x, t)
    
    assert unc_low < unc_high
