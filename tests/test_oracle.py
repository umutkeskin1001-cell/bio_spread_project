import torch
import pytest
from bio_spread_project.oracle_core import SovereignOracleNet, evidential_loss

def test_sovereign_oracle_net_dimensions():
    batch_size = 8
    vocab_size = 100
    seq_len = 50
    h_dim = 1024
    d_dim = 64
    
    model = SovereignOracleNet(vocab_size=vocab_size, h_dim=h_dim, d_dim=d_dim)
    
    # Random gene indices [B, seq_len]
    x_gene = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Time features [B, 1]
    t = torch.randn(batch_size, 1)
    
    prob, uncertainty, alpha = model(x_gene, t)
    
    assert prob.shape == (batch_size,), f"Expected prob shape {(batch_size,)}, got {prob.shape}"
    assert uncertainty.shape == (batch_size,), f"Expected uncertainty shape {(batch_size,)}, got {uncertainty.shape}"
    assert alpha.shape == (batch_size, 2), f"Expected alpha shape {(batch_size, 2)}, got {alpha.shape}"
    
    # Probabilities should be between 0 and 1
    assert torch.all(prob >= 0.0) and torch.all(prob <= 1.0)
    
    # Uncertainty should be positive
    assert torch.all(uncertainty >= 0.0)
    
    # Alpha should be >= 1.0
    assert torch.all(alpha >= 1.0)

def test_evidential_loss():
    batch_size = 8
    alpha = torch.rand(batch_size, 2) + 1.0
    y = torch.randint(0, 2, (batch_size,))
    
    loss = evidential_loss(alpha, y)
    
    assert loss.dim() == 0, f"Expected scalar loss, got shape {loss.shape}"
    assert loss.item() >= 0.0, "Loss should be non-negative"
