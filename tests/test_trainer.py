import torch
import pytest
from bio_spread_reborn.models.evidential import SovereignNet
from bio_spread_reborn.models.trainer import EvidentialTrainer

def test_evidential_loss():
    config = {
        'training': {
            'lr': 0.001,
            'epochs': 10,
            'kl_annealing': 0.1
        }
    }
    model = SovereignNet(vocab_size=10)
    trainer = EvidentialTrainer(model, config)
    
    alpha = torch.tensor([[2.0, 1.0], [1.0, 2.0]], requires_grad=True)
    y = torch.tensor([0, 1])
    
    loss = trainer.evidential_loss(alpha, y, epoch=5)
    assert loss.item() > 0
    loss.backward()
    assert alpha.grad is not None

def test_kl_divergence():
    config = {'training': {'lr': 0.001, 'epochs': 10, 'kl_annealing': 0.1}}
    trainer = EvidentialTrainer(SovereignNet(10), config)
    
    # KL between same distribution should be 0
    alpha = torch.tensor([[2.0, 2.0]])
    beta = torch.tensor([[2.0, 2.0]])
    kl = trainer._kl_dirichlet(alpha, beta)
    assert torch.allclose(kl, torch.tensor(0.0), atol=1e-5)
    
    # KL between different should be > 0
    beta = torch.tensor([[1.0, 1.0]])
    kl = trainer._kl_dirichlet(alpha, beta)
    assert kl > 0
