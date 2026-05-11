import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SovereignOracleNet(nn.Module):
    """
    Sovereign Oracle v17.
    A compact, spectral-normalized Evidential MLP for DNA-based risk detection.
    """
    def __init__(self, vocab_size: int, h_dim: int = 1024, d_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, h_dim, padding_idx=0)
        self.proj = nn.utils.spectral_norm(nn.Linear(h_dim, d_dim))
        self.temporal_proj = nn.Linear(1, d_dim)
        self.layer_norm = nn.LayerNorm(d_dim)
        self.dropout = nn.Dropout(0.3)
        self.evidential_head = nn.Linear(d_dim, 2)

    def forward(self, x_gene: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Embed and aggregate genes: [B, G] -> [B, h_dim]
        mask = (x_gene != 0).float().unsqueeze(-1)
        z_gene = (self.embedding(x_gene) * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        
        # Project and modulate with time
        z = self.proj(z_gene)
        z = z * torch.sigmoid(self.temporal_proj(t))
        
        # Normalize and predict
        z = self.dropout(self.layer_norm(z))
        evidence = F.softplus(self.evidential_head(z))
        alpha = evidence + 1.0
        prob = alpha[:, 1] / alpha.sum(dim=1)
        return prob, alpha

def dirichlet_kl(alpha: torch.Tensor) -> torch.Tensor:
    """KL(Dir(alpha) || Dir([1,1])) - The standard EDL regularizer."""
    K = alpha.shape[1]
    S = alpha.sum(dim=1, keepdim=True)
    kl = (
        torch.lgamma(S) - torch.lgamma(torch.tensor(float(K), device=alpha.device))
        - torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
        + torch.sum((alpha - 1.0) * (torch.digamma(alpha) - torch.digamma(S)), dim=1, keepdim=True)
    )
    return kl.squeeze()

def evidential_loss(alpha: torch.Tensor, y: torch.Tensor, lambda_reg: float = 0.1) -> torch.Tensor:
    """Correct EDL loss combining Log-Likelihood and KL regularization."""
    y_onehot = F.one_hot(y.long(), num_classes=2).float()
    S = alpha.sum(dim=1, keepdim=True)
    
    # Log-likelihood error
    error = torch.sum(y_onehot * (torch.digamma(S) - torch.digamma(alpha)), dim=1)
    
    # Only regularize the non-target evidence (standard practice)
    alpha_reg = y_onehot + (1 - y_onehot) * alpha
    kl = dirichlet_kl(alpha_reg)
    
    return torch.mean(error + lambda_reg * kl)
