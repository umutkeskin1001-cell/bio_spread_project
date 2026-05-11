import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SovereignNet(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.encoder = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.time_proj = nn.Linear(1, hidden_dim)
        self.evidential = nn.Linear(hidden_dim, 2)  # 2 units for binary evidence (alpha_0, alpha_1)
        
    def forward(self, x_gene: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (probability, uncertainty, alpha).
        x_gene: (batch, seq_len)
        t: (batch, 1)
        """
        # x_gene: (batch, seq_len)
        mask = (x_gene != 0).float().unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Embedding and pooling
        embedded = self.embed(x_gene) * mask       # (batch, seq_len, emb_dim)
        
        # Mean pooling over non-padding tokens
        sum_mask = mask.sum(dim=1).clamp_min(1.0)
        gene_repr = embedded.sum(dim=1) / sum_mask  # (batch, emb_dim)
        
        # Encoder
        gene_repr = self.encoder(gene_repr)        # (batch, hidden_dim)
        
        # Time effect (multiplicative fusion)
        time_effect = torch.sigmoid(self.time_proj(t))  # (batch, hidden_dim)
        fused = gene_repr * time_effect
        
        # Evidential head: evidence must be non-negative
        evidence = F.softplus(self.evidential(fused))   # (batch, 2)
        alpha = evidence + 1.0                          # alpha = evidence + 1
        
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha[:, 1:] / S                         # Probability for class 1
        
        # Uncertainty = K / S. For binary K=2.
        uncertainty = 2.0 / S.squeeze(-1)
        
        return prob.squeeze(-1), uncertainty, alpha
