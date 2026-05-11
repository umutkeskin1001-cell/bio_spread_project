import torch
import torch.nn as nn
from typing import Tuple
from bio_spread_reborn.models.components import GeneEncoder, TimeGate, EvidentialHead

class SovereignNet(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 emb_dim: int = 256, 
                 hidden_dim: int = 512,
                 num_heads: int = 4,
                 num_layers: int = 2,
                 time_freqs: int = 12):
        super().__init__()
        
        self.gene_encoder = GeneEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        self.time_gate = TimeGate(
            hidden_dim=emb_dim, # Fuse with gene representation dimension
            n_freq=time_freqs
        )
        
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.evidential_head = EvidentialHead(hidden_dim)
        
    def forward(self, x_gene: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass returning (probability, uncertainty, alpha).
        x_gene: (batch, seq_len)
        t: (batch, 1)
        """
        # 1. Encode genetic sequence
        gene_repr = self.gene_encoder(x_gene) # (batch, emb_dim)
        
        # 2. Compute temporal gating
        time_effect = self.time_gate(t) # (batch, emb_dim)
        
        # 3. Multiplicative fusion
        fused = gene_repr * time_effect
        
        # 4. Final projection
        features = self.fc(fused) # (batch, hidden_dim)
        
        # 5. Evidential head
        return self.evidential_head(features)
