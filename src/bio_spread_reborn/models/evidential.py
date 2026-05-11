import torch
import torch.nn as nn
from typing import Tuple, Optional
from bio_spread_reborn.models.components import GeneEncoder, TimeGate, EvidentialHead, TemporalContextEncoder

class FusionNet(nn.Module):
    def __init__(self, 
                 vocab_size: int, 
                 func_dim: int,
                 history_dim: int,
                 emb_dim: int = 128, 
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 1, # Lighter transformer as requested
                 time_freqs: int = 12):
        super().__init__()
        
        # 1. Raw Gene Stream (Transformer) - Lighter
        self.gene_encoder = GeneEncoder(
            vocab_size=vocab_size,
            emb_dim=emb_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            num_layers=num_layers
        )
        
        # 2. Functional Gene Stream (MLP)
        self.func_encoder = nn.Sequential(
            nn.Linear(func_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64)
        )
        
        # 3. Temporal History Stream (Backcast features)
        self.history_encoder = TemporalContextEncoder(input_dim=history_dim, hidden_dim=64)
        
        # 4. Time Gate
        self.time_gate = TimeGate(hidden_dim=emb_dim + 64 + 64, n_freq=time_freqs)
        
        # 5. Fusion & Final Projection
        total_dim = emb_dim + 64 + 64
        self.fc = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        self.evidential_head = EvidentialHead(hidden_dim)
        
    def forward(self, 
                x_gene: torch.Tensor, 
                x_func: torch.Tensor,
                x_history: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for FusionNet.
        """
        # Encode each stream
        e_gene = self.gene_encoder(x_gene) # (batch, emb_dim)
        e_func = self.func_encoder(x_func) # (batch, 64)
        e_history = self.history_encoder(x_history) # (batch, 64)
        
        # Concatenate
        fused = torch.cat([e_gene, e_func, e_history], dim=-1)
        
        # Temporal gating
        gate = self.time_gate(t)
        fused = fused * gate
        
        # Final layers
        features = self.fc(fused)
        return self.evidential_head(features)
