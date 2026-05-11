import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class TimeGate(nn.Module):
    """
    Time encoding using Fourier features with learnable frequencies.
    """
    def __init__(self, hidden_dim: int, n_freq: int = 12):
        super().__init__()
        # Fourier feature mapping
        self.freq = nn.Parameter(torch.randn(n_freq) * 0.1, requires_grad=True)
        # Input to linear is raw year + sin terms + cos terms
        self.linear = nn.Linear(2 * n_freq + 1, hidden_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch, 1)
        raw = t
        # (batch, 1) * (n_freq) -> (batch, n_freq)
        sin_terms = torch.sin(t * self.freq)
        cos_terms = torch.cos(t * self.freq)
        features = torch.cat([raw, sin_terms, cos_terms], dim=-1)
        return torch.tanh(self.linear(features)) # Gating in (-1, 1)

class GeneEncoder(nn.Module):
    """
    Transformer-based gene sequence encoder.
    """
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        # Small transformer to capture gene order/interactions
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len) token indices
        mask = (x == 0) # True means padding
        
        # If any row is ALL padding, PyTorch Transformer might fail.
        # We ensure at least one token is "unmasked" for the transformer pass, 
        # but we will zero it out later in pooling.
        safe_mask = mask.clone()
        all_padding = mask.all(dim=1)
        safe_mask[all_padding, 0] = False 
        
        emb = self.embed(x)
        
        # Transformer expects src_key_padding_mask (True = ignore)
        out = self.transformer(emb, src_key_padding_mask=safe_mask)
        
        # Mean pool over non-padding tokens
        # We use the ORIGINAL mask for pooling to ensure all-padding rows stay zero
        bool_mask = (~mask).unsqueeze(-1).float()
        sum_mask = bool_mask.sum(1).clamp_min(1.0)
        pooled = (out * bool_mask).sum(1) / sum_mask
        
        # Zero out rows that were entirely padding
        pooled[all_padding] = 0.0
        
        return pooled

class EvidentialHead(nn.Module):
    """
    Evidential output head for binary classification.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor):
        # Evidence must be non-negative
        evidence = F.softplus(self.linear(x))
        alpha = evidence + 1.0 # alpha = evidence + 1 (Dirichlet parameters)
        
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha[:, 1:] / S
        
        # Uncertainty = K / S. For binary K=2.
        uncertainty = 2.0 / S.squeeze(-1)
        
        return prob.squeeze(-1), uncertainty, alpha
