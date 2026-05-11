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
        self.freq = nn.Parameter(torch.randn(n_freq) * 0.1, requires_grad=True)
        self.linear = nn.Linear(2 * n_freq + 1, hidden_dim)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (batch, 1)
        raw = t
        sin_terms = torch.sin(t * self.freq)
        cos_terms = torch.cos(t * self.freq)
        features = torch.cat([raw, sin_terms, cos_terms], dim=-1)
        return torch.tanh(self.linear(features)) 

class GeneEncoder(nn.Module):
    """
    Transformer-based gene sequence encoder.
    """
    def __init__(self, vocab_size: int, emb_dim: int = 256, hidden_dim: int = 512, num_heads: int = 4, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=emb_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = (x == 0) 
        
        safe_mask = mask.clone()
        all_padding = mask.all(dim=1)
        safe_mask[all_padding, 0] = False 
        
        emb = self.embed(x)
        out = self.transformer(emb, src_key_padding_mask=safe_mask)
        
        bool_mask = (~mask).unsqueeze(-1).float()
        sum_mask = bool_mask.sum(1).clamp_min(1.0)
        pooled = (out * bool_mask).sum(1) / sum_mask
        pooled[all_padding] = 0.0
        return pooled

class TemporalContextEncoder(nn.Module):
    """
    MLP for processing backcast features (epidemiological pressure).
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

class EvidentialHead(nn.Module):
    """
    Evidential output head for binary classification.
    """
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 2)

    def forward(self, x: torch.Tensor):
        evidence = F.softplus(self.linear(x))
        alpha = evidence + 1.0 
        
        S = alpha.sum(dim=1, keepdim=True)
        prob = alpha[:, 1:] / S
        uncertainty = 2.0 / S.squeeze(-1)
        
        return prob.squeeze(-1), uncertainty, alpha
