"""KmerTransformer: lightweight Transformer over k-mer hash features."""
import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KmerTransformerConfig:
    n_kmer_features: int = 4096
    hidden_dim: int = 64
    n_heads: int = 4
    n_layers: int = 2
    ffn_ratio: int = 4
    dropout: float = 0.25
    max_windows: int = 28
    n_scales: int = 3

    def to_dict(self) -> dict:
        return asdict(self)

class KmerTransformer(nn.Module):
    def __init__(self, config: KmerTransformerConfig | None = None):
        super().__init__()
        self.config = config or KmerTransformerConfig()
        h = self.config.hidden_dim
        self.register_buffer("random_proj", torch.randn(self.config.n_kmer_features, h) / math.sqrt(h))
        self.pos_embed = nn.Embedding(self.config.max_windows, h)
        self.scale_embed = nn.Embedding(self.config.n_scales, h)
        self.input_norm = nn.LayerNorm(h)
        layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=self.config.n_heads,
            dim_feedforward=h * self.config.ffn_ratio,
            dropout=self.config.dropout,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.n_layers, norm=nn.LayerNorm(h))
        self.evidence_scorer = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        self.mobility_head = nn.Linear(h, 3)
        self.amr_head = nn.Linear(h, 1)
        self.expansion_head = nn.Linear(h, 1)
        self.logit_scale = nn.Parameter(torch.zeros(3))

    def forward(self, kmer_features, window_mask, scale_ids):
        B, W, _ = kmer_features.shape
        x = self.input_norm(kmer_features @ self.random_proj)
        pos = torch.arange(W, device=x.device)
        x = x + self.pos_embed(pos) + self.scale_embed(scale_ids)
        x = self.encoder(x, src_key_padding_mask=~window_mask)
        ev = self.evidence_scorer(x).squeeze(-1).masked_fill(~window_mask, -1e4)
        weights = F.softmax(ev, dim=-1)
        pooled = (x * weights.unsqueeze(-1)).sum(dim=1)
        temp = 1.0 + F.softplus(self.logit_scale)
        return {
            "mobility_logits": self.mobility_head(pooled) / temp[0],
            "amr_logits": self.amr_head(pooled).squeeze(-1) / temp[1],
            "expansion_logits": self.expansion_head(pooled).squeeze(-1) / temp[2],
            "evidence_weights": weights,
            "pooled": pooled,
        }

    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.config.to_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(KmerTransformerConfig(**state["config"]))
        model.load_state_dict(state["state_dict"])
        model.eval()
        return model
