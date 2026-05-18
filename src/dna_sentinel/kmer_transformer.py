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
        self.encoder = nn.TransformerEncoder(layer, num_layers=self.config.n_layers, norm=nn.LayerNorm(h), enable_nested_tensor=False)
        self.evidence_scorer = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))
        self.mobility_head = nn.Linear(h, 3)
        self.amr_head = nn.Linear(h, 1)
        self.expansion_head = nn.Linear(h, 1)
        self.logit_scale = nn.Parameter(torch.zeros(3))
        self.scale_weights = nn.Parameter(torch.zeros(3))

    def forward(self, kmer_features, window_mask, scale_ids):
        B, W, _ = kmer_features.shape
        x = self.input_norm(kmer_features @ self.random_proj)
        pos = torch.arange(W, device=x.device)
        x = x + self.pos_embed(pos) + self.scale_embed(scale_ids)
        x = self.encoder(x, src_key_padding_mask=~window_mask)
        ev = self.evidence_scorer(x).squeeze(-1).masked_fill(~window_mask, -1e4)
        ev0, ev1, ev2 = ev[:, :16], ev[:, 16:24], ev[:, 24:28]
        x0, x1, x2 = x[:, :16], x[:, 16:24], x[:, 24:28]
        m0, m1, m2 = window_mask[:, :16], window_mask[:, 16:24], window_mask[:, 24:28]
        w0 = F.softmax(ev0, dim=-1)
        w1 = F.softmax(ev1, dim=-1)
        w2 = F.softmax(ev2, dim=-1)
        has_0 = m0.any(dim=-1, keepdim=True).float()
        has_1 = m1.any(dim=-1, keepdim=True).float()
        has_2 = m2.any(dim=-1, keepdim=True).float()
        p0 = (x0 * w0.unsqueeze(-1)).sum(dim=1) * has_0
        p1 = (x1 * w1.unsqueeze(-1)).sum(dim=1) * has_1
        p2 = (x2 * w2.unsqueeze(-1)).sum(dim=1) * has_2
        has_scale = torch.cat([has_0, has_1, has_2], dim=-1)
        beta_logits = self.scale_weights.view(1, 3).expand(B, 3).clone()
        beta_logits = beta_logits.masked_fill(~has_scale.bool(), -1e4)
        beta = F.softmax(beta_logits, dim=-1)
        pooled = beta[:, 0:1] * p0 + beta[:, 1:2] * p1 + beta[:, 2:3] * p2
        weights = torch.cat([beta[:, 0:1] * w0, beta[:, 1:2] * w1, beta[:, 2:3] * w2], dim=-1)
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
        if "scale_weights" not in state["state_dict"]:
            state["state_dict"]["scale_weights"] = torch.zeros(3)
        model.load_state_dict(state["state_dict"])
        model.eval()
        return model
