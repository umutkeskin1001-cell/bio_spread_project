"""KmerTransformer: Genomic Coordinate Gated Bio-Spectral BDSG Transformer."""
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
        self.lex_proj = nn.Sequential(nn.Linear(self.config.n_kmer_features, h), nn.GELU())
        self.spec_proj = nn.Sequential(nn.Linear(512, h), nn.GELU())

        self.coord_proj = nn.Sequential(
            nn.Linear(2, h // 2),
            nn.GELU(),
            nn.Linear(h // 2, h)
        )
        self.input_norm = nn.LayerNorm(h)

        coords = []
        for i in range(16):
            coords.append([float(i * 256 + 256) / 10000.0, math.log(512.0) / 10.0])
        for i in range(8):
            coords.append([float(i * 1024 + 1024) / 10000.0, math.log(2048.0) / 10.0])
        for i in range(4):
            coords.append([float(i * 4096 + 4096) / 10000.0, math.log(8192.0) / 10.0])
        self.register_buffer("window_coords", torch.tensor(coords, dtype=torch.float32))

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

    def forward(self, kmer_features, spec_features, window_mask, scale_ids):
        B, W, _ = kmer_features.shape
        coords = self.coord_proj(self.window_coords).unsqueeze(0)
        h_lex = self.lex_proj(kmer_features) + coords
        h_spec = self.spec_proj(spec_features) + coords
        x = h_lex * torch.sigmoid(h_spec) + h_spec * torch.sigmoid(h_lex)
        x = self.input_norm(x)
        x = self.encoder(x, src_key_padding_mask=~window_mask)
        ev = self.evidence_scorer(x).squeeze(-1).masked_fill(~window_mask, -1e4)

        ev0, ev1, ev2 = ev[:, :16], ev[:, 16:24], ev[:, 24:28]
        x0, x1, x2 = x[:, :16], x[:, 16:24], x[:, 24:28]
        m0, m1, m2 = window_mask[:, :16], window_mask[:, 16:24], window_mask[:, 24:28]

        x_local = torch.cat([x0, x1], dim=1)
        x_macro = x2
        h_dim = x_local.shape[-1]
        attn = torch.bmm(x_local, x_macro.transpose(1, 2)) / math.sqrt(h_dim)
        attn = attn.masked_fill(~m2.unsqueeze(1), -1e4)
        attn = F.softmax(attn, dim=-1)
        x_macro_aligned = torch.bmm(attn, x_macro)
        x_local_fused = x_local * torch.sigmoid(x_macro_aligned)

        w0 = F.softmax(ev0, dim=-1)
        w1 = F.softmax(ev1, dim=-1)
        w2 = F.softmax(ev2, dim=-1)

        has_0 = m0.any(dim=-1, keepdim=True).float()
        has_1 = m1.any(dim=-1, keepdim=True).float()
        has_2 = m2.any(dim=-1, keepdim=True).float()

        p0 = (x_local_fused[:, :16] * w0.unsqueeze(-1)).sum(dim=1) * has_0
        p1 = (x_local_fused[:, 16:24] * w1.unsqueeze(-1)).sum(dim=1) * has_1
        p2 = (x2 * w2.unsqueeze(-1)).sum(dim=1) * has_2

        local_features = p0 + p1
        gated_local = local_features * torch.sigmoid(p2)
        gated_macro = p2 * torch.sigmoid(local_features)
        pooled = gated_local + gated_macro

        weights = torch.cat([w0, w1, w2], dim=-1) / 3.0
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
        model.load_state_dict(state["state_dict"], strict=False)
        model.eval()
        return model
