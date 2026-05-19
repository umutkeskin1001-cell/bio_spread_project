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


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * ffn_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * ffn_ratio, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, key_padding_mask=~mask)
        x = self.norm1(x + attn_out)
        return self.norm2(x + self.ffn(x))


def make_mlp(h: int, mid: int, out: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(h, mid),
        nn.LayerNorm(mid),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mid, out),
    )


class KmerTransformer(nn.Module):
    def __init__(self, config: KmerTransformerConfig | None = None):
        super().__init__()
        self.config = config or KmerTransformerConfig()
        h = self.config.hidden_dim
        self.lex_proj = nn.Sequential(
            nn.Linear(self.config.n_kmer_features, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )
        self.spec_proj = nn.Sequential(nn.Linear(512, h), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.LayerNorm(h),
            nn.GELU(),
        )
        self.scale_embed = nn.Embedding(3, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.config.max_windows, h))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.input_norm = nn.LayerNorm(h)

        self.encoder = nn.ModuleList([
            TransformerBlock(h, self.config.n_heads, self.config.ffn_ratio, self.config.dropout)
            for _ in range(self.config.n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(h)
        self.evidence_scorer = nn.Sequential(nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

        self.cross_gate = nn.Sequential(nn.Linear(h, h), nn.GELU())
        self.local_norm = nn.LayerNorm(h)
        self.macro_norm = nn.LayerNorm(h)

        self.mob_proj = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())
        self.amr_proj = nn.Sequential(nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU())
        self.exp_proj = nn.Sequential(nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU())

        self.mobility_head = make_mlp(h, h, 3, self.config.dropout)
        self.amr_head = make_mlp(h // 2, h // 2, 1, self.config.dropout)
        self.expansion_head = make_mlp(h // 2, h // 2, 1, self.config.dropout)

        self.logit_scale = nn.Parameter(torch.zeros(3))
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.register_buffer("amr_calib_w", torch.tensor(1.0))
        self.register_buffer("amr_calib_b", torch.tensor(0.0))
        self.register_buffer("exp_calib_w", torch.tensor(1.0))
        self.register_buffer("exp_calib_b", torch.tensor(0.0))
        self.register_buffer("mobility_calib_t", torch.tensor(1.0))

    def forward(self, kmer_features, spec_features, window_mask, scale_ids):
        h_lex = self.lex_proj(kmer_features)
        h_spec = self.spec_proj(spec_features)

        gate_x = h_lex * torch.sigmoid(h_spec) + h_spec * torch.sigmoid(h_lex)
        add_x = self.fusion(torch.cat([h_lex, h_spec], dim=-1))
        scale_emb = self.scale_embed(scale_ids)
        raw_x = gate_x + add_x + scale_emb

        x = self.input_norm(raw_x + self.pos_embed)

        for layer in self.encoder:
            x = layer(x, window_mask)
        x = self.encoder_norm(x)
        ev = self.evidence_scorer(x).squeeze(-1)

        x_local, x_macro = x[:, :24], x[:, 24:]
        m_macro = window_mask[:, 24:]
        attn = torch.bmm(x_local, x_macro.transpose(1, 2)) / math.sqrt(x_local.shape[-1])
        attn = attn.masked_fill(~m_macro.unsqueeze(1), -1e4).softmax(dim=-1)
        context = torch.bmm(attn, x_macro)
        x_local = self.local_norm(x_local + torch.sigmoid(self.cross_gate(context)) * context)
        x_macro = self.macro_norm(x_macro)

        pooled_scales = []
        slices = [(0, 16), (16, 24), (24, 28)]
        for i, (start, end) in enumerate(slices):
            sub_x = x_local[:, start:end] if i < 2 else x_macro
            sub_mask = window_mask[:, start:end]
            sub_ev = ev[:, start:end].masked_fill(~sub_mask, -1e4)
            w = sub_ev.softmax(dim=-1).unsqueeze(-1)
            p = (sub_x * w).sum(dim=1) * sub_mask.any(dim=-1, keepdim=True).float()
            pooled_scales.append(p)

        p0, p1, p2 = pooled_scales
        local_features = p0 + p1
        pooled = local_features * torch.sigmoid(p2) + p2 * torch.sigmoid(local_features)

        pooled_mob = self.mob_proj(pooled)
        pooled_amr = self.amr_proj(pooled)
        pooled_exp = self.exp_proj(pooled)

        weights = ev.masked_fill(~window_mask, -1e4).softmax(dim=-1)
        temp = 1.0 + F.softplus(self.logit_scale)
        return {
            "mobility_logits": (self.mobility_head(pooled_mob) / temp[0]) / self.mobility_calib_t,
            "amr_logits": (self.amr_head(pooled_amr).squeeze(-1) / temp[1]) * self.amr_calib_w + self.amr_calib_b,
            "expansion_logits": (self.expansion_head(pooled_exp).squeeze(-1) / temp[2]) * self.exp_calib_w + self.exp_calib_b,
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
        model.to(device)
        model.eval()
        return model
