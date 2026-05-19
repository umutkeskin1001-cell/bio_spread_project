import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KmerTransformerConfig:
    n_kmer_features: int = 5376  # Direct collision-free indexing
    hidden_dim: int = 56          # Scaled to fit comfortably under 450K budget
    n_heads: int = 8             # Highly expressive multi-head attention
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

        # Setup reverse complement index map
        if self.config.n_kmer_features == 4096:
            self.register_buffer("rev_comp_map", self._get_rev_comp_indices())
        else:
            self.register_buffer("rev_comp_map", torch.arange(self.config.n_kmer_features, dtype=torch.long))

        # Direct linear projections
        self.lex_proj = nn.Sequential(
            nn.Linear(self.config.n_kmer_features, h),
            nn.GELU()
        )
        self.spec_proj = nn.Sequential(
            nn.Linear(512, h),
            nn.GELU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(h * 2, h),
            nn.LayerNorm(h)
        )
        self.scale_embed = nn.Embedding(3, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.config.max_windows, h))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.input_norm = nn.LayerNorm(h)

        # Pyramidal Scale Context Routing Gate
        self.context_gate = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU()
        )
        self.context_norm = nn.LayerNorm(h)

        # Unified 2-layer encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(h, self.config.n_heads, self.config.ffn_ratio, self.config.dropout)
            for _ in range(self.config.n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(h)

        # Minimal Sequence Attention Scorer
        self.evidence_proj = nn.Linear(h, 1)

        # Minimal clean multi-task projection and heads
        self.mob_proj = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())
        self.amr_proj = nn.Sequential(nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU())
        self.exp_proj = nn.Sequential(nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU())

        self.mobility_head = make_mlp(h, h, 3, self.config.dropout)
        self.amr_head = make_mlp(h // 2, h // 2, 1, self.config.dropout)
        self.expansion_head = make_mlp(h // 2, h // 2, 1, self.config.dropout)

        self.logit_scale = nn.Parameter(torch.zeros(3))
        self.register_buffer("amr_calib_w", torch.tensor(1.0))
        self.register_buffer("amr_calib_b", torch.tensor(0.0))
        self.register_buffer("exp_calib_w", torch.tensor(1.0))
        self.register_buffer("exp_calib_b", torch.tensor(0.0))
        self.register_buffer("mobility_calib_t", torch.tensor(1.0))

    def _get_rev_comp_indices(self) -> torch.Tensor:
        comp = {0: 3, 1: 2, 2: 1, 3: 0}
        rev_comp_map = torch.zeros(4096, dtype=torch.long)
        for i in range(4096):
            val = i
            digits = []
            for _ in range(6):
                digits.append(val % 4)
                val //= 4
            rc_val = 0
            for d in digits:
                rc_val = rc_val * 4 + comp[d]
            rev_comp_map[i] = rc_val
        return rev_comp_map

    def forward(self, kmer_features, spec_features, window_mask, scale_ids):
        B, W, C = kmer_features.shape

        # 1. Zero-Parameter Input-Level Averaging
        kmer_rc = kmer_features.flip(dims=[1])
        if C == 4096:
            kmer_rc = kmer_rc.gather(dim=-1, index=self.rev_comp_map.view(1, 1, 4096).expand(B, W, 4096))
        kmer_features = 0.5 * (kmer_features + kmer_rc)

        spec_rc = spec_features.flip(dims=[1])
        spec_features = 0.5 * (spec_features + spec_rc)

        # 2. Embedding Projection & Multiscale Fusion
        h_lex = self.lex_proj(kmer_features)
        h_spec = self.spec_proj(spec_features)
        x = self.fusion(torch.cat([h_lex, h_spec], dim=-1)) + self.scale_embed(scale_ids)
        x = self.input_norm(x + self.pos_embed)

        # 3. Hierarchical Pyramidal Scale Routing
        x_local = x[:, :24]
        x_macro = x[:, 24:]
        m_macro = window_mask[:, 24:]

        # Aggregate macro context (representing global backbone features)
        w_macro = m_macro.float().unsqueeze(-1)
        macro_context = (x_macro * w_macro).sum(dim=1) / w_macro.sum(dim=1).clamp_min(1.0)

        # Inject macro scale context into local and intermediate window scales
        local_context = self.context_gate(macro_context).unsqueeze(1)
        x_local = self.context_norm(x_local + torch.sigmoid(local_context) * local_context)

        # Re-concatenate scales back
        x = torch.cat([x_local, x_macro], dim=1)

        # 4. Transformer Encoder Pass
        for layer in self.encoder:
            x = layer(x, window_mask)
        x = self.encoder_norm(x)

        # 5. Minimal Sequence Attention Scorer
        ev = self.evidence_proj(x).squeeze(-1)  # (B, W)
        evidence_weights = ev.masked_fill(~window_mask, -1e4).softmax(dim=-1)

        # 6. Attention-Weighted Pooling
        pooled = (x * evidence_weights.unsqueeze(-1)).sum(dim=1)

        # 7. Task Projections and Logit Heads
        pooled_mob = self.mob_proj(pooled)
        pooled_amr = self.amr_proj(pooled)
        pooled_exp = self.exp_proj(pooled)

        temp = 1.0 + F.softplus(self.logit_scale)
        return {
            "mobility_logits": (self.mobility_head(pooled_mob) / temp[0]) / self.mobility_calib_t,
            "amr_logits": (self.amr_head(pooled_amr).squeeze(-1) / temp[1]) * self.amr_calib_w + self.amr_calib_b,
            "expansion_logits": (self.expansion_head(pooled_exp).squeeze(-1) / temp[2]) * self.exp_calib_w + self.exp_calib_b,
            "evidence_weights": evidence_weights,
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
