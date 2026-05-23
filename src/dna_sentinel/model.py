from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class KmerTransformerConfig:
    n_kmer_features: int = 5376  # Direct, collision-free indexing
    hidden_dim: int = 56          # The absolute golden sweet spot width!
    n_heads: int = 8             # Highly expressive multi-head attention
    n_layers: int = 2            # Golden peak depth for 28-window plasmid sequences
    ffn_ratio: int = 4
    dropout: float = 0.25
    max_windows: int = 28
    n_scales: int = 3
    task_specific_pooling: bool = True
    scale_isolated_conv: bool = True

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

        # Direct flat linear projections (Preserves full-rank raw genomic resolution!)
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
        self.scale_embed = nn.Embedding(self.config.n_scales, h)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.config.max_windows, h))
        nn.init.normal_(self.pos_embed, std=0.02)
        self.input_norm = nn.LayerNorm(h)
        self.local_conv = nn.Sequential(
            nn.Conv1d(h, h, kernel_size=3, padding=1, groups=h),
            nn.GELU(),
            nn.Conv1d(h, h, kernel_size=1),
            nn.Dropout(self.config.dropout),
        )

        # Pyramidal Scale Context Routing Gate
        self.context_gate = nn.Sequential(
            nn.Linear(h, h),
            nn.GELU()
        )
        self.context_norm = nn.LayerNorm(h)

        # Golden Peak 2-layer encoder
        self.encoder = nn.ModuleList([
            TransformerBlock(h, self.config.n_heads, self.config.ffn_ratio, self.config.dropout)
            for _ in range(self.config.n_layers)
        ])
        self.encoder_norm = nn.LayerNorm(h)

        # Minimal Sequence Attention Scorer
        self.evidence_proj = nn.Linear(h, 1)
        self.mob_evidence_proj = nn.Linear(h, 1)
        self.amr_evidence_proj = nn.Linear(h, 1)
        self.exp_evidence_proj = nn.Linear(h, 1)

        # Minimal clean multi-task projection and heads
        self.mob_proj = nn.Sequential(nn.Linear(h, h), nn.LayerNorm(h), nn.GELU())
        self.amr_proj = nn.Sequential(nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU())
        self.exp_proj = nn.Sequential(nn.Linear(h, h // 2), nn.LayerNorm(h // 2), nn.GELU())

        self.mobility_head = make_mlp(h, h, 3, self.config.dropout)
        self.amr_head = make_mlp(h // 2, h // 2, 1, self.config.dropout)
        self.expansion_head = make_mlp(h // 2, h // 2, 1, self.config.dropout)
        self.recon_head = nn.Linear(h, h)

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

    def _embed_inputs(
        self,
        kmer_features: torch.Tensor,
        spec_features: torch.Tensor,
        window_mask: torch.Tensor,
        scale_ids: torch.Tensor,
    ) -> torch.Tensor:
        B, W, C = kmer_features.shape

        # Zero-parameter input-level reverse-complement averaging.
        kmer_rc = kmer_features.flip(dims=[1])
        if C == 4096:
            kmer_rc = kmer_rc.gather(dim=-1, index=self.rev_comp_map.view(1, 1, 4096).expand(B, W, 4096))
        kmer_features = 0.5 * (kmer_features + kmer_rc)

        spec_rc = spec_features.flip(dims=[1])
        spec_features = 0.5 * (spec_features + spec_rc)

        h_lex = self.lex_proj(kmer_features)
        h_spec = self.spec_proj(spec_features)
        x = self.fusion(torch.cat([h_lex, h_spec], dim=-1)) + self.scale_embed(scale_ids)
        x = self.input_norm(x + self.pos_embed[:, :W])

        mask_f = window_mask.unsqueeze(-1).to(dtype=x.dtype)
        if self.config.scale_isolated_conv and W == 28:
            # Scale-Isolated Convolutions to avoid resolution distortion across scale transitions
            x_conv = torch.zeros_like(x)
            # Scale 0: Windows 0-16
            x0 = x[:, :16]
            m0 = window_mask[:, :16].unsqueeze(-1).to(dtype=x.dtype)
            x_conv[:, :16] = self.local_conv((x0 * m0).transpose(1, 2)).transpose(1, 2) * m0
            # Scale 1: Windows 16-24
            x1 = x[:, 16:24]
            m1 = window_mask[:, 16:24].unsqueeze(-1).to(dtype=x.dtype)
            x_conv[:, 16:24] = self.local_conv((x1 * m1).transpose(1, 2)).transpose(1, 2) * m1
            # Scale 2: Windows 24-28
            x2 = x[:, 24:]
            m2 = window_mask[:, 24:].unsqueeze(-1).to(dtype=x.dtype)
            x_conv[:, 24:] = self.local_conv((x2 * m2).transpose(1, 2)).transpose(1, 2) * m2
        else:
            x_conv = self.local_conv((x * mask_f).transpose(1, 2)).transpose(1, 2)
            
        return x + x_conv * mask_f

    def _route_context(self, x: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        macro_start = min(24, x.shape[1])
        if macro_start == x.shape[1]:
            return x

        x_local = x[:, :macro_start]
        x_macro = x[:, macro_start:]
        m_macro = window_mask[:, macro_start:]

        # Aggregate macro context (representing global backbone features)
        w_macro = m_macro.float().unsqueeze(-1)
        macro_context = (x_macro * w_macro).sum(dim=1) / w_macro.sum(dim=1).clamp_min(1.0)

        # Inject macro scale context into local and intermediate window scales
        local_context = self.context_gate(macro_context).unsqueeze(1)
        x_local = self.context_norm(x_local + torch.sigmoid(local_context) * local_context)

        # Re-concatenate scales back
        return torch.cat([x_local, x_macro], dim=1)

    def _encode(self, x: torch.Tensor, window_mask: torch.Tensor) -> torch.Tensor:
        for layer in self.encoder:
            x = layer(x, window_mask)
        return self.encoder_norm(x)

    def forward(self, kmer_features, spec_features, window_mask, scale_ids):
        # 1. Embedding Projection, Multiscale Fusion, and Local K-mer Conv Prior
        x = self._embed_inputs(kmer_features, spec_features, window_mask, scale_ids)

        # 2. Hierarchical Pyramidal Scale Routing
        x = self._route_context(x, window_mask)

        # 3. Transformer Encoder Pass
        x = self._encode(x, window_mask)

        # 4. Attention-Weighted Pooling
        if self.config.task_specific_pooling:
            ev_mob = self.mob_evidence_proj(x).squeeze(-1).masked_fill(~window_mask, -1e4).softmax(dim=-1)
            ev_amr = self.amr_evidence_proj(x).squeeze(-1).masked_fill(~window_mask, -1e4).softmax(dim=-1)
            ev_exp = self.exp_evidence_proj(x).squeeze(-1).masked_fill(~window_mask, -1e4).softmax(dim=-1)
            
            pooled_mob = (x * ev_mob.unsqueeze(-1)).sum(dim=1)
            pooled_amr = (x * ev_amr.unsqueeze(-1)).sum(dim=1)
            pooled_exp = (x * ev_exp.unsqueeze(-1)).sum(dim=1)
            
            evidence_weights = ev_mob  # Backwards compatible default for weight reporting
            pooled_ret = pooled_mob
        else:
            ev = self.evidence_proj(x).squeeze(-1)  # (B, W)
            evidence_weights = ev.masked_fill(~window_mask, -1e4).softmax(dim=-1)
            pooled = (x * evidence_weights.unsqueeze(-1)).sum(dim=1)
            pooled_mob = pooled
            pooled_amr = pooled
            pooled_exp = pooled
            pooled_ret = pooled

        # 5. Task Projections and Logit Heads
        pooled_mob = self.mob_proj(pooled_mob)
        pooled_amr = self.amr_proj(pooled_amr)
        pooled_exp = self.exp_proj(pooled_exp)

        temp = 1.0 + F.softplus(self.logit_scale)
        return {
            "mobility_logits": (self.mobility_head(pooled_mob) / temp[0]) / self.mobility_calib_t,
            "amr_logits": (self.amr_head(pooled_amr).squeeze(-1) / temp[1]) * self.amr_calib_w + self.amr_calib_b,
            "expansion_logits": (self.expansion_head(pooled_exp).squeeze(-1) / temp[2]) * self.exp_calib_w + self.exp_calib_b,
            "evidence_weights": evidence_weights,
            "pooled": pooled_ret,
        }

    def forward_mwr(
        self,
        kmer_features: torch.Tensor,
        spec_features: torch.Tensor,
        window_mask: torch.Tensor,
        scale_ids: torch.Tensor,
        mask_ratio: float = 0.15,
        mwr_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        x = self._embed_inputs(kmer_features, spec_features, window_mask, scale_ids)
        target = x.detach()

        if mwr_mask is None:
            mwr_mask = (torch.rand(window_mask.shape, device=window_mask.device) < mask_ratio) & window_mask
            active = window_mask.sum(dim=1)
            needs_mask = (mwr_mask.sum(dim=1) == 0) & (active > 0)
            if needs_mask.any():
                first_active = window_mask.float().argmax(dim=1)
                mwr_mask[needs_mask, first_active[needs_mask]] = True
        else:
            mwr_mask = mwr_mask.to(device=window_mask.device, dtype=torch.bool) & window_mask

        x = x.masked_fill(mwr_mask.unsqueeze(-1), 0.0)
        x = self._route_context(x, window_mask)
        x = self._encode(x, window_mask)
        reconstruction = self.recon_head(x)
        if mwr_mask.any():
            loss = F.mse_loss(reconstruction[mwr_mask], target[mwr_mask])
        else:
            loss = reconstruction.sum() * 0.0

        return {
            "mwr_loss": loss,
            "reconstruction": reconstruction,
            "target": target,
            "mwr_mask": mwr_mask,
        }

    def save(self, path) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self.config.to_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=False)
        cfg_dict = state["config"]
        # Backwards compatibility: legacy checkpoints won't have new flags
        if "task_specific_pooling" not in cfg_dict:
            cfg_dict["task_specific_pooling"] = False
        if "scale_isolated_conv" not in cfg_dict:
            cfg_dict["scale_isolated_conv"] = False

        model = cls(KmerTransformerConfig(**cfg_dict))
        missing, unexpected = model.load_state_dict(state["state_dict"], strict=False)
        
        # Clone evidence_proj weights to new task-specific attention heads if loading legacy checkpoint
        if "evidence_proj.weight" in state["state_dict"] and not any(k.startswith(("mob_evidence_proj", "amr_evidence_proj", "exp_evidence_proj")) for k in state["state_dict"]):
            with torch.no_grad():
                model.mob_evidence_proj.weight.copy_(model.evidence_proj.weight)
                model.mob_evidence_proj.bias.copy_(model.evidence_proj.bias)
                model.amr_evidence_proj.weight.copy_(model.evidence_proj.weight)
                model.amr_evidence_proj.bias.copy_(model.evidence_proj.bias)
                model.exp_evidence_proj.weight.copy_(model.evidence_proj.weight)
                model.exp_evidence_proj.bias.copy_(model.evidence_proj.bias)
                
        allowed_missing = ("local_conv.", "recon_head.", "mob_evidence_proj.", "amr_evidence_proj.", "exp_evidence_proj.")
        bad_missing = [key for key in missing if not key.startswith(allowed_missing)]
        if bad_missing:
            raise RuntimeError(f"Checkpoint is incompatible; missing checkpoint keys: {bad_missing}")
        if unexpected:
            raise RuntimeError(f"Checkpoint is incompatible; unexpected checkpoint keys: {unexpected}")
        model.to(device)
        model.eval()
        return model
