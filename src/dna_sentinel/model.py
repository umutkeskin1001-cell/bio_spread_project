from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CassiopeiaConfig:
    n_canonical_features: int = 2728
    n_structural_features: int = 19
    hidden_dim: int = 128
    frp_out_dim: int = 256
    n_layers: int = 2
    lora_rank: int = 8
    n_evidence_heads: int = 1
    drop_path_rate: float = 0.1
    aux_loss_weight: float = 0.3
    dropout: float = 0.15
    max_windows: int = 28
    expansion_classes: int = 1
    amr_classes: int = 1
    label_smoothing: float = 0.1
    adapter_rank: int = 0
    use_scale_embedding: bool = True
    use_cppe: bool = False
    use_window_conv: bool = False
    window_conv_kernel: int = 5

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_yaml(cls, cfg: dict) -> CassiopeiaConfig:
        m = cfg.get("model", cfg)
        known = _config_fields()
        return cls(**{k: v for k, v in m.items() if k in known})


_FRP_SCALE = math.sqrt(3.0)


def _config_fields():
    return {f.name for f in fields(CassiopeiaConfig)}


@torch.no_grad()
def _make_frp(in_dim: int, out_dim: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(in_dim, out_dim, generator=g)
    return torch.where(r < 1.0 / 6, 1.0, torch.where(r < 2.0 / 6, -1.0, 0.0))


class GLUMixer(nn.Module):
    def __init__(self, n_tokens: int, hidden_dim: int, expansion: int = 3, dropout: float = 0.1):
        super().__init__()
        self.t_norm = nn.LayerNorm(hidden_dim)
        self.t_w1 = nn.Linear(n_tokens, n_tokens * 2)
        self.t_w2 = nn.Linear(n_tokens, n_tokens)
        self.t_drop = nn.Dropout(dropout)
        self.c_norm = nn.LayerNorm(hidden_dim)
        self.c_w1 = nn.Linear(hidden_dim, hidden_dim * expansion * 2)
        self.c_w2 = nn.Linear(hidden_dim * expansion, hidden_dim)
        self.c_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        t = self.t_norm(x).transpose(1, 2)
        if mask is not None:
            t = t * mask.unsqueeze(1).float()
        u, v = self.t_w1(t).chunk(2, dim=-1)
        x = x + self.t_w2(self.t_drop(u * v)).transpose(1, 2)
        u, v = self.c_w1(self.c_norm(x)).chunk(2, dim=-1)
        x = x + self.c_w2(self.c_drop(u * v))
        return x if mask is None else x * mask.unsqueeze(-1).to(dtype=x.dtype)


class ContextGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float().unsqueeze(-1)
        ctx = (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
        return x * torch.sigmoid(self.gate(torch.cat([x, ctx.expand(-1, x.shape[1], -1)], dim=-1)))


class CircularPositionEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, scale_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, w, _ = x.shape
        dev, dt = x.device, x.dtype
        pos = torch.arange(w, device=dev, dtype=dt).unsqueeze(0).expand(b, -1)
        denom = torch.full((b, 1), max(1, w), device=dev, dtype=dt)
        scale_norm = torch.zeros_like(pos)
        if scale_ids is not None:
            sid = scale_ids.to(dev)
            n_scales = int(sid.max().item()) + 1 if sid.numel() else 1
            positions = torch.arange(w, device=dev).unsqueeze(0).expand(b, w)
            for scale in range(n_scales):
                in_scale = (sid == scale).float()
                cnt = in_scale.sum(dim=1, keepdim=True).clamp_min(1)
                ranks = (in_scale * positions).cumsum(dim=1) * in_scale
                mask_s = in_scale > 0
                pos = torch.where(mask_s, (ranks - in_scale).clamp_min(0), pos)
                denom = torch.where(mask_s, cnt.expand_as(denom), denom)
            phase = 2.0 * math.pi * pos / denom.clamp_min(1.0)
            scale_norm = sid.to(dtype=dt) / float(max(1, n_scales - 1))
        else:
            phase = 2.0 * math.pi * pos / denom.clamp_min(1.0)
        coords = torch.stack([torch.sin(phase), torch.cos(phase), scale_norm], dim=-1)
        return self.proj(coords) * mask.unsqueeze(-1).to(dtype=dt)


class WindowMotifConv(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "window_conv_kernel must be odd"
        self.norm = nn.LayerNorm(hidden_dim)
        self.depthwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size, padding=kernel_size // 2, groups=hidden_dim)
        self.pointwise = nn.Linear(hidden_dim, hidden_dim * 2)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mf = mask.unsqueeze(-1).to(dtype=x.dtype)
        y = self.norm(x) * mf
        y = self.depthwise(y.transpose(1, 2)).transpose(1, 2)
        gate, val = self.pointwise(y).chunk(2, dim=-1)
        return (x + self.drop(torch.sigmoid(gate) * F.gelu(val))) * mf


class MultiQueryEvidencePool(nn.Module):
    def __init__(self, dim: int, n_heads: int = 1):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(n_heads)])
        self.mob_w = nn.Parameter(torch.zeros(n_heads))
        self.amr_w = nn.Parameter(torch.zeros(n_heads))
        self.exp_w = nn.Parameter(torch.zeros(n_heads))

    def forward(self, x_mob: torch.Tensor, x_amr: torch.Tensor, x_exp: torch.Tensor, mask: torch.Tensor):
        def _pool(x, w_param):
            s = torch.stack([h(x).squeeze(-1) for h in self.heads], dim=0).masked_fill(~mask, -1e9)
            a = torch.softmax(s, dim=-1)
            w = torch.softmax(w_param, dim=0)[:, None, None]
            ctx = (w * a[:, :, :, None] * x).sum(dim=(0, 2))
            return ctx, s[0]

        mob = _pool(x_mob, self.mob_w)
        amr = _pool(x_amr, self.amr_w)
        exp = _pool(x_exp, self.exp_w)
        return (mob[0], amr[0], exp[0]), {
            "mobility_evidence": mob[1], "amr_evidence": amr[1], "expansion_evidence": exp[1],
        }


class CassiopeiaEncoder(nn.Module):
    def __init__(self, cfg: CassiopeiaConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.frp_scale = _FRP_SCALE
        self.register_buffer("frp", _make_frp(cfg.n_canonical_features, cfg.frp_out_dim).float())

        self.lora_rank = cfg.lora_rank
        if self.lora_rank > 0:
            self.frp_lora_down = nn.Parameter(
                torch.empty(cfg.n_canonical_features, self.lora_rank).uniform_(-0.5, 0.5))
            self.frp_lora_up = nn.Parameter(torch.zeros(self.lora_rank, cfg.frp_out_dim))

        self.scale_embed = nn.Embedding(3, h) if cfg.use_scale_embedding else None
        self.cppe = CircularPositionEncoding(h) if cfg.use_cppe else None
        self.window_conv = WindowMotifConv(h, cfg.window_conv_kernel, cfg.dropout) if cfg.use_window_conv else None

        self.has_struct = cfg.n_structural_features > 0
        if self.has_struct:
            self.struct_proj = nn.Sequential(
                nn.LayerNorm(cfg.n_structural_features), nn.Linear(cfg.n_structural_features, 8), nn.GELU())
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(cfg.frp_out_dim), nn.Linear(cfg.frp_out_dim, h), nn.GELU())
        self.struct_fuse = nn.Linear(h + 8, h) if self.has_struct else None
        self.context_gate = ContextGate(h)
        self.mixers = nn.ModuleList(
            [GLUMixer(cfg.max_windows, h, dropout=cfg.dropout) for _ in range(cfg.n_layers)])
        self.drop_path_rate = cfg.drop_path_rate

    def forward(self, kmer_features, mask, struct_features=None, scale_ids=None):
        mf = mask.unsqueeze(-1)
        x = self.frp_scale * (kmer_features @ self.frp)
        if self.lora_rank > 0:
            x = x + (kmer_features @ self.frp_lora_down) @ self.frp_lora_up
        x = self.bottleneck(x)
        if self.has_struct and struct_features is not None:
            x = F.gelu(self.struct_fuse(torch.cat([x, self.struct_proj(struct_features)], dim=-1)))
        if self.scale_embed is not None and scale_ids is not None:
            x = x + self.scale_embed(scale_ids)
        if self.cppe is not None:
            x = x + self.cppe(x, mask, scale_ids)
        x = x * mf
        if self.window_conv is not None:
            x = self.window_conv(x, mask)
        x = self.context_gate(x, mask)
        if not self.training or self.drop_path_rate == 0:
            for mixer in self.mixers:
                x = mixer(x, mask)
            return x, []
        keep_prob = 1.0 - self.drop_path_rate
        aux = []
        for mixer in self.mixers:
            res = mixer(x, mask)
            scale = x.new_empty(x.shape[0], 1, 1).bernoulli_(keep_prob) / keep_prob
            x = x + scale * (res - x)
            aux.append(x)
        return x, aux


class Cassiopeia(nn.Module):
    def __init__(self, config: CassiopeiaConfig | dict | None = None):
        super().__init__()
        if isinstance(config, dict):
            config = CassiopeiaConfig(**{k: v for k, v in config.items()
                                         if k in _config_fields()})
        cfg = config or CassiopeiaConfig()
        self.config = cfg
        self.has_struct = cfg.n_structural_features > 0
        h = cfg.hidden_dim
        self.encoder = CassiopeiaEncoder(cfg)

        r = cfg.adapter_rank if cfg.adapter_rank > 0 else max(1, h // 16)
        self.mob_adapter = nn.Sequential(nn.Linear(h, r), nn.GELU(), nn.Linear(r, h))
        self.amr_adapter = nn.Sequential(nn.Linear(h, r), nn.GELU(), nn.Linear(r, h))
        self.exp_adapter = nn.Sequential(nn.Linear(h, r), nn.GELU(), nn.Linear(r, h))

        self.evidence = MultiQueryEvidencePool(h, cfg.n_evidence_heads)

        self.aux_mob = nn.Linear(h, 3) if cfg.n_layers > 1 and cfg.aux_loss_weight > 0 else None
        self.aux_amr = nn.Linear(h, 1) if cfg.n_layers > 1 and cfg.aux_loss_weight > 0 else None

        ao = max(1, cfg.amr_classes)
        eo = max(1, cfg.expansion_classes)
        self.mob_head = nn.Linear(h, 3)
        self.amr_head = nn.Linear(h, ao)
        self.exp_head = nn.Linear(h + 3 + ao, eo)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.register_buffer("mob_t", torch.tensor(1.0))
        self.register_buffer("amr_w", torch.full((ao,), 1.0))
        self.register_buffer("amr_b", torch.full((ao,), 0.0))
        self.register_buffer("exp_w", torch.full((eo,), 1.0))
        self.register_buffer("exp_b", torch.full((eo,), 0.0))

    def forward_from_encoder(self, x, mask, aux_features=None):
        B = x.shape[0]
        x_mob = x + self.mob_adapter(x)
        x_amr = x + self.amr_adapter(x)
        x_exp = x + self.exp_adapter(x)

        (mob_ctx, amr_ctx, exp_ctx), evidence = self.evidence(x_mob, x_amr, x_exp, mask)

        mob_logits = self.mob_head(mob_ctx) / self.mob_t.clamp(0.1)

        if self.config.amr_classes == 1:
            amr_logits = self.amr_head(amr_ctx).squeeze(-1) * self.amr_w.squeeze() + self.amr_b.squeeze()
        else:
            amr_logits = self.amr_head(amr_ctx) * self.amr_w + self.amr_b

        exp_in = torch.cat([exp_ctx, mob_logits.detach().reshape(B, -1),
                            amr_logits.detach().reshape(B, -1)], dim=-1)
        exp_logits = self.exp_head(exp_in) * self.exp_w + self.exp_b
        if self.config.expansion_classes == 1:
            exp_logits = exp_logits.squeeze(-1)

        result = {"mobility_logits": mob_logits, "amr_logits": amr_logits,
                  "expansion_logits": exp_logits, "mob_evidence": evidence["mobility_evidence"],
                  **evidence}

        if self.training and aux_features:
            m = mask.float().unsqueeze(-1)
            if self.aux_mob is not None:
                ap = (aux_features[0] * m).sum(1) / m.sum(1).clamp_min(1)
                result["aux_mob_logits"] = self.aux_mob(ap)
            if self.aux_amr is not None and len(aux_features) > 1:
                ap = (aux_features[-1] * m).sum(1) / m.sum(1).clamp_min(1)
                result["aux_amr_logits"] = self.aux_amr(ap).squeeze(-1)

        return result

    def forward(self, kmer_features, mask, struct_features=None, scale_ids=None):
        x, aux_features = self.encoder(kmer_features, mask, struct_features, scale_ids)
        return self.forward_from_encoder(x, mask, aux_features)

    def compute_loss(self, mob_logits, amr_logits, exp_logits, mob_target,
                     amr_target, exp_target, amr_pw=None, exp_pw=None, gamma=0.0, **kw):
        lm = F.cross_entropy(mob_logits, mob_target, label_smoothing=self.config.label_smoothing)
        la = _focal_bce(amr_logits, amr_target, amr_pw, gamma)
        le = (_focal_bce(exp_logits, exp_target, exp_pw, gamma)
              if self.config.expansion_classes == 1
              else F.cross_entropy(exp_logits, exp_target, label_smoothing=self.config.label_smoothing))
        lv = self.log_vars.clamp(-6, 6)
        losses = torch.stack([lm, la, le])
        return (0.5 * torch.exp(-lv) * losses + 0.5 * lv).sum()

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "config": self.config.to_dict()}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=True)
        cfg = CassiopeiaConfig(**{k: v for k, v in state["config"].items() if k in _config_fields()})
        model = cls(cfg)
        sd = {k: v for k, v in state["state_dict"].items() if k in model.state_dict()}
        model.load_state_dict(sd, strict=True)
        return model.to(device).eval()


def _focal_bce(logits, target, pw, gamma):
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    return ((1 - torch.exp(-loss)) ** gamma * loss).mean() if gamma > 0 else loss.mean()
