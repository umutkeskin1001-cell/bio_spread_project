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
    glu_expansion: int = 3
    learnable_frp: bool = False
    use_hierarchical: bool = True
    n_scale_layers: int = 2
    use_scale_gate: bool = True
    consistency_alpha: float = 0.1
    focal_loss_gamma: float = 0.0
    per_class_margin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    use_ordinal_mobility: bool = False
    risk_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_yaml(cls, cfg: dict) -> CassiopeiaConfig:
        kwargs = {k: v for k, v in cfg.get("model", cfg).items() if k in {f.name for f in fields(cls)}}
        if "risk_weights" in kwargs and isinstance(kwargs["risk_weights"], list):
            kwargs["risk_weights"] = tuple(kwargs["risk_weights"])
        if "per_class_margin" in kwargs and isinstance(kwargs["per_class_margin"], list):
            kwargs["per_class_margin"] = tuple(kwargs["per_class_margin"])
        return cls(**kwargs)


_FRP_SCALE = math.sqrt(3.0)


@torch.no_grad()
def _make_frp(in_dim: int, out_dim: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(in_dim, out_dim, generator=g)
    return torch.where(r < 1.0 / 6, 1.0, torch.where(r < 2.0 / 6, -1.0, 0.0))


class _STEThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return torch.where(x.abs() > 1.0 / 6, x, torch.zeros_like(x))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = x.new_empty(shape).bernoulli_(keep_prob) / keep_prob
        return x * mask


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
            mf = mask.to(dtype=x.dtype)
            t = t * mf.unsqueeze(1)
        u, v = self.t_w1(t).chunk(2, dim=-1)
        x = x + self.t_w2(self.t_drop(torch.sigmoid(u) * v)).transpose(1, 2)
        u, v = self.c_w1(self.c_norm(x)).chunk(2, dim=-1)
        x = x + self.c_w2(self.c_drop(torch.sigmoid(u) * v))
        return x if mask is None else x * mf.unsqueeze(-1)


class ContextGate(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim * 2, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.to(dtype=x.dtype).unsqueeze(-1)
        ctx = (x * m).sum(1, keepdim=True) / m.sum(1, keepdim=True).clamp_min(1)
        return x * torch.sigmoid(self.gate(torch.cat([x, ctx.expand(-1, x.shape[1], -1)], dim=-1)))


class CircularPositionEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, x: torch.Tensor, mask: torch.Tensor, scale_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, w, _ = x.shape
        dev, dt = x.device, x.dtype
        positions = torch.arange(w, device=dev, dtype=dt).unsqueeze(0).expand(b, -1)
        denom = torch.full((b, 1), max(1, w), device=dev, dtype=dt)
        scale_norm = torch.zeros_like(positions)
        if scale_ids is not None:
            sid = scale_ids.to(dev)
            n_scales = int(sid.max().item()) + 1 if sid.numel() else 1
            for scale in range(n_scales):
                in_scale = (sid == scale).float()
                cnt = in_scale.sum(dim=1, keepdim=True).clamp_min(1)
                rank = (in_scale * positions).cumsum(dim=1) * in_scale
                in_scale_mask = in_scale > 0
                positions = torch.where(in_scale_mask, (rank - in_scale).clamp_min(0), positions)
                denom = torch.where(in_scale_mask, cnt.expand_as(denom), denom)
            phase = 2.0 * math.pi * positions / denom.clamp_min(1.0)
            scale_norm = sid.to(dtype=dt) / float(max(1, n_scales - 1))
        else:
            phase = 2.0 * math.pi * positions / denom.clamp_min(1.0)
        coords = torch.stack([torch.sin(phase), torch.cos(phase), scale_norm], dim=-1)
        return self.proj(coords) * mask.unsqueeze(-1).to(dtype=dt)


class WindowMotifConv(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("window_conv_kernel must be odd")
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
        self.w = nn.Parameter(torch.zeros(3, n_heads)) if n_heads > 1 else None

    def forward(self, x_mob, x_amr, x_exp, mask):
        x = torch.stack([x_mob, x_amr, x_exp], dim=0)  # [3, B, W, D]
        n_h = len(self.heads)
        flat = x.reshape(-1, *x.shape[-2:])  # [3*B, W, D]
        s = torch.stack([h(flat).squeeze(-1) for h in self.heads], dim=0)
        s = s.reshape(n_h, 3, *mask.shape)  # [n_h, 3, B, W]
        s = s.masked_fill(~mask[None, None], torch.finfo(s.dtype).min / 2)
        a = torch.nan_to_num(torch.softmax(s, dim=-1))
        if n_h > 1:
            w = torch.softmax(self.w, dim=1)  # [3, n_h]
            ctx = torch.einsum('ht,htbw,tbwd->tbd', w, a, x)
        else:
            ctx = torch.einsum('tbw,tbwd->tbd', a[0], x)
        return (ctx[0], ctx[1], ctx[2]), {
            "mobility_evidence": s[0, 0], "amr_evidence": s[0, 1], "expansion_evidence": s[0, 2],
        }


class ScaleGate(nn.Module):
    def __init__(self, hidden_dim: int, n_scales: int = 3):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 4), nn.GELU(), nn.Linear(hidden_dim // 4, 1))
            for _ in range(n_scales)
        ])

    def forward(self, x: torch.Tensor, scale_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        n_scales = len(self.gates)
        for s in range(n_scales):
            s_mask = (scale_ids == s).unsqueeze(-1).to(dtype=x.dtype)
            gate = torch.sigmoid(self.gates[s](x))
            x = x * (1.0 + (2.0 * gate - 1.0) * s_mask)
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)


class CassiopeiaEncoder(nn.Module):
    def __init__(self, cfg: CassiopeiaConfig):
        super().__init__()
        h = cfg.hidden_dim
        self.frp_scale = _FRP_SCALE
        frp = _make_frp(cfg.n_canonical_features, cfg.frp_out_dim).float()
        if cfg.learnable_frp:
            self.frp_raw = nn.Parameter(frp)
        else:
            self.register_buffer("frp", frp)
        self._learnable_frp = cfg.learnable_frp

        self.lora_rank = cfg.lora_rank
        if self.lora_rank > 0:
            g = torch.Generator().manual_seed(42)
            self.frp_lora_down = nn.Parameter(
                torch.rand(cfg.n_canonical_features, self.lora_rank, generator=g).sub_(0.5)
            )
            self.frp_lora_up = nn.Parameter(torch.zeros(self.lora_rank, cfg.frp_out_dim))

        self.scale_embed = nn.Embedding(3, h) if cfg.use_scale_embedding else None
        self.cppe = CircularPositionEncoding(h) if cfg.use_cppe else None
        if not cfg.use_hierarchical and cfg.use_window_conv:
            self.window_conv = WindowMotifConv(h, cfg.window_conv_kernel, cfg.dropout)
        else:
            self.window_conv = None

        self.has_struct = cfg.n_structural_features > 0
        self._aux_lw = cfg.aux_loss_weight
        self._nl = cfg.n_layers
        if self.has_struct:
            self.struct_proj = nn.Sequential(
                nn.LayerNorm(cfg.n_structural_features), nn.Linear(cfg.n_structural_features, 8), nn.GELU()
            )
        self.bottleneck = nn.Sequential(nn.LayerNorm(cfg.frp_out_dim), nn.Linear(cfg.frp_out_dim, h), nn.GELU())
        self.struct_fuse = nn.Linear(h + 8, h) if self.has_struct else None

        self._hierarchical = cfg.use_hierarchical
        self._n_scale_layers = cfg.n_scale_layers
        self._n_scales = 3

        if self._hierarchical:
            self.scale_gate = ScaleGate(h, self._n_scales) if cfg.use_scale_gate else None
            self.scale_mixers = nn.ModuleList([
                GLUMixer(cfg.max_windows, h, expansion=cfg.glu_expansion, dropout=cfg.dropout)
                for _ in range(self._n_scale_layers)
            ])
            self.scale_dp_paths = nn.ModuleList([
                DropPath(cfg.drop_path_rate) for _ in range(self._n_scale_layers)
            ])
            n_global = max(0, cfg.n_layers - self._n_scale_layers)
            self.cross_scale_attn = nn.MultiheadAttention(h, num_heads=4, batch_first=True, dropout=cfg.dropout)
            self.cross_scale_norm = nn.LayerNorm(h)
            self.global_mixers = nn.ModuleList([
                GLUMixer(cfg.max_windows, h, expansion=cfg.glu_expansion, dropout=cfg.dropout)
                for _ in range(n_global)
            ])
            self.global_dp_paths = nn.ModuleList([
                DropPath(cfg.drop_path_rate) for _ in range(n_global)
            ])
        else:
            self.scale_gate = None
            self.context_gate = ContextGate(h)
            self.mixers = nn.ModuleList(
                [
                    GLUMixer(cfg.max_windows, h, expansion=cfg.glu_expansion, dropout=cfg.dropout)
                    for _ in range(cfg.n_layers)
                ]
            )
            self.dp_paths = nn.ModuleList([DropPath(cfg.drop_path_rate) for _ in range(cfg.n_layers)])

    def _apply_frp(self, kmer_features, frp_features=None):
        if frp_features is not None:
            x = self.frp_scale * frp_features
        elif self._learnable_frp:
            frp_eff = _STEThreshold.apply(self.frp_raw)
            x = self.frp_scale * (kmer_features @ frp_eff)
        else:
            x = self.frp_scale * (kmer_features @ self.frp)
        if self.lora_rank > 0:
            x = x + (kmer_features @ self.frp_lora_down) @ self.frp_lora_up
        return x

    def forward(self, kmer_features, mask, struct_features=None, scale_ids=None, frp_features=None):
        mf = mask.unsqueeze(-1)
        x = self._apply_frp(kmer_features, frp_features)
        x = self.bottleneck(x)
        if self.has_struct and struct_features is not None:
            x = F.gelu(self.struct_fuse(torch.cat([x, self.struct_proj(struct_features)], dim=-1)))
        if self.scale_embed is not None and scale_ids is not None:
            x = x + self.scale_embed(scale_ids)
        if self.cppe is not None:
            x = x + self.cppe(x, mask, scale_ids)
        x = x * mf

        if self._hierarchical:
            if self.scale_gate is not None and scale_ids is not None:
                x = self.scale_gate(x, scale_ids, mask)
            if scale_ids is None:
                scale_ids = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.long)
            x_orig = x
            for mixer, dp in zip(self.scale_mixers, self.scale_dp_paths):
                x_acc = torch.zeros_like(x)
                for s in range(self._n_scales):
                    s_mask = (scale_ids == s) & mask
                    x_s = x_orig * s_mask.unsqueeze(-1).to(dtype=x_orig.dtype)
                    residual = mixer(x_s, s_mask)
                    x_s = x_s + dp(residual)
                    x_acc = x_acc + x_s
                x_orig = x_acc
            x = x_orig
            sp_list = []
            for s in range(self._n_scales):
                s_mask = (scale_ids == s) & mask
                s_pool = (x * s_mask.unsqueeze(-1).to(dtype=x.dtype)).sum(dim=1)
                s_cnt = s_mask.sum(dim=1, keepdim=True).float().clamp_min(1)
                sp_list.append(s_pool / s_cnt)
            sp = torch.stack(sp_list, dim=1)
            x_fused, _ = self.cross_scale_attn(self.cross_scale_norm(x), sp, sp)
            x = x + x_fused
            x = x * mf
            aux = []
            for mixer, dp in zip(self.global_mixers, self.global_dp_paths):
                residual = mixer(x, mask)
                x = x + dp(residual)
                aux.append(x)
        else:
            if self.window_conv is not None:
                x = self.window_conv(x, mask)
            x = self.context_gate(x, mask)
            aux = []
            for mixer, dp in zip(self.mixers, self.dp_paths):
                residual = mixer(x, mask)
                x = x + dp(residual)
                aux.append(x)
        return x, aux if self.training and self._aux_lw > 0 and self._nl > 1 else []


class Cassiopeia(nn.Module):
    def __init__(self, config: CassiopeiaConfig | dict | None = None):
        super().__init__()
        if isinstance(config, dict):
            config = CassiopeiaConfig(
                **{k: v for k, v in config.items() if k in {f.name for f in fields(CassiopeiaConfig)}}
            )
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

        self.exp_proxy = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, 1))
        self.exp_gate_mob = nn.Sequential(nn.Linear(h, h // 4), nn.GELU(), nn.Linear(h // 4, 1))
        self.exp_gate_amr = nn.Sequential(nn.Linear(h, h // 4), nn.GELU(), nn.Linear(h // 4, 1))

        self.aux_mob = nn.Linear(h, 3) if cfg.n_layers > 1 and cfg.aux_loss_weight > 0 else None
        self.aux_amr = nn.Linear(h, 1) if cfg.n_layers > 1 and cfg.aux_loss_weight > 0 else None

        ao = max(1, cfg.amr_classes)
        eo = max(1, cfg.expansion_classes)
        self.mob_head = nn.Linear(h, 3)
        self.amr_head = nn.Linear(h, ao)
        self.exp_head = nn.Linear(h * 3, eo)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.register_buffer("mob_t", torch.tensor(1.0))
        self.register_buffer("amr_w", torch.full((ao,), 1.0))
        self.register_buffer("amr_b", torch.full((ao,), 0.0))
        self.register_buffer("exp_w", torch.full((eo,), 1.0))
        self.register_buffer("exp_b", torch.full((eo,), 0.0))
        self.register_buffer("exp_t", torch.tensor(1.0))

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

        gate_mob = torch.sigmoid(self.exp_gate_mob(exp_ctx))
        gate_amr = torch.sigmoid(self.exp_gate_amr(exp_ctx))
        exp_in = torch.cat([exp_ctx, gate_mob * mob_ctx.detach(), gate_amr * amr_ctx.detach()], dim=-1)
        exp_logits = self.exp_head(exp_in) * self.exp_w + self.exp_b
        if self.config.expansion_classes == 1:
            exp_logits = exp_logits.squeeze(-1) / self.exp_t.clamp(0.1)

        result = {"mobility_logits": mob_logits, "amr_logits": amr_logits, "expansion_logits": exp_logits, **evidence}

        if self.training and self.config.consistency_alpha > 0:
            exp_proxy_in = torch.cat([mob_ctx.detach(), amr_ctx.detach()], dim=-1)
            exp_proxy_logits = self.exp_proxy(exp_proxy_in).squeeze(-1)
            result["exp_proxy_logits"] = exp_proxy_logits

        if self.training and aux_features:
            m = mask.to(dtype=x.dtype).unsqueeze(-1)
            if self.aux_mob is not None:
                ap = (aux_features[0] * m).sum(1) / m.sum(1).clamp_min(1)
                result["aux_mob_logits"] = self.aux_mob(ap)
            if self.aux_amr is not None and len(aux_features) > 1:
                ap = (aux_features[-1] * m).sum(1) / m.sum(1).clamp_min(1)
                result["aux_amr_logits"] = self.aux_amr(ap).squeeze(-1)

        return result

    def forward(self, kmer_features, mask, struct_features=None, scale_ids=None, frp_features=None):
        x, aux_features = self.encoder(kmer_features, mask, struct_features, scale_ids, frp_features=frp_features)
        return self.forward_from_encoder(x, mask, aux_features)

    def compute_loss(
        self, mob_logits, amr_logits, exp_logits,
        mob_target, amr_target, exp_target,
        amr_pw=None, exp_pw=None, exp_pw_mc=None, gamma=0.0, **kw,
    ):
        aux = 0.0
        if self.config.use_ordinal_mobility:
            lm = _ordinal_ce(mob_logits, mob_target, label_smoothing=self.config.label_smoothing)
        elif self.config.focal_loss_gamma > 0:
            lm = _focal_ce(mob_logits, mob_target, self.config.focal_loss_gamma, self.config.label_smoothing)
        else:
            lm = F.cross_entropy(mob_logits, mob_target, label_smoothing=self.config.label_smoothing)
        la = _focal_bce(amr_logits, amr_target, amr_pw, gamma)
        if self.config.expansion_classes == 1:
            le = _focal_bce(exp_logits, exp_target, exp_pw, gamma)
        else:
            le = F.cross_entropy(exp_logits, exp_target, weight=exp_pw_mc, label_smoothing=self.config.label_smoothing)
        aux = 0.0
        if self.training and self.config.aux_loss_weight > 0:
            if kw.get("aux_mob_logits") is not None:
                aux += (_ordinal_ce(kw["aux_mob_logits"], mob_target, label_smoothing=0.0)
                        if self.config.use_ordinal_mobility
                        else F.cross_entropy(kw["aux_mob_logits"], mob_target))
            if kw.get("aux_amr_logits") is not None:
                aux += F.binary_cross_entropy_with_logits(kw["aux_amr_logits"], amr_target)
            aux *= self.config.aux_loss_weight
        lv = self.log_vars.clamp(-3, 3)
        loss = (0.5 * torch.exp(-lv) * torch.stack([lm, la, le]) + 0.5 * lv).sum() + aux

        if self.training and self.config.consistency_alpha > 0 and "exp_proxy_logits" in kw:
            le_consistency = F.mse_loss(
                torch.sigmoid(kw["exp_proxy_logits"]), torch.sigmoid(exp_logits.detach())
            )
            loss = loss + self.config.consistency_alpha * le_consistency

        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss detected: mob={lm.item():.4f} amr={la.item():.4f} exp={le.item():.4f}")
        return {"total": loss, "mob": lm, "amr": la, "exp": le}

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "config": self.config.to_dict()}, path)

    @torch.no_grad()
    def decouple_expansion(self):
        self.exp_w.data.fill_(1.0)
        self.exp_b.data.fill_(0.0)
        self.exp_t.data.fill_(1.0)

    @classmethod
    @torch.no_grad()
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=True)
        cfg = CassiopeiaConfig(
            **{k: v for k, v in state["config"].items() if k in {f.name for f in fields(CassiopeiaConfig)}}
        )
        model = cls(cfg)
        model.load_state_dict(state["state_dict"], strict=False)
        return model.to(device).eval()


def _focal_bce(logits, target, pw, gamma):
    if gamma <= 0:
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="mean")
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    return ((1 - torch.exp(-loss)) ** gamma * loss).mean()


def _focal_ce(logits, target, gamma, label_smoothing=0.0):
    """Multi-class focal loss with label smoothing."""
    n_classes = logits.shape[-1]
    log_probs = F.log_softmax(logits, dim=-1)
    if label_smoothing > 0:
        with torch.no_grad():
            smooth = torch.full_like(log_probs, label_smoothing / (n_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - label_smoothing)
        ce = -(smooth * log_probs).sum(dim=-1)
    else:
        ce = F.nll_loss(log_probs, target, reduction="none")
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


def _ordinal_ce(logits, target, label_smoothing=0.0):
    """Distance-weighted cross-entropy for ordered 3-class mobility.

    Non-mob ↔ Mobilizable: 1x penalty. Non-mob ↔ Conjugative: 2x penalty.
    Uses soft probability weighting — fully differentiable.
    """
    probs = F.softmax(logits, dim=-1)
    cost = torch.tensor([[0.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]],
                        dtype=probs.dtype, device=probs.device)
    per_sample_cost = (probs * cost[target]).sum(dim=-1)
    if label_smoothing > 0:
        n = probs.shape[-1]
        smooth = label_smoothing / (n - 1)
        reg = smooth * cost[target].sum(dim=-1)
        per_sample_cost = (1.0 - label_smoothing) * per_sample_cost + reg
    return per_sample_cost.mean()
