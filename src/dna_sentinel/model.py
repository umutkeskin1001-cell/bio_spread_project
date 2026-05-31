from __future__ import annotations

import logging
import math
from dataclasses import dataclass, fields
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CassiopeiaConfig:
    n_canonical_features: int = 2728
    n_structural_features: int = 49
    hidden_dim: int = 128
    frp_out_dim: int = 256
    n_layers: int = 2
    lora_rank: int = 8
    n_evidence_heads: int = 1
    drop_path_rate: float = 0.1
    dropout: float = 0.15
    max_windows: int = 28
    expansion_classes: int = 1
    amr_classes: int = 1
    label_smoothing: float = 0.1
    adapter_rank: int = 0
    use_cppe: bool = False
    learnable_frp: bool = False
    use_hierarchical: bool = True
    n_scale_layers: int = 2
    consistency_alpha: float = 0.1
    focal_loss_gamma: float = 0.0
    use_ordinal_mobility: bool = False
    risk_weights: tuple[float, float, float] = (0.4, 0.3, 0.3)
    ring_ssm_kernel: int = 7
    struct_proj_dim: int = 32

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}

    @classmethod
    def from_yaml(cls, cfg: dict) -> CassiopeiaConfig:
        valid = {f.name for f in fields(cls)}
        model_cfg = cfg.get("model", cfg)
        kwargs = {k: v for k, v in model_cfg.items() if k in valid}
        if "risk_weights" in kwargs and isinstance(kwargs["risk_weights"], list):
            kwargs["risk_weights"] = tuple(kwargs["risk_weights"])
        return cls(**kwargs)





@torch.no_grad()
def _make_frp(in_dim: int, out_dim: int, seed: int = 42) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    r = torch.rand(in_dim, out_dim, generator=g)
    return torch.where(r < 1 / 6, 1.0, torch.where(r < 2 / 6, -1.0, 0.0))


class _STEThreshold(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.where(x.abs() > 1 / 6, x, 0.0)

    @staticmethod
    def backward(ctx, g):
        return g * (ctx.saved_tensors[0].abs() > 1 / 6)


class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keep = 1 - self.p
        return x * x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep) / keep


class CircularPositionEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(3, dim)

    def forward(self, x, mask, scale_ids=None):
        b, w, _ = x.shape
        dev, dt = x.device, x.dtype
        pos = torch.arange(w, device=dev, dtype=dt).unsqueeze(0).expand(b, -1)
        denom = torch.full((b, 1), max(1, w), device=dev, dtype=dt)
        sn = torch.zeros_like(pos)
        if scale_ids is not None:
            sid = scale_ids.to(dev)
            ns = int(sid.max().item()) + 1 if sid.numel() else 1
            for s in range(ns):
                m = (sid == s).float()
                cnt = m.sum(1, keepdim=True).clamp_min(1)
                pos = torch.where(m > 0, ((m * pos).cumsum(1) * m - m).clamp_min(0), pos)
                denom = torch.where(m > 0, cnt.expand_as(denom), denom)
            sn = sid.to(dtype=dt) / max(1, ns - 1)
        phase = 2 * math.pi * pos / denom.clamp_min(1)
        return self.proj(torch.stack([phase.sin(), phase.cos(), sn], -1)) * mask.unsqueeze(-1).to(dtype=dt)


class RingSSMBlock(nn.Module):
    def __init__(self, dim, kernel=7, dropout=0.1):
        super().__init__()
        pad = kernel // 2
        self.norm = nn.LayerNorm(dim)
        self.dw_fwd = nn.Conv1d(dim, dim, kernel, padding=0, groups=dim)
        self.dw_bwd = nn.Conv1d(dim, dim, kernel, padding=0, groups=dim)
        self._pad = pad
        self.gate_fwd = nn.Linear(dim, dim)
        self.gate_bwd = nn.Linear(dim, dim)
        self.alpha = nn.Parameter(torch.zeros(dim))
        self.glu = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
        self.out_proj = nn.Linear(dim * 2, dim)
        self.out_drop = nn.Dropout(dropout)

    @staticmethod
    def _ema(x, a):
        B, W, D = x.shape
        a_d = a.view(1, 1, D)
        d = (1 - a_d).clamp(min=1e-4)
        acc_a = a_d.expand(B, W, D).contiguous()
        acc_b = d * x
        stride = 1
        while stride < W:
            n = W - stride
            src_a = torch.ones(B, stride, D, device=x.device, dtype=x.dtype)
            src_a = torch.cat([src_a, acc_a[:, :n]], dim=1)
            src_b = torch.cat([torch.zeros(B, stride, D, device=x.device, dtype=x.dtype),
                               acc_b[:, :n]], dim=1)
            acc_a = src_a * acc_a
            acc_b.mul_(src_a).add_(src_b)
            stride *= 2
        return acc_b

    def forward(self, x, mask):
        B, W, D = x.shape
        mf = mask.unsqueeze(-1).to(dtype=x.dtype)
        xn = self.norm(x * mf)
        a = self.alpha.sigmoid()
        if W >= self._pad * 2 + 1:
            xt = xn.transpose(1, 2)
            xf = self.dw_fwd(F.pad(xt, (self._pad, self._pad), mode="circular")).transpose(1, 2)
            xb = self.dw_bwd(F.pad(xt, (self._pad, self._pad), mode="circular")).transpose(1, 2).flip(1)
        else:
            xf = xb = xn
        g1, g2 = self.gate_fwd(xn).sigmoid(), self.gate_bwd(xn).sigmoid()
        ema_fwd = self._ema(xf, a)
        ema_bwd = self._ema(xb.flip(1), a).flip(1)
        of = g1 * ema_fwd + (1 - g1) * x
        ob = g2 * ema_bwd + (1 - g2) * x
        out = self.out_proj(torch.cat([of, ob], -1))
        return (x + self.out_drop(out + self.glu(out))) * mf


class MultiQueryEvidencePool(nn.Module):
    def __init__(self, dim, n_heads=1):
        super().__init__()
        self.heads = nn.ModuleList([nn.Linear(dim, 1) for _ in range(n_heads)])
        self.w = nn.Parameter(torch.zeros(3, n_heads)) if n_heads > 1 else None

    def forward(self, x_mob, x_amr, x_exp, mask):
        x = torch.stack([x_mob, x_amr, x_exp])
        s = torch.stack([h(x.reshape(-1, *x.shape[-2:])).squeeze(-1) for h in self.heads])
        s = s.reshape(len(self.heads), 3, *mask.shape).masked_fill(~mask[None, None], -65504)
        a = s.softmax(-1).nan_to_num()
        ctx = (torch.einsum('ht,htbw,tbwd->tbd', self.w.softmax(1), a, x) if len(self.heads) > 1
               else torch.einsum('tbw,tbwd->tbd', a[0], x))
        return (ctx[0], ctx[1], ctx[2]), {
            "mobility_evidence": s[0, 0], "amr_evidence": s[0, 1], "expansion_evidence": s[0, 2]}


class ScaleGate(nn.Module):
    def __init__(self, dim, n=3):
        super().__init__()
        self.gates = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, dim // 4), nn.GELU(), nn.Linear(dim // 4, 1))
            for _ in range(n)])

    def forward(self, x, scale_ids, mask):
        for s, gate in enumerate(self.gates):
            m = (scale_ids == s).unsqueeze(-1).to(dtype=x.dtype)
            x = x + m * (2 * gate(x).sigmoid() - 1) * x
        return x * mask.unsqueeze(-1).to(dtype=x.dtype)


class CassiopeiaEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        h = cfg.hidden_dim
        self.frp_scale = math.sqrt(3.0)
        frp = _make_frp(cfg.n_canonical_features, cfg.frp_out_dim).float()
        if cfg.learnable_frp:
            self.frp_raw = nn.Parameter(frp)
        else:
            self.register_buffer("frp", frp)
        self._learnable_frp = cfg.learnable_frp
        self.lora_rank = cfg.lora_rank
        if self.lora_rank:
            g = torch.Generator().manual_seed(42)
            self.frp_lora_down = nn.Parameter(torch.rand(cfg.n_canonical_features, self.lora_rank, generator=g).sub_(0.5))
            self.frp_lora_up = nn.Parameter(torch.zeros(self.lora_rank, cfg.frp_out_dim))
        self.cppe = CircularPositionEncoding(h) if cfg.use_cppe else None
        sp = cfg.struct_proj_dim
        self.has_struct = cfg.n_structural_features > 0
        if self.has_struct:
            self.struct_proj = nn.Sequential(
                nn.LayerNorm(cfg.n_structural_features),
                nn.Linear(cfg.n_structural_features, sp), nn.GELU())
        self.bottleneck = nn.Sequential(
            nn.LayerNorm(cfg.frp_out_dim),
            nn.Linear(cfg.frp_out_dim, h), nn.GELU())
        self.struct_fuse = nn.Linear(h + sp, h) if self.has_struct else None
        self._hier = cfg.use_hierarchical
        self._nsl = cfg.n_scale_layers
        self._ns = 3
        mw = (32, 16, 8)
        if self._hier:
            self.scale_gate = ScaleGate(h, self._ns)
            self.s_mixers = nn.ModuleList([
                RingSSMBlock(h, cfg.ring_ssm_kernel, cfg.dropout)
                for _ in range(self._nsl)])
            self.s_dp = nn.ModuleList([
                DropPath(cfg.drop_path_rate) for _ in range(self._nsl)])
            ng = max(0, cfg.n_layers - self._nsl)
            self.g_mixers = nn.ModuleList([
                RingSSMBlock(h, cfg.ring_ssm_kernel, cfg.dropout)
                for _ in range(ng)]) if ng else None
            self.g_dp = nn.ModuleList([
                DropPath(cfg.drop_path_rate) for _ in range(ng)]) if ng else None
        else:
            self.ctx_gate = nn.Linear(h * 2, h)
            self.mixers = nn.ModuleList([RingSSMBlock(h, cfg.ring_ssm_kernel, cfg.dropout) for _ in range(cfg.n_layers)])
            self.dp = nn.ModuleList([DropPath(cfg.drop_path_rate) for _ in range(cfg.n_layers)])
        self._mw = mw

    def _apply_frp(self, x, frp_features=None):
        if frp_features is not None:
            base = self.frp_scale * frp_features
        elif self._learnable_frp:
            base = self.frp_scale * (x @ _STEThreshold.apply(self.frp_raw))
        else:
            base = self.frp_scale * (x @ self.frp)
        if self.lora_rank:
            base = base + (x @ self.frp_lora_down) @ self.frp_lora_up
        return base

    def forward(self, kmer, mask, struct=None, scale_ids=None, frp_features=None):
        mf = mask.unsqueeze(-1)
        x = self.bottleneck(self._apply_frp(kmer, frp_features))
        if self.has_struct and struct is not None:
            x = F.gelu(self.struct_fuse(torch.cat([x, self.struct_proj(struct)], -1)))
        if self.cppe is not None:
            x = x + self.cppe(x, mask, scale_ids)
        x = x * mf
        if self._hier:
            if scale_ids is None:
                scale_ids = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=torch.long)
            x = self.scale_gate(x, scale_ids.clamp(0, self._ns - 1), mask)
            mw = self._mw
            for mixer, dp in zip(self.s_mixers, self.s_dp):
                parts = [dp(mixer(x[:, sum(mw[:i]):sum(mw[:i+1])], mask[:, sum(mw[:i]):sum(mw[:i+1])])) for i in range(len(mw))]
                x = torch.cat(parts, 1)
            if self.g_mixers:
                for mixer, dp in zip(self.g_mixers, self.g_dp):
                    x = x + dp(mixer(x, mask))
        else:
            ctx = (x * mf).sum(1, keepdim=True) / mf.sum(1, keepdim=True).clamp_min(1)
            x = x * self.ctx_gate(torch.cat([x, ctx.expand(-1, x.shape[1], -1)], -1)).sigmoid()
            for mixer, dp in zip(self.mixers, self.dp):
                x = x + dp(mixer(x, mask))
        return x, []


class Cassiopeia(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        if isinstance(config, dict):
            config = CassiopeiaConfig.from_yaml({"model": config})
        cfg = config or CassiopeiaConfig()
        self.config = cfg
        self.has_struct = cfg.n_structural_features > 0
        h = cfg.hidden_dim
        self.encoder = CassiopeiaEncoder(cfg)
        r = cfg.adapter_rank if cfg.adapter_rank > 0 else max(1, h // 16)
        self.mob_ad = nn.Sequential(nn.Linear(h, r), nn.GELU(), nn.Linear(r, h))
        self.amr_ad = nn.Sequential(nn.Linear(h, r), nn.GELU(), nn.Linear(r, h))
        self.exp_ad = nn.Sequential(nn.Linear(h, r), nn.GELU(), nn.Linear(r, h))
        self.evidence = MultiQueryEvidencePool(h, cfg.n_evidence_heads)
        ao, eo = max(1, cfg.amr_classes), max(1, cfg.expansion_classes)
        self.exp_proxy = nn.Sequential(nn.Linear(h * 2, h), nn.GELU(), nn.Linear(h, eo))
        self.exp_gate_m = nn.Sequential(nn.Linear(h, h // 4), nn.GELU(), nn.Linear(h // 4, 1))
        self.exp_gate_a = nn.Sequential(nn.Linear(h, h // 4), nn.GELU(), nn.Linear(h // 4, 1))
        self.mob_head = nn.Linear(h, 3)
        self.amr_head = nn.Linear(h, ao)
        self.exp_head = nn.Linear(h * 3, eo)
        self.log_vars = nn.Parameter(torch.zeros(3))
        self.register_buffer("mob_t", torch.tensor(1.0))
        self.register_buffer("amr_t", torch.tensor(1.0))
        self.register_buffer("amr_b", torch.tensor(0.0))
        self.register_buffer("exp_t", torch.tensor(1.0))
        self.register_buffer("exp_b", torch.tensor(0.0))

    def forward_from_encoder(self, x, mask):
        xm, xa, xe = x + self.mob_ad(x), x + self.amr_ad(x), x + self.exp_ad(x)
        (mc, ac, ec), ev = self.evidence(xm, xa, xe, mask)
        mob = self.mob_head(mc) / self.mob_t.clamp(0.1)
        amr = self.amr_head(ac) * self.amr_t + self.amr_b
        if self.config.amr_classes == 1:
            amr = amr.squeeze(-1)
        gm, ga = self.exp_gate_m(ec).sigmoid(), self.exp_gate_a(ec).sigmoid()
        exp = self.exp_head(torch.cat([ec, gm * mc.detach(), ga * ac.detach()], -1))
        if self.config.expansion_classes == 1:
            exp = (exp.squeeze(-1) * self.exp_t + self.exp_b)
        r = {"mobility_logits": mob, "amr_logits": amr, "expansion_logits": exp, **ev}
        if self.training and self.config.consistency_alpha > 0:
            r["exp_proxy_logits"] = self.exp_proxy(torch.cat([mc.detach(), ac.detach()], -1))
        return r

    def forward(self, kmer, mask, struct_features=None, scale_ids=None, frp_features=None):
        x, _ = self.encoder(kmer, mask, struct_features, scale_ids, frp_features=frp_features)
        return self.forward_from_encoder(x, mask)

    def compute_loss(self, mob_l, amr_l, exp_l, mob_t, amr_t, exp_t,
                     amr_pw=None, exp_pw=None, exp_pw_mc=None, gamma=0.0, **kw):
        cfg = self.config
        if cfg.use_ordinal_mobility:
            lm = _ordinal_ce(mob_l, mob_t, cfg.label_smoothing)
        elif cfg.focal_loss_gamma > 0:
            lm = _focal_ce(mob_l, mob_t, cfg.focal_loss_gamma, cfg.label_smoothing)
        else:
            lm = F.cross_entropy(mob_l, mob_t, label_smoothing=cfg.label_smoothing)
        la = _focal_bce(amr_l, amr_t, amr_pw, gamma)
        le = (
            _focal_bce(exp_l, exp_t, exp_pw, gamma)
            if cfg.expansion_classes == 1
            else F.cross_entropy(exp_l, exp_t, weight=exp_pw_mc, label_smoothing=cfg.label_smoothing)
        )
        lv = self.log_vars.clamp(0, 3)
        loss = (0.5 * (-lv).exp() * torch.stack([lm, la, le]) + 0.5 * lv).sum()
        if self.training and cfg.consistency_alpha > 0 and "exp_proxy_logits" in kw:
            p = kw["exp_proxy_logits"]
            fn = torch.sigmoid if cfg.expansion_classes == 1 else lambda t: t.softmax(-1)
            pred = fn(p.squeeze(-1) if cfg.expansion_classes == 1 else p)
            loss = loss + cfg.consistency_alpha * F.mse_loss(pred, fn(exp_l.detach()))
        if not torch.isfinite(loss):
            raise RuntimeError(f"NaN/Inf loss: mob={lm.item():.4f} amr={la.item():.4f} exp={le.item():.4f}")
        return {"total": loss, "mob": lm, "amr": la, "exp": le}

    def save(self, path):
        torch.save({"state_dict": self.state_dict(), "config": self.config.to_dict()}, path)

    @classmethod
    @torch.no_grad()
    def load(cls, path, device="cpu"):
        state = torch.load(path, map_location=device, weights_only=True)
        cfg = CassiopeiaConfig(**{k: v for k, v in state["config"].items() if k in {f.name for f in fields(CassiopeiaConfig)}})
        m = cls(cfg)
        r = m.load_state_dict(state["state_dict"], strict=False)
        if r.missing_keys or r.unexpected_keys:
            logger.warning("Load mismatch — missing: %s, unexpected: %s", r.missing_keys, r.unexpected_keys)
        return m.to(device).eval()


def _focal_bce(logits, target, pw, gamma):
    if gamma <= 0:
        return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw)
    loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pw, reduction="none")
    return ((1 - (-loss).exp()) ** gamma * loss).mean()


def _focal_ce(logits, target, gamma, ls=0.0):
    n = logits.shape[-1]
    lp = F.log_softmax(logits, -1)
    if ls > 0 and n > 1:
        with torch.no_grad():
            smooth = torch.full_like(lp, ls / (n - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1 - ls)
        ce = -(smooth * lp).sum(-1)
    else:
        ce = F.nll_loss(lp, target, reduction="none")
    return ((1 - (-ce).exp()) ** gamma * ce).mean()


_ORDINAL_COST = torch.tensor([[0, 1, 2], [1, 0, 1], [2, 1, 0]], dtype=torch.float32)


def _ordinal_ce(logits, target, ls=0.0):
    p = F.softmax(logits, -1)
    c = _ORDINAL_COST.to(device=logits.device, dtype=logits.dtype)
    cost_row = c[target]
    pc = (p * cost_row).sum(-1)
    if ls > 0:
        n = p.shape[-1]
        pc = (1 - ls) * pc + (ls / (n - 1)) * cost_row.sum(-1)
    return pc.mean()


@torch.no_grad()
def compress_checkpoint(src, dst, fmt="fp16"):
    state = torch.load(src, map_location="cpu", weights_only=True)
    sd = state["state_dict"]
    if fmt == "fp16":
        for k in sd:
            if sd[k].is_floating_point():
                sd[k] = sd[k].half()
    elif fmt == "int8":
        for k in list(sd):
            if sd[k].is_floating_point() and sd[k].ndim >= 2:
                s = sd[k].abs().max() / 127
                if s == 0:
                    continue
                sd[k] = (sd[k] / s).clamp(-128, 127).round().to(torch.int8)
                sd[f"{k}._scale"] = s
    state["state_dict"] = sd
    state["_compression"] = fmt
    torch.save(state, dst)
    return {"src_bytes": Path(src).stat().st_size, "dst_bytes": Path(dst).stat().st_size, "format": fmt}


def load_compressed(path, device="cpu"):
    state = torch.load(path, map_location=device, weights_only=True)
    sd = state["state_dict"]
    if state.get("_compression") == "int8":
        scales = {k: sd.pop(k) for k in list(sd) if k.endswith("._scale")}
        for k, s in scales.items():
            base = k[:-7]
            if base in sd:
                sd[base] = s.float() * sd.pop(base).float()
    cfg = CassiopeiaConfig(**{k: v for k, v in state["config"].items() if k in {f.name for f in fields(CassiopeiaConfig)}})
    m = Cassiopeia(cfg)
    m.load_state_dict(sd, strict=False)
    return m.to(device).eval()
