from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class DnaSentinelConfig:
    vocab_size: int = 5
    channels: int = 64
    layers: int = 4
    dropout: float = 0.10
    window_size: int = 4096
    stride: int = 2048
    max_windows: int = 32
    rc_consensus: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DnaSentinelOutput:
    mobility_logits: torch.Tensor
    amr_logits: torch.Tensor
    expansion_logits: torch.Tensor
    window_scores: torch.Tensor
    evidence_weights: torch.Tensor


class _ConvBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(channels, channels, kernel_size=7, padding=3 * dilation, dilation=dilation, groups=channels)
        self.pointwise = nn.Conv1d(channels, channels * 2, kernel_size=1)
        self.out = nn.Conv1d(channels, channels, kernel_size=1)
        self.norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        z = self.depthwise(x)
        z = self.pointwise(z)
        a, b = z.chunk(2, dim=1)
        z = self.out(F.gelu(a) * torch.sigmoid(b))
        z = self.dropout(z)
        return self.norm(z + residual)


class _WindowEncoder(nn.Module):
    def __init__(self, cfg: DnaSentinelConfig) -> None:
        super().__init__()
        self.embed = nn.Embedding(cfg.vocab_size, cfg.channels, padding_idx=0)
        dilations = [1, 2, 4, 8, 16, 32]
        self.blocks = nn.ModuleList(_ConvBlock(cfg.channels, dilations[i % len(dilations)], cfg.dropout) for i in range(cfg.layers))
        self.proj = nn.Sequential(nn.Linear(cfg.channels, cfg.channels), nn.GELU(), nn.LayerNorm(cfg.channels))

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x = self.embed(tokens).transpose(1, 2)
        for block in self.blocks:
            x = block(x)
        m = mask.unsqueeze(1).clamp(0, 1)
        pooled = (x * m).sum(dim=-1) / m.sum(dim=-1).clamp_min(1.0)
        return self.proj(pooled)


class DnaSentinel(nn.Module):
    def __init__(self, cfg: DnaSentinelConfig | None = None) -> None:
        super().__init__()
        self.cfg = cfg or DnaSentinelConfig()
        self.encoder = _WindowEncoder(self.cfg)
        c = self.cfg.channels
        self.window_score = nn.Sequential(nn.Linear(c, c // 2), nn.GELU(), nn.Linear(c // 2, 1))
        self.shared = nn.Sequential(nn.Linear(c, c), nn.GELU(), nn.Dropout(self.cfg.dropout), nn.LayerNorm(c))
        self.mobility = nn.Linear(c, 3)
        self.amr = nn.Linear(c, 1)
        self.expansion = nn.Linear(c, 1)

    def _forward_once(self, tokens: torch.Tensor, mask: torch.Tensor) -> DnaSentinelOutput:
        b, w, length = tokens.shape
        assert mask.shape == (b, w, length), f"Shape mismatch: tokens {tokens.shape} vs mask {mask.shape}"
        flat_tokens = tokens.reshape(b * w, length)
        flat_mask = mask.reshape(b * w, length)
        z = self.encoder(flat_tokens, flat_mask).reshape(b, w, -1)
        valid_windows = (mask.sum(dim=-1) > 0).float()
        raw_scores = self.window_score(z).squeeze(-1).masked_fill(valid_windows == 0, -1e4)
        weights = F.softmax(raw_scores, dim=1)
        pooled = (z * weights.unsqueeze(-1)).sum(dim=1)
        h = self.shared(pooled)
        return DnaSentinelOutput(
            mobility_logits=self.mobility(h),
            amr_logits=self.amr(h).squeeze(-1),
            expansion_logits=self.expansion(h).squeeze(-1),
            window_scores=raw_scores,
            evidence_weights=weights,
        )

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> DnaSentinelOutput:
        out = self._forward_once(tokens, mask)
        if not self.cfg.rc_consensus:
            return out
        rc_tokens = _reverse_complement_tokens(tokens)
        rc_mask = torch.flip(mask, dims=[-1])
        rc = self._forward_once(rc_tokens, rc_mask)
        return DnaSentinelOutput(
            mobility_logits=0.5 * (out.mobility_logits + rc.mobility_logits),
            amr_logits=0.5 * (out.amr_logits + rc.amr_logits),
            expansion_logits=0.5 * (out.expansion_logits + rc.expansion_logits),
            window_scores=0.5 * (out.window_scores + rc.window_scores),
            evidence_weights=0.5 * (out.evidence_weights + rc.evidence_weights),
        )


_RC_TOKENS_TABLE = None

def _reverse_complement_tokens(tokens: torch.Tensor) -> torch.Tensor:
    global _RC_TOKENS_TABLE
    if _RC_TOKENS_TABLE is None or _RC_TOKENS_TABLE.device != tokens.device or _RC_TOKENS_TABLE.dtype != tokens.dtype:
        _RC_TOKENS_TABLE = torch.tensor([0, 4, 3, 2, 1], device=tokens.device, dtype=tokens.dtype)
    return _RC_TOKENS_TABLE[tokens.flip(dims=[-1])]
