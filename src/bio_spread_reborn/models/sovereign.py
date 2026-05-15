"""
Sovereign-X: Dual-expert hazard model with per-timestep training.

Architecture:
    StaticExpert: backbone-level time-invariant features → 128
    TemporalExpert: GRU over snapshot sequences → 192
    ContextGate: context-dependent weighting for final prediction
    TimestepHead: per-timestep hazard predictions (trained on all snapshots)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_spread_reborn.models.components import MLP, ColdStartHead, GatedMLP, TaxonomyEncoder


@dataclass
class ModelOutput:
    """SovereignX forward output container.

    Attributes:
        hazard_logits: (B, n_hazard) gated final prediction.
        hazard_logits_all: (B, L, n_hazard) per-timestep predictions.
        count_logits: (B,) log1p(n_new_countries) prediction.
        cold_logits: (B, n_hazard) static-only prediction.
        fused: (B, static_dim) fused embedding.
        gate_weights: (B, 2) static/temporal expert weights.
        mask: (B, L) padding mask.
    """

    hazard_logits: torch.Tensor
    hazard_logits_all: torch.Tensor
    count_logits: torch.Tensor
    cold_logits: torch.Tensor
    fused: torch.Tensor
    gate_weights: torch.Tensor
    mask: torch.Tensor

    def __iter__(self):
        """Allow tuple unpacking for backward compatibility."""
        return iter((
            self.hazard_logits,
            self.hazard_logits_all,
            self.count_logits,
            self.cold_logits,
            self.fused,
            self.gate_weights,
            self.mask,
        ))


class StaticExpert(nn.Module):
    """Encodes time-invariant backbone properties."""

    def __init__(self, input_dim: int, static_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.encoder = GatedMLP(input_dim, static_dim * 2, static_dim, dropout=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TemporalExpert(nn.Module):
    """GRU over per-snapshot features with position encoding.

    Returns ALL hidden states (B, L, D) for per-timestep heads,
    AND a pooled representation (B, D) for final prediction.

    Supports temporal masking for cold-start robustness:
    when ``temporal_mask[i]`` is True, the GRU output for sample i
    is replaced with a learned ``null_embed`` vector, simulating
    "no temporal history available".
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 192, num_layers: int = 2, max_seq_len: int = 50, dropout: float = 0.15
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        # Learned position embeddings for temporal awareness
        self.pos_embed = nn.Parameter(torch.randn(1, max_seq_len, hidden_dim) * 0.1)
        self.gru = nn.GRU(
            hidden_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        # Learned "no temporal data" embedding for cold-start masking
        self.null_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, temporal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, L, input_dim) snapshot features
            mask: (B, L) padding mask (1 = valid, 0 = pad)
            temporal_mask: optional (B,) bool — True = replace with null_embed
        Returns:
            (B, L, hidden_dim) all hidden states
            (B, hidden_dim) pooled temporal representation
        """
        B, L, _ = x.shape
        h = self.input_proj(x)  # (B, L, hidden_dim)
        # Add position encoding
        h = h + self.pos_embed[:, :L, :]

        lens = mask.sum(dim=1).long()
        if lens.max() > 0:
            packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            h_all, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)
        else:
            h_all, _ = self.gru(h)

        # Attentive pooling for final aggregated representation
        scores = self.attn(h_all).squeeze(-1)  # (B, L)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(scores, dim=-1)
        h_pooled = (h_all * weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden_dim)

        # Replace hidden states with null_embed for temporally masked samples
        # Uses broadcasting instead of explicit expand to avoid extra allocation
        if temporal_mask is not None and temporal_mask.any():
            mask_t = temporal_mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, 1)
            h_all = torch.where(mask_t, self.null_embed.unsqueeze(0), h_all)
            h_pooled = torch.where(
                temporal_mask.unsqueeze(-1), self.null_embed, h_pooled
            )

        return h_all, h_pooled


class ContextGate(nn.Module):
    """Mixture-of-Experts gate."""

    def __init__(self, static_dim: int, temporal_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(static_dim + temporal_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1),
        )

    def forward(self, static: torch.Tensor, temporal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = self.gate(torch.cat([static, temporal], dim=-1))
        fused = weights[:, 0:1] * static + weights[:, 1:2] * temporal
        return fused, weights


class CountHead(nn.Module):
    """Predicts log1p(n_new_countries) from fused embedding.

    2-layer MLP for non-linear count regression.
    """

    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, max(input_dim // 2, 1)),
            nn.ReLU(),
            nn.Linear(max(input_dim // 2, 1), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SovereignX(nn.Module):
    """Sovereign-X Pro: Dual-expert hazard model with per-timestep training.

    Key innovation: predicts hazard at EVERY timestep (not just the last one),
    allowing ALL snapshot targets to contribute to training.

    Args:
        n_static: number of numeric static (backbone-level) features
        n_snapshot: number of per-snapshot (time-varying) features
        taxonomy_vocab_sizes: [n_phyla, n_classes, n_orders, n_families, n_genera]
        taxonomy_embed_dim: per-level embedding dim (default 8, yielding 5*8=40)
        static_dim: static expert output (128)
        temporal_dim: temporal expert projection (128)
        hidden_dim: GRU hidden (192)
        num_layers: GRU layers (2)
        n_hazard: hazard steps (3)
        max_seq_len: maximum sequence length for position embeddings (50)
        dropout: dropout rate
    """

    def __init__(
        self,
        n_static: int,
        n_snapshot: int,
        taxonomy_vocab_sizes: list[int] | None = None,
        taxonomy_embed_dim: int = 8,
        static_dim: int = 128,
        temporal_dim: int = 128,
        hidden_dim: int = 192,
        num_layers: int = 2,
        n_hazard: int = 3,
        max_seq_len: int = 50,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.use_taxonomy = taxonomy_vocab_sizes is not None and len(taxonomy_vocab_sizes) > 0

        if self.use_taxonomy:
            self.taxonomy_encoder = TaxonomyEncoder(taxonomy_vocab_sizes, taxonomy_embed_dim, dropout)
            static_input_dim = n_static + self.taxonomy_encoder.output_dim
        else:
            static_input_dim = n_static

        self.static_expert = StaticExpert(static_input_dim, static_dim, dropout)
        # TemporalExpert now returns ALL hidden states (B, L, D) + pooled (B, D)
        self.temporal_expert = TemporalExpert(n_snapshot, hidden_dim, num_layers, max_seq_len, dropout)
        self.temporal_proj = nn.Linear(hidden_dim, temporal_dim)  # projects pooled output
        self.gate = ContextGate(static_dim, temporal_dim)

        fused_dim = static_dim  # ContextGate outputs static_dim
        self.hazard_proj = MLP([fused_dim, fused_dim // 2], dropout=dropout)
        self.hazard_head = nn.Linear(fused_dim // 2, n_hazard)
        self.count_head = CountHead(fused_dim // 2)

        # === Per-timestep hazard head (GENIUS: trains on ALL snapshots) ===
        # Concatenates static + temporal at each timestep, predicts hazard directly
        # No MoE gate here — simpler, more direct gradient flow to both experts
        self.timestep_proj = nn.Linear(hidden_dim, temporal_dim)
        self.timestep_head = nn.Sequential(
            nn.Linear(static_dim + temporal_dim, static_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_dim // 2, n_hazard),
        )

        # Cold-start head: predicts hazard from static expert only
        self.cold_start_head = ColdStartHead(static_dim, n_hazard, dropout)

    def forward(
        self,
        static: torch.Tensor,
        snapshots: torch.Tensor,
        mask: torch.Tensor,
        taxonomy_idxs: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
    ) -> ModelOutput:
        B, L = snapshots.shape[:2]

        # Static expert
        if self.use_taxonomy and taxonomy_idxs is not None:
            tax_emb = self.taxonomy_encoder(taxonomy_idxs)
            static_input = torch.cat([static, tax_emb], dim=-1)
        else:
            static_input = static
        z_static = self.static_expert(static_input)

        # Temporal expert
        h_all, h_pooled = self.temporal_expert(snapshots, mask, temporal_mask)

        # Per-timestep predictions
        z_static_exp = z_static.unsqueeze(1).expand(-1, L, -1)
        h_all_proj = self.timestep_proj(h_all)
        ts_input = torch.cat([z_static_exp, h_all_proj], dim=-1)
        hazard_logits_all = self.timestep_head(ts_input)

        # Final prediction (pooled)
        h_pooled_proj = self.temporal_proj(h_pooled)
        fused, gate_weights = self.gate(z_static, h_pooled_proj)
        h = self.hazard_proj(fused)
        hazard_logits = self.hazard_head(h)
        count_logits = self.count_head(h)

        # Cold-start head
        cold_logits = self.cold_start_head(z_static)

        return ModelOutput(
            hazard_logits=hazard_logits,
            hazard_logits_all=hazard_logits_all,
            count_logits=count_logits,
            cold_logits=cold_logits,
            fused=fused,
            gate_weights=gate_weights,
            mask=mask,
        )

    def get_embedding(
        self,
        static: torch.Tensor,
        snapshots: torch.Tensor,
        mask: torch.Tensor,
        taxonomy_idxs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.use_taxonomy and taxonomy_idxs is not None:
            tax_emb = self.taxonomy_encoder(taxonomy_idxs)
            static_input = torch.cat([static, tax_emb], dim=-1)
        else:
            static_input = static
        z_static = self.static_expert(static_input)
        _, h_pooled = self.temporal_expert(snapshots, mask)
        h_pooled_proj = self.temporal_proj(h_pooled)
        fused, _ = self.gate(z_static, h_pooled_proj)
        return fused
