from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from bio_spread.models.components import (
    CategoricalEncoder,
    ColdStartEncoder,
    CountHead,
    GatedResidualMLP,
    MLP,
    PretrainHead,
    TaxonomyEncoder,
)


@dataclass
class ModelOutput:
    hazard_logits: torch.Tensor
    hazard_logits_all: torch.Tensor
    count_logits: torch.Tensor
    cold_logits: torch.Tensor
    fused: torch.Tensor
    gate_weights: torch.Tensor
    mask: torch.Tensor


class StaticEncoder(nn.Module):
    def __init__(self, input_dim: int, static_dim: int = 128, dropout: float = 0.15):
        super().__init__()
        self.encoder = GatedResidualMLP(
            [input_dim, static_dim * 2, static_dim * 2, static_dim],
            dropout=dropout,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class TemporalEncoder(nn.Module):
    def __init__(
        self, input_dim: int, hidden_dim: int = 192, num_layers: int = 2, max_seq_len: int = 50, dropout: float = 0.15
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(
            hidden_dim, hidden_dim, num_layers=num_layers, batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.null_embed = nn.Parameter(torch.randn(1, hidden_dim) * 0.01)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, temporal_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B, L, _ = x.shape
        h = self.input_proj(x)

        lens = mask.sum(dim=1).long()
        if lens.max() > 0:
            packed = nn.utils.rnn.pack_padded_sequence(h, lens.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, _ = self.gru(packed)
            h_all, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=L)
        else:
            h_all, _ = self.gru(h)

        scores = self.attn(h_all).squeeze(-1)
        scores = scores.masked_fill(~mask.bool(), -1e9)
        weights = F.softmax(scores, dim=-1)
        h_pooled = (h_all * weights.unsqueeze(-1)).sum(dim=1)

        if temporal_mask is not None and temporal_mask.any():
            mask_t = temporal_mask.unsqueeze(1).unsqueeze(-1)
            h_all = torch.where(mask_t, self.null_embed.unsqueeze(0), h_all)
            h_pooled = torch.where(temporal_mask.unsqueeze(-1), self.null_embed, h_pooled)

        return h_all, h_pooled


class FusionGate(nn.Module):
    def __init__(self, static_dim: int, temporal_dim: int, use_cross_attention: bool = False):
        super().__init__()
        self.use_cross_attention = use_cross_attention
        if static_dim != temporal_dim:
            self.temporal_proj = nn.Linear(temporal_dim, static_dim)
        else:
            self.temporal_proj = nn.Identity()

        gate_input_dim = static_dim + static_dim
        if use_cross_attention:
            gate_input_dim += static_dim
            self.cross_attn_q = nn.Linear(static_dim, static_dim)
            self.cross_attn_k = nn.Linear(temporal_dim, static_dim)
            self.cross_attn_v = nn.Linear(temporal_dim, static_dim)

        self.gate = nn.Sequential(
            nn.Linear(gate_input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softmax(dim=-1),
        )

    def forward(
        self,
        static: torch.Tensor,
        temporal: torch.Tensor,
        h_all_proj: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        temporal = self.temporal_proj(temporal)
        gate_input = [static, temporal]

        if self.use_cross_attention and h_all_proj is not None and mask is not None:
            q = self.cross_attn_q(static).unsqueeze(1)
            k = self.cross_attn_k(h_all_proj)
            v = self.cross_attn_v(h_all_proj)
            attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(static.size(-1))
            attn = attn.masked_fill(~mask.bool().unsqueeze(1), -1e9)
            attn = F.softmax(attn, dim=-1)
            context = torch.matmul(attn, v).squeeze(1)
            gate_input.append(context)

        weights = self.gate(torch.cat(gate_input, dim=-1))
        fused = weights[:, 0:1] * static + weights[:, 1:2] * temporal
        return fused, weights


class BioSpreadModel(nn.Module):
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
        use_cross_attention: bool = False,
        categorical_vocab_sizes: dict[str, int] | None = None,
        categorical_embed_dim: int = 16,
    ):
        super().__init__()
        self.use_taxonomy = taxonomy_vocab_sizes is not None and len(taxonomy_vocab_sizes) > 0
        self.use_cross_attention = use_cross_attention

        self.use_categorical = categorical_vocab_sizes is not None and len(categorical_vocab_sizes) > 0
        if self.use_categorical:
            self.categorical_encoder = CategoricalEncoder(
                categorical_vocab_sizes, categorical_embed_dim, dropout
            )
            cat_embed_dim = self.categorical_encoder.output_dim
        else:
            cat_embed_dim = 0

        if self.use_taxonomy:
            self.taxonomy_encoder = TaxonomyEncoder(taxonomy_vocab_sizes, taxonomy_embed_dim, dropout)
            static_input_dim = n_static + self.taxonomy_encoder.output_dim
        else:
            static_input_dim = n_static

        self.static_encoder = StaticEncoder(static_input_dim, static_dim, dropout)
        self.temporal_encoder = TemporalEncoder(n_snapshot, hidden_dim, num_layers, max_seq_len, dropout)
        self.temporal_proj = nn.Linear(hidden_dim, temporal_dim)

        self.gate = FusionGate(static_dim, temporal_dim, use_cross_attention)

        fused_dim = static_dim
        self.hazard_proj = MLP([fused_dim, fused_dim // 2, fused_dim // 2], dropout=dropout)
        self.hazard_head = nn.Linear(fused_dim // 2, n_hazard)
        self.count_head = CountHead(fused_dim // 2)

        self.timestep_head = nn.Sequential(
            nn.Linear(static_dim + temporal_dim, static_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(static_dim // 2, n_hazard),
        )

        tax_embed_dim_total = taxonomy_embed_dim * len(taxonomy_vocab_sizes) if self.use_taxonomy else 0
        self.cold_start_encoder = ColdStartEncoder(
            static_dim=n_static,
            tax_embed_dim=tax_embed_dim_total,
            cat_embed_dim=cat_embed_dim,
            hidden_dim=max(static_dim * 2, 256),
            n_hazard=n_hazard,
            dropout=dropout,
        )

        self.pretrain_head: PretrainHead | None = None

    def enable_pretrain(self, n_features: int, hidden_dim: int = 192):
        self.pretrain_head = PretrainHead(hidden_dim, n_features)

    def forward(
        self,
        static: torch.Tensor,
        snapshots: torch.Tensor,
        mask: torch.Tensor,
        taxonomy_idxs: torch.Tensor | None = None,
        temporal_mask: torch.Tensor | None = None,
        cat_inputs: dict[str, torch.Tensor] | None = None,
    ) -> ModelOutput:
        B, L = snapshots.shape[:2]

        tax_emb = None
        if self.use_taxonomy and taxonomy_idxs is not None:
            tax_emb, _ = self.taxonomy_encoder(taxonomy_idxs)
            static_input = torch.cat([static, tax_emb], dim=-1)
        else:
            static_input = static
        z_static = self.static_encoder(static_input)

        h_all, h_pooled = self.temporal_encoder(snapshots, mask, temporal_mask)

        z_static_exp = z_static.unsqueeze(1).expand(-1, L, -1)
        h_all_proj = self.temporal_proj(h_all)
        ts_input = torch.cat([z_static_exp, h_all_proj], dim=-1)
        hazard_logits_all = self.timestep_head(ts_input)

        h_pooled_proj = self.temporal_proj(h_pooled)
        fused, gate_weights = self.gate(z_static, h_pooled_proj, h_all_proj, mask)
        h = self.hazard_proj(fused)
        hazard_logits = self.hazard_head(h)
        count_logits = self.count_head(h)

        cold_input_parts = [static]
        if tax_emb is not None:
            cold_input_parts.append(tax_emb)
        if self.use_categorical and cat_inputs is not None:
            cat_emb = self.categorical_encoder(cat_inputs)
            cold_input_parts.append(cat_emb)
        cold_input = torch.cat(cold_input_parts, dim=-1)
        cold_logits = self.cold_start_encoder(cold_input)

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
        temporal_mask: torch.Tensor | None = None,
        cat_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.use_taxonomy and taxonomy_idxs is not None:
            tax_emb, _ = self.taxonomy_encoder(taxonomy_idxs)
            static_input = torch.cat([static, tax_emb], dim=-1)
        else:
            static_input = static
        z_static = self.static_encoder(static_input)
        _, h_pooled = self.temporal_encoder(snapshots, mask, temporal_mask)
        h_pooled_proj = self.temporal_proj(h_pooled)
        fused, _ = self.gate(z_static, h_pooled_proj)
        return fused
