"""
Sovereign-X: Clean reusable neural modules. No dead code.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Minimal MLP with configurable depth and dropout."""

    def __init__(
        self, dims: list[int], dropout: float = 0.1, activation: type[nn.Module] = nn.ReLU
    ):
        super().__init__()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedMLP(nn.Module):
    """MLP with sigmoid gating. Used for static expert fusion."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.15):
        super().__init__()
        self.main = MLP([input_dim, hidden_dim, output_dim], dropout=dropout)
        self.gate = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.main(x) * self.gate(x)


class ColdStartHead(nn.Module):
    """Static-only hazard prediction head for cold-start auxiliary loss.

    Takes static expert output and predicts hazard directly,
    bypassing the MoE gate entirely. Forces the static expert
    to learn useful representations on its own.
    """

    def __init__(self, static_dim: int, n_hazard: int = 3, dropout: float = 0.1):
        super().__init__()
        self.net = MLP([static_dim, static_dim // 2, n_hazard], dropout=dropout)

    def forward(self, z_static: torch.Tensor) -> torch.Tensor:
        return self.net(z_static)  # (B, n_hazard)


class PlattScaler(nn.Module):
    """Platt scaling: sigma(a * logits + b).

    More expressive than temperature scaling (which is a special case with b=0).
    Learns both slope ``a`` and intercept ``b`` to calibrate logits.
    Initialized near identity transform (a≈1, b≈0).
    """

    def __init__(self):
        super().__init__()
        self.log_a = nn.Parameter(torch.log(torch.tensor(1.0)), requires_grad=True)
        self.b = nn.Parameter(torch.zeros(1), requires_grad=True)

    @property
    def a(self) -> torch.Tensor:
        return torch.exp(self.log_a)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits * self.a + self.b


class TaxonomyEncoder(nn.Module):
    """Embeds hierarchical taxonomy (phylum, class, order, family, genus).

    Each taxonomic level has its own embedding table.
    Vocab sizes: [n_phyla, n_classes, n_orders, n_families, n_genera]
    Output dim = sum(embed_dims) = 5 * embed_dim
    """

    def __init__(self, vocab_sizes: list[int], embed_dim: int = 8, dropout: float = 0.1):
        super().__init__()
        self.embeddings = nn.ModuleList([nn.Embedding(v, embed_dim, padding_idx=0) for v in vocab_sizes])
        self.dropout = nn.Dropout(dropout)
        self.output_dim = len(vocab_sizes) * embed_dim

    def forward(self, idxs: torch.Tensor) -> torch.Tensor:
        """idxs: (B, 5) or (B, L, 5) — int64 indices.
        Returns: (B, output_dim) or (B, L, output_dim).
        """
        embs = [emb(idxs[..., i]) for i, emb in enumerate(self.embeddings)]
        out = torch.cat(embs, dim=-1)
        return self.dropout(out)
