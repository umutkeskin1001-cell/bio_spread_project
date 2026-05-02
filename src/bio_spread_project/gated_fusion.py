from __future__ import annotations

import torch
import torch.nn as nn


class KnownnessGatedFusion(nn.Module):  # type: ignore[misc]
    def __init__(self, hist_dim: int, intrin_dim: int, latent_dim: int = 64):
        super().__init__()
        self.hist_dim = hist_dim
        self.intrin_dim = intrin_dim
        self.gate_net = nn.Sequential(nn.Linear(1, hist_dim + intrin_dim), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Linear(hist_dim + intrin_dim, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
        )

    def forward(self, hist_feat: torch.Tensor, intrin_feat: torch.Tensor, knownness: torch.Tensor) -> torch.Tensor:
        gate_weights = self.gate_net(knownness)
        hist_w = gate_weights[:, : self.hist_dim] * hist_feat
        intrin_w = gate_weights[:, self.hist_dim :] * intrin_feat
        fused = torch.cat([hist_w, intrin_w], dim=1)
        return self.fusion(fused)
