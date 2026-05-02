from __future__ import annotations

import torch
import torch.nn as nn

from bio_spread_project.gated_fusion import KnownnessGatedFusion


class BioSpreadJointEncoder(nn.Module):  # type: ignore[misc]
    def __init__(self, hist_dim: int, intrin_dim: int, latent_dim: int = 64):
        super().__init__()
        self.gated_fusion = KnownnessGatedFusion(hist_dim, intrin_dim, latent_dim)
        self.projection = nn.Linear(latent_dim, latent_dim)

    def forward(self, hist_feat: torch.Tensor, intrin_feat: torch.Tensor, knownness: torch.Tensor) -> torch.Tensor:
        fused = self.gated_fusion(hist_feat, intrin_feat, knownness)
        return self.projection(fused)


class MultiHeadRiskPredictor(nn.Module):  # type: ignore[misc]
    def __init__(self, encoder: BioSpreadJointEncoder, latent_dim: int, num_countries: int | None = None):
        super().__init__()
        self.encoder = encoder
        self.spread_head = nn.Linear(latent_dim, 1)
        self.host_jump_head = nn.Linear(latent_dim, 1)
        self.impact_head = nn.Linear(latent_dim, 1)
        self.country_head = nn.Linear(latent_dim, num_countries) if num_countries else None

    def forward(self, hist_feat: torch.Tensor, intrin_feat: torch.Tensor, knownness: torch.Tensor) -> dict[str, torch.Tensor | None]:
        latent = self.encoder(hist_feat, intrin_feat, knownness)
        return {
            "spread": self.spread_head(latent),
            "host_jump": self.host_jump_head(latent),
            "impact": torch.nn.functional.softplus(self.impact_head(latent)),
            "country_logits": self.country_head(latent) if self.country_head else None,
        }
