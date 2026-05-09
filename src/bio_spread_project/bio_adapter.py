from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch


class BioAdapter(torch.nn.Module):  # type: ignore[misc]
    """
    Multitask adapter for sequence-safe latent extraction.
    """
    def __init__(self, input_dim: int = 1280, hidden: int = 64, latent: int = 16) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden, latent)
        )
    def forward(self, x: torch.Tensor) -> Any:
        return self.net(x)

def load_bio_adapter(model_path: Path) -> BioAdapter:
    model = BioAdapter()
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def transform_bio_features(features_df: pl.DataFrame, esm_df: pl.DataFrame, adapter: BioAdapter) -> pl.DataFrame:
    """
    Applies time-restricted adapter inference.
    """
    if esm_df.is_empty():
        cols = [f'bio_adapt_{i}' for i in range(16)]
        return features_df.with_columns([pl.lit(0.0).alias(c) for c in cols])

    embed_cols = [c for c in esm_df.columns if c.startswith('esm2_')]
    join_df = features_df.join(esm_df, on='backbone_id', how='inner')

    if join_df.is_empty():
        cols = [f'bio_adapt_{i}' for i in range(16)]
        return features_df.with_columns([pl.lit(0.0).alias(c) for c in cols])

    X = join_df.select(embed_cols).to_numpy().astype(np.float32)
    with torch.no_grad():
        latent = adapter(torch.from_numpy(X)).numpy()

    latent_df = pl.DataFrame({'backbone_id': join_df['backbone_id']})
    for i in range(latent.shape[1]):
        latent_df = latent_df.with_columns(pl.Series(f'bio_adapt_{i}', latent[:, i]))

    out = features_df.join(latent_df, on='backbone_id', how='left')
    for i in range(16):
        c = f'bio_adapt_{i}'
        if c in out.columns:
            out = out.with_columns(pl.col(c).fill_null(0.0))
    return out
